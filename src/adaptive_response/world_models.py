from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, log
from random import Random
from statistics import median
from typing import Mapping, Protocol, Sequence

from .models import Edge, HiddenWorld, Site
from .spatial_belief import EcologicalHypothesis


@dataclass(frozen=True, slots=True)
class WorldModelContext:
    """Static observable context supplied to synthetic ecological world models.

    The context contains monitoring-site geometry/metadata and the known initial
    detection only. It deliberately contains no future observations and no latent
    occupancy. World models use it to create synthetic hidden incidents.
    """

    sites: tuple[Site, ...]
    edges: tuple[Edge, ...]
    initial_detection: str

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("WorldModelContext requires at least one site.")
        site_ids = [site.id for site in self.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("WorldModelContext site ids must be unique.")
        if self.initial_detection not in set(site_ids):
            raise ValueError("initial_detection must reference a context site.")
        known = set(site_ids)
        for edge in self.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError("WorldModelContext edge references an unknown site.")


@dataclass(frozen=True, slots=True)
class GeneratedWorld:
    """One synthetic latent incident plus transparent generator provenance."""

    family_id: str
    occupied_by_site: Mapping[str, bool]
    seed: int
    generator_parameters: Mapping[str, float | int | str] = field(default_factory=dict)

    def as_hidden_world(self) -> HiddenWorld:
        return HiddenWorld(
            occupied_by_site=dict(self.occupied_by_site),
            generator_parameters={
                "family_id": self.family_id,
                "seed": self.seed,
                **dict(self.generator_parameters),
            },
        )


class WorldModel(Protocol):
    """Shared simulator-family interface; implementations must be seedable."""

    family_id: str

    def sample(self, context: WorldModelContext, *, seed: int) -> GeneratedWorld:
        ...


@dataclass(frozen=True, slots=True)
class GraphDiffusionWorldModel:
    """Family A: occupancy grows along graph adjacency from the initial detection.

    `edge_transmission` and the habitat modifier are model assumptions sampled over
    broad design ranges. Missing/open `connectivity_weight` values are intentionally
    not converted into dispersal probabilities here.
    """

    family_id: str = "A_graph_diffusion"
    transmission_range: tuple[float, float] = (0.35, 0.75)
    habitat_effect_range: tuple[float, float] = (0.0, 0.35)
    wave_range: tuple[int, int] = (1, 4)

    def sample(self, context: WorldModelContext, *, seed: int) -> GeneratedWorld:
        _validate_context(context)
        rng = Random(seed)
        transmission = _uniform(rng, self.transmission_range)
        habitat_effect = _uniform(rng, self.habitat_effect_range)
        waves = rng.randint(*self.wave_range)

        site_by_id = {site.id: site for site in context.sites}
        adjacency = _adjacency(context)
        occupied = {context.initial_detection}
        frontier = {context.initial_detection}

        for _ in range(waves):
            new_frontier: set[str] = set()
            for src in sorted(frontier):
                for dst in sorted(adjacency[src]):
                    if dst in occupied:
                        continue
                    habitat = _clamp01(float(site_by_id[dst].habitat_score))
                    probability = transmission * (1.0 - habitat_effect + habitat_effect * habitat)
                    if rng.random() < _clamp01(probability):
                        new_frontier.add(dst)
            if not new_frontier:
                break
            occupied.update(new_frontier)
            frontier = new_frontier

        return _generated(
            self.family_id,
            context,
            occupied,
            seed,
            {
                "edge_transmission": transmission,
                "habitat_effect": habitat_effect,
                "waves": waves,
            },
        )


@dataclass(frozen=True, slots=True)
class SpatialClusterWorldModel:
    """Family B: distance-decay cluster centered on the known initial detection.

    Radius is defined relative to the incident graph's own geometry rather than as a
    universal ecological distance threshold. `patchiness` randomly drops otherwise
    plausible cluster members.
    """

    family_id: str = "B_spatial_cluster"
    radius_scale_range: tuple[float, float] = (0.65, 1.75)
    patchiness_range: tuple[float, float] = (0.05, 0.35)

    def sample(self, context: WorldModelContext, *, seed: int) -> GeneratedWorld:
        _validate_context(context)
        rng = Random(seed)
        geometry_scale = _geometry_scale(context.sites)
        radius_scale = _uniform(rng, self.radius_scale_range)
        radius = max(geometry_scale * radius_scale, 1e-12)
        patchiness = _uniform(rng, self.patchiness_range)

        center = _site_by_id(context)[context.initial_detection]
        occupied = {context.initial_detection}
        for site in context.sites:
            if site.id == context.initial_detection:
                continue
            distance = _distance(center, site)
            cluster_probability = exp(-0.5 * (distance / radius) ** 2)
            probability = cluster_probability * (1.0 - patchiness)
            if rng.random() < _clamp01(probability):
                occupied.add(site.id)

        return _generated(
            self.family_id,
            context,
            occupied,
            seed,
            {
                "geometry_scale": geometry_scale,
                "radius_scale": radius_scale,
                "patchiness": patchiness,
            },
        )


@dataclass(frozen=True, slots=True)
class HabitatDrivenWorldModel:
    """Family C: habitat suitability dominates occupancy with mild local correlation.

    This is intentionally a structurally different assumption from graph diffusion.
    Habitat scores are treated as model inputs, not as observed occupancy labels.
    """

    family_id: str = "C_habitat_driven"
    prevalence_range: tuple[float, float] = (0.08, 0.35)
    habitat_coefficient_range: tuple[float, float] = (1.0, 4.0)
    local_boost_range: tuple[float, float] = (0.0, 1.25)

    def sample(self, context: WorldModelContext, *, seed: int) -> GeneratedWorld:
        _validate_context(context)
        rng = Random(seed)
        prevalence = _uniform(rng, self.prevalence_range)
        coefficient = _uniform(rng, self.habitat_coefficient_range)
        local_boost = _uniform(rng, self.local_boost_range)

        habitat_values = [_clamp01(float(site.habitat_score)) for site in context.sites]
        habitat_mean = sum(habitat_values) / len(habitat_values)
        adjacency = _adjacency(context)
        baseline_logit = log(prevalence / (1.0 - prevalence))

        occupied = {context.initial_detection}
        for site in context.sites:
            if site.id == context.initial_detection:
                continue
            habitat = _clamp01(float(site.habitat_score))
            near_initial = context.initial_detection in adjacency[site.id]
            linear = baseline_logit + coefficient * (habitat - habitat_mean)
            if near_initial:
                linear += local_boost
            probability = 1.0 / (1.0 + exp(-linear))
            if rng.random() < probability:
                occupied.add(site.id)

        return _generated(
            self.family_id,
            context,
            occupied,
            seed,
            {
                "prevalence": prevalence,
                "habitat_coefficient": coefficient,
                "local_boost": local_boost,
            },
        )


@dataclass(frozen=True, slots=True)
class FragmentedPatchyWorldModel:
    """Family E: structurally held-out fragmented/patchy alternative.

    Multiple geographic patch anchors may create occupancy islands that do not follow
    graph-diffusion assumptions. This family is the CURRENT DEFAULT OOD-model holdout,
    not a claim that real invasions follow this mechanism.
    """

    family_id: str = "E_fragmented_patchy"
    extra_patch_range: tuple[int, int] = (1, 2)
    patch_radius_scale_range: tuple[float, float] = (0.25, 0.65)
    dropout_range: tuple[float, float] = (0.15, 0.45)

    def sample(self, context: WorldModelContext, *, seed: int) -> GeneratedWorld:
        _validate_context(context)
        rng = Random(seed)
        geometry_scale = _geometry_scale(context.sites)
        radius_scale = _uniform(rng, self.patch_radius_scale_range)
        radius = max(geometry_scale * radius_scale, 1e-12)
        dropout = _uniform(rng, self.dropout_range)
        extra_patches = rng.randint(*self.extra_patch_range)

        initial = _site_by_id(context)[context.initial_detection]
        candidates = [site for site in context.sites if site.id != context.initial_detection]
        candidates.sort(key=lambda site: _distance(initial, site), reverse=True)
        remote_pool_size = max(extra_patches, max(1, len(candidates) // 2))
        anchor_pool = candidates[:remote_pool_size]
        anchors = [initial]
        if anchor_pool:
            anchors.extend(rng.sample(anchor_pool, k=min(extra_patches, len(anchor_pool))))

        occupied = {context.initial_detection}
        for site in context.sites:
            if site.id == context.initial_detection:
                continue
            nearest = min(_distance(anchor, site) for anchor in anchors)
            patch_probability = exp(-0.5 * (nearest / radius) ** 2) * (1.0 - dropout)
            if rng.random() < _clamp01(patch_probability):
                occupied.add(site.id)

        return _generated(
            self.family_id,
            context,
            occupied,
            seed,
            {
                "geometry_scale": geometry_scale,
                "patch_radius_scale": radius_scale,
                "dropout": dropout,
                "extra_patches": extra_patches,
            },
        )


def default_world_model_split() -> dict[str, tuple[WorldModel, ...]]:
    """Return the CURRENT DEFAULT family split without freezing numeric parameters."""
    return {
        "train": (
            GraphDiffusionWorldModel(),
            SpatialClusterWorldModel(),
            HabitatDrivenWorldModel(),
        ),
        "ood_model_holdout": (FragmentedPatchyWorldModel(),),
    }


def sample_ecological_hypotheses(
    models: Sequence[WorldModel],
    context: WorldModelContext,
    *,
    draws_per_model: int,
    seed: int,
) -> list[EcologicalHypothesis]:
    """Sample/deduplicate plausible worlds for the explicit spatial belief engine.

    Duplicate occupancy maps are collapsed and their empirical frequency becomes the
    prior weight. Which model families are supplied is a caller decision; notably the
    OOD holdout can be excluded from the inference/training ensemble.
    """
    if not models:
        raise ValueError("At least one world model is required.")
    if isinstance(draws_per_model, bool) or not isinstance(draws_per_model, int) or draws_per_model <= 0:
        raise ValueError("draws_per_model must be a positive integer.")

    counts: dict[tuple[tuple[str, bool], ...], int] = {}
    labels: dict[tuple[tuple[str, bool], ...], set[str]] = {}
    draw_seed = int(seed)
    for model_index, model in enumerate(models):
        for draw_index in range(draws_per_model):
            world_seed = draw_seed + model_index * 100_003 + draw_index
            generated = model.sample(context, seed=world_seed)
            key = tuple(sorted((str(site_id), bool(value)) for site_id, value in generated.occupied_by_site.items()))
            counts[key] = counts.get(key, 0) + 1
            labels.setdefault(key, set()).add(generated.family_id)

    return [
        EcologicalHypothesis(
            presence_by_site=dict(key),
            prior_weight=float(counts[key]),
            label="|".join(sorted(labels[key])),
        )
        for key in sorted(counts)
    ]


def _validate_context(context: WorldModelContext) -> None:
    # Re-run dataclass contract for callers that construct/deserialize unusual values.
    if not context.sites:
        raise ValueError("WorldModelContext requires at least one site.")
    known = {site.id for site in context.sites}
    if context.initial_detection not in known:
        raise ValueError("initial_detection must reference a context site.")


def _generated(
    family_id: str,
    context: WorldModelContext,
    occupied: set[str],
    seed: int,
    parameters: Mapping[str, float | int | str],
) -> GeneratedWorld:
    occupied.add(context.initial_detection)
    return GeneratedWorld(
        family_id=family_id,
        occupied_by_site={site.id: site.id in occupied for site in context.sites},
        seed=seed,
        generator_parameters=dict(parameters),
    )


def _site_by_id(context: WorldModelContext) -> dict[str, Site]:
    return {site.id: site for site in context.sites}


def _adjacency(context: WorldModelContext) -> dict[str, set[str]]:
    adjacency = {site.id: set() for site in context.sites}
    for edge in context.edges:
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)
    return adjacency


def _distance(a: Site, b: Site) -> float:
    return hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def _geometry_scale(sites: Sequence[Site]) -> float:
    distances = [
        _distance(a, b)
        for index, a in enumerate(sites)
        for b in sites[index + 1 :]
        if _distance(a, b) > 0.0
    ]
    return median(distances) if distances else 1.0


def _uniform(rng: Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low > high:
        raise ValueError("World-model parameter range lower bound exceeds upper bound.")
    return rng.uniform(float(low), float(high))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
