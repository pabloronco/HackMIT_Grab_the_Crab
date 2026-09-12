from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .incident_subgraph import audit_all_incident_seeds, extract_incident_subgraph, load_graph_edges
from .models import Edge, Site
from .world_models import GeneratedWorld, WorldModel, WorldModelContext


@dataclass(frozen=True, slots=True)
class StructuralSignature:
    occupied_count: int
    occupied_fraction: float
    occupied_components: int
    largest_component_fraction: float
    occupied_with_neighbor_fraction: float | None
    median_nearest_neighbor_km: float | None
    mean_habitat_score: float | None


@dataclass(frozen=True, slots=True)
class RealContextBundle:
    context: WorldModelContext
    canonical_rows: tuple[dict[str, Any], ...]
    selected_site_ids: tuple[str, ...]
    seed_site_id: str
    incident_summary: Mapping[str, Any]
    calibration_years: tuple[int, ...]
    reality_check_years: tuple[int, ...]
    habitat_score_by_category: Mapping[str, float]
    habitat_support_by_category: Mapping[str, int]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _site_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(str(value)))
    except ValueError:
        return (1, str(value))


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def read_canonical_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "year", "month", "habitat", "trap_sets", "effort_missing", "detected"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical table missing fields: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            effort_text = str(row.get("trap_sets") or "").strip()
            rows.append(
                {
                    "site_id": str(row["site_id"]).strip(),
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "habitat": str(row.get("habitat") or "").strip(),
                    "trap_sets": None if not effort_text else float(effort_text),
                    "effort_missing": _as_bool(row.get("effort_missing")),
                    "detected": _as_bool(row.get("detected")),
                }
            )
    return rows


def read_real_sites(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "latitude", "longitude", "crabteam_habitat"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Real-site table missing fields: {sorted(missing)}")
        return [
            {
                "site_id": str(row["site_id"]).strip(),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "crabteam_habitat": str(row.get("crabteam_habitat") or "").strip(),
            }
            for row in reader
        ]


def split_years(years: Sequence[int], *, holdout_years: int = 2) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = tuple(sorted(set(int(year) for year in years)))
    if len(unique) < 3:
        raise ValueError("At least three observed years are required for calibration/reality split.")
    holdout = max(1, min(int(holdout_years), len(unique) - 1))
    return unique[:-holdout], unique[-holdout:]


def habitat_detection_proxy(
    canonical_rows: Sequence[Mapping[str, Any]],
    calibration_years: Iterable[int],
) -> tuple[dict[str, float], dict[str, int], dict[str, tuple[int, int]]]:
    """Weak observation-derived habitat proxy; never latent suitability truth."""
    allowed = set(int(year) for year in calibration_years)
    by_site_year: dict[tuple[str, int], dict[str, Any]] = {}
    for row in canonical_rows:
        year = int(row["year"])
        if year not in allowed:
            continue
        value = by_site_year.setdefault(
            (str(row["site_id"]), year), {"habitats": set(), "detected": False}
        )
        habitat = str(row.get("habitat") or "").strip()
        if habitat:
            value["habitats"].add(habitat)
        value["detected"] = bool(value["detected"] or bool(row["detected"]))

    totals: dict[str, int] = defaultdict(int)
    positives: dict[str, int] = defaultdict(int)
    for value in by_site_year.values():
        habitats = sorted(value["habitats"])
        if len(habitats) != 1:
            continue
        habitat = habitats[0]
        totals[habitat] += 1
        positives[habitat] += int(bool(value["detected"]))

    scores = {
        habitat: (positives[habitat] + 1.0) / (totals[habitat] + 2.0)
        for habitat in sorted(totals)
    }
    support = {habitat: totals[habitat] for habitat in sorted(totals)}
    counts = {habitat: (positives[habitat], totals[habitat]) for habitat in sorted(totals)}
    return scores, support, counts


def _local_xy_km(lat: float, lon: float, *, lat0: float, lon0: float) -> tuple[float, float]:
    return (
        (lon - lon0) * 111.32 * math.cos(math.radians(lat0)),
        (lat - lat0) * 110.54,
    )


def _habitat_score(category_text: str, scores: Mapping[str, float]) -> float:
    categories = [part.strip() for part in str(category_text).split("|") if part.strip()]
    values = [float(scores[category]) for category in categories if category in scores]
    if values:
        return sum(values) / len(values)
    return median(scores.values()) if scores else 0.5


def build_real_incident_context(
    canonical_path: Path,
    real_sites_path: Path,
    edges_path: Path,
    *,
    holdout_years: int = 2,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
) -> RealContextBundle:
    canonical_rows = read_canonical_rows(canonical_path)
    real_sites = read_real_sites(real_sites_path)
    edge_rows = load_graph_edges(edges_path)
    site_ids = [str(row["site_id"]) for row in real_sites]

    audit = audit_all_incident_seeds(
        site_ids, edge_rows, max_sites=max_sites, preferred_min_sites=preferred_min_sites
    )
    eligible = [
        row for row in audit["seed_rows"]
        if preferred_min_sites <= int(row["selected_sites"]) <= max_sites
    ]
    if not eligible:
        raise ValueError("No topology-only seed yields the preferred incident-graph size range.")
    seed = min((str(row["seed_site_id"]) for row in eligible), key=_site_sort_key)
    selected_ids, selected_edges, incident_summary = extract_incident_subgraph(
        site_ids,
        edge_rows,
        seed_site_id=seed,
        max_sites=max_sites,
        preferred_min_sites=preferred_min_sites,
    )

    calibration_years, reality_years = split_years(
        [int(row["year"]) for row in canonical_rows], holdout_years=holdout_years
    )
    habitat_scores, habitat_support, _ = habitat_detection_proxy(canonical_rows, calibration_years)

    real_by_id = {str(row["site_id"]): row for row in real_sites}
    lat0 = median(float(real_by_id[site_id]["latitude"]) for site_id in selected_ids)
    lon0 = median(float(real_by_id[site_id]["longitude"]) for site_id in selected_ids)
    sites: list[Site] = []
    for site_id in selected_ids:
        row = real_by_id[site_id]
        x, y = _local_xy_km(float(row["latitude"]), float(row["longitude"]), lat0=lat0, lon0=lon0)
        sites.append(
            Site(
                id=site_id,
                x=x,
                y=y,
                habitat_score=_habitat_score(row["crabteam_habitat"], habitat_scores),
                q_model={"status": "OPEN", "semantics": "effective_protocol_detectability"},
            )
        )

    edges = tuple(
        Edge(
            src=str(row["src"]),
            dst=str(row["dst"]),
            distance=float(row["distance_km"]),
            connectivity_weight=None,
            travel_cost=None,
        )
        for row in selected_edges
    )
    return RealContextBundle(
        context=WorldModelContext(tuple(sites), edges, seed),
        canonical_rows=tuple(canonical_rows),
        selected_site_ids=tuple(selected_ids),
        seed_site_id=seed,
        incident_summary=incident_summary,
        calibration_years=calibration_years,
        reality_check_years=reality_years,
        habitat_score_by_category=habitat_scores,
        habitat_support_by_category=habitat_support,
    )


def _adjacency(context: WorldModelContext) -> dict[str, set[str]]:
    result = {site.id: set() for site in context.sites}
    for edge in context.edges:
        result[edge.src].add(edge.dst)
        result[edge.dst].add(edge.src)
    return result


def _components(nodes: set[str], adjacency: Mapping[str, set[str]]) -> list[set[str]]:
    remaining = set(nodes)
    result: list[set[str]] = []
    while remaining:
        start = min(remaining, key=_site_sort_key)
        remaining.remove(start)
        component = {start}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining and neighbor in nodes:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def structural_signature(
    context: WorldModelContext,
    occupied_by_site: Mapping[str, bool],
) -> StructuralSignature:
    site_by_id = {site.id: site for site in context.sites}
    if set(occupied_by_site) != set(site_by_id):
        raise ValueError("occupied_by_site must align exactly with WorldModelContext sites.")
    occupied = {site_id for site_id, present in occupied_by_site.items() if bool(present)}
    if not occupied:
        return StructuralSignature(0, 0.0, 0, 0.0, None, None, None)

    adjacency = _adjacency(context)
    components = _components(occupied, adjacency)
    with_neighbor = sum(
        any(neighbor in occupied for neighbor in adjacency[site_id]) for site_id in occupied
    )
    nearest: list[float] = []
    if len(occupied) >= 2:
        for site_id in occupied:
            site = site_by_id[site_id]
            nearest.append(
                min(
                    math.hypot(site.x - site_by_id[other].x, site.y - site_by_id[other].y)
                    for other in occupied
                    if other != site_id
                )
            )
    habitat = [float(site_by_id[site_id].habitat_score) for site_id in occupied]
    return StructuralSignature(
        occupied_count=len(occupied),
        occupied_fraction=len(occupied) / len(site_by_id),
        occupied_components=len(components),
        largest_component_fraction=max(len(component) for component in components) / len(occupied),
        occupied_with_neighbor_fraction=with_neighbor / len(occupied),
        median_nearest_neighbor_km=median(nearest) if nearest else None,
        mean_habitat_score=sum(habitat) / len(habitat),
    )


def observed_positive_signature(
    bundle: RealContextBundle,
    *,
    years: Iterable[int] | None = None,
) -> StructuralSignature:
    allowed = None if years is None else set(int(year) for year in years)
    selected = set(bundle.selected_site_ids)
    positive = {
        str(row["site_id"])
        for row in bundle.canonical_rows
        if str(row["site_id"]) in selected
        and (allowed is None or int(row["year"]) in allowed)
        and bool(row["detected"])
    }
    return structural_signature(
        bundle.context,
        {site_id: site_id in positive for site_id in bundle.selected_site_ids},
    )


def effort_summary(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    site_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    selected = None if site_ids is None else set(str(site_id) for site_id in site_ids)
    relevant = [row for row in canonical_rows if selected is None or str(row["site_id"]) in selected]
    effort = [
        float(row["trap_sets"])
        for row in relevant
        if not bool(row["effort_missing"]) and row["trap_sets"] is not None
    ]
    return {
        "known_rows": len(effort),
        "missing_rows": sum(bool(row["effort_missing"]) for row in relevant),
        "min": min(effort) if effort else None,
        "p10": _quantile(effort, 0.10),
        "median": median(effort) if effort else None,
        "p90": _quantile(effort, 0.90),
        "max": max(effort) if effort else None,
    }


def summarize_family(
    model: WorldModel,
    context: WorldModelContext,
    *,
    draws: int,
    seed_start: int,
) -> dict[str, Any]:
    if draws <= 0:
        raise ValueError("draws must be positive.")
    signatures: list[StructuralSignature] = []
    patterns: set[tuple[tuple[str, bool], ...]] = set()
    for offset in range(draws):
        generated: GeneratedWorld = model.sample(context, seed=seed_start + offset)
        signatures.append(structural_signature(context, generated.occupied_by_site))
        patterns.add(tuple(sorted((str(k), bool(v)) for k, v in generated.occupied_by_site.items())))

    def stats(attribute: str) -> dict[str, float | None]:
        values = [
            float(value)
            for signature in signatures
            if (value := getattr(signature, attribute)) is not None
        ]
        return {
            "min": min(values) if values else None,
            "p10": _quantile(values, 0.10),
            "median": median(values) if values else None,
            "p90": _quantile(values, 0.90),
            "max": max(values) if values else None,
        }

    counts = [signature.occupied_count for signature in signatures]
    n_sites = len(context.sites)
    return {
        "family_id": model.family_id,
        "draws": draws,
        "unique_occupancy_patterns": len(patterns),
        "fraction_single_site": sum(count == 1 for count in counts) / draws,
        "fraction_full_graph": sum(count == n_sites for count in counts) / draws,
        "occupied_count": stats("occupied_count"),
        "occupied_components": stats("occupied_components"),
        "largest_component_fraction": stats("largest_component_fraction"),
        "occupied_with_neighbor_fraction": stats("occupied_with_neighbor_fraction"),
        "median_nearest_neighbor_km": stats("median_nearest_neighbor_km"),
        "mean_habitat_score": stats("mean_habitat_score"),
    }


def _outside(value: float | None, stats: Mapping[str, Any]) -> bool:
    if value is None or stats.get("p10") is None or stats.get("p90") is None:
        return False
    return not (float(stats["p10"]) <= float(value) <= float(stats["p90"]))


def simulator_criticism_report(
    bundle: RealContextBundle,
    models: Sequence[WorldModel],
    *,
    draws_per_family: int = 200,
    seed_start: int = 10_000,
) -> dict[str, Any]:
    if not models:
        raise ValueError("At least one world model is required.")

    observed_all = observed_positive_signature(bundle)
    observed_reality = observed_positive_signature(bundle, years=bundle.reality_check_years)
    families: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        summary = summarize_family(
            model,
            bundle.context,
            draws=draws_per_family,
            seed_start=seed_start + index * 100_000,
        )
        warnings: list[str] = []
        if summary["unique_occupancy_patterns"] < max(3, draws_per_family // 20):
            warnings.append("low_seed_variation")
        if float(summary["fraction_single_site"]) > 0.90:
            warnings.append("mostly_single_site_worlds")
        if float(summary["fraction_full_graph"]) > 0.90:
            warnings.append("mostly_full_graph_worlds")
        p90_count = summary["occupied_count"].get("p90")
        if p90_count is not None and observed_all.occupied_count > float(p90_count):
            warnings.append("synthetic_extent_p90_below_all_year_observed_positive_sites")
        if _outside(observed_all.occupied_with_neighbor_fraction, summary["occupied_with_neighbor_fraction"]):
            warnings.append("observed_positive_cohesion_outside_synthetic_p10_p90")
        if _outside(observed_all.median_nearest_neighbor_km, summary["median_nearest_neighbor_km"]):
            warnings.append("observed_positive_spacing_outside_synthetic_p10_p90")
        summary["warnings"] = warnings
        families.append(summary)

    scores, _, counts = habitat_detection_proxy(bundle.canonical_rows, bundle.calibration_years)
    if scores != dict(bundle.habitat_score_by_category):
        raise RuntimeError("Habitat proxy changed between context construction and criticism report.")

    return {
        "status": "SIMULATOR_CRITICISM_REALITY_CHECK_NOT_FIELD_VALIDATION",
        "incident_context": {
            "topology_only_seed": bundle.seed_site_id,
            "sites": len(bundle.selected_site_ids),
            "edges": len(bundle.context.edges),
            "selected_site_ids": list(bundle.selected_site_ids),
            "selection_inputs": list(bundle.incident_summary["selection_inputs"]),
            "forbidden_inputs": list(bundle.incident_summary["forbidden_inputs"]),
        },
        "real_data_split": {
            "calibration_years": list(bundle.calibration_years),
            "reality_check_years": list(bundle.reality_check_years),
            "rule": "last two observed years held out from habitat-proxy construction",
            "status": "CURRENT_DEFAULT_FOR_SIMULATOR_CRITICISM",
        },
        "real_effort": effort_summary(bundle.canonical_rows, site_ids=bundle.selected_site_ids),
        "habitat_proxy": {
            "semantics": "Beta(1,1)-smoothed site-year observed-detection probability on calibration years; not latent habitat suitability",
            "scores": dict(bundle.habitat_score_by_category),
            "support_site_years": dict(bundle.habitat_support_by_category),
            "positive_over_total_site_years": {
                habitat: {"positive": positive, "total": total}
                for habitat, (positive, total) in counts.items()
            },
        },
        "observed_positive_pattern_all_years": asdict(observed_all),
        "observed_positive_pattern_reality_years": asdict(observed_reality),
        "families": families,
        "interpretation": [
            "Observed positive sites are imperfect-detection outcomes and therefore are not complete occupancy truth.",
            "All-year positive-site count is a descriptive lower-bound-style anchor, not a target prevalence for a single synthetic incident.",
            "Warnings are criticism prompts, not automatic parameter fitting or proof that a family is realistic/unrealistic.",
            "Do not tune simulator ranges to make any planner look good.",
            "Hidden synthetic truth remains available only for simulator/evaluator metrics, never planner inputs.",
        ],
    }
