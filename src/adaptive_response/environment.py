from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping

import networkx as nx
import numpy as np

from .models import (
    HiddenWorld,
    IncidentConfig,
    MissionAction,
    Observation,
    ObservationBatch,
    PublicState,
    Site,
)
from .world_models import WorldModel, WorldModelContext


class Environment:
    """Field-simulation environment with a strict hidden-truth boundary.

    Backwards-compatible default behavior still uses the original toy hidden-world
    generator and each site's scalar ``q_model`` as simulator q. For formal R6+
    benchmark work callers may instead inject:

    - ``world_model``: one explicit synthetic ecological family implementing the
      shared ``WorldModel`` protocol;
    - ``q_true``: a hidden simulator-only scalar or per-site map;
    - ``observation_uniform_by_site``: optional simulator/evaluator-only fixed
      uniforms used for paired experiments. It is never exposed to planners,
      observations, PublicState, or GraphState.

    This separation matters because the inference engine must not be handed the
    simulator's true q. Public observations therefore never include the q value used
    to generate them. Belief-side q assumptions/posteriors live outside Environment.
    """

    def __init__(
        self,
        config: IncidentConfig,
        *,
        world_model: WorldModel | None = None,
        q_true: float | Mapping[str, float] | None = None,
        observation_uniform_by_site: Mapping[str, float] | None = None,
    ) -> None:
        self._validate_config(config)
        self._config = deepcopy(config)
        self._world_model = world_model
        self._q_true_by_site = self._normalize_q_true(q_true, self._config.sites)
        self._observation_uniform_by_site = self._normalize_observation_uniforms(
            observation_uniform_by_site, self._config.sites
        )
        self._rng: np.random.Generator | None = None
        self._hidden_world: HiddenWorld | None = None
        self._public_state: PublicState | None = None
        self._reveal_allowed = False

    def reset(self, seed: int | None = None) -> PublicState:
        """Reset the incident and return only observable state.

        Same seed + same config/model produces the same hidden world. The returned
        object never contains latent occupancy or simulator q_true. Reveal is locked
        again on every reset and can only be enabled by the orchestration layer.
        """

        resolved_seed = self._config.seed if seed is None else seed
        self._rng = np.random.default_rng(resolved_seed)
        if self._world_model is None:
            self._hidden_world = self._generate_toy_hidden_world(self._rng)
        else:
            context = WorldModelContext(
                sites=tuple(deepcopy(self._config.sites)),
                edges=tuple(deepcopy(self._config.edges)),
                initial_detection=self._config.initial_detection,
            )
            generated = self._world_model.sample(context, seed=resolved_seed)
            expected = {site.id for site in self._config.sites}
            if set(generated.occupied_by_site) != expected:
                raise ValueError("WorldModel output must align exactly with IncidentConfig sites.")
            if not bool(generated.occupied_by_site[self._config.initial_detection]):
                raise ValueError("WorldModel must keep the confirmed initial detection occupied.")
            self._hidden_world = generated.as_hidden_world()
        self._reveal_allowed = False

        public_sites = [self._reset_site(site) for site in self._config.sites]
        self._public_state = PublicState(
            sites=public_sites,
            edges=deepcopy(self._config.edges),
            initial_detection=self._config.initial_detection,
            remaining_budget=self._config.budget,
            round=0,
            teams=self._config.teams,
            protocol=self._config.protocol,
            seed=resolved_seed,
            world_model_id=(
                self._config.world_model_id
                if self._config.world_model_id is not None
                else getattr(self._world_model, "family_id", None)
            ),
        )
        return deepcopy(self._public_state)

    def step(
        self, action: MissionAction
    ) -> tuple[ObservationBatch, bool, dict[str, Any]]:
        """Execute one field mission against the hidden incident.

        CURRENT DEFAULT: one effort unit consumes one budget unit. Multiple
        allocations to the same site are aggregated before simulating a single
        field return for that site.

        Detection semantics remain explicit:
        P(detection | occupied, effort=e) = 1 - (1-q_true)^e.
        An unoccupied site cannot generate a false positive in the MVP.

        ``q_true`` is simulator-only and is intentionally absent from Observation
        metadata and planner-facing state.
        """

        self._require_reset()
        assert self._public_state is not None
        assert self._hidden_world is not None
        assert self._rng is not None

        effort_by_site = self._validate_and_aggregate_action(action)
        next_round = self._public_state.round + 1
        observations: list[Observation] = []

        site_lookup = {site.id: site for site in self._public_state.sites}
        for site_id, effort in effort_by_site.items():
            site = site_lookup[site_id]
            q_true = self._resolve_simulator_q(site)
            occupied = self._hidden_world.occupied_by_site[site_id]
            detection_probability = 1.0 - (1.0 - q_true) ** effort if occupied else 0.0
            observation_uniform = (
                float(self._rng.random())
                if self._observation_uniform_by_site is None
                else self._observation_uniform_by_site[site_id]
            )
            detection = bool(observation_uniform < detection_probability)

            observation = Observation(
                site_id=site_id,
                effort=effort,
                detection=detection,
                round=next_round,
                metadata={
                    "protocol": self._public_state.protocol,
                    "observation_model": "binary_effort_imperfect_detection",
                },
            )
            observations.append(observation)

            site.observed_effort += effort
            if detection:
                site.detections += 1
                site.status = "detected"
            else:
                site.status = "surveyed_no_detection"

        self._public_state.remaining_budget -= action.total_cost
        self._public_state.round = next_round

        batch = ObservationBatch(
            observations=tuple(observations),
            round=next_round,
            total_effort=sum(effort_by_site.values()),
        )
        done = self._public_state.remaining_budget == 0
        metrics = {
            "round": next_round,
            "effort_spent": action.total_cost,
            "remaining_budget": self._public_state.remaining_budget,
            "detections": sum(obs.detection for obs in observations),
            "sites_surveyed": len(observations),
        }
        return batch, done, metrics

    @property
    def current_public_state(self) -> PublicState:
        self._require_reset()
        assert self._public_state is not None
        return deepcopy(self._public_state)

    def reveal(self) -> HiddenWorld:
        """Return latent truth only after the state-machine layer unlocks reveal.

        This method is evaluator/demo-only. Planner code must never receive the
        Environment object or call this path.
        """

        self._require_reset()
        if not self._reveal_allowed:
            raise RuntimeError("Hidden truth reveal is not allowed in the current state.")
        assert self._hidden_world is not None
        return deepcopy(self._hidden_world)

    def _allow_reveal(self) -> None:
        """Internal hook used by the orchestration layer when the episode ends."""

        self._require_reset()
        self._reveal_allowed = True

    def _validate_and_aggregate_action(self, action: MissionAction) -> dict[str, int]:
        assert self._public_state is not None

        if action.total_cost < 0:
            raise ValueError("MissionAction.total_cost cannot be negative.")
        if action.total_cost > self._public_state.remaining_budget:
            raise ValueError("MissionAction exceeds remaining budget.")

        known_sites = {site.id for site in self._public_state.sites}
        effort_by_site: defaultdict[str, int] = defaultdict(int)
        for allocation in action.allocations:
            if allocation.site_id not in known_sites:
                raise ValueError(
                    f"Mission allocation references unknown site {allocation.site_id!r}."
                )
            if not isinstance(allocation.effort_units, int) or allocation.effort_units <= 0:
                raise ValueError("effort_units must be a positive integer.")
            effort_by_site[allocation.site_id] += allocation.effort_units

        allocated_effort = sum(effort_by_site.values())
        if allocated_effort != action.total_cost:
            raise ValueError(
                "For the current MVP, MissionAction.total_cost must equal allocated effort units."
            )
        if allocated_effort == 0:
            raise ValueError("MissionAction must allocate positive effort.")

        return dict(effort_by_site)

    @staticmethod
    def _normalize_observation_uniforms(
        values: Mapping[str, float] | None,
        sites: list[Site],
    ) -> dict[str, float] | None:
        if values is None:
            return None

        site_ids = {site.id for site in sites}
        if set(values) != site_ids:
            raise ValueError(
                "observation_uniform_by_site keys must exactly match IncidentConfig sites."
            )

        normalized: dict[str, float] = {}
        for site_id, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("observation uniforms must be numeric.")
            uniform = float(value)
            if not 0.0 <= uniform < 1.0:
                raise ValueError("observation uniforms must be in [0, 1).")
            normalized[site_id] = uniform
        return normalized

    def _resolve_simulator_q(self, site: Site) -> float:
        if self._q_true_by_site is not None:
            return self._q_true_by_site[site.id]
        return self._resolve_public_q_model(site)

    @staticmethod
    def _resolve_public_q_model(site: Site) -> float:
        if isinstance(site.q_model, bool) or not isinstance(site.q_model, (int, float)):
            raise ValueError(
                "Without an explicit simulator q_true, Environment requires scalar site.q_model."
            )
        q = float(site.q_model)
        if not 0.0 <= q <= 1.0:
            raise ValueError("q_model must be between 0 and 1.")
        return q

    @staticmethod
    def _normalize_q_true(
        q_true: float | Mapping[str, float] | None,
        sites: list[Site],
    ) -> dict[str, float] | None:
        if q_true is None:
            return None
        site_ids = {site.id for site in sites}
        if isinstance(q_true, Mapping):
            if set(q_true) != site_ids:
                raise ValueError("Per-site q_true keys must exactly match IncidentConfig sites.")
            values = {site_id: value for site_id, value in q_true.items()}
        else:
            values = {site_id: q_true for site_id in site_ids}

        normalized: dict[str, float] = {}
        for site_id, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("q_true values must be numeric probabilities.")
            q = float(value)
            if not 0.0 <= q <= 1.0:
                raise ValueError("q_true values must be between 0 and 1.")
            normalized[site_id] = q
        return normalized

    def _reset_site(self, site: Site) -> Site:
        public_site = deepcopy(site)
        public_site.observed_effort = 0
        public_site.detections = 0
        public_site.status = "unsurveyed"

        if public_site.id == self._config.initial_detection:
            # The product starts after a confirmed first detection. We represent
            # that fact explicitly without inventing effort for the pre-incident
            # confirmation protocol.
            public_site.detections = 1
            public_site.status = "confirmed_detection"

        return public_site

    def _generate_toy_hidden_world(
        self, rng: np.random.Generator
    ) -> HiddenWorld:
        """Generate a simple connected-ish latent footprint for the legacy MVP kernel.

        MODEL ASSUMPTION: nearby nodes around the confirmed detection are more
        likely to be occupied. This path is deliberately legacy/testing-only; formal
        benchmark work should inject one of the explicit multiple world-model families.
        """

        graph = nx.Graph()
        for site in self._config.sites:
            graph.add_node(site.id)
        for edge in self._config.edges:
            graph.add_edge(edge.src, edge.dst)

        distances = nx.single_source_shortest_path_length(
            graph, self._config.initial_detection
        )
        radius = int(rng.integers(1, 4))

        occupied: dict[str, bool] = {}
        for site in self._config.sites:
            if site.id == self._config.initial_detection:
                occupied[site.id] = True
                continue

            distance = distances.get(site.id)
            if distance is None or distance > radius:
                occupied[site.id] = False
                continue

            habitat = float(np.clip(site.habitat_score, 0.0, 1.0))
            probability = float(
                np.clip(0.72 - 0.16 * distance + 0.18 * habitat, 0.05, 0.95)
            )
            occupied[site.id] = bool(rng.random() < probability)

        return HiddenWorld(
            occupied_by_site=occupied,
            generator_parameters={
                "family": "toy_graph_cluster_m1",
                "radius": radius,
            },
        )

    def _require_reset(self) -> None:
        if self._public_state is None or self._hidden_world is None or self._rng is None:
            raise RuntimeError("Environment has not been reset yet.")

    @staticmethod
    def _validate_config(config: IncidentConfig) -> None:
        if not config.sites:
            raise ValueError("IncidentConfig must contain at least one site.")

        site_ids = [site.id for site in config.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("Site ids must be unique.")

        if config.initial_detection not in set(site_ids):
            raise ValueError("initial_detection must reference an existing site.")

        if config.budget < 0:
            raise ValueError("budget cannot be negative.")

        if config.teams < 1:
            raise ValueError("teams must be at least 1.")

        known = set(site_ids)
        for edge in config.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError(
                    f"Edge {edge.src!r}->{edge.dst!r} references an unknown site."
                )
            if edge.distance < 0:
                raise ValueError("edge distance cannot be negative.")
