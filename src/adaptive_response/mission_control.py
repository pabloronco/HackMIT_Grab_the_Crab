from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from typing import Any

from .environment import Environment
from .graph_state import NODE_FEATURE_NAMES
from .mission_loop import LoopPhase, RoundTransition
from .models import HiddenWorld, MissionAction
from .planners import FrontierPlanner, Planner
from .real_incident_source import build_real_incident, eligible_incident_seed_sites
from .spatial_belief import QHypothesis
from .spatial_mission_loop import SpatialAdaptiveMissionLoop
from .world_models import WorldModelContext, default_world_model_split, sample_ecological_hypotheses

# CURRENT DEFAULT parameters, mirrored from scripts/train_spatial_gnn_policy.py so the
# UI demo runs the same real-incident/belief/action-space configuration already used
# for training and evaluation, not a UI-invented scenario.
_BUDGET = 18
_INCIDENT_MAX_SITES = 20
_INCIDENT_PREFERRED_MIN_SITES = 12
_DRAWS_PER_MODEL = 60
_Q_BELIEF_VALUES = (0.05, 0.10, 0.20)
_EFFORT_LEVELS = (1, 3, 6)
_TOP_PROPAGATED_CHANGES = 5

# Keep the UI's seed input well clear of the belief-seed namespace offset below,
# so a truth seed and a derived belief seed can never collide (see
# _derive_belief_seed).
_SEED_MODULUS = 1_000_000

_BELIEF_SEED_NAMESPACE_OFFSET = 10_000_000_000


def _derive_belief_seed(case_id: str, world_seed: int) -> int:
    """Hash-derive a belief-ensemble sample seed, disjoint from the truth seed.

    Duplicated from rl/r8_manifest_cases.py's derive_belief_seed (same formula,
    same namespace offset) rather than imported, for the same reason
    real_incident_source.py re-derives build_real_incident_case: importing
    anything under adaptive_response.rl pulls in that package's eager
    GNN/torch imports, which a product web app must not require. If truth and
    belief hypotheses were sampled from the same seed, sample_ecological_hypotheses's
    first draw (model_index=0, draw_index=0) reuses the seed unmodified, so it
    can reproduce the exact hidden world as one of the belief ensemble's own
    hypotheses - a real truth/belief independence violation, not a style
    choice (this is exactly the bug the R8 benchmark review fixed).
    """
    digest = hashlib.sha256(f"ui-belief-ensemble::{case_id}::{world_seed}".encode("utf-8")).hexdigest()
    return _BELIEF_SEED_NAMESPACE_OFFSET + (int(digest[:16], 16) % 1_000_000_000)


class MissionControlSession:
    """Thin product adapter over the real spatial adaptive mission loop.

    The UI consumes only observable/public state exposed here. No ecological
    decision rule is duplicated in the frontend: missions come from the active
    Planner, evidence comes from Environment.step, and beliefs come from the
    explicit spatial Bayes ensemble (SpatialBeliefEngine). Hidden truth appears
    in snapshots only after the explicit reveal gate.

    Incidents are real monitoring-site graphs (real_graph_v0), not a toy
    scenario - see real_incident_source.py. The ecological hypothesis ensemble
    is the same synthetic world-model family split used for training/
    benchmarking (world_models.default_world_model_split()); it is a belief
    prior, not ground truth.
    """

    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner or FrontierPlanner(max_sites=1, effort_levels=_EFFORT_LEVELS)
        self._eligible_seed_sites = sorted(
            eligible_incident_seed_sites(
                max_sites=_INCIDENT_MAX_SITES,
                preferred_min_sites=_INCIDENT_PREFERRED_MIN_SITES,
            )
        )
        if not self._eligible_seed_sites:
            raise RuntimeError("No eligible real incident seed sites found in the committed graph.")
        self._loop: SpatialAdaptiveMissionLoop | None = None
        self._mission: MissionAction | None = None
        self._last_transition: RoundTransition | None = None
        self._revealed_world: HiddenWorld | None = None
        self._events: list[dict[str, Any]] = []
        self._seed = 0
        self._seed_site_id = ""
        self._budget = _BUDGET
        self.reset(seed=0)

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        resolved_seed = (self._seed if seed is None else int(seed)) % _SEED_MODULUS
        self._seed = resolved_seed
        rng = random.Random(resolved_seed)

        seed_site_id = rng.choice(self._eligible_seed_sites)
        self._seed_site_id = seed_site_id

        real_incident = build_real_incident(
            seed_site_id,
            budget=_BUDGET,
            max_sites=_INCIDENT_MAX_SITES,
            preferred_min_sites=_INCIDENT_PREFERRED_MIN_SITES,
            rl_seed=resolved_seed,
        )
        self._budget = real_incident.incident.budget

        train_models = default_world_model_split()["train"]
        world_model = rng.choice(train_models)
        q_true = rng.choice(_Q_BELIEF_VALUES)

        context = WorldModelContext(
            tuple(real_incident.incident.sites), tuple(real_incident.incident.edges), seed_site_id,
        )
        belief_seed = _derive_belief_seed(f"ui/{seed_site_id}", resolved_seed)
        hypotheses = sample_ecological_hypotheses(
            train_models, context, draws_per_model=_DRAWS_PER_MODEL, seed=belief_seed,
        )
        q_hypotheses = [QHypothesis(q) for q in _Q_BELIEF_VALUES]

        self._loop = SpatialAdaptiveMissionLoop(
            Environment(real_incident.incident, world_model=world_model, q_true=q_true),
            self._planner,
            hypotheses,
            q_hypotheses,
        )
        self._loop.reset(seed=resolved_seed)
        self._mission = None
        self._last_transition = None
        self._revealed_world = None
        self._events = [
            {
                "kind": "incident",
                "round": 0,
                "title": "New confirmed detection",
                "detail": (
                    f"Real monitoring-site incident at site {seed_site_id} "
                    f"({len(real_incident.incident.sites)} sites in the response graph)."
                ),
            }
        ]
        return self.snapshot()

    def plan(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.READY_TO_PLAN:
            raise RuntimeError(f"Cannot plan while loop phase is {loop.phase.value!r}.")
        self._mission = loop.plan_next()
        self._events.append(
            {
                "kind": "mission",
                "round": loop.current_public_state.round,
                "title": "Mission generated",
                "detail": self._mission_text(self._mission),
            }
        )
        return self.snapshot()

    def execute(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.MISSION_PLANNED:
            raise RuntimeError("A mission must be planned before it can be executed.")
        previous_mission = self._mission
        transition = loop.execute_pending()
        self._last_transition = transition
        self._events.append(
            {
                "kind": "return",
                "round": transition.observations.round,
                "title": "Field return received",
                "detail": self._observation_text(transition),
            }
        )

        if transition.next_mission is not None:
            self._mission = transition.next_mission
            self._events.append(
                {
                    "kind": "replan",
                    "round": transition.observations.round,
                    "title": "Mission updated",
                    "detail": self._mission_text(transition.next_mission),
                }
            )
        else:
            self._mission = None
            self._events.append(
                {
                    "kind": "complete",
                    "round": transition.observations.round,
                    "title": "Field budget exhausted",
                    "detail": "Reveal gate is now available for evaluator/demo use.",
                }
            )

        if previous_mission is None:
            raise RuntimeError("Mission-control adapter lost the pending mission reference.")
        return self.snapshot()

    def reveal(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.COMPLETE:
            raise RuntimeError("True extent can only be revealed after mission completion.")
        self._revealed_world = loop.reveal()
        self._events.append(
            {
                "kind": "reveal",
                "round": loop.current_public_state.round,
                "title": "True extent revealed",
                "detail": "Evaluator/demo-only latent occupancy is now visible.",
            }
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        loop = self._require_loop()
        public = loop.current_public_state
        belief = loop.current_belief
        graph = loop.current_graph_state
        spatial = loop.current_spatial_belief

        feature_index = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}
        site_by_id = {site.id: site for site in public.sites}
        mission_effort = {
            allocation.site_id: allocation.effort_units
            for allocation in (self._mission.allocations if self._mission else ())
        }

        nodes: list[dict[str, Any]] = []
        for i, site_id in enumerate(graph.node_ids):
            site = site_by_id[site_id]
            row = graph.node_features[i]
            node = {
                "id": site_id,
                "label": f"Monitoring site {site_id}",
                # CURRENT DEFAULT simplification: real monitoring sites carry no
                # coast/harbor/offshore classification - that was M5 toy-scenario
                # decoration (DEMO_SITE_META). A single constant zone keeps the
                # kept M5 rendering paths (zone-based CSS/edge styling) working
                # without fabricating geography the real data doesn't support.
                "zone": "coast",
                "x": site.x,
                "y": site.y,
                "belief": belief.p_by_site[site_id],
                "uncertainty": belief.uncertainty_by_site[site_id],
                "effort": site.observed_effort,
                "detections": site.detections,
                "status": site.status,
                "frontier": bool(row[feature_index["frontier"]] > 0.5),
                "feasible": bool(graph.feasibility_mask[i]),
                "mission_effort": mission_effort.get(site_id, 0),
            }
            if self._revealed_world is not None:
                node["true_occupied"] = bool(self._revealed_world.occupied_by_site[site_id])
            nodes.append(node)

        last_round = self._serialize_transition(self._last_transition)
        mission_changed = False
        if self._last_transition is not None and self._last_transition.next_mission is not None:
            mission_changed = self._allocation_signature(
                self._last_transition.mission
            ) != self._allocation_signature(self._last_transition.next_mission)

        replan = None
        if last_round is not None:
            replan = {
                "changed": mission_changed,
                "from": last_round["previous_mission"],
                "to": last_round["next_mission"],
            }

        return {
            "phase": loop.phase.value,
            "incident": {
                "label": "Marine invasive species - confirmed first detection (real monitoring graph)",
                "scenario_name": f"Real incident / seed site {public.initial_detection}",
                "initial_detection": public.initial_detection,
                "seed": public.seed,
                "world_model_id": public.world_model_id,
                "truth_locked": self._revealed_world is None,
            },
            "resources": {
                "initial_budget": self._budget,
                "remaining_budget": public.remaining_budget,
                "spent_budget": self._budget - public.remaining_budget,
                "teams": public.teams,
                "round": public.round,
            },
            "mission": self._serialize_mission(self._mission),
            "mission_changed": mission_changed,
            "replan": replan,
            "last_round": last_round,
            "nodes": nodes,
            "edges": [asdict(edge) for edge in public.edges],
            "events": list(self._events),
            "can_plan": loop.phase is LoopPhase.READY_TO_PLAN,
            "can_execute": loop.phase is LoopPhase.MISSION_PLANNED,
            "can_reveal": loop.phase is LoopPhase.COMPLETE,
            "revealed": loop.phase is LoopPhase.REVEALED,
            # Real planner-facing q posterior/mean (belief_posterior_not_simulator_truth,
            # see SpatialAdaptiveMissionLoop._planner_constraints) - never simulator q_true.
            "q_posterior": {str(q): weight for q, weight in spatial.q_posterior().items()},
            "q_mean": spatial.q_mean(),
        }

    @staticmethod
    def _serialize_mission(mission: MissionAction | None) -> dict[str, Any] | None:
        if mission is None:
            return None
        diagnostics = mission.diagnostics or {}
        return {
            "allocations": [
                {
                    "site_id": allocation.site_id,
                    "effort_units": allocation.effort_units,
                    "team_id": allocation.team_id,
                }
                for allocation in mission.allocations
            ],
            "total_cost": mission.total_cost,
            "planner": diagnostics.get("planner", "unknown"),
            # Real ranking diagnostics from the active Planner (e.g. FrontierPlanner's
            # ranked_selected_sites) so the UI can explain "why this mission" from
            # actual ranking signals instead of invented copy.
            "diagnostics": {
                "fallback_used": diagnostics.get("fallback_used"),
                "ranked_selected_sites": [
                    dict(row) for row in diagnostics.get("ranked_selected_sites", ())
                ],
            },
        }

    @classmethod
    def _serialize_transition(
        cls, transition: RoundTransition | None
    ) -> dict[str, Any] | None:
        if transition is None:
            return None
        observations = []
        surveyed_site_ids = set()
        for observation in transition.observations.observations:
            surveyed_site_ids.add(observation.site_id)
            observations.append(
                {
                    "site_id": observation.site_id,
                    "effort": observation.effort,
                    "detection": observation.detection,
                    "belief_before": transition.belief_before.p_by_site[observation.site_id],
                    "belief_after": transition.belief_after.p_by_site[observation.site_id],
                }
            )

        # Spatial belief propagation: sites nobody surveyed this round can still move,
        # because the ensemble ties occupancy across sites. Surfacing the largest of
        # these makes that propagation visible instead of implying only surveyed
        # sites ever change.
        propagated = [
            {
                "site_id": site_id,
                "belief_before": transition.belief_before.p_by_site[site_id],
                "belief_after": transition.belief_after.p_by_site[site_id],
                "delta": transition.belief_after.p_by_site[site_id] - transition.belief_before.p_by_site[site_id],
            }
            for site_id in transition.belief_before.p_by_site
            if site_id not in surveyed_site_ids
        ]
        propagated.sort(key=lambda row: abs(row["delta"]), reverse=True)

        return {
            "round": transition.observations.round,
            "observations": observations,
            "propagated_belief_changes": propagated[:_TOP_PROPAGATED_CHANGES],
            "done": transition.done,
            "previous_mission": cls._serialize_mission(transition.mission),
            "next_mission": cls._serialize_mission(transition.next_mission),
        }

    @staticmethod
    def _allocation_signature(mission: MissionAction) -> tuple[tuple[str, int], ...]:
        return tuple(
            (allocation.site_id, allocation.effort_units)
            for allocation in mission.allocations
        )

    @staticmethod
    def _mission_text(mission: MissionAction) -> str:
        return ", ".join(
            f"{allocation.site_id}: {allocation.effort_units} checks"
            for allocation in mission.allocations
        )

    @staticmethod
    def _observation_text(transition: RoundTransition) -> str:
        return ", ".join(
            f"{obs.site_id}: {int(obs.detection)} detection / {obs.effort} checks"
            for obs in transition.observations.observations
        )

    def _require_loop(self) -> SpatialAdaptiveMissionLoop:
        if self._loop is None:
            raise RuntimeError("Mission-control session has not been initialized.")
        return self._loop
