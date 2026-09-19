from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environment import Environment
from .graph_state import GraphStateExporter, NODE_FEATURE_NAMES
from .mission_loop import LoopPhase, RoundTransition
from .model_mismatch import posterior_predictive_surprise
from .models import GraphState, HiddenWorld, MissionAction, MissionAllocation
from .planners import FrontierPlanner, Planner
from .real_incident_source import build_real_incident, eligible_incident_seed_sites
from .spatial_belief import QHypothesis, SpatialBeliefEngine, SpatialBeliefState
from .spatial_mission_loop import SpatialAdaptiveMissionLoop
from .world_models import (
    WorldModel,
    WorldModelContext,
    default_world_model_split,
    sample_ecological_hypotheses,
)


_BUDGET = 18
_INCIDENT_MAX_SITES = 20
_INCIDENT_PREFERRED_MIN_SITES = 12
_DRAWS_PER_MODEL = 60
_Q_BELIEF_VALUES = (0.05, 0.10, 0.20)
_EFFORT_LEVELS = (1, 3, 6)
_TOP_PROPAGATED_CHANGES = 5
_TOP_RECOMMENDATIONS = 5
_TOP_WORLDS = 5
_SEED_MODULUS = 1_000_000
_BELIEF_SEED_NAMESPACE_OFFSET = 10_000_000_000
_CASE_OUTCOME_VERSION = "ui-v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CASE_MANIFEST = _REPO_ROOT / "configs" / "ui_case_manifest_v1.json"


def _derive_belief_seed(case_id: str, world_seed: int) -> int:
    digest = hashlib.sha256(
        f"ui-belief-ensemble::{case_id}::{world_seed}".encode("utf-8")
    ).hexdigest()
    return _BELIEF_SEED_NAMESPACE_OFFSET + (
        int(digest[:16], 16) % 1_000_000_000
    )


def _paired_uniform(case_id: str, site_id: str) -> float:
    key = f"{_CASE_OUTCOME_VERSION}|{case_id}|{site_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64)


def _load_case_manifest() -> list[dict[str, Any]]:
    if _CASE_MANIFEST.exists():
        payload = json.loads(_CASE_MANIFEST.read_text(encoding="utf-8"))
        rows = list(payload.get("cases", ()))
        if rows:
            return rows

    # Defensive fallback for editable installs that omit configs. The committed
    # manifest is the product source of truth.
    return [
        {
            "case_id": f"incident_{index:03d}",
            "seed": index - 1,
            "label": f"Response incident {index:03d}",
        }
        for index in range(1, 101)
    ]


def _masked_for_product_recommendations(graph_state: GraphState) -> GraphState:
    """Exclude already-observed nodes from Marine's recommendation list.

    This is a product/demo choice, not a change to the benchmark Frontier baseline.
    A human override may still revisit a site; only automatic recommendations are
    kept on unsurveyed/unconfirmed nodes so the mission remains legible.
    """

    effort_idx = NODE_FEATURE_NAMES.index("observed_effort")
    detections_idx = NODE_FEATURE_NAMES.index("detections")
    mask = tuple(
        feasible
        and float(features[effort_idx]) <= 0.0
        and float(features[detections_idx]) <= 0.0
        for feasible, features in zip(
            graph_state.feasibility_mask,
            graph_state.node_features,
        )
    )
    return replace(graph_state, feasibility_mask=mask)


def _masked_for_product_revisit_fallback(graph_state: GraphState) -> GraphState:
    """Allow revisits only when every fresh delimitation site has been used.

    The confirmed initial-detection node is identifiable by detections > 0 with
    zero observed response effort, so it stays blocked. This fallback exists only
    to keep low-effort interactive sessions from dead-ending before budget reaches
    zero on small incident subgraphs.
    """

    effort_idx = NODE_FEATURE_NAMES.index("observed_effort")
    detections_idx = NODE_FEATURE_NAMES.index("detections")
    mask = tuple(
        feasible
        and not (
            float(features[detections_idx]) > 0.0
            and float(features[effort_idx]) <= 0.0
        )
        for feasible, features in zip(
            graph_state.feasibility_mask,
            graph_state.node_features,
        )
    )
    return replace(graph_state, feasibility_mask=mask)


def _product_recommendation_graph(graph_state: GraphState) -> GraphState:
    fresh = _masked_for_product_recommendations(graph_state)
    if any(fresh.feasibility_mask):
        return fresh
    return _masked_for_product_revisit_fallback(graph_state)


class MissionControlFrontierPlanner:
    """UI planner: exact Frontier ordering, with observed sites masked."""

    def __init__(self) -> None:
        self._base = FrontierPlanner(
            max_sites=1,
            effort_levels=_EFFORT_LEVELS,
        )

    def rank_candidates(self, graph_state: GraphState) -> tuple[dict[str, Any], ...]:
        return self._base.rank_candidates(
            _product_recommendation_graph(graph_state)
        )

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        return self._base.plan(
            _product_recommendation_graph(graph_state),
            remaining_budget=remaining_budget,
            constraints=constraints,
        )


class _StaticResponsePlanner:
    """Precommit three effort-6 Frontier sites from the initial observable state."""

    def __init__(self) -> None:
        self._ranker = MissionControlFrontierPlanner()
        self._actions: tuple[MissionAction, ...] | None = None
        self._cursor = 0

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        del constraints
        if self._actions is None:
            ranked = self._ranker.rank_candidates(graph_state)
            actions: list[MissionAction] = []
            budget = remaining_budget
            for row in ranked[:3]:
                effort = min(6, budget)
                if effort <= 0:
                    break
                actions.append(
                    MissionAction(
                        allocations=(
                            MissionAllocation(
                                site_id=str(row["site_id"]),
                                effort_units=effort,
                            ),
                        ),
                        total_cost=effort,
                        diagnostics={
                            "planner": "static_response",
                            "precommitted_at_t0": True,
                            "ranked_selected_sites": (
                                {**row, "effort_units": effort},
                            ),
                        },
                    )
                )
                budget -= effort
            self._actions = tuple(actions)

        if not self._actions or self._cursor >= len(self._actions):
            raise RuntimeError("Static response planner exhausted its committed plan.")
        action = self._actions[self._cursor]
        if action.total_cost > remaining_budget:
            raise RuntimeError("Static response action exceeds remaining budget.")
        self._cursor += 1
        return action


class MissionControlSession:
    """Interactive product adapter over the real spatial mission loop.

    The browser receives observable state only. Hidden truth and simulator q remain
    evaluator-only until the reveal gate opens. Marine recommends; the operator may
    follow or override both site and effort. Field evidence is still interpreted by
    the explicit spatial Bayesian engine before the next recommendation is produced.
    """

    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner or MissionControlFrontierPlanner()
        self._eligible_seed_sites = sorted(
            eligible_incident_seed_sites(
                max_sites=_INCIDENT_MAX_SITES,
                preferred_min_sites=_INCIDENT_PREFERRED_MIN_SITES,
            )
        )
        if not self._eligible_seed_sites:
            raise RuntimeError("No eligible real incident seed sites found.")

        self._cases = _load_case_manifest()
        self._loop: SpatialAdaptiveMissionLoop | None = None
        self._mission: MissionAction | None = None
        self._last_transition: RoundTransition | None = None
        self._last_surprise: dict[str, Any] | None = None
        self._last_counterfactual_next: MissionAction | None = None
        self._revealed_world: HiddenWorld | None = None
        self._events: list[dict[str, Any]] = []
        self._judge_history: list[dict[str, Any]] = []
        self._comparison_receipt: dict[str, Any] | None = None
        self._world_mass_before: dict[str, float] = {}
        self._seed = 0
        self._seed_site_id = ""
        self._case_id = ""
        self._case_index = 0
        self._budget = _BUDGET

        first_case = self._cases[0]
        self.reset(case_id=str(first_case["case_id"]))

    def case_library(self) -> dict[str, Any]:
        return {
            "count": len(self._cases),
            "current_case_id": self._case_id,
            "cases": [
                {
                    "case_id": str(row["case_id"]),
                    "label": str(row.get("label", row["case_id"])),
                    "index": index + 1,
                }
                for index, row in enumerate(self._cases)
            ],
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        if case_id is not None:
            matches = [
                (index, row)
                for index, row in enumerate(self._cases)
                if str(row["case_id"]) == str(case_id)
            ]
            if not matches:
                raise ValueError(f"Unknown case_id {case_id!r}.")
            self._case_index, case = matches[0]
            resolved_seed = int(case["seed"]) % _SEED_MODULUS
            resolved_case_id = str(case["case_id"])
        else:
            resolved_seed = (
                self._seed if seed is None else int(seed)
            ) % _SEED_MODULUS
            resolved_case_id = f"ad_hoc_{resolved_seed:06d}"
            self._case_index = -1

        self._seed = resolved_seed
        self._case_id = resolved_case_id
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
            tuple(real_incident.incident.sites),
            tuple(real_incident.incident.edges),
            seed_site_id,
        )
        belief_seed = _derive_belief_seed(resolved_case_id, resolved_seed)
        hypotheses = sample_ecological_hypotheses(
            train_models,
            context,
            draws_per_model=_DRAWS_PER_MODEL,
            seed=belief_seed,
        )
        q_hypotheses = tuple(QHypothesis(q) for q in _Q_BELIEF_VALUES)
        uniforms = {
            site.id: _paired_uniform(resolved_case_id, site.id)
            for site in real_incident.incident.sites
        }

        self._loop = SpatialAdaptiveMissionLoop(
            Environment(
                real_incident.incident,
                world_model=world_model,
                q_true=q_true,
                observation_uniform_by_site=uniforms,
            ),
            self._planner,
            hypotheses,
            q_hypotheses,
        )
        self._loop.reset(seed=resolved_seed)

        self._mission = None
        self._last_transition = None
        self._last_surprise = None
        self._last_counterfactual_next = None
        self._revealed_world = None
        self._judge_history = []
        self._world_mass_before = self._ecological_world_mass_map(
            self._loop.current_spatial_belief
        )
        self._events = [
            {
                "kind": "incident",
                "round": 0,
                "title": "New confirmed detection",
                "detail": (
                    f"Incident {resolved_case_id}: real monitoring graph, "
                    f"confirmed site {seed_site_id}."
                ),
            }
        ]

        self._comparison_receipt = self._precompute_comparators(
            incident=real_incident.incident,
            world_model=world_model,
            q_true=q_true,
            hypotheses=hypotheses,
            q_hypotheses=q_hypotheses,
            uniforms=uniforms,
            seed=resolved_seed,
        )
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
                "title": "Marine recommendation accepted",
                "detail": self._mission_text(self._mission),
            }
        )
        return self.snapshot()

    def deploy(self, *, site_id: str, effort: int) -> dict[str, Any]:
        loop = self._require_loop()
        public = loop.current_public_state

        if site_id == public.initial_detection:
            raise ValueError(
                "The confirmed initial-detection site is already known; choose a delimitation site."
            )
        if site_id not in {site.id for site in public.sites}:
            raise ValueError(f"Unknown survey site {site_id!r}.")
        if effort not in _EFFORT_LEVELS:
            raise ValueError(
                f"effort must be one of {list(_EFFORT_LEVELS)!r}."
            )
        if effort > public.remaining_budget:
            raise ValueError("Selected effort exceeds remaining field budget.")
        if loop.phase not in (LoopPhase.READY_TO_PLAN, LoopPhase.MISSION_PLANNED):
            raise RuntimeError("No new mission can be deployed in the current phase.")

        recommendations = self._global_recommendations(limit=None)
        rank_by_site = {
            row["site_id"]: int(row["rank"]) for row in recommendations
        }
        marine_rank = rank_by_site.get(site_id)
        source = (
            "marine_recommendation"
            if marine_rank == 1
            else "human_override"
        )

        diagnostics_row = next(
            (row for row in recommendations if row["site_id"] == site_id),
            {
                "site_id": site_id,
                "belief": loop.current_belief.p_by_site[site_id],
                "uncertainty": loop.current_belief.uncertainty_by_site[site_id],
                "frontier": False,
            },
        )
        mission = MissionAction(
            allocations=(
                MissionAllocation(site_id=site_id, effort_units=effort),
            ),
            total_cost=effort,
            diagnostics={
                "planner": "human_in_loop",
                "selection_source": source,
                "marine_rank": marine_rank,
                "ranked_selected_sites": (
                    {**diagnostics_row, "effort_units": effort},
                ),
            },
        )
        loop.set_pending_mission(mission)
        self._mission = mission

        self._events.append(
            {
                "kind": "mission",
                "round": public.round,
                "title": (
                    "Marine recommendation selected"
                    if source == "marine_recommendation"
                    else "Human override selected"
                ),
                "detail": (
                    f"{site_id}: effort {effort}"
                    + (
                        ""
                        if marine_rank is None
                        else f" / Marine rank #{marine_rank}"
                    )
                ),
            }
        )
        return self._execute_current()

    def execute(self) -> dict[str, Any]:
        loop = self._require_loop()
        if loop.phase is not LoopPhase.MISSION_PLANNED:
            raise RuntimeError("A mission must be planned before it can be executed.")
        return self._execute_current()

    def _execute_current(self) -> dict[str, Any]:
        loop = self._require_loop()
        spatial_before = loop.current_spatial_belief
        self._world_mass_before = self._ecological_world_mass_map(spatial_before)

        transition = loop.execute_pending()
        self._last_transition = transition
        self._last_counterfactual_next = self._counterfactual_next_without_evidence(
            transition
        )

        surprise_rows = []
        for observation in transition.observations.observations:
            surprise = posterior_predictive_surprise(spatial_before, observation)
            surprise_rows.append(
                {
                    "site_id": surprise.site_id,
                    "effort": surprise.effort,
                    "detection": surprise.detection,
                    "predictive_detection_probability": surprise.predictive_detection_probability,
                    "observation_probability": surprise.observation_probability,
                    "surprise_bits": (
                        surprise.surprise_bits
                        if isfinite(surprise.surprise_bits)
                        else None
                    ),
                    "impossible_under_current_ensemble": surprise.impossible_under_current_ensemble,
                }
            )
        self._last_surprise = surprise_rows[0] if surprise_rows else None

        self._judge_history.append(
            {
                "mission": transition.mission,
                "observations": transition.observations,
                "effort_spent": int(transition.simulator_metrics["effort_spent"]),
            }
        )
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
            evidence_changed_mission = (
                self._last_counterfactual_next is not None
                and self._allocation_signature(self._last_counterfactual_next)
                != self._allocation_signature(transition.next_mission)
            )
            self._events.append(
                {
                    "kind": "replan" if evidence_changed_mission else "mission",
                    "round": transition.observations.round,
                    "title": (
                        "Mission updated by evidence"
                        if evidence_changed_mission
                        else "Next mission confirmed"
                    ),
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
                    "detail": "Reveal gate is now available.",
                }
            )

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
                "detail": "Latent occupancy is now visible for evaluation.",
            }
        )
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        loop = self._require_loop()
        public = loop.current_public_state
        belief = loop.current_belief
        graph = loop.current_graph_state
        spatial = loop.current_spatial_belief

        feature_index = {
            name: index for index, name in enumerate(NODE_FEATURE_NAMES)
        }
        site_by_id = {site.id: site for site in public.sites}
        mission_effort = {
            allocation.site_id: allocation.effort_units
            for allocation in (
                self._mission.allocations if self._mission else ()
            )
        }
        all_recommendations = self._global_recommendations(limit=None)
        recommendation_by_site = {
            str(row["site_id"]): row for row in all_recommendations
        }

        nodes: list[dict[str, Any]] = []
        for index, site_id in enumerate(graph.node_ids):
            site = site_by_id[site_id]
            row = graph.node_features[index]
            node = {
                "id": site_id,
                "label": f"Monitoring site {site_id}",
                "zone": "coast",
                "x": site.x,
                "y": site.y,
                "belief": belief.p_by_site[site_id],
                "uncertainty": belief.uncertainty_by_site[site_id],
                "habitat": float(site.habitat_score),
                "effort": site.observed_effort,
                "detections": site.detections,
                "status": site.status,
                "frontier": bool(
                    row[feature_index["frontier"]] > 0.5
                ),
                "feasible": bool(graph.feasibility_mask[index]),
                "mission_effort": mission_effort.get(site_id, 0),
                "marine_rank": (
                    int(recommendation_by_site[site_id]["rank"])
                    if site_id in recommendation_by_site
                    else None
                ),
                "predictive_detection": (
                    dict(recommendation_by_site[site_id]["predictive_detection"])
                    if site_id in recommendation_by_site
                    else {}
                ),
            }
            if self._revealed_world is not None:
                node["true_occupied"] = bool(
                    self._revealed_world.occupied_by_site[site_id]
                )
            nodes.append(node)

        last_round = self._serialize_transition(self._last_transition)
        mission_changed = False
        if (
            self._last_counterfactual_next is not None
            and self._last_transition is not None
            and self._last_transition.next_mission is not None
        ):
            mission_changed = self._allocation_signature(
                self._last_counterfactual_next
            ) != self._allocation_signature(
                self._last_transition.next_mission
            )

        counterfactual_next = self._serialize_mission(
            self._last_counterfactual_next
        )
        replan = None
        if last_round is not None:
            replan = {
                "changed": mission_changed,
                "from": counterfactual_next,
                "to": last_round["next_mission"],
                "semantics": "same_post_survey_public_state_without_new_evidence",
            }

        top_worlds = self._top_worlds(spatial)
        recommendations = all_recommendations[:_TOP_RECOMMENDATIONS]

        performance = None
        if self._revealed_world is not None:
            performance = {
                "marine": self._comparison_receipt["marine"],
                "static": self._comparison_receipt["static"],
                "you": self._judge_performance(self._revealed_world),
            }

        return {
            "phase": loop.phase.value,
            "case": {
                "case_id": self._case_id,
                "index": (
                    self._case_index + 1 if self._case_index >= 0 else None
                ),
                "count": len(self._cases),
                "outcome_protocol": _CASE_OUTCOME_VERSION,
            },
            "incident": {
                "label": "Marine invasive species - confirmed first detection",
                "scenario_name": (
                    f"Response incident {self._case_id} / seed site "
                    f"{public.initial_detection}"
                ),
                "initial_detection": public.initial_detection,
                "seed": public.seed,
                "truth_locked": self._revealed_world is None,
                "truth_family": (
                    public.world_model_id
                    if self._revealed_world is not None
                    else None
                ),
            },
            "resources": {
                "initial_budget": self._budget,
                "remaining_budget": public.remaining_budget,
                "spent_budget": self._budget - public.remaining_budget,
                "teams": public.teams,
                "round": public.round,
                "effort_levels": list(_EFFORT_LEVELS),
            },
            "mission": self._serialize_mission(self._mission),
            "global_recommendations": recommendations,
            "top_worlds": top_worlds,
            "mission_changed": mission_changed,
            "replan": replan,
            "last_round": last_round,
            "model_stress": self._last_surprise,
            "nodes": nodes,
            "edges": [asdict(edge) for edge in public.edges],
            "events": list(self._events),
            "can_plan": loop.phase is LoopPhase.READY_TO_PLAN,
            "can_execute": loop.phase is LoopPhase.MISSION_PLANNED,
            "can_reveal": loop.phase is LoopPhase.COMPLETE,
            "revealed": loop.phase is LoopPhase.REVEALED,
            "q_posterior": {
                str(q): weight
                for q, weight in spatial.q_posterior().items()
            },
            "q_mean": spatial.q_mean(),
            "performance": performance,
        }

    def _global_recommendations(
        self,
        *,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        loop = self._require_loop()
        if loop.phase in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
            return []

        graph = loop.current_graph_state
        spatial = loop.current_spatial_belief
        public = loop.current_public_state

        if hasattr(self._planner, "rank_candidates"):
            ranked = list(self._planner.rank_candidates(graph))
        else:
            ranked = list(
                MissionControlFrontierPlanner().rank_candidates(graph)
            )

        rows: list[dict[str, Any]] = []
        for index, row in enumerate(ranked, start=1):
            predictive: dict[str, float | None] = {}
            for effort in _EFFORT_LEVELS:
                predictive[str(effort)] = (
                    SpatialBeliefEngine.predictive_detection_probability(
                        spatial,
                        site_id=str(row["site_id"]),
                        effort=effort,
                    )
                    if effort <= public.remaining_budget
                    else None
                )
            rows.append(
                {
                    **row,
                    "rank": index,
                    "predictive_detection": predictive,
                }
            )

        return rows if limit is None else rows[:limit]

    def _counterfactual_next_without_evidence(
        self,
        transition: RoundTransition,
    ) -> MissionAction | None:
        """Compute the next Frontier mission with the survey recorded but evidence ignored.

        This isolates the causal question shown in the UI: did the *field result*
        change Marine's next recommendation, rather than merely advancing from the
        just-executed site to another site? Public state after the survey is held
        fixed (budget spent, site observed, detections recorded for operational
        status), while occupancy marginals are held at their pre-observation values.
        """

        if transition.done:
            return None

        counterfactual_graph = GraphStateExporter().export(
            transition.public_state_after,
            transition.belief_before,
        )
        planner = (
            self._planner
            if isinstance(self._planner, MissionControlFrontierPlanner)
            else MissionControlFrontierPlanner()
        )
        mission = planner.plan(
            counterfactual_graph,
            remaining_budget=transition.public_state_after.remaining_budget,
            constraints={},
        )
        return mission

    @staticmethod
    def _ecological_world_mass_map(
        belief: SpatialBeliefState,
    ) -> dict[str, float]:
        masses: dict[str, float] = {}
        for hypothesis, weight in zip(
            belief.hypotheses,
            belief.weights,
        ):
            key = MissionControlSession._world_id(
                belief.site_ids,
                hypothesis.presence,
            )
            masses[key] = masses.get(key, 0.0) + float(weight)
        return masses

    def _top_worlds(
        self,
        belief: SpatialBeliefState,
    ) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for hypothesis, weight in zip(
            belief.hypotheses,
            belief.weights,
        ):
            world_id = self._world_id(
                belief.site_ids,
                hypothesis.presence,
            )
            row = grouped.setdefault(
                world_id,
                {
                    "world_id": world_id,
                    "posterior": 0.0,
                    "families": set(),
                    "occupied_site_ids": [
                        site_id
                        for site_id, present in zip(
                            belief.site_ids,
                            hypothesis.presence,
                        )
                        if present
                    ],
                },
            )
            row["posterior"] += float(weight)
            if hypothesis.ecological_label:
                row["families"].update(
                    str(hypothesis.ecological_label).split("|")
                )

        ranked = sorted(
            grouped.values(),
            key=lambda row: (
                -float(row["posterior"]),
                str(row["world_id"]),
            ),
        )
        top = []
        for index, row in enumerate(ranked[:_TOP_WORLDS], start=1):
            posterior = float(row["posterior"])
            top.append(
                {
                    "world_id": row["world_id"],
                    "rank": index,
                    "posterior": posterior,
                    "delta": posterior
                    - self._world_mass_before.get(
                        str(row["world_id"]),
                        posterior,
                    ),
                    "families": sorted(row["families"]),
                    "occupied_site_ids": row["occupied_site_ids"],
                    "occupied_count": len(row["occupied_site_ids"]),
                }
            )

        top_mass = sum(float(row["posterior"]) for row in top)
        return {
            "items": top,
            "remaining_mass": max(0.0, 1.0 - top_mass),
            "unique_world_count": len(ranked),
        }

    @staticmethod
    def _world_id(
        site_ids: Sequence[str],
        presence: Sequence[bool],
    ) -> str:
        payload = "|".join(
            f"{site_id}:{int(value)}"
            for site_id, value in zip(site_ids, presence)
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"world_{digest[:10]}"

    def _precompute_comparators(
        self,
        *,
        incident,
        world_model: WorldModel,
        q_true: float,
        hypotheses,
        q_hypotheses,
        uniforms: Mapping[str, float],
        seed: int,
    ) -> dict[str, Any]:
        marine = self._run_comparator_track(
            name="marine",
            incident=incident,
            world_model=world_model,
            q_true=q_true,
            hypotheses=hypotheses,
            q_hypotheses=q_hypotheses,
            uniforms=uniforms,
            seed=seed,
            planner=MissionControlFrontierPlanner(),
        )
        static = self._run_comparator_track(
            name="static",
            incident=incident,
            world_model=world_model,
            q_true=q_true,
            hypotheses=hypotheses,
            q_hypotheses=q_hypotheses,
            uniforms=uniforms,
            seed=seed,
            planner=_StaticResponsePlanner(),
        )
        if marine["occupied_total"] != static["occupied_total"]:
            raise RuntimeError("Comparator hidden worlds diverged.")
        return {"marine": marine, "static": static}

    @staticmethod
    def _run_comparator_track(
        *,
        name: str,
        incident,
        world_model: WorldModel,
        q_true: float,
        hypotheses,
        q_hypotheses,
        uniforms: Mapping[str, float],
        seed: int,
        planner: Planner,
    ) -> dict[str, Any]:
        loop = SpatialAdaptiveMissionLoop(
            Environment(
                incident,
                world_model=world_model,
                q_true=q_true,
                observation_uniform_by_site=uniforms,
            ),
            planner,
            hypotheses,
            q_hypotheses,
        )
        loop.reset(seed=seed)

        cumulative_effort = 0
        detected_snapshots: list[tuple[int, set[str]]] = [
            (0, {incident.initial_detection})
        ]
        mission_sites: list[str] = []

        while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
            transition = loop.run_round()
            cumulative_effort += int(
                transition.simulator_metrics["effort_spent"]
            )
            detected = set(detected_snapshots[-1][1])
            for observation in transition.observations.observations:
                mission_sites.append(observation.site_id)
                if observation.detection:
                    detected.add(observation.site_id)
            detected_snapshots.append((cumulative_effort, detected))

        hidden = loop.reveal()
        occupied = {
            site_id
            for site_id, value in hidden.occupied_by_site.items()
            if value
        }
        occupied_total = len(occupied)
        curve = [
            {
                "effort": effort,
                "detected_occupied": len(detected & occupied),
                "detected_fraction": (
                    len(detected & occupied) / occupied_total
                    if occupied_total
                    else 0.0
                ),
            }
            for effort, detected in detected_snapshots
        ]
        final_detected = int(curve[-1]["detected_occupied"])
        return {
            "name": name,
            "occupied_total": occupied_total,
            "detected_occupied": final_detected,
            "undetected_occupied": occupied_total - final_detected,
            "mission_sites": mission_sites,
            "curve": curve,
        }

    def _judge_performance(
        self,
        hidden_world: HiddenWorld,
    ) -> dict[str, Any]:
        occupied = {
            site_id
            for site_id, value in hidden_world.occupied_by_site.items()
            if value
        }
        occupied_total = len(occupied)
        detected = {self._seed_site_id}
        cumulative_effort = 0
        curve = [
            {
                "effort": 0,
                "detected_occupied": len(detected & occupied),
                "detected_fraction": (
                    len(detected & occupied) / occupied_total
                    if occupied_total
                    else 0.0
                ),
            }
        ]
        mission_sites: list[str] = []

        for row in self._judge_history:
            cumulative_effort += int(row["effort_spent"])
            for observation in row["observations"].observations:
                mission_sites.append(observation.site_id)
                if observation.detection:
                    detected.add(observation.site_id)
            curve.append(
                {
                    "effort": cumulative_effort,
                    "detected_occupied": len(detected & occupied),
                    "detected_fraction": (
                        len(detected & occupied) / occupied_total
                        if occupied_total
                        else 0.0
                    ),
                }
            )

        final_detected = len(detected & occupied)
        return {
            "name": "you",
            "occupied_total": occupied_total,
            "detected_occupied": final_detected,
            "undetected_occupied": occupied_total - final_detected,
            "mission_sites": mission_sites,
            "curve": curve,
        }

    @staticmethod
    def _serialize_mission(
        mission: MissionAction | None,
    ) -> dict[str, Any] | None:
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
            "selection_source": diagnostics.get("selection_source"),
            "marine_rank": diagnostics.get("marine_rank"),
            "diagnostics": {
                "fallback_used": diagnostics.get("fallback_used"),
                "ranked_selected_sites": [
                    dict(row)
                    for row in diagnostics.get(
                        "ranked_selected_sites",
                        (),
                    )
                ],
            },
        }

    @classmethod
    def _serialize_transition(
        cls,
        transition: RoundTransition | None,
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
                    "belief_before": transition.belief_before.p_by_site[
                        observation.site_id
                    ],
                    "belief_after": transition.belief_after.p_by_site[
                        observation.site_id
                    ],
                }
            )

        propagated = [
            {
                "site_id": site_id,
                "belief_before": transition.belief_before.p_by_site[
                    site_id
                ],
                "belief_after": transition.belief_after.p_by_site[
                    site_id
                ],
                "delta": (
                    transition.belief_after.p_by_site[site_id]
                    - transition.belief_before.p_by_site[site_id]
                ),
            }
            for site_id in transition.belief_before.p_by_site
            if site_id not in surveyed_site_ids
        ]
        propagated.sort(
            key=lambda row: abs(row["delta"]),
            reverse=True,
        )

        return {
            "round": transition.observations.round,
            "observations": observations,
            "propagated_belief_changes": propagated[
                :_TOP_PROPAGATED_CHANGES
            ],
            "done": transition.done,
            "previous_mission": cls._serialize_mission(
                transition.mission
            ),
            "next_mission": cls._serialize_mission(
                transition.next_mission
            ),
        }

    @staticmethod
    def _allocation_signature(
        mission: MissionAction,
    ) -> tuple[tuple[str, int], ...]:
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
            (
                f"{obs.site_id}: "
                f"{int(obs.detection)} detection / {obs.effort} checks"
            )
            for obs in transition.observations.observations
        )

    def _require_loop(self) -> SpatialAdaptiveMissionLoop:
        if self._loop is None:
            raise RuntimeError(
                "Mission-control session has not been initialized."
            )
        return self._loop
