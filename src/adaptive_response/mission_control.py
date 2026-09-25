from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, replace
from math import atan2, cos, isfinite, radians, sin, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environment import Environment
from .graph_state import GraphStateExporter, NODE_FEATURE_NAMES
from .mission_loop import LoopPhase, RoundTransition
from .model_mismatch import posterior_predictive_surprise
from .models import (
    GraphState,
    HiddenWorld,
    MissionAction,
    MissionAllocation,
    Observation,
    ObservationBatch,
)
from .planners import FrontierPlanner, Planner
from .real_incident_source import (
    build_real_incident,
    eligible_incident_seed_sites,
    real_site_display_metadata,
)
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
# Defensive cap only: every legal action costs >=1 effort unit out of an
# 18-unit budget, so a normal campaign reaches remaining_budget == 0 (and
# therefore `transition.done`) at or before this many windows on its own.
# This is NOT a target number of deployments - a policy choosing smaller
# efforts legitimately runs more windows before the budget is exhausted.
_MAX_RESPONSE_WINDOWS = 18
# Delimitation-band thresholds only. Empirically, e3's information-gain
# retention in this belief model consistently lands around 0.55 (never
# ~0.65) whenever e3 clears the 0.55 detection-power bar - so a 0.65 info
# floor reproduced the exact same structural exclusion this rule was
# written to fix (e1 was excluded from exploratory before this change; e3
# was excluded from delimitation at this threshold). 0.50 leaves headroom
# below the observed range while still requiring e3 to capture most of the
# information a full-effort survey would.
_EFFORT_INFORMATION_RETENTION = 0.50
_EFFORT_DETECTION_RETENTION_LOW = 0.55
_EFFORT_MEDIUM_OCCUPANCY = 0.25
_EFFORT_HIGH_OCCUPANCY = 0.55
_TOP_PROPAGATED_CHANGES = 5
_TOP_RECOMMENDATIONS = 5
_TOP_WORLDS = 5
_SEED_MODULUS = 1_000_000
_BELIEF_SEED_NAMESPACE_OFFSET = 10_000_000_000
_CASE_OUTCOME_VERSION = "ui-v1"
_DEFAULT_DEMO_CASE_ID = "incident_079"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CASE_MANIFEST = _REPO_ROOT / "configs" / "ui_case_manifest_v1.json"


def _derive_belief_seed(case_id: str, world_seed: int) -> int:
    digest = hashlib.sha256(
        f"ui-belief-ensemble::{case_id}::{world_seed}".encode("utf-8")
    ).hexdigest()
    return _BELIEF_SEED_NAMESPACE_OFFSET + (
        int(digest[:16], 16) % 1_000_000_000
    )


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = (
        sin(dphi / 2.0) ** 2
        + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * atan2(sqrt(a), sqrt(max(0.0, 1.0 - a)))


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
    """Exclude already-observed nodes from the adaptive planner's recommendation list.

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
    """Product planner: Frontier site ranking + Bayesian resource-aware effort.

    Site selection deliberately preserves the benchmark-supported Frontier ordering.
    The product layer optimizes *how much* effort to spend at the selected site
    using only the observable joint spatial/q posterior, with a distinct rule per
    occupancy-belief band (PROBE -> DELIMIT -> CONFIRM; see `recommend_effort`):

    - exploratory (low belief): the cheapest, highest-information-per-effort
      probe, tie-broken toward the smaller effort;
    - delimitation (medium belief): the smallest effort that still retains most
      of the information gain and conditional detection power a full-effort
      survey would give, otherwise full effort;
    - confirmation (high belief): the largest feasible effort outright.

    This is a transparent DESIGN CHOICE for the resource-efficiency demo.
    It is not an ecological constant and it does not use hidden truth.
    """

    def __init__(self) -> None:
        self._base = FrontierPlanner(max_sites=1)

    def rank_candidates(self, graph_state: GraphState) -> tuple[dict[str, Any], ...]:
        return self._base.rank_candidates(
            _product_recommendation_graph(graph_state)
        )

    @staticmethod
    def _expected_information_gain_bits(
        spatial: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> float:
        current_entropy = sum(spatial.uncertainty_by_site().values())
        p_detection = SpatialBeliefEngine.predictive_detection_probability(
            spatial,
            site_id=site_id,
            effort=effort,
        )
        expected_after = 0.0
        outcomes = (
            (True, p_detection),
            (False, 1.0 - p_detection),
        )
        for detection, probability in outcomes:
            if probability <= 0.0:
                continue
            posterior = SpatialBeliefEngine.update(
                spatial,
                ObservationBatch(
                    observations=(
                        Observation(
                            site_id=site_id,
                            effort=effort,
                            detection=detection,
                            round=0,
                        ),
                    ),
                    round=0,
                    total_effort=effort,
                ),
            )
            expected_after += probability * sum(
                posterior.uncertainty_by_site().values()
            )
        return max(0.0, current_entropy - expected_after)

    @staticmethod
    def _conditional_detection_if_occupied(
        spatial: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> float:
        index = spatial.site_ids.index(site_id)
        occupied_mass = 0.0
        detection_mass = 0.0
        for hypothesis, weight in zip(spatial.hypotheses, spatial.weights):
            if not hypothesis.presence[index]:
                continue
            occupied_mass += float(weight)
            detection_mass += float(weight) * (
                1.0 - (1.0 - float(hypothesis.q)) ** effort
            )
        if occupied_mass <= 1e-12:
            # Degenerate fallback only: the candidate is effectively impossible
            # under the current occupancy posterior, so use the marginal q belief.
            return sum(
                float(weight) * (1.0 - (1.0 - float(q)) ** effort)
                for q, weight in spatial.q_posterior().items()
            )
        return detection_mass / occupied_mass

    def effort_options(
        self,
        *,
        spatial: SpatialBeliefState,
        site_id: str,
        remaining_budget: int,
    ) -> tuple[dict[str, Any], ...]:
        feasible = [
            effort
            for effort in _EFFORT_LEVELS
            if effort <= remaining_budget
        ]
        if not feasible:
            return ()

        rows: list[dict[str, Any]] = []
        for effort in feasible:
            information_gain = self._expected_information_gain_bits(
                spatial,
                site_id=site_id,
                effort=effort,
            )
            predictive_detection = SpatialBeliefEngine.predictive_detection_probability(
                spatial,
                site_id=site_id,
                effort=effort,
            )
            conditional_detection = self._conditional_detection_if_occupied(
                spatial,
                site_id=site_id,
                effort=effort,
            )
            rows.append(
                {
                    "effort": effort,
                    "expected_information_gain_bits": information_gain,
                    "information_gain_per_effort": information_gain / effort,
                    "predictive_detection_probability": predictive_detection,
                    "conditional_detection_if_occupied": conditional_detection,
                }
            )

        max_row = rows[-1]
        max_ig = float(max_row["expected_information_gain_bits"])
        max_conditional = float(max_row["conditional_detection_if_occupied"])
        for row in rows:
            row["information_retention"] = (
                1.0
                if max_ig <= 1e-12
                else float(row["expected_information_gain_bits"]) / max_ig
            )
            row["detection_power_retention"] = (
                1.0
                if max_conditional <= 1e-12
                else float(row["conditional_detection_if_occupied"]) / max_conditional
            )
            row["resource_fraction_vs_max"] = float(row["effort"]) / float(max_row["effort"])
        return tuple(rows)

    def recommend_effort(
        self,
        *,
        spatial: SpatialBeliefState,
        site_id: str,
        remaining_budget: int,
    ) -> dict[str, Any]:
        options = self.effort_options(
            spatial=spatial,
            site_id=site_id,
            remaining_budget=remaining_budget,
        )
        if not options:
            raise ValueError("No allowed effort level fits remaining budget.")

        occupancy_belief = float(spatial.p_by_site()[site_id])
        max_effort = int(options[-1]["effort"])

        # PROBE -> DELIMIT -> CONFIRM (product decision rule, not an
        # ecological constant): at low belief a fixed high-retention
        # threshold structurally excludes e1 (e1/e6 conditional-detection
        # retention is only ~19-27% across the belief q support), so every
        # site converged on the largest feasible effort regardless of belief.
        # Each band now has its own, distinct decision rule instead of one
        # shared "smallest effort clearing a retention bar" rule applied at
        # three thresholds.
        if occupancy_belief >= _EFFORT_HIGH_OCCUPANCY:
            # Confirmation: the site is already strongly suspected - buy
            # detection power outright with the largest feasible effort.
            occupancy_band = "confirmation"
            chosen = options[-1]
        elif occupancy_belief >= _EFFORT_MEDIUM_OCCUPANCY:
            # Delimitation: spend the smallest effort that still keeps most
            # of the information gain and conditional detection power a
            # full-effort survey would give; otherwise spend full effort.
            occupancy_band = "delimitation"
            chosen = options[-1]
            for row in options:
                if (
                    float(row["information_retention"]) >= _EFFORT_INFORMATION_RETENTION
                    and float(row["detection_power_retention"]) >= _EFFORT_DETECTION_RETENTION_LOW
                ):
                    chosen = row
                    break
        else:
            # Exploratory: cheaply probe the frontier. Maximize information
            # gain per effort unit (not a retention-vs-max-effort ratio), so
            # e1 can win outright when it is the most efficient probe;
            # ties are broken toward the smaller effort to preserve field
            # capacity for another branch.
            occupancy_band = "exploratory"
            chosen = max(
                options,
                key=lambda row: (
                    float(row["information_gain_per_effort"]),
                    -float(row["effort"]),
                ),
            )

        chosen_effort = int(chosen["effort"])
        return {
            **chosen,
            "recommended_effort": chosen_effort,
            "effort_saved_vs_max": max_effort - chosen_effort,
            "max_feasible_effort": max_effort,
            "occupancy_belief": occupancy_belief,
            "occupancy_band": occupancy_band,
            "options": [dict(row) for row in options],
            "rule": "probe_delimit_confirm",
            "design_note": (
                "Exploratory (low belief): cheapest highest-information-per-effort "
                "probe, tied toward the smaller effort, to preserve field capacity "
                "for other branches. Delimitation (medium belief): smallest effort "
                "that keeps most of the information gain and conditional detection "
                "power a full-effort survey would give, otherwise full effort. "
                "Confirmation (high belief): buy detection power outright with the "
                "largest feasible effort. Thresholds are product design choices for "
                "the resource-efficiency demo, not ecological constants."
            ),
        }

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        if remaining_budget <= 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={
                    "planner": "resource_aware_frontier",
                    "reason": "no_remaining_budget",
                },
            )

        ranked = list(self.rank_candidates(graph_state))
        if not ranked:
            raise RuntimeError("No feasible product recommendation sites remain.")

        selected = ranked[0]
        spatial = constraints.get("spatial_belief_state")
        if isinstance(spatial, SpatialBeliefState):
            effort_diag = self.recommend_effort(
                spatial=spatial,
                site_id=str(selected["site_id"]),
                remaining_budget=remaining_budget,
            )
            effort = int(effort_diag["recommended_effort"])
        else:
            feasible = [e for e in _EFFORT_LEVELS if e <= remaining_budget]
            if not feasible:
                raise RuntimeError("No allowed effort level fits remaining budget.")
            effort = feasible[-1]
            effort_diag = {
                "recommended_effort": effort,
                "effort_saved_vs_max": 0,
                "max_feasible_effort": effort,
                "rule": "fallback_max_effort_without_spatial_posterior",
                "options": [],
            }

        selected_diag = {
            **selected,
            "effort_units": effort,
            "effort_recommendation": effort_diag,
        }
        return MissionAction(
            allocations=(
                MissionAllocation(
                    site_id=str(selected["site_id"]),
                    effort_units=effort,
                ),
            ),
            total_cost=effort,
            diagnostics={
                "planner": "resource_aware_frontier",
                "site_objective": "frontier_then_belief_then_uncertainty",
                "effort_objective": "retain_value_with_minimum_field_effort",
                "effort_recommendation": effort_diag,
                "ranked_selected_sites": (selected_diag,),
                "ranked_candidates": tuple(ranked),
                "hidden_truth_used": False,
            },
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
    evaluator-only until the reveal gate opens. The adaptive planner recommends; the operator may
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
        self._pending_decision_context: dict[str, Any] | None = None
        self._last_decision_receipt: dict[str, Any] | None = None
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
        self._site_display_metadata: dict[str, dict[str, object]] = {}
        self._static_plan_sites: tuple[str, ...] = ()

        default_case = next(
            (
                row
                for row in self._cases
                if str(row["case_id"]) == _DEFAULT_DEMO_CASE_ID
            ),
            self._cases[0],
        )
        self.reset(case_id=str(default_case["case_id"]))

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
        self._site_display_metadata = real_site_display_metadata(
            [site.id for site in real_incident.incident.sites]
        )
        initial_ranked = MissionControlFrontierPlanner().rank_candidates(
            self._loop.current_graph_state
        )
        self._static_plan_sites = tuple(
            str(row["site_id"]) for row in initial_ranked[:3]
        )

        self._mission = None
        self._last_transition = None
        self._last_surprise = None
        self._last_counterfactual_next = None
        self._pending_decision_context = None
        self._last_decision_receipt = None
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
                "title": "Adaptive recommendation accepted",
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
        selected_recommendation = next(
            (row for row in recommendations if row["site_id"] == site_id),
            None,
        )
        top_recommendation = recommendations[0] if recommendations else None
        selected_recommended_effort = (
            int(selected_recommendation["recommended_effort"])
            if selected_recommendation is not None
            else None
        )
        self._pending_decision_context = {
            "selected_site_id": site_id,
            "selected_effort": effort,
            "marine_rank": marine_rank,
            "marine_top_site_id": (
                str(top_recommendation["site_id"])
                if top_recommendation is not None
                else None
            ),
            "marine_top_effort": (
                int(top_recommendation["recommended_effort"])
                if top_recommendation is not None
                else None
            ),
            "marine_recommended_effort_for_selected_site": selected_recommended_effort,
            "site_followed": bool(marine_rank == 1),
            "effort_followed": bool(
                selected_recommended_effort is not None
                and effort == selected_recommended_effort
            ),
            "selected_recommendation": (
                dict(selected_recommendation)
                if selected_recommendation is not None
                else None
            ),
        }

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
                "marine_recommended_effort": selected_recommended_effort,
                "effort_followed": bool(
                    selected_recommended_effort is not None
                    and effort == selected_recommended_effort
                ),
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
                    "Adaptive recommendation selected"
                    if source == "marine_recommendation"
                    else "Human override selected"
                ),
                "detail": (
                    f"{site_id}: effort {effort}"
                    + (
                        ""
                        if marine_rank is None
                        else f" / recommendation rank #{marine_rank}"
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
        if (
            not transition.done
            and transition.public_state_after.round >= _MAX_RESPONSE_WINDOWS
        ):
            loop.force_complete()
            transition = replace(
                transition,
                next_mission=None,
                done=True,
            )

        self._last_transition = transition
        self._last_counterfactual_next = self._counterfactual_next_without_evidence(
            transition,
            spatial_before=spatial_before,
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

        decision_context = self._pending_decision_context or {}
        first_observation = (
            transition.observations.observations[0]
            if transition.observations.observations
            else None
        )
        selected_effort = (
            int(first_observation.effort)
            if first_observation is not None
            else int(transition.mission.total_cost)
        )
        conditional_detection_if_occupied = (
            MissionControlFrontierPlanner._conditional_detection_if_occupied(
                spatial_before,
                site_id=first_observation.site_id,
                effort=selected_effort,
            )
            if first_observation is not None
            else None
        )
        selected_recommendation = decision_context.get("selected_recommendation")
        if first_observation is not None:
            site_followed = bool(decision_context.get("site_followed", False))
            effort_followed = bool(decision_context.get("effort_followed", False))
            aligned = site_followed and effort_followed
            if aligned and first_observation.detection:
                interpretation = (
                    "Recommendation-aligned decision; the field return produced a detection."
                )
            elif aligned:
                interpretation = (
                    "Recommendation-aligned decision selected before the outcome. "
                    "A non-detection remains possible under imperfect detection; "
                    "the result does not retroactively make the decision wrong."
                )
            else:
                interpretation = (
                    "Operator override. Compare the realized outcome with the adaptive planner's "
                    "pre-outcome recommendation rather than treating one stochastic "
                    "return as a policy benchmark."
                )
            self._last_decision_receipt = {
                "site_id": first_observation.site_id,
                "effort": selected_effort,
                "detection": bool(first_observation.detection),
                "site_followed": site_followed,
                "effort_followed": effort_followed,
                "marine_aligned": aligned,
                "marine_rank": decision_context.get("marine_rank"),
                "marine_top_site_id": decision_context.get("marine_top_site_id"),
                "marine_top_effort": decision_context.get("marine_top_effort"),
                "marine_recommended_effort_for_selected_site": decision_context.get(
                    "marine_recommended_effort_for_selected_site"
                ),
                "predictive_detection_probability": (
                    selected_recommendation.get("predictive_detection", {}).get(
                        str(selected_effort)
                    )
                    if isinstance(selected_recommendation, dict)
                    else None
                ),
                "conditional_detection_if_occupied": conditional_detection_if_occupied,
                "conditional_miss_if_occupied": (
                    1.0 - conditional_detection_if_occupied
                    if conditional_detection_if_occupied is not None
                    else None
                ),
                "observation_probability": (
                    self._last_surprise["observation_probability"]
                    if self._last_surprise is not None
                    else None
                ),
                "surprise_bits": (
                    self._last_surprise["surprise_bits"]
                    if self._last_surprise is not None
                    else None
                ),
                "interpretation": interpretation,
            }
        else:
            self._last_decision_receipt = None
        self._pending_decision_context = None

        self._judge_history.append(
            {
                "mission": transition.mission,
                "observations": transition.observations,
                "effort_spent": int(transition.simulator_metrics["effort_spent"]),
                "decision_receipt": deepcopy(self._last_decision_receipt),
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
                    "title": "Field-response budget allocated",
                    "detail": (
                        "The 18-unit response budget has been allocated. Hidden "
                        "extent can now be revealed for evaluation."
                    ),
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
            display_meta = self._site_display_metadata.get(site_id, {})
            node = {
                "id": site_id,
                "label": f"Monitoring site {site_id}",
                "zone": "Salish Sea",
                "x": site.x,
                "y": site.y,
                "latitude": display_meta.get("latitude"),
                "longitude": display_meta.get("longitude"),
                "habitat_label": display_meta.get("habitat_label"),
                "substrate": display_meta.get("substrate"),
                "shoreline_type": display_meta.get("shoreline_type"),
                "exposure": display_meta.get("exposure"),
                "eelgrass": display_meta.get("eelgrass"),
                "salt_marsh": display_meta.get("salt_marsh"),
                "temperature_median_c": display_meta.get("temperature_median_c"),
                "temperature_min_c": display_meta.get("temperature_min_c"),
                "temperature_max_c": display_meta.get("temperature_max_c"),
                "temperature_n": display_meta.get("temperature_n", 0),
                "temperature_years": display_meta.get("temperature_years", []),
                "temperature_source": display_meta.get("temperature_source"),
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
                "recommended_effort": (
                    int(recommendation_by_site[site_id]["recommended_effort"])
                    if site_id in recommendation_by_site
                    else None
                ),
                "effort_recommendation": (
                    dict(recommendation_by_site[site_id]["effort_recommendation"])
                    if site_id in recommendation_by_site
                    else None
                ),
            }
            if self._revealed_world is not None:
                node["true_occupied"] = bool(
                    self._revealed_world.occupied_by_site[site_id]
                )
            nodes.append(node)

        initial_meta = self._site_display_metadata.get(public.initial_detection, {})
        initial_lat = initial_meta.get("latitude")
        initial_lon = initial_meta.get("longitude")
        if initial_lat is not None and initial_lon is not None:
            for node in nodes:
                if node["latitude"] is None or node["longitude"] is None:
                    node["distance_from_detection_km"] = None
                    continue
                node["distance_from_detection_km"] = _haversine_km(
                    float(initial_lat),
                    float(initial_lon),
                    float(node["latitude"]),
                    float(node["longitude"]),
                )
        else:
            for node in nodes:
                node["distance_from_detection_km"] = None

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
            marine_performance = self._comparison_receipt["marine"]
            static_performance = self._comparison_receipt["static"]
            you_performance = self._judge_performance(self._revealed_world)
            effort_saved_vs_static = (
                int(static_performance["effort_spent"])
                - int(marine_performance["effort_spent"])
            )
            performance = {
                "marine": marine_performance,
                "static": static_performance,
                "you": you_performance,
                "resource_receipt": {
                    "marine_effort_spent": int(marine_performance["effort_spent"]),
                    "static_effort_spent": int(static_performance["effort_spent"]),
                    "effort_saved_vs_static": effort_saved_vs_static,
                    "effort_reduction_fraction_vs_static": (
                        effort_saved_vs_static / float(static_performance["effort_spent"])
                        if static_performance["effort_spent"]
                        else 0.0
                    ),
                    "marine_detected_occupied": int(
                        marine_performance["detected_occupied"]
                    ),
                    "static_detected_occupied": int(
                        static_performance["detected_occupied"]
                    ),
                    "detected_delta_marine_minus_static": int(
                        marine_performance["detected_occupied"]
                    )
                    - int(static_performance["detected_occupied"]),
                    "claim_scope": (
                        "Illustrative blinded synthetic incident on the real monitoring "
                        "graph; effort units are field-capacity units, not dollars."
                    ),
                },
                "interpretation": (
                    "Realized detections are stochastic. A single interactive path can "
                    "beat the adaptive response by luck or lose by luck. Decision receipts therefore "
                    "separate ex-ante action quality from realized outcome; aggregate "
                    "policy claims require the frozen case audit."
                ),
            }

        q_posterior = spatial.q_posterior()
        q_values = sorted(q_posterior)
        low_q = q_values[0]
        high_q = q_values[-1]
        low_mass = float(q_posterior[low_q])
        high_mass = float(q_posterior[high_q])
        dominant_edge = "low" if low_mass >= high_mass else "high"
        dominant_edge_mass = max(low_mass, high_mass)
        q_diagnostics = {
            "support": q_values,
            "posterior": {str(q): float(weight) for q, weight in q_posterior.items()},
            "mean": spatial.q_mean(),
            "low_edge_mass": low_mass,
            "high_edge_mass": high_mass,
            "dominant_edge": dominant_edge,
            "dominant_edge_mass": dominant_edge_mass,
            "boundary_pressure": dominant_edge_mass,
            "edge_note": (
                "Mass near one edge of the tested q support is a boundary-pressure "
                "diagnostic, not a stress score. It can suggest that q support is narrow "
                "or that occupancy and detectability remain confounded. Posterior-"
                "predictive model stress is computed separately from observation surprise."
            ),
        }

        temperature_count = sum(
            node["temperature_median_c"] is not None for node in nodes
        )
        layers = {
            "belief": {"available": True, "kind": "continuous"},
            "uncertainty": {"available": True, "kind": "continuous"},
            "habitat": {"available": True, "kind": "continuous"},
            "temperature": {
                "available": temperature_count > 0,
                "kind": "continuous",
                "observed_sites": temperature_count,
                "total_sites": len(nodes),
                "provenance": "direct Crab Team logger summaries; no imputation",
            },
            "exposure": {"available": any(node["exposure"] for node in nodes), "kind": "ordinal"},
            "eelgrass": {"available": any(node["eelgrass"] for node in nodes), "kind": "ordinal"},
            "salt_marsh": {"available": any(node["salt_marsh"] for node in nodes), "kind": "ordinal"},
            "field_effort": {"available": True, "kind": "continuous"},
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
                "label": "Marine invasive species — confirmed first detection",
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
                "capacity_preserved": public.remaining_budget,
                "teams": public.teams,
                "round": public.round,
                "mission_horizon": _MAX_RESPONSE_WINDOWS,
                # Upper bound on further windows (each costs >=1 unit), not a
                # target - the tighter, more meaningful bound is whatever
                # field budget remains.
                "missions_remaining": public.remaining_budget,
                "effort_levels": list(_EFFORT_LEVELS),
                "horizon_semantics": (
                    "Response continues until the 18-unit field budget is spent, "
                    f"up to {_MAX_RESPONSE_WINDOWS} windows. Lower effort now preserves "
                    "capacity for additional searches later."
                ),
            },
            "mission": self._serialize_mission(self._mission),
            "static_response": {
                "label": "Static response",
                "semantics": "same_frontier_ranking_precommitted_at_t0",
                "plan_sites": list(self._static_plan_sites),
            },
            "global_recommendations": recommendations,
            "top_worlds": top_worlds,
            "mission_changed": mission_changed,
            "replan": replan,
            "last_round": last_round,
            "model_stress": self._last_surprise,
            "decision_receipt": self._last_decision_receipt,
            "environment_layers": layers,
            "nodes": nodes,
            "edges": [asdict(edge) for edge in public.edges],
            "events": list(self._events),
            "can_plan": loop.phase is LoopPhase.READY_TO_PLAN,
            "can_execute": loop.phase is LoopPhase.MISSION_PLANNED,
            "can_reveal": loop.phase is LoopPhase.COMPLETE,
            "revealed": loop.phase is LoopPhase.REVEALED,
            "q_posterior": q_diagnostics["posterior"],
            "q_mean": q_diagnostics["mean"],
            "q_diagnostics": q_diagnostics,
            "live_curve": self._observable_live_curve(),
            "resource_curve": self._resource_curve(),
            "resource_summary": self._resource_summary(),
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
            if hasattr(self._planner, "recommend_effort"):
                effort_recommendation = self._planner.recommend_effort(
                    spatial=spatial,
                    site_id=str(row["site_id"]),
                    remaining_budget=public.remaining_budget,
                )
            else:
                feasible = [
                    effort for effort in _EFFORT_LEVELS
                    if effort <= public.remaining_budget
                ]
                recommended = feasible[-1] if feasible else None
                effort_recommendation = {
                    "recommended_effort": recommended,
                    "effort_saved_vs_max": 0,
                    "max_feasible_effort": recommended,
                    "options": [],
                    "rule": "planner_has_no_effort_optimizer",
                }
            rows.append(
                {
                    **row,
                    "rank": index,
                    "predictive_detection": predictive,
                    "recommended_effort": effort_recommendation["recommended_effort"],
                    "effort_recommendation": effort_recommendation,
                }
            )

        return rows if limit is None else rows[:limit]

    def _counterfactual_next_without_evidence(
        self,
        transition: RoundTransition,
        *,
        spatial_before: SpatialBeliefState,
    ) -> MissionAction | None:
        """Compute the next Frontier mission with the survey recorded but evidence ignored.

        This isolates the causal question shown in the UI: did the *field result*
        change the adaptive planner's next recommendation, rather than merely advancing from the
        just-executed site to another site? The counterfactual keeps the survey action
        itself (effort spent, budget and round advanced) but removes the newly returned
        ecological outcome: detections/status and occupancy belief stay at their
        pre-observation values.
        """

        if transition.done:
            return None

        counterfactual_public = deepcopy(transition.public_state_after)
        before_by_site = {
            site.id: site for site in transition.public_state_before.sites
        }
        for site in counterfactual_public.sites:
            before = before_by_site[site.id]
            # Keep the fact that the survey occurred (effort/budget/round), but
            # erase the just-returned ecological outcome itself. That prevents a
            # detection from creating a new frontier in the counterfactual.
            site.detections = before.detections
            site.status = before.status

        counterfactual_graph = GraphStateExporter().export(
            counterfactual_public,
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
            constraints={
                "spatial_belief_state": spatial_before,
                "q_posterior": spatial_before.q_posterior(),
                "q_semantics": "belief_posterior_not_simulator_truth",
            },
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
        mission_efforts: list[int] = []
        new_field_detections = 0

        mission_receipts: list[dict[str, Any]] = []

        while (
            loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED)
            and len(mission_efforts) < _MAX_RESPONSE_WINDOWS
        ):
            spatial_before = loop.current_spatial_belief
            transition = loop.run_round()
            spent = int(transition.simulator_metrics["effort_spent"])
            cumulative_effort += spent
            mission_efforts.append(spent)
            detected = set(detected_snapshots[-1][1])

            first_observation = (
                transition.observations.observations[0]
                if transition.observations.observations
                else None
            )
            if first_observation is not None:
                predictive_detection = SpatialBeliefEngine.predictive_detection_probability(
                    spatial_before,
                    site_id=first_observation.site_id,
                    effort=first_observation.effort,
                )
                conditional_detection = (
                    MissionControlFrontierPlanner._conditional_detection_if_occupied(
                        spatial_before,
                        site_id=first_observation.site_id,
                        effort=first_observation.effort,
                    )
                )
                expected_information_gain = (
                    MissionControlFrontierPlanner._expected_information_gain_bits(
                        spatial_before,
                        site_id=first_observation.site_id,
                        effort=first_observation.effort,
                    )
                )
                mission_receipts.append(
                    {
                        "site_id": first_observation.site_id,
                        "effort": int(first_observation.effort),
                        "detection": bool(first_observation.detection),
                        "occupancy_belief_before": float(
                            spatial_before.p_by_site()[first_observation.site_id]
                        ),
                        "predictive_detection_probability": float(
                            predictive_detection
                        ),
                        "conditional_detection_if_occupied": float(
                            conditional_detection
                        ),
                        "conditional_miss_if_occupied": float(
                            1.0 - conditional_detection
                        ),
                        "expected_information_gain_bits": float(
                            expected_information_gain
                        ),
                    }
                )

            for observation in transition.observations.observations:
                mission_sites.append(observation.site_id)
                if observation.detection:
                    detected.add(observation.site_id)
                    new_field_detections += 1
            detected_snapshots.append((cumulative_effort, detected))

            if (
                len(mission_efforts) >= _MAX_RESPONSE_WINDOWS
                and loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED)
            ):
                loop.force_complete()

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
        for receipt in mission_receipts:
            true_occupied = receipt["site_id"] in occupied
            receipt["true_occupied"] = true_occupied
            if true_occupied and receipt["detection"]:
                receipt["realization"] = "occupied_and_detected"
            elif true_occupied:
                receipt["realization"] = "occupied_but_missed"
            else:
                receipt["realization"] = "surveyed_not_occupied"

        final_detected = int(curve[-1]["detected_occupied"])
        high_effort_equivalent = 6 * len(mission_efforts)
        expected_detection_sum = sum(
            float(row["predictive_detection_probability"])
            for row in mission_receipts
        )
        expected_information_sum = sum(
            float(row["expected_information_gain_bits"])
            for row in mission_receipts
        )
        return {
            "name": name,
            "occupied_total": occupied_total,
            "detected_occupied": final_detected,
            "undetected_occupied": occupied_total - final_detected,
            "mission_sites": mission_sites,
            "mission_efforts": mission_efforts,
            "mission_receipts": mission_receipts,
            "missions_completed": len(mission_efforts),
            "mission_horizon": _MAX_RESPONSE_WINDOWS,
            "field_detections_beyond_initial": new_field_detections,
            "effort_spent": cumulative_effort,
            "capacity_preserved": max(0, int(incident.budget) - cumulative_effort),
            "expected_detection_sum": expected_detection_sum,
            "expected_information_gain_bits_sum": expected_information_sum,
            "expected_information_gain_per_effort": (
                expected_information_sum / cumulative_effort
                if cumulative_effort > 0
                else None
            ),
            "effort_per_detected_occupied": (
                cumulative_effort / final_detected
                if final_detected > 0
                else None
            ),
            "high_effort_equivalent_for_same_missions": high_effort_equivalent,
            "effort_avoided_vs_always_high_for_same_missions": max(
                0, high_effort_equivalent - cumulative_effort
            ),
            "curve": curve,
        }

    def _observable_live_curve(self) -> list[dict[str, Any]]:
        """Observable live-demo trajectory; contains no hidden occupancy truth."""

        cumulative_effort = 0
        cumulative_detections = 1  # confirmed initial detection
        rows: list[dict[str, Any]] = [
            {
                "round": 0,
                "effort": 0,
                "field_detections": cumulative_detections,
                "budget_used_fraction": 0.0,
            }
        ]
        for index, row in enumerate(self._judge_history, start=1):
            cumulative_effort += int(row["effort_spent"])
            cumulative_detections += sum(
                int(bool(observation.detection))
                for observation in row["observations"].observations
            )
            rows.append(
                {
                    "round": index,
                    "effort": cumulative_effort,
                    "field_detections": cumulative_detections,
                    "budget_used_fraction": (
                        cumulative_effort / self._budget if self._budget else 0.0
                    ),
                }
            )
        return rows

    def _resource_curve(self) -> list[dict[str, Any]]:
        """Per-mission resource trajectory for the resource-efficiency product surface."""

        cumulative_effort = 0
        cumulative_detections = 1
        rows: list[dict[str, Any]] = [
            {
                "round": 0,
                "cumulative_effort": 0,
                "mission_effort": 0,
                "field_detections": cumulative_detections,
                "budget_remaining": self._budget,
            }
        ]
        for index, row in enumerate(self._judge_history, start=1):
            mission_effort = int(row["effort_spent"])
            cumulative_effort += mission_effort
            cumulative_detections += sum(
                int(bool(observation.detection))
                for observation in row["observations"].observations
            )
            rows.append(
                {
                    "round": index,
                    "cumulative_effort": cumulative_effort,
                    "mission_effort": mission_effort,
                    "field_detections": cumulative_detections,
                    "budget_remaining": max(0, self._budget - cumulative_effort),
                }
            )
        return rows

    def _resource_summary(self) -> dict[str, Any]:
        curve = self._resource_curve()
        spent = int(curve[-1]["cumulative_effort"]) if curve else 0
        completed = max(0, len(curve) - 1)
        high_equivalent = 6 * completed
        latest_recommendation = self._global_recommendations(limit=1)
        next_effort = (
            int(latest_recommendation[0]["recommended_effort"])
            if latest_recommendation
            else None
        )
        return {
            "effort_spent": spent,
            "budget_remaining": max(0, self._budget - spent),
            "capacity_preserved": max(0, self._budget - spent),
            "capacity_preserved_fraction": (
                max(0, self._budget - spent) / float(self._budget)
                if self._budget
                else 0.0
            ),
            "missions_completed": completed,
            "mission_horizon": _MAX_RESPONSE_WINDOWS,
            # Upper bound on further windows, not a target: each remaining
            # window costs at least 1 effort unit, so this equals whatever
            # field budget is left. The actual number of future windows
            # depends on the effort the policy chooses at each step.
            "missions_remaining": max(0, self._budget - spent),
            "confirmed_detections": int(curve[-1]["field_detections"]) if curve else 1,
            "next_recommended_effort": next_effort,
            "high_effort_equivalent_for_completed_missions": high_equivalent,
            "effort_avoided_vs_always_high_for_completed_missions": max(
                0, high_equivalent - spent
            ),
            "semantics": (
                "The response continues until the 18-unit field budget is spent (at most "
                f"{_MAX_RESPONSE_WINDOWS} windows, since every action costs at least 1 unit). "
                "Choosing lower effort now preserves capacity for additional searches later; "
                "choosing higher effort spends more of the budget for stronger detection power "
                "at the current site. No dollar conversion is assumed."
            ),
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
        mission_efforts: list[int] = []
        mission_receipts: list[dict[str, Any]] = []
        new_field_detections = 0

        for row in self._judge_history:
            spent = int(row["effort_spent"])
            cumulative_effort += spent
            mission_efforts.append(spent)
            decision_receipt = row.get("decision_receipt")
            if isinstance(decision_receipt, dict):
                receipt = dict(decision_receipt)
                site_id = str(receipt.get("site_id"))
                true_occupied = site_id in occupied
                receipt["true_occupied"] = true_occupied
                if true_occupied and bool(receipt.get("detection")):
                    receipt["realization"] = "occupied_and_detected"
                elif true_occupied:
                    receipt["realization"] = "occupied_but_missed"
                else:
                    receipt["realization"] = "surveyed_not_occupied"
                mission_receipts.append(receipt)

            for observation in row["observations"].observations:
                mission_sites.append(observation.site_id)
                if observation.detection:
                    detected.add(observation.site_id)
                    new_field_detections += 1
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
        high_effort_equivalent = 6 * len(mission_efforts)
        expected_detection_sum = sum(
            float(row["predictive_detection_probability"])
            for row in mission_receipts
            if row.get("predictive_detection_probability") is not None
        )
        return {
            "name": "you",
            "occupied_total": occupied_total,
            "detected_occupied": final_detected,
            "undetected_occupied": occupied_total - final_detected,
            "mission_sites": mission_sites,
            "mission_efforts": mission_efforts,
            "mission_receipts": mission_receipts,
            "missions_completed": len(mission_efforts),
            "mission_horizon": _MAX_RESPONSE_WINDOWS,
            "field_detections_beyond_initial": new_field_detections,
            "effort_spent": cumulative_effort,
            "capacity_preserved": max(0, self._budget - cumulative_effort),
            "expected_detection_sum": expected_detection_sum,
            "effort_per_detected_occupied": (
                cumulative_effort / final_detected
                if final_detected > 0
                else None
            ),
            "high_effort_equivalent_for_same_missions": high_effort_equivalent,
            "effort_avoided_vs_always_high_for_same_missions": max(
                0, high_effort_equivalent - cumulative_effort
            ),
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
            "recommended_effort": diagnostics.get("marine_recommended_effort")
            or diagnostics.get("effort_recommendation", {}).get("recommended_effort"),
            "diagnostics": {
                "fallback_used": diagnostics.get("fallback_used"),
                "effort_recommendation": diagnostics.get("effort_recommendation"),
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
