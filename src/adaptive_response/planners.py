from __future__ import annotations

from math import log2
from typing import Any, Mapping, Protocol

from .graph_state import NODE_FEATURE_NAMES
from .models import (
    GraphState,
    MissionAction,
    MissionAllocation,
    Observation,
    ObservationBatch,
)
from .spatial_belief import SpatialBeliefEngine, SpatialBeliefState


class Planner(Protocol):
    """Shared planner contract used by heuristic, information-gain and RL planners."""

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        ...


class FrontierPlanner:
    """Transparent frontier-first baseline for rapid-response delimitation.

    CURRENT DEFAULT ranking for M3:
    1. frontier nodes first;
    2. higher occupancy belief;
    3. higher uncertainty;
    4. deterministic site_id tie-break.

    If budget remains after feasible frontier nodes are ranked, the same
    belief/uncertainty ordering is used for remaining feasible nodes. This is a
    baseline decision rule, not an ecological optimality claim.

    `effort_per_site` is configurable so M3 does not freeze the later RL action
    space. The planner never sees HiddenWorld and uses only GraphState + budget.
    """

    def __init__(self, *, effort_per_site: int = 1, max_sites: int | None = None) -> None:
        if not isinstance(effort_per_site, int) or effort_per_site <= 0:
            raise ValueError("effort_per_site must be a positive integer.")
        if max_sites is not None and (
            not isinstance(max_sites, int) or max_sites <= 0
        ):
            raise ValueError("max_sites must be a positive integer when provided.")
        self.effort_per_site = effort_per_site
        self.max_sites = max_sites

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        del constraints  # Frontier v0 intentionally does not use q or hidden context.

        if not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")
        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")
        if len(graph_state.node_ids) != len(graph_state.feasibility_mask):
            raise ValueError("GraphState feasibility_mask is misaligned with nodes.")
        if remaining_budget == 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={
                    "planner": "frontier",
                    "reason": "no_remaining_budget",
                    "fallback_used": False,
                },
            )

        belief_idx = NODE_FEATURE_NAMES.index("belief")
        uncertainty_idx = NODE_FEATURE_NAMES.index("uncertainty")
        frontier_idx = NODE_FEATURE_NAMES.index("frontier")

        candidates: list[tuple[str, float, float, bool]] = []
        for i, site_id in enumerate(graph_state.node_ids):
            if not graph_state.feasibility_mask[i]:
                continue
            features = graph_state.node_features[i]
            if len(features) != len(NODE_FEATURE_NAMES):
                raise ValueError(
                    f"Node {site_id!r} has {len(features)} features; "
                    f"expected {len(NODE_FEATURE_NAMES)}."
                )
            candidates.append(
                (
                    site_id,
                    float(features[belief_idx]),
                    float(features[uncertainty_idx]),
                    bool(features[frontier_idx] > 0.5),
                )
            )

        candidates.sort(
            key=lambda row: (
                -int(row[3]),
                -row[1],
                -row[2],
                row[0],
            )
        )
        if self.max_sites is not None:
            candidates = candidates[: self.max_sites]

        budget_left = remaining_budget
        allocations: list[MissionAllocation] = []
        ranking_diagnostics: list[dict[str, Any]] = []

        for site_id, belief, uncertainty, is_frontier in candidates:
            if budget_left <= 0:
                break
            effort = min(self.effort_per_site, budget_left)
            allocations.append(
                MissionAllocation(site_id=site_id, effort_units=effort)
            )
            ranking_diagnostics.append(
                {
                    "site_id": site_id,
                    "belief": belief,
                    "uncertainty": uncertainty,
                    "frontier": is_frontier,
                    "effort_units": effort,
                }
            )
            budget_left -= effort

        total_cost = sum(a.effort_units for a in allocations)
        selected_non_frontier = any(not row["frontier"] for row in ranking_diagnostics)
        no_frontier_available = not any(row[3] for row in candidates)
        return MissionAction(
            allocations=tuple(allocations),
            total_cost=total_cost,
            diagnostics={
                "planner": "frontier",
                "fallback_used": selected_non_frontier or no_frontier_available,
                "effort_per_site": self.effort_per_site,
                "ranked_selected_sites": tuple(ranking_diagnostics),
            },
        )


class InformationGainPlanner:
    """Greedy one-step entropy/VOI baseline under imperfect detection.

    Strong mode consumes an explicit ``SpatialBeliefState`` supplied through the
    observable planner constraints under ``spatial_belief_state``. For each feasible
    site it computes the exact finite-ensemble expected reduction in the *sum of
    marginal occupancy entropies* after either a detection or non-detection at the
    proposed effort. This captures spatial belief propagation and marginalizes the
    posterior q uncertainty represented by ``SpatialBeliefEngine``.

    A site-local fallback is retained for integration tests and legacy loops. It uses
    only the node occupancy marginal and scalar ``q_by_site`` and therefore does not
    capture cross-site information. Formal competitive benchmarking should require
    ``mode == 'spatial_joint'``.

    The planner is deliberately myopic: it values the next survey before seeing its
    outcome, then the outer adaptive loop replans after field evidence arrives. With
    ``max_sites=1`` (CURRENT DEFAULT for this class) this cleanly demonstrates the
    project's sequential OBSERVE -> INFER -> DECIDE -> SURVEY -> LEARN -> REPLAN loop.
    ``max_sites`` and ``effort_per_site`` remain configurable and are not frozen by
    this implementation.
    """

    def __init__(
        self,
        *,
        effort_per_site: int = 1,
        max_sites: int | None = 1,
        require_spatial_belief: bool = False,
    ) -> None:
        if isinstance(effort_per_site, bool) or not isinstance(effort_per_site, int) or effort_per_site <= 0:
            raise ValueError("effort_per_site must be a positive integer.")
        if max_sites is not None and (
            isinstance(max_sites, bool) or not isinstance(max_sites, int) or max_sites <= 0
        ):
            raise ValueError("max_sites must be a positive integer when provided.")
        self.effort_per_site = effort_per_site
        self.max_sites = max_sites
        self.require_spatial_belief = bool(require_spatial_belief)

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")
        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")
        if len(graph_state.node_ids) != len(graph_state.feasibility_mask):
            raise ValueError("GraphState feasibility_mask is misaligned with nodes.")
        if remaining_budget == 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={
                    "planner": "information_gain",
                    "reason": "no_remaining_budget",
                    "mode": "not_evaluated",
                },
            )

        spatial = constraints.get("spatial_belief_state")
        if spatial is not None and not isinstance(spatial, SpatialBeliefState):
            raise ValueError("constraints['spatial_belief_state'] must be SpatialBeliefState.")
        if self.require_spatial_belief and spatial is None:
            raise ValueError(
                "InformationGainPlanner formal mode requires constraints['spatial_belief_state']."
            )
        if spatial is not None and set(spatial.site_ids) != set(graph_state.node_ids):
            raise ValueError("Spatial belief site ids must match GraphState node ids.")

        effort = min(self.effort_per_site, remaining_budget)
        belief_idx = NODE_FEATURE_NAMES.index("belief")
        uncertainty_idx = NODE_FEATURE_NAMES.index("uncertainty")
        q_by_site = constraints.get("q_by_site", {})
        if spatial is None and not isinstance(q_by_site, Mapping):
            raise ValueError("constraints['q_by_site'] must be a mapping in site-local mode.")

        scored: list[dict[str, Any]] = []
        for index, site_id in enumerate(graph_state.node_ids):
            if not graph_state.feasibility_mask[index]:
                continue
            features = graph_state.node_features[index]
            if len(features) != len(NODE_FEATURE_NAMES):
                raise ValueError(
                    f"Node {site_id!r} has {len(features)} features; expected {len(NODE_FEATURE_NAMES)}."
                )
            belief = float(features[belief_idx])
            uncertainty = float(features[uncertainty_idx])
            if spatial is not None:
                score, p_detection, expected_after = self._spatial_score(
                    spatial,
                    site_id=site_id,
                    effort=effort,
                )
                mode = "spatial_joint"
            else:
                if site_id not in q_by_site:
                    raise ValueError(f"q_by_site is missing site {site_id!r}.")
                q = float(q_by_site[site_id])
                score, p_detection, expected_after = self._local_score(
                    belief=belief,
                    q=q,
                    effort=effort,
                )
                mode = "site_local_fallback"

            scored.append(
                {
                    "site_id": site_id,
                    "expected_information_gain_bits": score,
                    "predictive_detection_probability": p_detection,
                    "expected_entropy_after_bits": expected_after,
                    "belief": belief,
                    "uncertainty": uncertainty,
                }
            )

        scored.sort(
            key=lambda row: (
                -float(row["expected_information_gain_bits"]),
                -float(row["uncertainty"]),
                -float(row["belief"]),
                str(row["site_id"]),
            )
        )
        selected = scored if self.max_sites is None else scored[: self.max_sites]

        budget_left = remaining_budget
        allocations: list[MissionAllocation] = []
        selected_diagnostics: list[dict[str, Any]] = []
        for row in selected:
            if budget_left <= 0:
                break
            allocated_effort = min(self.effort_per_site, budget_left)
            allocations.append(
                MissionAllocation(
                    site_id=str(row["site_id"]),
                    effort_units=allocated_effort,
                )
            )
            selected_diagnostics.append({**row, "effort_units": allocated_effort})
            budget_left -= allocated_effort

        total_cost = sum(allocation.effort_units for allocation in allocations)
        resolved_mode = "spatial_joint" if spatial is not None else "site_local_fallback"
        return MissionAction(
            allocations=tuple(allocations),
            total_cost=total_cost,
            diagnostics={
                "planner": "information_gain",
                "mode": resolved_mode,
                "objective": "expected_reduction_sum_marginal_occupancy_entropy_bits",
                "effort_per_site": self.effort_per_site,
                "max_sites": self.max_sites,
                "ranked_candidates": tuple(scored),
                "ranked_selected_sites": tuple(selected_diagnostics),
                "hidden_truth_used": False,
            },
        )

    @staticmethod
    def _spatial_score(
        belief: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> tuple[float, float, float]:
        current_entropy = sum(belief.uncertainty_by_site().values())
        p_detection = SpatialBeliefEngine.predictive_detection_probability(
            belief,
            site_id=site_id,
            effort=effort,
        )
        p_no_detection = 1.0 - p_detection
        expected_after = 0.0

        if p_detection > 0.0:
            detected = Observation(
                site_id=site_id,
                effort=effort,
                detection=True,
                round=0,
            )
            posterior = SpatialBeliefEngine.update(
                belief,
                ObservationBatch(
                    observations=(detected,),
                    round=0,
                    total_effort=effort,
                ),
            )
            expected_after += p_detection * sum(posterior.uncertainty_by_site().values())

        if p_no_detection > 0.0:
            not_detected = Observation(
                site_id=site_id,
                effort=effort,
                detection=False,
                round=0,
            )
            posterior = SpatialBeliefEngine.update(
                belief,
                ObservationBatch(
                    observations=(not_detected,),
                    round=0,
                    total_effort=effort,
                ),
            )
            expected_after += p_no_detection * sum(posterior.uncertainty_by_site().values())

        information_gain = max(0.0, current_entropy - expected_after)
        return information_gain, p_detection, expected_after

    @staticmethod
    def _local_score(
        *,
        belief: float,
        q: float,
        effort: int,
    ) -> tuple[float, float, float]:
        if not 0.0 <= belief <= 1.0:
            raise ValueError("belief must be in [0, 1].")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        detection_if_occupied = 1.0 - (1.0 - q) ** effort
        p_detection = belief * detection_if_occupied
        p_no_detection = 1.0 - p_detection
        prior_entropy = _bernoulli_entropy(belief)
        if p_no_detection <= 0.0:
            expected_after = 0.0
        else:
            posterior_no_detection = (
                belief * (1.0 - detection_if_occupied) / p_no_detection
            )
            expected_after = p_no_detection * _bernoulli_entropy(posterior_no_detection)
        return max(0.0, prior_entropy - expected_after), p_detection, expected_after


def _bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p))
