from __future__ import annotations

from math import log2
from typing import Any, Mapping, Protocol, Sequence

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

    rank_candidates exposes the exact same ranking used by plan so product
    surfaces can explain and display several alternatives without re-implementing
    planner semantics in the UI layer.
    """

    def __init__(
        self,
        *,
        effort_per_site: int = 1,
        max_sites: int | None = None,
        effort_levels: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(effort_per_site, int) or effort_per_site <= 0:
            raise ValueError("effort_per_site must be a positive integer.")
        if max_sites is not None and (
            not isinstance(max_sites, int) or max_sites <= 0
        ):
            raise ValueError("max_sites must be a positive integer when provided.")
        if effort_levels is not None:
            if not effort_levels or any(
                isinstance(e, bool) or not isinstance(e, int) or e <= 0
                for e in effort_levels
            ):
                raise ValueError(
                    "effort_levels must be a non-empty sequence of positive ints."
                )
        self.effort_per_site = effort_per_site
        self.max_sites = max_sites
        self.effort_levels = (
            tuple(sorted(effort_levels, reverse=True))
            if effort_levels is not None
            else None
        )

    def rank_candidates(self, graph_state: GraphState) -> tuple[dict[str, Any], ...]:
        """Return all feasible candidates in the exact deterministic Frontier order."""
        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")
        if len(graph_state.node_ids) != len(graph_state.feasibility_mask):
            raise ValueError("GraphState feasibility_mask is misaligned with nodes.")

        belief_idx = NODE_FEATURE_NAMES.index("belief")
        uncertainty_idx = NODE_FEATURE_NAMES.index("uncertainty")
        frontier_idx = NODE_FEATURE_NAMES.index("frontier")

        candidates: list[dict[str, Any]] = []
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
                {
                    "site_id": site_id,
                    "belief": float(features[belief_idx]),
                    "uncertainty": float(features[uncertainty_idx]),
                    "frontier": bool(features[frontier_idx] > 0.5),
                }
            )

        candidates.sort(
            key=lambda row: (
                -int(bool(row["frontier"])),
                -float(row["belief"]),
                -float(row["uncertainty"]),
                str(row["site_id"]),
            )
        )
        return tuple(candidates)

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        del constraints

        if not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")
        if remaining_budget == 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={
                    "planner": "frontier",
                    "reason": "no_remaining_budget",
                    "fallback_used": False,
                    "ranked_candidates": (),
                },
            )

        all_candidates = list(self.rank_candidates(graph_state))
        selected_candidates = (
            all_candidates
            if self.max_sites is None
            else all_candidates[: self.max_sites]
        )

        budget_left = remaining_budget
        allocations: list[MissionAllocation] = []
        selected_diagnostics: list[dict[str, Any]] = []

        for row in selected_candidates:
            if budget_left <= 0:
                break
            if self.effort_levels is not None:
                feasible_levels = [e for e in self.effort_levels if e <= budget_left]
                if not feasible_levels:
                    break
                effort = feasible_levels[0]
            else:
                effort = min(self.effort_per_site, budget_left)

            allocations.append(
                MissionAllocation(
                    site_id=str(row["site_id"]),
                    effort_units=effort,
                )
            )
            selected_diagnostics.append({**row, "effort_units": effort})
            budget_left -= effort

        total_cost = sum(a.effort_units for a in allocations)
        selected_non_frontier = any(
            not bool(row["frontier"]) for row in selected_diagnostics
        )
        no_frontier_available = not any(
            bool(row["frontier"]) for row in all_candidates
        )
        return MissionAction(
            allocations=tuple(allocations),
            total_cost=total_cost,
            diagnostics={
                "planner": "frontier",
                "fallback_used": selected_non_frontier or no_frontier_available,
                "effort_per_site": self.effort_per_site,
                "ranked_selected_sites": tuple(selected_diagnostics),
                "ranked_candidates": tuple(all_candidates),
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
        effort_levels: Sequence[int] | None = None,
    ) -> None:
        if isinstance(effort_per_site, bool) or not isinstance(effort_per_site, int) or effort_per_site <= 0:
            raise ValueError("effort_per_site must be a positive integer.")
        if max_sites is not None and (
            isinstance(max_sites, bool) or not isinstance(max_sites, int) or max_sites <= 0
        ):
            raise ValueError("max_sites must be a positive integer when provided.")
        if effort_levels is not None:
            if not effort_levels or any(
                isinstance(e, bool) or not isinstance(e, int) or e <= 0 for e in effort_levels
            ):
                raise ValueError("effort_levels must be a non-empty sequence of positive ints.")
        self.effort_per_site = effort_per_site
        self.max_sites = max_sites
        self.require_spatial_belief = bool(require_spatial_belief)
        # R7 action contract (docs/DEMU_HANDOFF_R7.md): when set, `plan()`
        # searches (site, effort) jointly for every effort in effort_levels
        # that fits the remaining budget, ranking by expected information gain
        # *per effort unit* (absolute IG as tie-break), per
        # benchmark_protocol_r7.json's recommended_information_gain_contract.
        # None (default) preserves the original fixed-effort_per_site behavior
        # exactly, unchanged - this is purely additive.
        self.effort_levels = tuple(sorted(effort_levels)) if effort_levels is not None else None

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

        candidate_efforts = (
            [e for e in self.effort_levels if e <= remaining_budget]
            if self.effort_levels is not None
            else None
        )

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

            if candidate_efforts is not None:
                if not candidate_efforts:
                    continue
                mode = "spatial_joint" if spatial is not None else "site_local_fallback"
                best: dict[str, Any] | None = None
                for candidate_effort in candidate_efforts:
                    if spatial is not None:
                        c_score, c_p_detection, c_expected_after = self._spatial_score(
                            spatial, site_id=site_id, effort=candidate_effort,
                        )
                    else:
                        if site_id not in q_by_site:
                            raise ValueError(f"q_by_site is missing site {site_id!r}.")
                        c_score, c_p_detection, c_expected_after = self._local_score(
                            belief=belief, q=float(q_by_site[site_id]), effort=candidate_effort,
                        )
                    c_ig_per_effort = c_score / candidate_effort
                    # R8 baseline-fairness correction: absolute IG is the
                    # primary objective, IG-per-effort only a tie-break -
                    # reversed from the R7-provisional per-effort-first
                    # ranking, which systematically preferred effort=1 and
                    # left most of the budget unused (docs/R8_DEMU_BENCHMARK_REVIEW.md
                    # BLOCKER 2).
                    if best is None or (c_score, c_ig_per_effort) > (best["score"], best["ig_per_effort"]):
                        best = {
                            "effort": candidate_effort,
                            "score": c_score,
                            "ig_per_effort": c_ig_per_effort,
                            "p_detection": c_p_detection,
                            "expected_after": c_expected_after,
                        }
                assert best is not None
                score, p_detection, expected_after = best["score"], best["p_detection"], best["expected_after"]
                effort_for_site = best["effort"]
                ig_per_effort = best["ig_per_effort"]
            else:
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
                effort_for_site = effort
                ig_per_effort = score / effort if effort > 0 else 0.0

            scored.append(
                {
                    "site_id": site_id,
                    "expected_information_gain_bits": score,
                    "information_gain_per_effort_unit": ig_per_effort,
                    "predictive_detection_probability": p_detection,
                    "expected_entropy_after_bits": expected_after,
                    "belief": belief,
                    "uncertainty": uncertainty,
                    "effort_units": effort_for_site,
                }
            )

        if candidate_efforts is not None:
            # R8: absolute expected IG is the primary ranking objective across
            # sites too; IG-per-effort is only a tie-break (see comment above).
            scored.sort(
                key=lambda row: (
                    -float(row["expected_information_gain_bits"]),
                    -float(row["information_gain_per_effort_unit"]),
                    -float(row["uncertainty"]),
                    -float(row["belief"]),
                    str(row["site_id"]),
                )
            )
        else:
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
            allocated_effort = (
                min(int(row["effort_units"]), budget_left)
                if candidate_efforts is not None
                else min(self.effort_per_site, budget_left)
            )
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
