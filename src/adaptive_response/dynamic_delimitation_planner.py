"""R11 Dynamic Delimitation baseline (docs/R11_DYNAMIC_DELIMITATION_REPORT.md).

A deterministic, adaptive-but-not-probabilistic reactive delimitation
strategy: confirmed positive -> inspect local frontier -> survey at full
effort -> positive expands the frontier -> negative closes/deprioritizes that
branch -> replan -> repeat. This is the "credible but deliberately simpler"
operational comparator for `SiteEffortRoundPolicy` (Grab the Crab GNN/RL),
NOT a replacement for the R8/R10 planner-to-planner strong-baseline
comparison (FrontierPlanner, InformationGainPlanner).

INFORMATION FIREWALL (frozen, tested by test_dynamic_delimitation_planner.py):

Allowed  - graph adjacency (`GraphState.edge_index`), surveyed/unsurveyed and
           observed-positive/observed-negative (derived from the
           `observed_effort`/`detections` node features), static habitat
           proxy (`habitat_score`), static accessibility / water-route proxy
           (`access_cost`), remaining effort budget.
Forbidden - posterior occupancy belief, uncertainty/entropy, q posterior,
           predictive detection probability, possible worlds, world-model
           weights, any `SpatialBeliefState`, GNN embeddings, hidden truth,
           future observations.

`GraphState.node_features` columns 0 (`belief`) and 1 (`uncertainty`) are
never read by this planner - see `_ALLOWED_FEATURE_NAMES` below, which is
deliberately missing them. `constraints` (which may carry
`spatial_belief_state`/`q_by_site` for other planners) is accepted only to
satisfy the shared `Planner` protocol and is immediately discarded.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

from .graph_state import NODE_FEATURE_NAMES
from .models import GraphState, MissionAction, MissionAllocation

# Deliberately excludes "belief" and "uncertainty" - the information firewall.
_ALLOWED_FEATURE_NAMES = ("observed_effort", "detections", "habitat_score", "access_cost")
_FORBIDDEN_FEATURE_NAMES = ("belief", "uncertainty")


class DynamicDelimitationMax6Planner:
    """Local, reactive, fixed-effort delimitation baseline. Label: "Dynamic Delimitation".

    Tier 1 (active frontier): unsurveyed sites directly adjacent to >=1
    confirmed-positive site, ranked by (higher habitat suitability, lower
    access/water-route cost, deterministic site id).

    Tier 2 (ring expansion, used only when Tier 1 is empty): unsurveyed sites
    ranked by (minimum graph-hop distance from any confirmed-positive site,
    higher habitat suitability, lower access/water-route cost, deterministic
    site id).

    Effort is fixed (`effort=6` by default - DYNAMIC_EFFORT in the handoff),
    falling back to the largest feasible level in `effort_levels` only when
    the remaining budget cannot cover it (mirrors the R8 baseline-fairness
    Frontier effort rule; irrelevant for the frozen 18/e6x3 same-budget
    benchmark, matters for the Effort-to-Parity continuation at higher
    budgets that are not exact multiples of 6). No revisits: a site is
    removed from the candidate pool the moment it is surveyed (positive or
    searched-negative) and never reconsidered within a run.
    """

    def __init__(
        self,
        *,
        effort: int = 6,
        effort_levels: Sequence[int] = (1, 3, 6),
    ) -> None:
        if isinstance(effort, bool) or not isinstance(effort, int) or effort <= 0:
            raise ValueError("effort must be a positive integer.")
        if not effort_levels or any(
            isinstance(e, bool) or not isinstance(e, int) or e <= 0 for e in effort_levels
        ):
            raise ValueError("effort_levels must be a non-empty sequence of positive ints.")
        if effort not in effort_levels:
            raise ValueError("effort must be one of effort_levels.")
        self.effort = effort
        # Descending, so "largest feasible level <= remaining_budget" is levels[0].
        self.effort_levels = tuple(sorted(set(effort_levels), reverse=True))

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        del constraints  # Dynamic never reads belief/uncertainty/q/possible-worlds/hidden truth.

        if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")
        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")

        if remaining_budget == 0:
            return _empty_mission(reason="no_remaining_budget")

        feasible_effort = next((e for e in self.effort_levels if e <= remaining_budget), None)
        if feasible_effort is None:
            return _empty_mission(reason="insufficient_budget_for_any_effort_level")

        n = len(graph_state.node_ids)
        obs_idx = NODE_FEATURE_NAMES.index("observed_effort")
        det_idx = NODE_FEATURE_NAMES.index("detections")
        habitat_idx = NODE_FEATURE_NAMES.index("habitat_score")
        access_idx = NODE_FEATURE_NAMES.index("access_cost")

        detections = [graph_state.node_features[i][det_idx] for i in range(n)]
        observed_effort = [graph_state.node_features[i][obs_idx] for i in range(n)]
        habitat = [graph_state.node_features[i][habitat_idx] for i in range(n)]
        access = [graph_state.node_features[i][access_idx] for i in range(n)]

        positive = {i for i in range(n) if detections[i] > 0}
        # Unsurveyed excludes the initial confirmed-positive site too (it has
        # observed_effort == 0 but is already known-positive; see
        # Environment._reset_site) - "positive" always wins the partition.
        unsurveyed = {
            i
            for i in range(n)
            if i not in positive
            and observed_effort[i] <= 0
            and graph_state.feasibility_mask[i]
        }

        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
        src_indices, dst_indices = graph_state.edge_index
        for s, d in zip(src_indices, dst_indices):
            adjacency[s].add(d)

        active_frontier = {i for i in unsurveyed if adjacency[i] & positive}

        if active_frontier:
            tier = "active_frontier"
            candidates = active_frontier
            hop_by_index: dict[int, int] = {}
        else:
            hop_by_index = _min_hops_from(positive, adjacency, n)
            reachable = {i for i in unsurveyed if i in hop_by_index}
            if reachable:
                tier = "ring_expansion"
                candidates = reachable
            elif unsurveyed:
                # Disconnected component with no positive site reachable at all
                # (should not occur on a connected monitoring graph); fall back
                # to the same allowed static features rather than crashing.
                tier = "ring_expansion_disconnected_fallback"
                candidates = unsurveyed
            else:
                return _empty_mission(reason="no_unsurveyed_sites_remaining")

        if tier == "active_frontier":
            def sort_key(i: int) -> tuple[Any, ...]:
                return (-habitat[i], access[i], graph_state.node_ids[i])
        else:
            def sort_key(i: int) -> tuple[Any, ...]:
                return (hop_by_index.get(i, float("inf")), -habitat[i], access[i], graph_state.node_ids[i])

        chosen_index = min(candidates, key=sort_key)
        site_id = graph_state.node_ids[chosen_index]

        return MissionAction(
            allocations=(MissionAllocation(site_id=site_id, effort_units=feasible_effort),),
            total_cost=feasible_effort,
            diagnostics={
                "planner": "dynamic_delimitation",
                "tier": tier,
                "candidate_count": len(candidates),
                "positive_site_count": len(positive),
                "allowed_features": _ALLOWED_FEATURE_NAMES,
                "forbidden_features_not_read": _FORBIDDEN_FEATURE_NAMES,
            },
        )


def _empty_mission(*, reason: str) -> MissionAction:
    return MissionAction(
        allocations=(),
        total_cost=0,
        diagnostics={"planner": "dynamic_delimitation", "reason": reason},
    )


def _min_hops_from(
    sources: set[int], adjacency: dict[int, set[int]], n: int
) -> dict[int, int]:
    """Unweighted multi-source BFS hop distance from any site in `sources`."""
    hops: dict[int, int] = {s: 0 for s in sources}
    frontier = deque(sources)
    while frontier:
        current = frontier.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in hops:
                hops[neighbor] = hops[current] + 1
                frontier.append(neighbor)
    return hops
