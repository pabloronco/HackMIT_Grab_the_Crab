import pytest

from adaptive_response import FrontierPlanner, GraphState, NODE_FEATURE_NAMES


def node_row(
    *,
    belief: float,
    uncertainty: float,
    frontier: float,
    observed_effort: float = 0.0,
    detections: float = 0.0,
    habitat_score: float = 0.5,
    access_cost: float = 1.0,
) -> tuple[float, ...]:
    values = {
        "belief": belief,
        "uncertainty": uncertainty,
        "observed_effort": observed_effort,
        "detections": detections,
        "habitat_score": habitat_score,
        "access_cost": access_cost,
        "frontier": frontier,
    }
    return tuple(float(values[name]) for name in NODE_FEATURE_NAMES)


def graph(
    rows: list[tuple[str, tuple[float, ...], bool]],
) -> GraphState:
    return GraphState(
        node_ids=tuple(site_id for site_id, _, _ in rows),
        node_features=tuple(features for _, features, _ in rows),
        edge_index=((), ()),
        edge_features=(),
        global_features=(10.0, 0.0, 2.0, 0.8),
        feasibility_mask=tuple(feasible for _, _, feasible in rows),
    )


def selected_ids(action):
    return [allocation.site_id for allocation in action.allocations]


def test_frontier_nodes_are_preferred_over_non_frontier_nodes() -> None:
    state = graph(
        [
            ("site_a", node_row(belief=0.95, uncertainty=0.2, frontier=0), True),
            ("site_b", node_row(belief=0.40, uncertainty=0.9, frontier=1), True),
            ("site_c", node_row(belief=0.60, uncertainty=0.8, frontier=1), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=2, constraints={})

    assert selected_ids(action) == ["site_c", "site_b"]
    assert "site_a" not in selected_ids(action)
    assert action.diagnostics["fallback_used"] is False


def test_frontier_ranking_uses_belief_then_uncertainty() -> None:
    state = graph(
        [
            ("site_a", node_row(belief=0.6, uncertainty=0.2, frontier=1), True),
            ("site_b", node_row(belief=0.7, uncertainty=0.1, frontier=1), True),
            ("site_c", node_row(belief=0.6, uncertainty=0.9, frontier=1), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=3, constraints={})

    assert selected_ids(action) == ["site_b", "site_c", "site_a"]


def test_site_id_is_deterministic_final_tie_break() -> None:
    state = graph(
        [
            ("site_c", node_row(belief=0.5, uncertainty=0.5, frontier=1), True),
            ("site_a", node_row(belief=0.5, uncertainty=0.5, frontier=1), True),
            ("site_b", node_row(belief=0.5, uncertainty=0.5, frontier=1), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=3, constraints={})

    assert selected_ids(action) == ["site_a", "site_b", "site_c"]


def test_infeasible_nodes_are_never_selected() -> None:
    state = graph(
        [
            ("blocked", node_row(belief=0.99, uncertainty=1.0, frontier=1), False),
            ("open", node_row(belief=0.4, uncertainty=0.4, frontier=1), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=2, constraints={})

    assert selected_ids(action) == ["open"]
    assert action.total_cost == 1


def test_fallback_ranks_all_feasible_nodes_when_no_frontier_exists() -> None:
    state = graph(
        [
            ("site_a", node_row(belief=0.2, uncertainty=0.9, frontier=0), True),
            ("site_b", node_row(belief=0.8, uncertainty=0.1, frontier=0), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=1, constraints={})

    assert selected_ids(action) == ["site_b"]
    assert action.diagnostics["fallback_used"] is True


def test_remaining_budget_can_flow_to_non_frontier_after_frontier_nodes() -> None:
    state = graph(
        [
            ("frontier", node_row(belief=0.3, uncertainty=0.3, frontier=1), True),
            ("non_frontier", node_row(belief=0.9, uncertainty=0.9, frontier=0), True),
        ]
    )

    action = FrontierPlanner().plan(state, remaining_budget=2, constraints={})

    assert selected_ids(action) == ["frontier", "non_frontier"]
    assert action.total_cost == 2
    assert action.diagnostics["fallback_used"] is True


def test_configurable_effort_respects_budget_and_partial_last_allocation() -> None:
    state = graph(
        [
            ("site_a", node_row(belief=0.9, uncertainty=0.9, frontier=1), True),
            ("site_b", node_row(belief=0.8, uncertainty=0.8, frontier=1), True),
        ]
    )

    action = FrontierPlanner(effort_per_site=3).plan(
        state, remaining_budget=5, constraints={}
    )

    assert [(a.site_id, a.effort_units) for a in action.allocations] == [
        ("site_a", 3),
        ("site_b", 2),
    ]
    assert action.total_cost == 5


def test_zero_budget_returns_empty_valid_action() -> None:
    state = graph(
        [("site_a", node_row(belief=0.9, uncertainty=0.9, frontier=1), True)]
    )

    action = FrontierPlanner().plan(state, remaining_budget=0, constraints={})

    assert action.allocations == ()
    assert action.total_cost == 0
    assert action.diagnostics["reason"] == "no_remaining_budget"


def test_effort_levels_use_standard_event_effort_when_budget_permits() -> None:
    """R8 baseline-fairness correction (docs/R8_DEMU_BENCHMARK_REVIEW.md,
    configs/benchmark_protocol_r8.json): Frontier must use the standard-event
    effort level (the largest in effort_levels) whenever the remaining budget
    allows it, not a fixed small effort_per_site that strands most of the
    budget unused."""

    state = graph(
        [
            ("site_a", node_row(belief=0.9, uncertainty=0.9, frontier=1), True),
            ("site_b", node_row(belief=0.5, uncertainty=0.5, frontier=1), True),
            ("site_c", node_row(belief=0.1, uncertainty=0.1, frontier=1), True),
        ]
    )
    planner = FrontierPlanner(effort_levels=(1, 3, 6))

    action = planner.plan(state, remaining_budget=18, constraints={})

    assert [(a.site_id, a.effort_units) for a in action.allocations] == [
        ("site_a", 6), ("site_b", 6), ("site_c", 6),
    ]
    assert action.total_cost == 18


def test_effort_levels_fall_back_to_largest_feasible_level() -> None:
    state = graph(
        [
            ("site_a", node_row(belief=0.9, uncertainty=0.9, frontier=1), True),
            ("site_b", node_row(belief=0.5, uncertainty=0.5, frontier=1), True),
        ]
    )
    planner = FrontierPlanner(effort_levels=(1, 3, 6))

    action = planner.plan(state, remaining_budget=4, constraints={})

    # site_a takes the standard effort=3 (largest <=4 remaining), leaving 1
    # for site_b - not stuck at a fixed effort_per_site=1 the whole time.
    assert [(a.site_id, a.effort_units) for a in action.allocations] == [
        ("site_a", 3), ("site_b", 1),
    ]
    assert action.total_cost == 4


def test_effort_levels_default_none_preserves_original_fixed_effort_behavior() -> None:
    state = graph([("site_a", node_row(belief=0.9, uncertainty=0.9, frontier=1), True)])
    action = FrontierPlanner(effort_per_site=1).plan(state, remaining_budget=6, constraints={})
    assert action.allocations[0].effort_units == 1


def test_invalid_feature_width_is_rejected() -> None:
    state = GraphState(
        node_ids=("site_a",),
        node_features=((0.5, 0.5),),
        edge_index=((), ()),
        edge_features=(),
        global_features=(10.0, 0.0, 2.0, 0.5),
        feasibility_mask=(True,),
    )

    with pytest.raises(ValueError, match="expected"):
        FrontierPlanner().plan(state, remaining_budget=1, constraints={})


def test_rank_candidates_matches_plan_order_without_spending_budget() -> None:
    state = graph(
        [
            ("site_c", node_row(belief=0.7, uncertainty=0.4, frontier=0), True),
            ("site_a", node_row(belief=0.4, uncertainty=0.8, frontier=1), True),
            ("site_b", node_row(belief=0.6, uncertainty=0.2, frontier=1), True),
        ]
    )
    planner = FrontierPlanner()

    ranked = planner.rank_candidates(state)
    action = planner.plan(state, remaining_budget=3, constraints={})

    assert [row["site_id"] for row in ranked] == ["site_b", "site_a", "site_c"]
    assert selected_ids(action) == [row["site_id"] for row in ranked]
