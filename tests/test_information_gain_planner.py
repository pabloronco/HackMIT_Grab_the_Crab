import pytest

from adaptive_response import (
    EcologicalHypothesis,
    GraphState,
    InformationGainPlanner,
    QHypothesis,
    SpatialBeliefEngine,
)
from adaptive_response.graph_state import NODE_FEATURE_NAMES


def graph_from_beliefs(p_by_site: dict[str, float], uncertainty_by_site: dict[str, float]) -> GraphState:
    node_ids = tuple(sorted(p_by_site))
    rows = []
    for site_id in node_ids:
        values = {
            "belief": p_by_site[site_id],
            "uncertainty": uncertainty_by_site[site_id],
            "observed_effort": 0.0,
            "detections": 0.0,
            "habitat_score": 0.5,
            "access_cost": 1.0,
            "frontier": 0.0,
        }
        rows.append(tuple(float(values[name]) for name in NODE_FEATURE_NAMES))
    return GraphState(
        node_ids=node_ids,
        node_features=tuple(rows),
        edge_index=((), ()),
        edge_features=(),
        global_features=(10.0, 0.0, 1.0, 1.0),
        feasibility_mask=tuple(True for _ in node_ids),
    )


def correlated_spatial_belief():
    # A and B are perfectly correlated while C is independent. All three sites
    # have the same marginal occupancy probability (0.5), so a local entropy
    # rule cannot see that observing A also informs B.
    worlds = [
        EcologicalHypothesis({"a": False, "b": False, "c": False}),
        EcologicalHypothesis({"a": False, "b": False, "c": True}),
        EcologicalHypothesis({"a": True, "b": True, "c": False}),
        EcologicalHypothesis({"a": True, "b": True, "c": True}),
    ]
    return SpatialBeliefEngine.initialize(worlds, [QHypothesis(0.5)])


def test_spatial_information_gain_prefers_site_that_resolves_correlated_extent() -> None:
    spatial = correlated_spatial_belief()
    graph = graph_from_beliefs(spatial.p_by_site(), spatial.uncertainty_by_site())
    planner = InformationGainPlanner(
        effort_per_site=1,
        max_sites=1,
        require_spatial_belief=True,
    )

    mission = planner.plan(
        graph,
        remaining_budget=5,
        constraints={"spatial_belief_state": spatial},
    )

    assert mission.allocations[0].site_id == "a"
    assert mission.total_cost == 1
    assert mission.diagnostics["mode"] == "spatial_joint"
    scores = {
        row["site_id"]: row["expected_information_gain_bits"]
        for row in mission.diagnostics["ranked_candidates"]
    }
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] > scores["c"]


def test_more_effort_does_not_reduce_expected_information_gain() -> None:
    spatial = correlated_spatial_belief()
    graph = graph_from_beliefs(spatial.p_by_site(), spatial.uncertainty_by_site())

    one = InformationGainPlanner(
        effort_per_site=1,
        max_sites=1,
        require_spatial_belief=True,
    ).plan(graph, 5, {"spatial_belief_state": spatial})
    four = InformationGainPlanner(
        effort_per_site=4,
        max_sites=1,
        require_spatial_belief=True,
    ).plan(graph, 5, {"spatial_belief_state": spatial})

    ig_one = one.diagnostics["ranked_selected_sites"][0]["expected_information_gain_bits"]
    ig_four = four.diagnostics["ranked_selected_sites"][0]["expected_information_gain_bits"]
    assert ig_four >= ig_one


def test_formal_mode_rejects_missing_spatial_posterior() -> None:
    graph = graph_from_beliefs(
        {"a": 0.5},
        {"a": 1.0},
    )
    planner = InformationGainPlanner(require_spatial_belief=True)

    with pytest.raises(ValueError, match="requires"):
        planner.plan(graph, 1, {"q_by_site": {"a": 0.2}})


def test_site_local_fallback_uses_effort_aware_q_semantics() -> None:
    graph = graph_from_beliefs(
        {"a": 0.5, "b": 0.5},
        {"a": 1.0, "b": 1.0},
    )
    planner = InformationGainPlanner(effort_per_site=2, max_sites=1)
    mission = planner.plan(
        graph,
        remaining_budget=2,
        constraints={"q_by_site": {"a": 0.5, "b": 0.1}},
    )

    assert mission.allocations[0].site_id == "a"
    assert mission.diagnostics["mode"] == "site_local_fallback"


def test_zero_budget_returns_empty_action() -> None:
    spatial = correlated_spatial_belief()
    graph = graph_from_beliefs(spatial.p_by_site(), spatial.uncertainty_by_site())
    mission = InformationGainPlanner(require_spatial_belief=True).plan(
        graph,
        remaining_budget=0,
        constraints={"spatial_belief_state": spatial},
    )
    assert mission.allocations == ()
    assert mission.total_cost == 0


def test_information_gain_diagnostics_explicitly_deny_hidden_truth_use() -> None:
    spatial = correlated_spatial_belief()
    graph = graph_from_beliefs(spatial.p_by_site(), spatial.uncertainty_by_site())
    mission = InformationGainPlanner(require_spatial_belief=True).plan(
        graph,
        remaining_budget=1,
        constraints={"spatial_belief_state": spatial},
    )

    assert mission.diagnostics["hidden_truth_used"] is False
    assert "hidden_world" not in mission.diagnostics
    assert "occupied_by_site" not in mission.diagnostics
