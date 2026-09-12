import pytest

from adaptive_response import (
    EcologicalHypothesis,
    Edge,
    Environment,
    IncidentConfig,
    InformationGainPlanner,
    QHypothesis,
    Site,
    SpatialAdaptiveMissionLoop,
)


def make_loop(*, budget: int = 2) -> SpatialAdaptiveMissionLoop:
    sites = [
        Site("seed", 0.0, 0.0, 0.5, 0.2),
        Site("a", 1.0, 0.0, 0.5, 0.2),
        Site("b", 2.0, 0.0, 0.5, 0.2),
        Site("c", 0.0, 1.0, 0.5, 0.2),
    ]
    config = IncidentConfig(
        sites=sites,
        edges=[
            Edge("seed", "a", 1.0),
            Edge("a", "b", 1.0),
            Edge("seed", "c", 1.0),
        ],
        initial_detection="seed",
        budget=budget,
        teams=1,
        protocol="test_protocol",
        seed=7,
    )
    environment = Environment(config)
    worlds = [
        EcologicalHypothesis({"seed": True, "a": False, "b": False, "c": False}),
        EcologicalHypothesis({"seed": True, "a": False, "b": False, "c": True}),
        EcologicalHypothesis({"seed": True, "a": True, "b": True, "c": False}),
        EcologicalHypothesis({"seed": True, "a": True, "b": True, "c": True}),
    ]
    planner = InformationGainPlanner(
        effort_per_site=1,
        max_sites=1,
        require_spatial_belief=True,
    )
    return SpatialAdaptiveMissionLoop(
        environment,
        planner,
        worlds,
        [QHypothesis(0.2), QHypothesis(0.4)],
    )


def test_spatial_loop_runs_observe_infer_decide_replan_with_joint_voi() -> None:
    loop = make_loop(budget=2)
    loop.reset(seed=7)

    before = loop.current_belief
    assert before.p_by_site["seed"] == pytest.approx(1.0)
    assert before.p_by_site["a"] == pytest.approx(0.5)
    assert before.p_by_site["b"] == pytest.approx(0.5)
    assert before.p_by_site["c"] == pytest.approx(0.5)

    first = loop.plan_next()
    assert first.allocations[0].site_id == "a"
    assert first.diagnostics["mode"] == "spatial_joint"

    transition = loop.execute_pending()
    assert transition.observations.observations[0].site_id == "a"
    assert transition.belief_after.p_by_site["b"] != pytest.approx(0.5)
    assert transition.graph_after != transition.graph_before
    assert transition.next_mission is not None
    assert transition.next_mission.diagnostics["mode"] == "spatial_joint"
    assert transition.done is False


def test_spatial_loop_reveal_remains_gated_and_posterior_is_not_hidden_truth() -> None:
    loop = make_loop(budget=2)
    loop.reset(seed=7)

    with pytest.raises(RuntimeError, match="only available"):
        loop.reveal()

    spatial = loop.current_spatial_belief
    assert not hasattr(spatial, "hidden_world")
    assert not hasattr(spatial, "occupied_by_site")

    loop.run_round()
    second = loop.execute_pending()
    assert second.done is True
    hidden = loop.reveal()
    assert set(hidden.occupied_by_site) == {"seed", "a", "b", "c"}


def test_spatial_loop_q_context_is_belief_posterior_not_environment_truth() -> None:
    loop = make_loop(budget=1)
    loop.reset(seed=7)
    mission = loop.plan_next()
    assert mission.diagnostics["mode"] == "spatial_joint"
    assert mission.diagnostics["hidden_truth_used"] is False
