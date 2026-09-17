from types import SimpleNamespace

import pytest

from adaptive_response.models import (
    Edge,
    GraphState,
    IncidentConfig,
    MissionAction,
    MissionAllocation,
    Site,
)
from adaptive_response.environment import Environment
from adaptive_response.r10_static_adaptive import (
    AdaptiveFrontierPlanner,
    StaticFrontierPlanner,
    paired_uniform,
    run_r10_case,
)
from adaptive_response.spatial_belief import EcologicalHypothesis, QHypothesis
from adaptive_response.world_models import GeneratedWorld


def _features(
    belief: float,
    uncertainty: float,
    *,
    frontier: bool,
) -> tuple[float, ...]:
    return (
        belief,
        uncertainty,
        0.0,
        0.0,
        0.5,
        1.0,
        float(frontier),
    )


def _graph(
    *,
    b_frontier: bool,
    c_frontier: bool,
) -> GraphState:
    return GraphState(
        node_ids=("A", "B", "C", "D"),
        node_features=(
            _features(1.0, 0.0, frontier=False),
            _features(0.9, 0.2, frontier=b_frontier),
            _features(0.5, 0.8, frontier=c_frontier),
            _features(0.8, 0.4, frontier=False),
        ),
        edge_index=((), ()),
        edge_features=(),
        global_features=(18.0, 0.0, 1.0, 0.5),
        feasibility_mask=(True, True, True, True),
    )


def test_paired_uniform_exact_frozen_value() -> None:
    assert paired_uniform("case-A", "site-1") == pytest.approx(
        0.6147598643798865,
        abs=0.0,
        rel=1e-15,
    )


def test_paired_uniform_is_deterministic_and_site_specific() -> None:
    a = paired_uniform("case-A", "site-1")
    b = paired_uniform("case-A", "site-1")
    c = paired_uniform("case-A", "site-2")

    assert 0.0 <= a < 1.0
    assert a == b
    assert a != c


def test_static_precommitment_ignores_later_graph_changes() -> None:
    static = StaticFrontierPlanner(initially_blocked_sites=("A",))

    first = static.plan(_graph(b_frontier=True, c_frontier=False), 18, {})
    assert first.allocations[0].site_id == "B"

    # After evidence, imagine C becomes frontier.
    # STATIC must still execute its t0-precommitted second mission.
    second = static.plan(_graph(b_frontier=False, c_frontier=True), 12, {})

    assert static.precommitted_sites == ("B", "D", "C")
    assert second.allocations[0].site_id == "D"


def test_adaptive_frontier_replans_after_graph_change() -> None:
    adaptive = AdaptiveFrontierPlanner(initially_blocked_sites=("A",))

    first = adaptive.plan(_graph(b_frontier=True, c_frontier=False), 18, {})
    assert first.allocations[0].site_id == "B"

    second = adaptive.plan(_graph(b_frontier=False, c_frontier=True), 12, {})

    assert second.allocations[0].site_id == "C"
    assert adaptive.selected_sites == ("B", "C")


def _incident() -> IncidentConfig:
    sites = [
        Site(id="A", x=0.0, y=0.0, habitat_score=0.5, q_model=0.2),
        Site(id="B", x=1.0, y=0.0, habitat_score=0.5, q_model=0.2),
        Site(id="C", x=2.0, y=0.0, habitat_score=0.5, q_model=0.2),
        Site(id="D", x=3.0, y=0.0, habitat_score=0.5, q_model=0.2),
    ]
    edges = [
        Edge(src="A", dst="B", distance=1.0),
        Edge(src="B", dst="C", distance=1.0),
        Edge(src="C", dst="D", distance=1.0),
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="A",
        budget=18,
        teams=1,
        protocol="r10_test",
        seed=7,
    )


class _FixedWorldModel:
    family_id = "r10_test_world"

    def sample(self, context, *, seed: int) -> GeneratedWorld:
        return GeneratedWorld(
            family_id=self.family_id,
            occupied_by_site={
                site.id: True
                for site in context.sites
            },
            seed=seed,
        )


def test_environment_fixed_uniform_is_simulator_only() -> None:
    incident = _incident()
    uniforms = {site.id: 0.0 for site in incident.sites}

    env = Environment(
        incident,
        world_model=_FixedWorldModel(),
        q_true=0.1,
        observation_uniform_by_site=uniforms,
    )
    env.reset(seed=7)

    action = MissionAction(
        allocations=(MissionAllocation(site_id="B", effort_units=6),),
        total_cost=6,
    )
    batch, _, _ = env.step(action)

    assert batch.observations[0].detection is True
    assert "q_true" not in batch.observations[0].metadata
    assert "uniform" not in batch.observations[0].metadata


def test_environment_rejects_incomplete_uniform_map() -> None:
    with pytest.raises(ValueError, match="keys must exactly match"):
        Environment(
            _incident(),
            observation_uniform_by_site={"A": 0.5},
        )


def test_r10_end_to_end_first_mission_is_paired_and_later_plan_can_diverge() -> None:
    incident = _incident()

    hypotheses = (
        EcologicalHypothesis(
            presence_by_site={
                "A": True,
                "B": True,
                "C": False,
                "D": True,
            },
            label="h1",
        ),
        EcologicalHypothesis(
            presence_by_site={
                "A": True,
                "B": True,
                "C": True,
                "D": True,
            },
            label="h2",
        ),
    )

    case = SimpleNamespace(
        label="r10_test_case",
        incident=incident,
        world_model=_FixedWorldModel(),
        q_true=1.0,
        ecological_hypotheses=hypotheses,
        q_hypotheses=(QHypothesis(0.5),),
        seed=7,
        group="unit_test",
    )

    result = run_r10_case(case)

    assert result.static.mission_sites[0] == result.adaptive.mission_sites[0]
    assert result.static.mission_detections[0] == result.adaptive.mission_detections[0]

    assert result.static.mission_sites == ("B", "D", "C")
    assert result.adaptive.mission_sites == ("B", "C", "D")

    assert result.mission_2_divergence is True
    assert result.mission_3_divergence is True

    assert len(set(result.static.mission_sites)) == 3
    assert len(set(result.adaptive.mission_sites)) == 3

    assert result.static.effort_spent == 18
    assert result.adaptive.effort_spent == 18


def test_r10_initial_confirmed_detection_is_not_eligible_for_resurvey() -> None:
    static = StaticFrontierPlanner(initially_blocked_sites=("A",))
    adaptive = AdaptiveFrontierPlanner(initially_blocked_sites=("A",))

    static_action = static.plan(
        _graph(b_frontier=False, c_frontier=False),
        18,
        {},
    )
    adaptive_action = adaptive.plan(
        _graph(b_frontier=False, c_frontier=False),
        18,
        {},
    )

    assert static_action.allocations[0].site_id != "A"
    assert adaptive_action.allocations[0].site_id != "A"
