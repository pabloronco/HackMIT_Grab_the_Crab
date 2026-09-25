"""R11 Dynamic Delimitation baseline (handoff: R11_DYNAMIC_DELIMITATION_RESOURCE).

Covers the required unit tests from the handoff's list that are properly
scoped to the planner and the frozen resource contract:
  1-6.  effort/window contract (budget<=18, windows<=18, e1x18/e3x6/e6x3/mixed legal)
  7.    Dynamic always emits effort=6 (when budget allows)
  8.    Dynamic never reads belief/uncertainty/q/world state
  9.    positive observation expands the active frontier
  10.   non-detection removes a site but does not expand the frontier
  11.   Dynamic ranking is deterministic
  12.   Dynamic never sees hidden truth
  18.   old R8 config remains unchanged

Item 13 (same seed + same action -> same observation) is covered by the
existing simulator-determinism tests; items 14-17 (parity continuation,
parity target definition, parity overhead calculation, route-cost gating)
belong to the Effort-to-Parity benchmark script, not this planner.
"""

from __future__ import annotations

import json

import pytest

from adaptive_response import (
    DynamicDelimitationMax6Planner,
    Edge,
    EcologicalHypothesis,
    Environment,
    GraphDiffusionWorldModel,
    GraphState,
    IncidentConfig,
    LoopPhase,
    NODE_FEATURE_NAMES,
    QHypothesis,
    Site,
    SpatialAdaptiveMissionLoop,
)


def node_row(
    *,
    belief: float = 0.0,
    uncertainty: float = 0.0,
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
        "frontier": 0.0,  # never read by Dynamic; left at a fixed value.
    }
    return tuple(float(values[name]) for name in NODE_FEATURE_NAMES)


def graph(
    rows: list[tuple[str, tuple[float, ...], bool]],
    edges: list[tuple[str, str]] | None = None,
) -> GraphState:
    node_ids = tuple(site_id for site_id, _, _ in rows)
    index = {site_id: i for i, site_id in enumerate(node_ids)}
    src: list[int] = []
    dst: list[int] = []
    for a, b in edges or []:
        src.extend((index[a], index[b]))
        dst.extend((index[b], index[a]))
    return GraphState(
        node_ids=node_ids,
        node_features=tuple(features for _, features, _ in rows),
        edge_index=(tuple(src), tuple(dst)),
        edge_features=tuple((1.0, 1.0) for _ in src),
        global_features=(18.0, 0.0, 1.0, 0.5),
        feasibility_mask=tuple(feasible for _, _, feasible in rows),
    )


# a - b - c - d  (chain), a is the confirmed first detection.
CHAIN_EDGES = [("a", "b"), ("b", "c"), ("c", "d")]


def chain_graph(*, a_positive: bool = True, **overrides) -> GraphState:
    """a is always positive (confirmed first detection). b/c/d vary."""
    rows = {
        "a": node_row(detections=1.0 if a_positive else 0.0, observed_effort=0.0),
        "b": node_row(),
        "c": node_row(),
        "d": node_row(),
    }
    rows.update(overrides)
    return graph(
        [(site_id, features, True) for site_id, features in rows.items()],
        edges=CHAIN_EDGES,
    )


def test_dynamic_always_emits_effort_six_when_budget_allows() -> None:
    state = chain_graph()
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=18, constraints={})
    assert len(action.allocations) == 1
    assert action.allocations[0].effort_units == 6
    assert action.total_cost == 6


def test_dynamic_falls_back_to_largest_feasible_effort_below_six() -> None:
    state = chain_graph()
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=2, constraints={})
    assert action.allocations[0].effort_units == 1  # largest feasible level <= 2 from {1,3,6}


def test_dynamic_returns_empty_mission_at_zero_budget() -> None:
    state = chain_graph()
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=0, constraints={})
    assert action.allocations == ()
    assert action.total_cost == 0


def test_positive_observation_expands_active_frontier() -> None:
    """a positive -> frontier is {b}. After b also turns positive, frontier becomes {c}."""
    planner = DynamicDelimitationMax6Planner()

    round1 = chain_graph()
    action1 = planner.plan(round1, remaining_budget=18, constraints={})
    assert action1.allocations[0].site_id == "b"  # only unsurveyed neighbor of positive "a"

    round2 = chain_graph(b=node_row(detections=1.0))  # b came back positive
    action2 = planner.plan(round2, remaining_budget=12, constraints={})
    assert action2.allocations[0].site_id == "c"  # frontier expanded past the new positive


def test_non_detection_removes_site_but_does_not_expand_frontier() -> None:
    """b surveyed negative -> b leaves the candidate pool, but c is NOT unlocked
    (c is only adjacent to b, not to any positive site)."""
    planner = DynamicDelimitationMax6Planner()
    state = chain_graph(b=node_row(observed_effort=6.0, detections=0.0))
    action = planner.plan(state, remaining_budget=12, constraints={})
    # No unsurveyed site is adjacent to a positive ("a") anymore -> Tier 2 (ring
    # expansion) by hop distance from "a": b is surveyed (excluded), c is 2 hops,
    # d is 3 hops -> c is picked, not "reopening" b.
    assert action.allocations[0].site_id == "c"
    assert action.diagnostics["tier"] == "ring_expansion"


def test_ring_expansion_used_only_when_active_frontier_is_empty() -> None:
    state = chain_graph(b=node_row(observed_effort=6.0, detections=0.0))
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=12, constraints={})
    assert action.diagnostics["tier"] == "ring_expansion"

    state_with_frontier = chain_graph()  # b is unsurveyed and adjacent to positive a
    action2 = DynamicDelimitationMax6Planner().plan(state_with_frontier, remaining_budget=12, constraints={})
    assert action2.diagnostics["tier"] == "active_frontier"


def test_no_revisit_of_already_surveyed_or_positive_sites() -> None:
    """a is positive, b is searched-negative: neither is ever re-selected."""
    state = chain_graph(b=node_row(observed_effort=6.0, detections=0.0))
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=12, constraints={})
    chosen = action.allocations[0].site_id
    assert chosen not in ("a", "b")


def test_tier1_ranking_prefers_higher_habitat_then_lower_access_cost() -> None:
    """Both b and (a hypothetical second neighbor) are on the frontier; habitat wins first."""
    rows = [
        ("a", node_row(detections=1.0), True),
        ("b", node_row(habitat_score=0.9, access_cost=5.0), True),
        ("c", node_row(habitat_score=0.9, access_cost=1.0), True),  # same habitat, cheaper access
        ("d", node_row(habitat_score=0.2, access_cost=0.1), True),  # low habitat wins nothing
    ]
    state = graph(rows, edges=[("a", "b"), ("a", "c"), ("a", "d")])
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=18, constraints={})
    assert action.allocations[0].site_id == "c"  # tied habitat with b, but cheaper access


def test_tier2_ranking_prefers_minimum_hop_distance_first() -> None:
    # a(positive) - b(unsurveyed, hop=1 but pre-surveyed to force tier2) - c(hop=2) ; a - e(hop=1, pre-surveyed)
    rows = [
        ("a", node_row(detections=1.0), True),
        ("b", node_row(observed_effort=6.0), True),  # searched-negative, hop=1, excluded
        ("c", node_row(habitat_score=0.1), True),     # hop=2 via b
        ("e", node_row(observed_effort=6.0), True),  # searched-negative, hop=1, excluded
        ("f", node_row(habitat_score=0.9), True),     # unreachable within 2 hops (hop=3)
    ]
    state = graph(rows, edges=[("a", "b"), ("a", "e"), ("b", "c"), ("c", "f")])
    action = DynamicDelimitationMax6Planner().plan(state, remaining_budget=18, constraints={})
    assert action.allocations[0].site_id == "c"  # hop=2, closer than f's hop=3, despite lower habitat


def test_dynamic_ranking_is_deterministic_across_repeated_calls() -> None:
    state = chain_graph()
    planner = DynamicDelimitationMax6Planner()
    first = planner.plan(state, remaining_budget=18, constraints={})
    for _ in range(20):
        again = planner.plan(state, remaining_budget=18, constraints={})
        assert again.allocations == first.allocations


def test_dynamic_output_is_invariant_to_belief_uncertainty_and_constraints() -> None:
    """The information firewall: only observed_effort/detections/habitat/access/
    edges may influence the decision. belief, uncertainty and constraints (which
    may carry spatial_belief_state/q_by_site for other planners) must not."""
    planner = DynamicDelimitationMax6Planner()

    baseline = chain_graph()
    baseline_action = planner.plan(baseline, remaining_budget=18, constraints={})

    perturbed = chain_graph(
        a=node_row(detections=1.0, belief=0.99, uncertainty=0.01),
        b=node_row(belief=0.01, uncertainty=0.99),
        c=node_row(belief=0.5, uncertainty=0.5),
        d=node_row(belief=1.0, uncertainty=1.0),
    )
    perturbed_action = planner.plan(
        perturbed,
        remaining_budget=18,
        constraints={"spatial_belief_state": object(), "q_by_site": {"a": 0.9, "b": 0.01}},
    )

    assert perturbed_action.allocations == baseline_action.allocations


def test_dynamic_never_reads_forbidden_feature_indices_by_construction() -> None:
    """belief (index 0) and uncertainty (index 1) are absent from the allowlist
    the planner documents and uses; this pins that contract against regressions."""
    from adaptive_response.dynamic_delimitation_planner import (
        _ALLOWED_FEATURE_NAMES,
        _FORBIDDEN_FEATURE_NAMES,
    )

    assert "belief" not in _ALLOWED_FEATURE_NAMES
    assert "uncertainty" not in _ALLOWED_FEATURE_NAMES
    assert set(_FORBIDDEN_FEATURE_NAMES) == {"belief", "uncertainty"}


def test_dynamic_plan_signature_never_receives_hidden_truth() -> None:
    """Structural guarantee: the shared Planner protocol's plan() has no
    parameter through which HiddenWorld could reach the planner at all."""
    import inspect

    sig = inspect.signature(DynamicDelimitationMax6Planner.plan)
    assert set(sig.parameters) == {"self", "graph_state", "remaining_budget", "constraints"}


def test_invalid_construction_rejected() -> None:
    with pytest.raises(ValueError):
        DynamicDelimitationMax6Planner(effort=0)
    with pytest.raises(ValueError):
        DynamicDelimitationMax6Planner(effort=6, effort_levels=(1, 3))  # 6 not in levels
    with pytest.raises(ValueError):
        DynamicDelimitationMax6Planner(effort_levels=())


# --- Frozen resource contract (budget<=18, windows<=18, effort-level legality) ---

def _small_incident(*, initial_detection: str = "a") -> IncidentConfig:
    sites = [
        Site(id=s, x=float(i), y=0.0, habitat_score=0.5, q_model=0.1)
        for i, s in enumerate(("a", "b", "c", "d"))
    ]
    edges = [Edge("a", "b", 1.0), Edge("b", "c", 1.0), Edge("c", "d", 1.0)]
    return IncidentConfig(
        sites=sites, edges=edges, initial_detection=initial_detection,
        budget=18, teams=1, protocol="r11_test", seed=7,
    )


def _small_loop(planner) -> SpatialAdaptiveMissionLoop:
    hypotheses = [
        EcologicalHypothesis({"a": True, "b": True, "c": False, "d": False}, prior_weight=1.0),
        EcologicalHypothesis({"a": True, "b": False, "c": True, "d": False}, prior_weight=1.0),
    ]
    loop = SpatialAdaptiveMissionLoop(
        Environment(_small_incident(), world_model=GraphDiffusionWorldModel(), q_true=0.15),
        planner,
        hypotheses,
        [QHypothesis(q) for q in (0.05, 0.15, 0.3)],
    )
    loop.reset(seed=7)
    return loop


def test_dynamic_total_effort_never_exceeds_eighteen() -> None:
    loop = _small_loop(DynamicDelimitationMax6Planner())
    total_effort = 0
    rounds = 0
    while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
        transition = loop.run_round()
        total_effort += transition.mission.total_cost
        rounds += 1
        assert rounds <= 18  # max_windows contract
    assert total_effort <= 18


def test_dynamic_spends_exactly_eighteen_in_three_normal_budget_missions() -> None:
    """Section 3 of the handoff: with budget 18 and effort always 6, Dynamic
    executes exactly 3 missions."""
    loop = _small_loop(DynamicDelimitationMax6Planner())
    rounds = 0
    total_effort = 0
    while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
        transition = loop.run_round()
        rounds += 1
        total_effort += transition.mission.total_cost
    assert rounds == 3
    assert total_effort == 18


class _FixedEffortPlanner:
    """Minimal scripted planner for the generic (non-Dynamic-specific) effort/
    window contract tests: always spends `effort` on the first unsurveyed site."""

    def __init__(self, effort: int) -> None:
        self.effort = effort

    def plan(self, graph_state, remaining_budget, constraints):
        from adaptive_response import MissionAction, MissionAllocation

        del constraints
        if remaining_budget < self.effort:
            return MissionAction(allocations=(), total_cost=0, diagnostics={})
        obs_idx = NODE_FEATURE_NAMES.index("observed_effort")
        for i, site_id in enumerate(graph_state.node_ids):
            if graph_state.node_features[i][obs_idx] <= 0 and graph_state.feasibility_mask[i]:
                return MissionAction(
                    allocations=(MissionAllocation(site_id=site_id, effort_units=self.effort),),
                    total_cost=self.effort,
                    diagnostics={},
                )
        return MissionAction(allocations=(), total_cost=0, diagnostics={})


def _big_incident(n: int, *, initial_detection: str = "s0") -> IncidentConfig:
    site_ids = [f"s{i}" for i in range(n)]
    sites = [
        Site(id=s, x=float(i), y=0.0, habitat_score=0.5, q_model=0.1)
        for i, s in enumerate(site_ids)
    ]
    edges = [Edge(site_ids[i], site_ids[i + 1], 1.0) for i in range(n - 1)]
    return IncidentConfig(
        sites=sites, edges=edges, initial_detection=initial_detection,
        budget=18, teams=1, protocol="r11_test", seed=11,
    )


def _big_loop(planner, n: int) -> SpatialAdaptiveMissionLoop:
    # All-occupied: every observation (detect or not) has nonzero likelihood
    # under this hypothesis regardless of what the hidden world model draws,
    # since these tests only exercise the effort/window contract, not belief
    # accuracy.
    hypotheses = [
        EcologicalHypothesis({f"s{i}": True for i in range(n)}, prior_weight=1.0),
    ]
    loop = SpatialAdaptiveMissionLoop(
        Environment(_big_incident(n), world_model=GraphDiffusionWorldModel(), q_true=0.15),
        planner,
        hypotheses,
        [QHypothesis(q) for q in (0.05, 0.15, 0.3)],
    )
    loop.reset(seed=11)
    return loop


@pytest.mark.parametrize(
    "effort, expected_windows",
    [(1, 18), (3, 6), (6, 3)],
)
def test_effort_levels_produce_exactly_the_expected_window_count(effort, expected_windows) -> None:
    loop = _big_loop(_FixedEffortPlanner(effort), n=20)
    rounds = 0
    total_effort = 0
    while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
        transition = loop.run_round()
        rounds += 1
        total_effort += transition.mission.total_cost
        assert rounds <= 18
    assert rounds == expected_windows
    assert total_effort == 18


class _MixedEffortPlanner:
    """Scripted planner spending a fixed pre-declared sequence of efforts,
    used to prove a mixed e1/e3/e6 trajectory can spend exactly 18."""

    def __init__(self, sequence: tuple[int, ...]) -> None:
        assert sum(sequence) == 18
        self._sequence = list(sequence)

    def plan(self, graph_state, remaining_budget, constraints):
        from adaptive_response import MissionAction, MissionAllocation

        del constraints
        obs_idx = NODE_FEATURE_NAMES.index("observed_effort")
        effort = self._sequence.pop(0)
        assert effort <= remaining_budget
        for i, site_id in enumerate(graph_state.node_ids):
            if graph_state.node_features[i][obs_idx] <= 0 and graph_state.feasibility_mask[i]:
                return MissionAction(
                    allocations=(MissionAllocation(site_id=site_id, effort_units=effort),),
                    total_cost=effort,
                    diagnostics={},
                )
        raise AssertionError("ran out of unsurveyed sites before spending the full sequence")


def test_mixed_effort_trajectory_can_spend_exactly_eighteen() -> None:
    sequence = (1, 3, 6, 1, 1, 6)  # sums to 18, mixes all three levels
    loop = _big_loop(_MixedEffortPlanner(sequence), n=20)
    rounds = 0
    total_effort = 0
    while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
        transition = loop.run_round()
        rounds += 1
        total_effort += transition.mission.total_cost
        assert rounds <= 18
    assert rounds == len(sequence)
    assert total_effort == 18


def test_r8_frozen_config_is_unchanged_by_the_r11_lane() -> None:
    """Item 18: old R8 configs/reports must remain unchanged by this new lane."""
    with open("configs/benchmark_protocol_r8.json") as f:
        r8 = json.load(f)
    assert r8["frozen_action_contract"]["total_budget"] == 18
    assert r8["frozen_action_contract"]["max_rounds"] == 6
    assert r8["frozen_action_contract"]["effort_levels"] == [1, 3, 6]
    assert r8["frozen_case_manifest"]["total_cases"] == 240


def test_r11_config_declares_a_separate_resource_contract() -> None:
    with open("configs/benchmark_protocol_r11_dynamic.json") as f:
        r11 = json.load(f)
    assert r11["frozen_resource_contract"]["total_effort_budget"] == 18
    assert r11["frozen_resource_contract"]["max_windows"] == 18
    assert r11["gnn_horizon_change"]["max_rounds"] == 18
    assert "configs/benchmark_protocol_r8.json" not in r11.get("new_artifacts", [])
