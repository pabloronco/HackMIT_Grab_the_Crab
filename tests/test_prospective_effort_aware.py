from __future__ import annotations

import math

import pytest

from adaptive_response.effort_aware_planner import (
    EffortAwareInformationGainPlanner,
    EffortAwarePlannerConfig,
    FixedHighEffortPlanner,
    OperationalCostConfig,
    is_stop_action,
    make_stop_action,
)
from adaptive_response.environment import Environment
from adaptive_response.graph_state import GraphStateExporter
from adaptive_response.mission_loop import LoopPhase
from adaptive_response.models import Edge, IncidentConfig, Observation, ObservationBatch, Site
from adaptive_response.prospective import ProspectiveEvaluator, detectability_support_pressure
from adaptive_response.spatial_belief import EcologicalHypothesis, QHypothesis, SpatialBeliefEngine
from adaptive_response.spatial_mission_loop import SpatialAdaptiveMissionLoop
from adaptive_response.world_models import GraphDiffusionWorldModel

SITES = ("a", "b", "c", "d")
EFFORTS = (1, 3, 6)


def _belief():
    hypotheses = [
        EcologicalHypothesis({"a": True, "b": True, "c": False, "d": False}, prior_weight=3.0, label="A"),
        EcologicalHypothesis({"a": True, "b": False, "c": True, "d": False}, prior_weight=2.0, label="B"),
        EcologicalHypothesis({"a": True, "b": True, "c": True, "d": True}, prior_weight=1.0, label="C"),
        EcologicalHypothesis({"a": True, "b": False, "c": False, "d": False}, prior_weight=1.0, label="A"),
    ]
    q = [QHypothesis(0.05), QHypothesis(0.10), QHypothesis(0.20)]
    return SpatialBeliefEngine.initialize(hypotheses, q, confirmed_sites={"a"})


def _incident() -> IncidentConfig:
    sites = [Site(id=s, x=float(i), y=0.0, habitat_score=0.5, q_model=0.1) for i, s in enumerate(SITES)]
    edges = [Edge("a", "b", 1.0), Edge("b", "c", 1.0), Edge("c", "d", 1.0)]
    return IncidentConfig(sites=sites, edges=edges, initial_detection="a", budget=18, teams=1, protocol="test", seed=3)


def test_predictive_detection_is_monotone_in_effort() -> None:
    belief = _belief()
    for site in SITES:
        p = [SpatialBeliefEngine.predictive_detection_probability(belief, site_id=site, effort=e) for e in EFFORTS]
        assert p[0] <= p[1] <= p[2]
        assert all(0.0 <= v <= 1.0 for v in p)


def test_conditional_miss_probability_is_a_valid_probability_and_none_when_unoccupied() -> None:
    belief = _belief()
    for site in ("a", "b", "c", "d"):
        for e in EFFORTS:
            miss = SpatialBeliefEngine.conditional_miss_probability_if_occupied(belief, site_id=site, effort=e)
            assert miss is not None and 0.0 <= miss <= 1.0
    # site "a" is occupied in every hypothesis: miss = E_q[(1-q)^e] exactly
    q_post = belief.q_posterior()
    expected = sum(w * (1 - q) ** 3 for q, w in q_post.items())
    assert math.isclose(SpatialBeliefEngine.conditional_miss_probability_if_occupied(belief, site_id="a", effort=3), expected, rel_tol=1e-12)
    # a belief with zero mass on occupancy at "d"
    hyp = [EcologicalHypothesis({"a": True, "b": True, "c": False, "d": False})]
    zero = SpatialBeliefEngine.initialize(hyp, [QHypothesis(0.1)], confirmed_sites={"a"})
    assert SpatialBeliefEngine.conditional_miss_probability_if_occupied(zero, site_id="d", effort=6) is None


def test_ecological_entropy_marginalizes_q_duplicates() -> None:
    belief = _belief()
    # 4 unique extents; joint has 12 hypotheses. Extent entropy must equal the
    # entropy of the 4 ecological prior weights (all consistent with "a").
    weights = [3.0, 2.0, 1.0, 1.0]
    total = sum(weights)
    expected = -sum(w / total * math.log2(w / total) for w in weights)
    assert math.isclose(SpatialBeliefEngine.ecological_extent_entropy_bits(belief), expected, rel_tol=1e-12)
    assert belief.joint_entropy_bits() > SpatialBeliefEngine.ecological_extent_entropy_bits(belief)
    assert ProspectiveEvaluator(belief).unique_extent_count == 4


def test_hypothetical_eig_does_not_mutate_live_belief_and_matches_vectorized_path() -> None:
    belief = _belief()
    weights_before = tuple(belief.weights)
    history_before = tuple(belief.observed_history)
    evaluator = ProspectiveEvaluator(belief)
    rows = evaluator.evaluate_all(EFFORTS)
    for row in rows:
        ref = SpatialBeliefEngine.expected_information_gain_bits(belief, site_id=row.site_id, effort=row.effort)
        assert math.isclose(row.expected_information_gain_bits, ref.expected_information_gain_bits, abs_tol=1e-10)
        assert math.isclose(row.predictive_detection_probability, ref.predictive_detection_probability, abs_tol=1e-12)
        assert math.isclose(row.expected_marginal_information_gain_bits, ref.expected_marginal_information_gain_bits, abs_tol=1e-10)
        assert row.expected_information_gain_bits >= 0.0
        if ref.conditional_miss_probability_if_occupied is None:
            assert row.conditional_miss_probability_if_occupied is None
        else:
            assert math.isclose(row.conditional_miss_probability_if_occupied, ref.conditional_miss_probability_if_occupied, abs_tol=1e-12)
    assert tuple(belief.weights) == weights_before
    assert tuple(belief.observed_history) == history_before
    # EIG is non-decreasing in effort for every site
    by_site: dict[str, dict[int, float]] = {}
    for row in rows:
        by_site.setdefault(row.site_id, {})[row.effort] = row.expected_information_gain_bits
    for curve in by_site.values():
        assert curve[1] <= curve[3] + 1e-12 <= curve[6] + 2e-12


def test_q_boundary_concentration_does_not_set_observation_stress_flag() -> None:
    from adaptive_response.model_mismatch import posterior_predictive_surprise

    belief = _belief()
    # Many non-detections at the confirmed occupied site push q mass to q_min.
    for _ in range(6):
        belief = SpatialBeliefEngine.update(
            belief, ObservationBatch(observations=(Observation("a", 6, False, 1),), round=1, total_effort=6)
        )
    pressure = detectability_support_pressure(belief, boundary_mass_threshold=0.6)
    assert pressure.q_min == 0.05 and pressure.mass_at_low_boundary > 0.6
    assert pressure.concentrated_at_boundary is True and pressure.boundary_side == "low"
    # Without a configured threshold no flag is asserted at all.
    assert detectability_support_pressure(belief).concentrated_at_boundary is None
    # Observation surprise for one more non-detection is LOW (expected), not stress.
    surprise = posterior_predictive_surprise(belief, Observation("a", 6, False, 2))
    assert surprise.observation_probability > 0.5
    assert not surprise.impossible_under_current_ensemble


def _loop(planner, budget: int = 18):
    config = _incident()
    config.budget = budget
    hypotheses = [
        EcologicalHypothesis({"a": True, "b": True, "c": False, "d": False}, label="A"),
        EcologicalHypothesis({"a": True, "b": False, "c": True, "d": False}, label="B"),
        EcologicalHypothesis({"a": True, "b": True, "c": True, "d": True}, label="C"),
    ]
    loop = SpatialAdaptiveMissionLoop(
        Environment(config, world_model=GraphDiffusionWorldModel(), q_true=0.1),
        planner,
        hypotheses,
        [QHypothesis(q) for q in (0.05, 0.1, 0.2)],
    )
    loop.reset(seed=3)
    return loop


def test_effort_aware_planner_recommends_a_valid_site_effort_pair_with_full_diagnostics() -> None:
    planner = EffortAwareInformationGainPlanner(EffortAwarePlannerConfig(cost=OperationalCostConfig(dispatch_overhead_cost=0.5)))
    loop = _loop(planner)
    mission = loop.plan_next()
    assert len(mission.allocations) == 1
    allocation = mission.allocations[0]
    assert allocation.effort_units in EFFORTS
    assert allocation.effort_units <= loop.current_public_state.remaining_budget
    diag = mission.diagnostics
    assert diag["planner"] == "adaptive_effort_aware" and diag["hidden_truth_used"] is False
    selected = diag["selected"]
    for key in ("expected_information_gain_bits", "predictive_detection_probability", "cost", "information_efficiency", "utility", "frontier", "belief", "uncertainty", "conditional_miss_probability_if_occupied"):
        assert key in selected
    alternatives = diag["effort_alternatives"]
    assert [row["effort_units"] for row in alternatives] == [1, 3, 6]
    assert alternatives[-1]["eig_fraction_of_max_effort"] == pytest.approx(1.0)
    # every candidate row respects budget
    for row in diag["ranked_candidates"]:
        assert row["effort_units"] <= 18


def test_effort_aware_planner_respects_small_remaining_budget() -> None:
    planner = EffortAwareInformationGainPlanner()
    loop = _loop(planner, budget=2)
    mission = loop.plan_next()
    assert mission.allocations[0].effort_units == 1


def test_fixed_high_effort_uses_largest_feasible_effort() -> None:
    loop = _loop(FixedHighEffortPlanner())
    efforts = []
    while loop.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
        tr = loop.run_round()
        efforts.append(tr.mission.allocations[0].effort_units)
    assert efforts == [6, 6, 6]


def test_make_stop_action_roundtrip() -> None:
    action = make_stop_action(reason="x")
    assert is_stop_action(action) and action.total_cost == 0 and action.allocations == ()


def test_safari_target_no_horizontal_overflow_placeholder() -> None:
    """Layout/overflow regressions belong to a browser rehearsal, not pytest.

    See scripts/rehearse_ui.py for the Playwright check at 1440x900 / 1512x982
    that asserts document.documentElement.scrollWidth <= clientWidth and that
    the decision-log tape / possible-worlds panel never sit behind another
    panel. This placeholder documents the pointer so the acceptance-gate item
    in docs/INTERACTIVE_JUDGE_MODE_UI_SPEC.md is traceable from the test suite.
    """


def test_belief_frontier_site_ranking_matches_frontier_planner_order() -> None:
    """Regression test for the R11 revision-4 site-ranking defect.

    An EIG/cost site ranking can prefer a lower-belief, higher-uncertainty
    site over a higher-belief one (a real, measured failure on live incident
    data), because EIG rewards uncertainty reduction, not occupancy
    likelihood. site_ranking="belief_frontier" must reproduce FrontierPlanner's
    site order exactly, regardless of the effort-selection objective.
    """
    from adaptive_response.planners import FrontierPlanner

    belief = _belief()
    config = _incident()
    hypotheses = [
        EcologicalHypothesis({"a": True, "b": True, "c": False, "d": False}, prior_weight=3.0),
        EcologicalHypothesis({"a": True, "b": False, "c": True, "d": False}, prior_weight=2.0),
        EcologicalHypothesis({"a": True, "b": True, "c": True, "d": True}, prior_weight=1.0),
        EcologicalHypothesis({"a": True, "b": False, "c": False, "d": False}, prior_weight=1.0),
    ]
    loop = SpatialAdaptiveMissionLoop(
        Environment(config, world_model=GraphDiffusionWorldModel(), q_true=0.1),
        EffortAwareInformationGainPlanner(EffortAwarePlannerConfig(site_ranking="belief_frontier")),
        hypotheses,
        [QHypothesis(q) for q in (0.05, 0.1, 0.2)],
    )
    loop.reset(seed=3)
    graph = loop.current_graph_state
    frontier_order = [row["site_id"] for row in FrontierPlanner(max_sites=20).rank_candidates(graph)]

    scored = EffortAwareInformationGainPlanner(
        EffortAwarePlannerConfig(site_ranking="belief_frontier")
    ).score_actions(graph, loop.current_spatial_belief, 18)
    effort_aware_order = [row["site_id"] for row in scored["ranked_sites"]]
    assert effort_aware_order == frontier_order

    # And the information_efficiency site_ranking mode must NOT be forced to
    # match Frontier -- it is kept only for experimentation/comparison.
    scored_ig = EffortAwareInformationGainPlanner(
        EffortAwarePlannerConfig(site_ranking="information_efficiency")
    ).score_actions(graph, loop.current_spatial_belief, 18)
    ig_order = [row["site_id"] for row in scored_ig["ranked_sites"]]
    assert ig_order != frontier_order


def test_frozen_default_uses_belief_frontier_site_ranking() -> None:
    config = EffortAwarePlannerConfig.load_frozen()
    assert config.site_ranking == "belief_frontier"


def test_regime_adaptive_dispatch_cost_is_a_pure_function_of_observables() -> None:
    """Regression coverage for the R11 regime-adaptive dispatch cost (methods 1+3)."""
    cfg = OperationalCostConfig(
        regime_adaptive=True, dispatch_base=0.5, q_reference=0.1167,
        q_slope=10.0, prevalence_reference=0.35, prevalence_slope=2.0,
        dispatch_min=0.0, dispatch_max=3.0,
    )
    # At the reference point (prior q_mean, reference prevalence) -> dispatch_base unchanged.
    assert cfg.resolved_dispatch_overhead_cost(q_mean=0.1167, expected_prevalence=0.35) == pytest.approx(0.5)
    # Low q_mean (below reference) and low prevalence -> higher dispatch cost (push toward more effort).
    low = cfg.resolved_dispatch_overhead_cost(q_mean=0.05, expected_prevalence=0.15)
    assert low > 0.5
    # High q_mean and high prevalence -> lower dispatch cost (push toward less effort), clamped at dispatch_min.
    high = cfg.resolved_dispatch_overhead_cost(q_mean=0.20, expected_prevalence=0.65)
    assert high == pytest.approx(0.0)
    # Missing an observable falls back to leaving that term at its reference (no contribution).
    only_q = cfg.resolved_dispatch_overhead_cost(q_mean=0.05, expected_prevalence=None)
    assert only_q == pytest.approx(0.5 + 10.0 * (0.1167 - 0.05))


def test_regime_adaptive_disabled_matches_fixed_dispatch_cost_exactly() -> None:
    """regime_adaptive=False (the pre-R11-regime default) must be untouched by the new fields."""
    fixed = OperationalCostConfig(dispatch_overhead_cost=0.5)
    adaptive_off = OperationalCostConfig(regime_adaptive=False, dispatch_overhead_cost=0.5, q_slope=999.0, prevalence_slope=999.0)
    for q, p in [(0.05, 0.1), (0.20, 0.9), (None, None)]:
        assert fixed.action_cost(3, q_mean=q, expected_prevalence=p) == adaptive_off.action_cost(3, q_mean=q, expected_prevalence=p)
    assert fixed.action_cost(3) == pytest.approx(0.5 + 3.0)


def test_planner_shifts_effort_with_regime_when_adaptive_enabled() -> None:
    """End-to-end: the planner actually picks lower effort in a high-prevalence-looking
    incident and higher effort in a low-prevalence-looking one, holding everything else fixed."""
    cfg = EffortAwarePlannerConfig(
        cost=OperationalCostConfig(regime_adaptive=True, dispatch_base=0.5, q_slope=10.0, prevalence_slope=6.0)
    )
    planner = EffortAwareInformationGainPlanner(cfg)

    # A belief where every site looks likely-occupied (high expected prevalence).
    high_prevalence_hyps = [
        EcologicalHypothesis({"a": True, "b": True, "c": True, "d": True}, prior_weight=1.0),
    ]
    high_belief = SpatialBeliefEngine.initialize(high_prevalence_hyps, [QHypothesis(0.1)], confirmed_sites={"a"})
    incident = _incident()
    from adaptive_response.graph_state import GraphStateExporter
    from adaptive_response.models import PublicState
    public = PublicState(
        sites=incident.sites, edges=incident.edges, initial_detection="a",
        remaining_budget=18, round=0, teams=1, protocol="test", seed=1,
    )
    graph_high = GraphStateExporter().export(public, high_belief.as_belief_state())
    scored_high = planner.score_actions(graph_high, high_belief, 18)
    dispatch_high = scored_high["ranked_sites"][0]["dispatch_overhead_cost_used"]

    # A belief where only one site out of many looks occupied (low expected prevalence).
    low_prevalence_hyps = [
        EcologicalHypothesis({"a": True, "b": False, "c": False, "d": False}, prior_weight=1.0),
    ]
    low_belief = SpatialBeliefEngine.initialize(low_prevalence_hyps, [QHypothesis(0.1)], confirmed_sites={"a"})
    graph_low = GraphStateExporter().export(public, low_belief.as_belief_state())
    scored_low = planner.score_actions(graph_low, low_belief, 18)
    dispatch_low = scored_low["ranked_sites"][0]["dispatch_overhead_cost_used"]

    assert dispatch_low > dispatch_high


def test_revisit_requires_escalation_blocks_same_or_lower_effort_revisits() -> None:
    """R11 revisit-masking ablation: a revisit is only ever offered at a
    strictly higher effort than what was already spent at that site."""
    from dataclasses import replace as _replace

    cfg = _replace(EffortAwarePlannerConfig(), revisit_requires_escalation=True)
    planner = EffortAwareInformationGainPlanner(cfg)
    loop = _loop(planner, budget=18)

    first = loop.plan_next()
    first_site = first.allocations[0].site_id
    first_effort = first.allocations[0].effort_units
    loop.run_round()

    # Force the loop to re-consider the just-surveyed site by scoring directly.
    graph = loop.current_graph_state
    scored = planner.score_actions(graph, loop.current_spatial_belief, loop.current_public_state.remaining_budget)
    same_site_rows = [row for row in scored["ranked_actions"] if row["site_id"] == first_site]
    assert all(row["effort_units"] > first_effort for row in same_site_rows)


def test_revisit_requires_escalation_disabled_by_default() -> None:
    assert EffortAwarePlannerConfig().revisit_requires_escalation is False
    assert EffortAwarePlannerConfig.load_frozen().revisit_requires_escalation is False


