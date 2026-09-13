import pytest

from adaptive_response import (
    EcologicalHypothesis,
    Edge,
    GraphStateExporter,
    Observation,
    ObservationBatch,
    PublicState,
    QHypothesis,
    Site,
    SpatialBeliefEngine,
)


def batch(*observations: Observation, round: int = 1) -> ObservationBatch:
    return ObservationBatch(
        observations=tuple(observations),
        round=round,
        total_effort=sum(observation.effort for observation in observations),
    )


def correlated_worlds() -> list[EcologicalHypothesis]:
    return [
        EcologicalHypothesis({"a": True, "b": True}, label="both_present"),
        EcologicalHypothesis({"a": False, "b": False}, label="both_absent"),
    ]


def test_confirmed_site_conditions_worlds_without_changing_q_support() -> None:
    belief = SpatialBeliefEngine.initialize(
        correlated_worlds(),
        [QHypothesis(0.1), QHypothesis(0.3)],
        confirmed_sites={"a"},
    )

    assert belief.p_by_site()["a"] == pytest.approx(1.0)
    assert belief.p_by_site()["b"] == pytest.approx(1.0)
    assert belief.q_posterior()[0.1] == pytest.approx(0.5)
    assert belief.q_posterior()[0.3] == pytest.approx(0.5)
    assert sum(belief.weights) == pytest.approx(1.0)


def test_zero_after_ten_checks_is_stronger_than_zero_after_one_and_propagates() -> None:
    prior = SpatialBeliefEngine.initialize(
        correlated_worlds(),
        [QHypothesis(0.25)],
    )

    after_one = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=1, detection=False, round=1)),
    )
    after_ten = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=10, detection=False, round=1)),
    )

    assert after_ten.p_by_site()["a"] < after_one.p_by_site()["a"] < 0.5
    assert after_ten.p_by_site()["b"] < after_one.p_by_site()["b"] < 0.5


def test_positive_detection_changes_correlated_site_via_world_reweighting() -> None:
    prior = SpatialBeliefEngine.initialize(
        correlated_worlds(),
        [QHypothesis(0.2)],
    )
    updated = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=2, detection=True, round=1)),
    )

    assert updated.p_by_site()["a"] == pytest.approx(1.0)
    assert updated.p_by_site()["b"] == pytest.approx(1.0)


def test_evidence_does_not_move_independent_site_without_model_correlation() -> None:
    independent_worlds = [
        EcologicalHypothesis({"a": False, "b": False}),
        EcologicalHypothesis({"a": False, "b": True}),
        EcologicalHypothesis({"a": True, "b": False}),
        EcologicalHypothesis({"a": True, "b": True}),
    ]
    prior = SpatialBeliefEngine.initialize(independent_worlds, [QHypothesis(0.3)])
    updated = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=5, detection=False, round=1)),
    )

    assert updated.p_by_site()["a"] < 0.5
    assert updated.p_by_site()["b"] == pytest.approx(0.5)


def test_q_uncertainty_is_updated_instead_of_assuming_true_q_known() -> None:
    prior = SpatialBeliefEngine.initialize(
        [EcologicalHypothesis({"a": True})],
        [QHypothesis(0.05), QHypothesis(0.30)],
    )
    assert prior.q_mean() == pytest.approx(0.175)

    updated = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=6, detection=False, round=1)),
    )

    assert updated.p_by_site()["a"] == pytest.approx(1.0)
    assert updated.q_mean() < prior.q_mean()
    assert updated.q_posterior()[0.05] > updated.q_posterior()[0.30]


def test_predictive_detection_probability_marginalizes_world_and_q_uncertainty() -> None:
    belief = SpatialBeliefEngine.initialize(
        [
            EcologicalHypothesis({"a": True}),
            EcologicalHypothesis({"a": False}),
        ],
        [QHypothesis(0.2)],
    )

    # P(occupied)=0.5 and P(detect | occupied,e=2)=1-(0.8)^2=0.36.
    assert SpatialBeliefEngine.predictive_detection_probability(
        belief,
        site_id="a",
        effort=2,
    ) == pytest.approx(0.18)


def test_projection_to_existing_belief_state_preserves_observable_contract() -> None:
    observation = Observation("a", effort=2, detection=False, round=1)
    prior = SpatialBeliefEngine.initialize(correlated_worlds(), [QHypothesis(0.2)])
    updated = SpatialBeliefEngine.update(prior, batch(observation))

    projected = updated.as_belief_state()

    assert projected.p_by_site == updated.p_by_site()
    assert projected.uncertainty_by_site == updated.uncertainty_by_site()
    assert projected.observed_history == (observation,)
    assert not hasattr(projected, "hypotheses")
    assert not hasattr(projected, "hidden_world")


def test_spatial_projection_exports_through_existing_graphstate_schema() -> None:
    prior = SpatialBeliefEngine.initialize(correlated_worlds(), [QHypothesis(0.2)])
    updated = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=3, detection=False, round=1)),
    )
    public_state = PublicState(
        sites=[
            Site("a", 0.0, 0.0, habitat_score=0.4, q_model=0.2),
            Site("b", 1.0, 0.0, habitat_score=0.6, q_model=0.2),
        ],
        edges=[Edge("a", "b", distance=1.0)],
        initial_detection="a",
        remaining_budget=4,
        round=1,
        teams=1,
        protocol="test",
        seed=7,
    )

    graph_state = GraphStateExporter().export(public_state, updated.as_belief_state())
    node_index = {site_id: index for index, site_id in enumerate(graph_state.node_ids)}

    assert graph_state.node_features[node_index["a"]][0] == pytest.approx(
        updated.p_by_site()["a"]
    )
    assert graph_state.node_features[node_index["b"]][0] == pytest.approx(
        updated.p_by_site()["b"]
    )


def test_joint_weights_remain_normalized_and_effective_sample_size_is_valid() -> None:
    prior = SpatialBeliefEngine.initialize(
        correlated_worlds(),
        [QHypothesis(0.1), QHypothesis(0.3)],
    )
    updated = SpatialBeliefEngine.update(
        prior,
        batch(Observation("a", effort=3, detection=False, round=1)),
    )

    assert sum(updated.weights) == pytest.approx(1.0)
    assert 1.0 <= updated.effective_sample_size() <= len(updated.weights)
    assert updated.joint_entropy_bits() >= 0.0


def test_impossible_positive_batch_is_rejected() -> None:
    belief = SpatialBeliefEngine.initialize(
        [EcologicalHypothesis({"a": False})],
        [QHypothesis(0.2)],
    )

    with pytest.raises(ValueError, match="zero probability"):
        SpatialBeliefEngine.update(
            belief,
            batch(Observation("a", effort=1, detection=True, round=1)),
        )


def test_mismatched_ecological_site_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="same site ids"):
        SpatialBeliefEngine.initialize(
            [
                EcologicalHypothesis({"a": True, "b": False}),
                EcologicalHypothesis({"a": True}),
            ],
            [QHypothesis(0.2)],
        )
