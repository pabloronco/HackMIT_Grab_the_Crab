from math import isinf

from adaptive_response.model_mismatch import posterior_predictive_surprise
from adaptive_response.models import Observation, ObservationBatch
from adaptive_response.spatial_belief import (
    EcologicalHypothesis,
    QHypothesis,
    SpatialBeliefEngine,
)


def _belief(*, occupied_weight: float = 1.0, absent_weight: float = 1.0):
    return SpatialBeliefEngine.initialize(
        (
            EcologicalHypothesis(
                presence_by_site={"A": True, "B": False},
                prior_weight=occupied_weight,
                label="occupied_A",
            ),
            EcologicalHypothesis(
                presence_by_site={"A": False, "B": False},
                prior_weight=absent_weight,
                label="absent_A",
            ),
        ),
        (QHypothesis(q=0.2, prior_weight=1.0, label="q_02"),),
    )


def test_predictive_surprise_matches_observation_probability() -> None:
    belief = _belief()
    observation = Observation(site_id="A", effort=1, detection=True, round=1)

    result = posterior_predictive_surprise(belief, observation)

    # P(A occupied)=0.5 and P(detect|occupied,e=1)=0.2.
    assert abs(result.predictive_detection_probability - 0.1) < 1e-12
    assert abs(result.observation_probability - 0.1) < 1e-12
    assert abs(result.surprise_bits - 3.321928094887362) < 1e-12
    assert result.impossible_under_current_ensemble is False


def test_more_effort_changes_predictive_surprise_without_using_hidden_truth() -> None:
    belief = _belief()
    weak = posterior_predictive_surprise(
        belief,
        Observation(site_id="A", effort=1, detection=False, round=1),
    )
    strong = posterior_predictive_surprise(
        belief,
        Observation(site_id="A", effort=6, detection=False, round=1),
    )

    assert strong.observation_probability < weak.observation_probability
    assert strong.surprise_bits > weak.surprise_bits


def test_detection_at_site_absent_in_every_hypothesis_is_explicitly_impossible() -> None:
    belief = _belief()
    result = posterior_predictive_surprise(
        belief,
        Observation(site_id="B", effort=6, detection=True, round=1),
    )

    assert result.observation_probability == 0.0
    assert isinf(result.surprise_bits)
    assert result.impossible_under_current_ensemble is True


def test_surprise_is_preupdate_and_q_shift_can_be_inspected_separately() -> None:
    belief = SpatialBeliefEngine.initialize(
        (
            EcologicalHypothesis(
                presence_by_site={"A": True},
                prior_weight=1.0,
                label="A_occupied",
            ),
        ),
        (
            QHypothesis(q=0.05, prior_weight=1.0, label="low_q"),
            QHypothesis(q=0.30, prior_weight=1.0, label="high_q"),
        ),
    )
    observation = Observation(site_id="A", effort=6, detection=False, round=1)

    diagnostic = posterior_predictive_surprise(belief, observation)
    posterior = SpatialBeliefEngine.update(
        belief,
        ObservationBatch(
            observations=(observation,),
            round=1,
            total_effort=6,
        ),
    )

    assert abs(diagnostic.q_mean_before - 0.175) < 1e-12
    assert posterior.q_mean() < diagnostic.q_mean_before
    assert diagnostic.observation_probability > 0.0
