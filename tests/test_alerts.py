from math import isinf

import pytest

from adaptive_response.alerts import (
    DEFAULT_ALERT_BITS,
    evaluate_round_surprise,
    format_alert_facts,
    template_alert_explanation,
)
from adaptive_response.models import Observation, ObservationBatch
from adaptive_response.spatial_belief import (
    EcologicalHypothesis,
    QHypothesis,
    SpatialBeliefEngine,
)


def _belief():
    # P(A occupied) = 0.5, B absent everywhere, q = 0.2.
    return SpatialBeliefEngine.initialize(
        (
            EcologicalHypothesis(presence_by_site={"A": True, "B": False}, prior_weight=1.0, label="occ"),
            EcologicalHypothesis(presence_by_site={"A": False, "B": False}, prior_weight=1.0, label="abs"),
        ),
        (QHypothesis(q=0.2, prior_weight=1.0, label="q02"),),
    )


def _batch(*observations: Observation) -> ObservationBatch:
    return ObservationBatch(observations=observations, round=1, total_effort=sum(o.effort for o in observations))


def test_expected_outcome_is_not_an_alert() -> None:
    # No detection at A with 1 check: P = 1 - 0.5*0.2 = 0.9 -> ~0.15 bits.
    alert = evaluate_round_surprise(_belief(), _batch(Observation("A", 1, False, 1)))

    assert alert.level == "none"
    assert not alert.triggered
    assert alert.alert_bits == DEFAULT_ALERT_BITS
    assert abs(alert.max_surprise_bits - 0.152003093445) < 1e-9
    assert "consistent" in template_alert_explanation(alert)


def test_surprising_outcome_crosses_a_low_setting() -> None:
    # Detection at A with 1 check: P = 0.1 -> 3.32 bits.
    alert = evaluate_round_surprise(_belief(), _batch(Observation("A", 1, True, 1)), alert_bits=3.0)

    assert alert.level == "high"
    assert alert.triggered
    assert alert.most_surprising is not None and alert.most_surprising.site_id == "A"
    text = template_alert_explanation(alert)
    assert "model-mismatch warning" in text
    assert "A" in text and "10.0%" in text
    assert "not proof" in text


def test_default_setting_flags_a_one_in_ten_outcome() -> None:
    alert = evaluate_round_surprise(_belief(), _batch(Observation("A", 1, True, 1)))
    assert DEFAULT_ALERT_BITS == 3.0
    assert alert.level == "high"  # 3.32 bits >= 3.0: outcome probability 0.1 <= 1/8


def test_a_likely_outcome_is_below_the_default_setting() -> None:
    # No detection at A with 6 checks: P = 1 - 0.5*(1-0.8^6) = 0.631 -> 0.66 bits.
    alert = evaluate_round_surprise(_belief(), _batch(Observation("A", 6, False, 1)))
    assert alert.level == "none"


def test_impossible_observation_is_flagged_without_any_threshold() -> None:
    alert = evaluate_round_surprise(_belief(), _batch(Observation("B", 1, True, 1)), alert_bits=1000.0)

    assert alert.level == "impossible"
    assert isinf(alert.max_surprise_bits) and isinf(alert.total_surprise_bits)
    assert alert.per_observation[0].impossible_under_current_ensemble
    text = template_alert_explanation(alert)
    assert "MODEL MISMATCH" in text and "zero probability" in text


def test_round_max_is_over_all_observations() -> None:
    alert = evaluate_round_surprise(
        _belief(),
        _batch(Observation("A", 1, False, 1), Observation("A", 1, True, 1)),
        alert_bits=3.0,
    )
    assert alert.level == "high"
    assert alert.most_surprising is not None and alert.most_surprising.detection is True
    assert abs(alert.total_surprise_bits - sum(s.surprise_bits for s in alert.per_observation)) < 1e-12


def test_facts_block_carries_every_number_the_explainer_may_use() -> None:
    alert = evaluate_round_surprise(_belief(), _batch(Observation("A", 1, True, 1)), alert_bits=3.0)
    facts = format_alert_facts(alert)
    assert "round 1" in facts and "HIGH" in facts
    assert "A: DETECTION after 1 checks" in facts
    assert "3.32 bits" in facts and "0.1000" in facts
    assert "not a calibrated failure criterion" in facts


def test_alert_bits_must_be_positive() -> None:
    with pytest.raises(ValueError):
        evaluate_round_surprise(_belief(), _batch(Observation("A", 1, False, 1)), alert_bits=0.0)
