from __future__ import annotations

import pytest

from adaptive_response.q_sensitivity import (
    build_q_sensitivity_table,
    detection_probability,
    posterior_after_nondetection,
)


def test_more_effort_increases_detection_if_occupied() -> None:
    assert detection_probability(0.1, 10) > detection_probability(0.1, 1)


def test_more_effort_makes_nondetection_stronger_evidence() -> None:
    p1 = posterior_after_nondetection(0.5, 0.1, 1)
    p10 = posterior_after_nondetection(0.5, 0.1, 10)
    assert p10 < p1


def test_higher_q_makes_same_nondetection_stronger() -> None:
    low_q = posterior_after_nondetection(0.5, 0.05, 6)
    high_q = posterior_after_nondetection(0.5, 0.20, 6)
    assert high_q < low_q


def test_default_table_labels_q_as_probe_not_estimate() -> None:
    rows = build_q_sensitivity_table(q_values=(0.1,), effort_values=(6,))
    assert len(rows) == 1
    assert rows[0]["q_status"] == "DESIGN_SENSITIVITY_PROBE_NOT_EMPIRICAL_ESTIMATE"
    assert rows[0]["detection_if_occupied"] == pytest.approx(1 - 0.9**6)
