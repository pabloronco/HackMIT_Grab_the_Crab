from __future__ import annotations

from adaptive_response.q_identifiability import summarize_q_identifiability


def _rows() -> list[dict[str, object]]:
    return [
        {
            "site_id": "A",
            "year": 2020,
            "month": 5,
            "trap_sets": 2.0,
            "effort_missing": False,
            "cama_count": 0,
            "detected": False,
        },
        {
            "site_id": "A",
            "year": 2020,
            "month": 6,
            "trap_sets": 6.0,
            "effort_missing": False,
            "cama_count": 1,
            "detected": True,
        },
        {
            "site_id": "A",
            "year": 2020,
            "month": 7,
            "trap_sets": 6.0,
            "effort_missing": False,
            "cama_count": 0,
            "detected": False,
        },
        {
            "site_id": "B",
            "year": 2020,
            "month": 6,
            "trap_sets": None,
            "effort_missing": True,
            "cama_count": 0,
            "detected": False,
        },
    ]


def test_current_monthly_table_does_not_claim_q_identifiability() -> None:
    summary = summarize_q_identifiability(_rows())
    assert summary["q_identifiability"]["per_effort_q_identified_from_current_table"] is False
    assert summary["same_month_replicates_available"] is False


def test_effort_conditioned_detection_fraction_is_descriptive_only() -> None:
    summary = summarize_q_identifiability(_rows())
    effort = summary["effort"]
    assert effort["known_rows"] == 3
    assert effort["missing_rows"] == 1
    assert effort["by_exact_trap_sets"]["2"]["observed_detection_fraction"] == 0.0
    assert effort["by_exact_trap_sets"]["6"]["observed_detection_fraction"] == 0.5
    assert any("not direct detectability estimates" in note for note in summary["notes"])


def test_post_detection_months_are_reported_without_assuming_closure() -> None:
    summary = summarize_q_identifiability(_rows())
    temporal = summary["temporal_repetition"]
    assert temporal["site_years_with_any_detection"] == 1
    assert temporal["known_effort_months_after_first_detection"] == 1
    assert temporal["non_detections_after_first_detection"] == 1
    assert "closure" in summary["q_identifiability"]["reason"].lower()


def test_missing_effort_never_enters_effort_conditioned_outcomes() -> None:
    summary = summarize_q_identifiability(_rows())
    outcomes = summary["outcomes_with_known_effort"]
    assert outcomes["detections"] == 1
    assert outcomes["non_detections"] == 2
