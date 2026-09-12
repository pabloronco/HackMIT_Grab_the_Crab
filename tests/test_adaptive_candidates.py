from __future__ import annotations

import pytest

from adaptive_response.adaptive_candidates import (
    build_adaptive_candidate_pool,
    compare_adaptive_candidate_radii,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.00},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.02},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.04},
        {"site_id": "D", "latitude": 48.0, "longitude": -123.00},
    ]


def test_local_radius_allows_variable_degree() -> None:
    rows, summary = build_adaptive_candidate_pool(
        _sites(),
        local_radius_km=5.0,
        sparse_degree_threshold=0,
        sparse_review_k=0,
    )
    by_pair = {(row["src"], row["dst"]): row for row in rows}

    assert ("A", "B") in by_pair
    assert ("B", "C") in by_pair
    assert all("D" not in pair for pair in by_pair)
    assert summary["local_degree_min"] == 0
    assert summary["local_degree_max"] == 2


def test_sparse_site_gets_review_candidate_without_auto_acceptance() -> None:
    rows, summary = build_adaptive_candidate_pool(
        _sites(),
        local_radius_km=5.0,
        sparse_degree_threshold=1,
        sparse_review_k=1,
    )
    review_only = [
        row
        for row in rows
        if row["candidate_sparse_review"] and not row["candidate_local_radius"]
    ]

    assert summary["sampling_sparse_site_ids"] == ["D"]
    assert len(review_only) == 1
    assert review_only[0]["sparse_review_requested_by"] == "D"
    assert review_only[0]["candidate_tier"] == "sampling_isolation_review"


def test_sparse_review_does_not_impose_minimum_ecological_degree() -> None:
    _, summary = build_adaptive_candidate_pool(
        _sites(),
        local_radius_km=5.0,
        sparse_degree_threshold=1,
        sparse_review_k=1,
    )
    # The local-radius graph still records D as isolated. The review pool is a
    # diagnostic expansion, not a claim that D has a validated ecological edge.
    assert "D" in summary["local_radius_graph"]["isolated_sites"]
    assert summary["candidate_pool_graph"]["isolated_sites"] == []


def test_candidate_builder_validates_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_adaptive_candidate_pool(_sites(), local_radius_km=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        build_adaptive_candidate_pool(_sites(), sparse_degree_threshold=-1)
    with pytest.raises(ValueError, match="smaller than number of sites"):
        build_adaptive_candidate_pool(_sites(), sparse_review_k=4)


def test_radius_comparison_is_monotonic_for_local_edges() -> None:
    result = compare_adaptive_candidate_radii(
        _sites(),
        radii_km=(2.0, 5.0, 10.0),
        sparse_degree_threshold=1,
        sparse_review_k=1,
    )
    counts = [result[key]["local_radius_edges"] for key in ("2", "5", "10")]
    assert counts == sorted(counts)
