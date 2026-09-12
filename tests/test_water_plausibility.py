from __future__ import annotations

import pytest

from adaptive_response.water_plausibility import (
    annotate_open_saltwater_support,
    sample_interior_segment_points,
    summarize_water_support,
)


def test_sample_interior_segment_points_excludes_endpoints() -> None:
    points = sample_interior_segment_points(0.0, 0.0, 6.0, 12.0, samples=5)
    assert len(points) == 5
    assert points[0] == pytest.approx((1.0, 2.0))
    assert points[-1] == pytest.approx((5.0, 10.0))


def test_sample_interior_segment_points_validates_count() -> None:
    with pytest.raises(ValueError, match="samples"):
        sample_interior_segment_points(0.0, 0.0, 1.0, 1.0, samples=0)


def test_annotate_open_saltwater_support_preserves_diagnostic_semantics() -> None:
    edges = [
        {
            "src": "A",
            "dst": "B",
            "distance_km": "10.0",
            "max_local_distance_ratio": "2.0",
        }
    ]
    coordinates = {"A": (0.0, 0.0), "B": (0.0, 6.0)}

    # Sample longitudes are 1,2,3,4,5. Mark 1,2,4 as open saltwater.
    result = annotate_open_saltwater_support(
        edges,
        coordinates,
        samples_per_edge=5,
        checker=lambda lat, lon: lon in {1.0, 2.0, 4.0},
    )[0]

    assert result["marine_sample_count"] == 5
    assert result["marine_sample_hits"] == 3
    assert result["straight_marine_fraction"] == pytest.approx(0.6)
    assert result["straight_marine_all"] is False
    assert result["straight_marine_none"] is False
    assert result["straight_marine_pattern"] == "11010"
    assert "accepted" not in result


def test_annotate_rejects_unknown_site_reference() -> None:
    edges = [{"src": "A", "dst": "B", "distance_km": 1.0, "max_local_distance_ratio": 1.0}]
    with pytest.raises(ValueError, match="unknown site"):
        annotate_open_saltwater_support(
            edges,
            {"A": (0.0, 0.0)},
            checker=lambda lat, lon: True,
        )


def test_summarize_water_support_counts_all_mixed_and_none() -> None:
    edges = [
        {"straight_marine_fraction": 1.0, "straight_marine_all": True, "straight_marine_none": False},
        {"straight_marine_fraction": 0.6, "straight_marine_all": False, "straight_marine_none": False},
        {"straight_marine_fraction": 0.0, "straight_marine_all": False, "straight_marine_none": True},
    ]
    summary = summarize_water_support(edges)
    assert summary["edges"] == 3
    assert summary["all_samples_open_saltwater_edges"] == 1
    assert summary["mixed_open_saltwater_edges"] == 1
    assert summary["no_samples_open_saltwater_edges"] == 1
    assert summary["straight_marine_fraction_median"] == pytest.approx(0.6)
