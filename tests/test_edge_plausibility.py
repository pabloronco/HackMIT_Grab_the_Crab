from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptive_response.edge_plausibility import (
    annotate_candidate_edges,
    build_ranked_candidate_edges,
    parse_phy_ident,
    summarize_edge_candidates,
)


def test_parse_phy_ident_documented_format() -> None:
    parsed = parse_phy_ident("12/03/0552/0")
    assert parsed["region"] == "12"
    assert parsed["area"] == "03"
    assert parsed["physical_unit"] == "0552"
    assert parsed["subunit"] == "0"


def test_parse_phy_ident_invalid_is_explicit() -> None:
    parsed = parse_phy_ident("bad")
    assert parsed["phy_ident"] == "bad"
    assert parsed["region"] is None
    assert parsed["area"] is None


def test_ranked_candidates_allow_variable_reciprocity() -> None:
    # Deliberately avoid equal-distance ties: the test is about reciprocity,
    # not floating-point tie-breaking between geographically equidistant sites.
    sites = [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.00},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.01},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.03},
        {"site_id": "D", "latitude": 48.0, "longitude": -122.50},
    ]
    edges = build_ranked_candidate_edges(sites, k_max=1)
    by_pair = {(row["src"], row["dst"]): row for row in edges}

    assert ("A", "B") in by_pair
    assert ("B", "C") in by_pair
    assert ("C", "D") in by_pair
    assert by_pair[("A", "B")]["mutual_knn"] is True
    assert by_pair[("B", "C")]["mutual_knn"] is False


def test_ranked_candidates_validate_k() -> None:
    sites = [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.0},
        {"site_id": "B", "latitude": 48.1, "longitude": -122.1},
    ]
    with pytest.raises(ValueError, match="k_max"):
        build_ranked_candidate_edges(sites, k_max=2)


def test_annotation_keeps_mapping_context_separate_from_acceptance() -> None:
    edges = [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 5.0,
            "src_rank": 1,
            "dst_rank": 2,
            "mutual_knn": True,
            "src_local_distance_ratio": 1.0,
            "dst_local_distance_ratio": 1.5,
            "max_local_distance_ratio": 1.5,
        }
    ]
    site_units = {"A": 1, "B": 2}
    metadata = {
        1: {
            "PHY_IDENT": "12/03/0552/0",
            "SHORENAME": "Example Bay",
            "shorezone_region": "12",
            "shorezone_area": "03",
        },
        2: {
            "PHY_IDENT": "12/03/0553/0",
            "SHORENAME": "Example Bay",
            "shorezone_region": "12",
            "shorezone_area": "03",
        },
    }

    result = annotate_candidate_edges(edges, site_units, metadata)[0]
    assert result["same_shorezone_region"] is True
    assert result["same_shorezone_area"] is True
    assert result["same_shorename"] is True
    assert "accepted" not in result

    summary = summarize_edge_candidates([result])
    assert summary["same_shorezone_area_edges"] == 1
    assert summary["mutual_knn_edges"] == 1


def test_summary_handles_missing_mapping_metadata() -> None:
    edges = [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 3.0,
            "mutual_knn": False,
            "max_local_distance_ratio": 4.0,
            "same_shorezone_region": None,
            "same_shorezone_area": None,
            "same_shorename": None,
        }
    ]
    summary = summarize_edge_candidates(edges)

    # Missing optional ShoreZone mapping metadata must not crash the summary
    # and must not be silently counted as supporting connectivity.
    assert summary["edges"] == 1
    assert summary["same_shorezone_region_edges"] == 0
    assert summary["same_shorezone_area_edges"] == 0
    assert summary["same_shorename_edges"] == 0
    assert summary["local_distance_ratio_max"] == 4.0
