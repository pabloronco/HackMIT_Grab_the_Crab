from __future__ import annotations

from adaptive_response.graph_filter_sensitivity import compare_water_filter_topologies


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.00, "longitude": -122.00},
        {"site_id": "B", "latitude": 48.00, "longitude": -122.01},
        {"site_id": "C", "latitude": 48.00, "longitude": -122.02},
        {"site_id": "D", "latitude": 48.00, "longitude": -122.03},
    ]


def _rows() -> list[dict[str, object]]:
    return [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 1.0,
        },
        {
            "src": "B",
            "dst": "C",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 0.4,
        },
        {
            "src": "C",
            "dst": "D",
            "distance_km": 1.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "straight_marine_fraction": 0.0,
        },
        {
            "src": "A",
            "dst": "D",
            "distance_km": 8.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "straight_marine_fraction": 1.0,
        },
    ]


def test_sparse_review_edges_never_enter_primary_variants() -> None:
    summary = compare_water_filter_topologies(_sites(), _rows())
    assert summary["topology_variants"]["all_local"]["edges"] == 3
    assert summary["sampling_isolation_review_only"]["edges"] == 1


def test_exclude_zero_water_is_less_aggressive_than_point_four_filter() -> None:
    summary = compare_water_filter_topologies(_sites(), _rows())
    assert summary["topology_variants"]["exclude_zero_water"]["edges"] == 2
    assert summary["topology_variants"]["water_ge_0_4"]["edges"] == 2
    assert summary["topology_variants"]["water_ge_0_6"]["edges"] == 1


def test_stricter_filters_can_create_isolation_without_repairing_it() -> None:
    summary = compare_water_filter_topologies(_sites(), _rows())
    strict = summary["topology_variants"]["water_ge_0_6"]
    assert set(strict["isolated_sites"]) == {"C", "D"}
    assert strict["components"] == 3


def test_review_summary_reports_long_edges_without_accepting_them() -> None:
    rows = _rows()
    rows[-1]["distance_km"] = 40.0
    summary = compare_water_filter_topologies(_sites(), rows)
    review = summary["sampling_isolation_review_only"]
    assert review["all_sampled_water"] == 1
    assert review["over_35km"] == 1
