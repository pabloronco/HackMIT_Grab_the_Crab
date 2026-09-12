from __future__ import annotations

import pytest

from adaptive_response.route_graph_sensitivity import (
    compare_route_distance_topologies,
    extract_site_snap_distances,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.0},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.1},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.2},
        {"site_id": "D", "latitude": 48.0, "longitude": -122.3},
    ]


def _row(
    src: str,
    dst: str,
    direct: float,
    grid_route: float,
    src_snap: float,
    dst_snap: float,
    *,
    local: bool,
) -> dict[str, object]:
    total = grid_route + src_snap + dst_snap
    return {
        "src": src,
        "dst": dst,
        "distance_km": direct,
        "candidate_local_radius": local,
        "candidate_sparse_review": not local,
        "salishseacast_route_found": True,
        "salishseacast_water_route_km": grid_route,
        "salishseacast_detour_ratio": grid_route / direct,
        "salishseacast_total_route_proxy_km": total,
        "salishseacast_total_detour_ratio": total / direct,
        "salishseacast_src_snap_km": src_snap,
        "salishseacast_dst_snap_km": dst_snap,
    }


def _rows() -> list[dict[str, object]]:
    return [
        _row("A", "B", 8.0, 9.0, 0.2, 0.3, local=True),
        _row("B", "C", 10.0, 24.0, 0.3, 1.4, local=True),
        _row("C", "D", 12.0, 42.0, 1.4, 0.4, local=True),
        _row("A", "D", 40.0, 45.0, 0.2, 0.4, local=False),
    ]


def test_route_thresholds_use_endpoint_corrected_proxy_on_local_edges_only() -> None:
    summary = compare_route_distance_topologies(
        _sites(), _rows(), route_thresholds_km=(20.0, 30.0, 50.0)
    )
    variants = summary["topology_variants"]
    assert variants["all_local_with_route"]["edges"] == 3
    assert variants["route_le_20km"]["edges"] == 1
    assert variants["route_le_30km"]["edges"] == 2
    assert variants["route_le_50km"]["edges"] == 3
    assert summary["review_only_edges"] == 1


def test_snap_quality_is_reported_not_used_as_hidden_filter() -> None:
    summary = compare_route_distance_topologies(_sites(), _rows())
    assert summary["site_snap"]["sites_gt_1km"] == 1
    assert summary["local_route"]["edges_with_endpoint_snap_gt_1km"] == 2
    assert summary["topology_variants"]["all_local_with_route"]["edges"] == 3


def test_extract_site_snap_distances_requires_consistency() -> None:
    rows = _rows()
    rows[1]["salishseacast_src_snap_km"] = 0.9
    with pytest.raises(ValueError, match="Inconsistent snap distance"):
        extract_site_snap_distances(rows)


def test_detour_summary_uses_total_route_proxy() -> None:
    summary = compare_route_distance_topologies(_sites(), _rows())
    route = summary["local_route"]
    assert route["total_detour_ratio_gt_2"] == 2
    assert route["total_detour_ratio_gt_3"] == 1
    assert route["total_detour_ratio_gt_5"] == 0
    assert route["most_extreme_detours"][0]["src"] == "C"
    assert route["most_extreme_detours"][0]["dst"] == "D"
    assert route["total_route_proxy_km_median"] > route["grid_route_km_median"]
