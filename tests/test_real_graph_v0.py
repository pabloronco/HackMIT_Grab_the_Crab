from __future__ import annotations

from adaptive_response.real_graph_v0 import (
    build_real_graph_v0_audit,
    select_real_graph_v0_edges,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "A", "latitude": 48.0, "longitude": -122.00},
        {"site_id": "B", "latitude": 48.0, "longitude": -122.01},
        {"site_id": "C", "latitude": 48.0, "longitude": -122.02},
        {"site_id": "D", "latitude": 48.0, "longitude": -122.03},
    ]


def _row(
    src: str,
    dst: str,
    *,
    local: bool,
    review: bool,
    straight_water: float,
    route_found: bool = True,
    direct_km: float = 1.0,
    route_km: float = 2.0,
    src_snap: float = 0.2,
    dst_snap: float = 0.3,
) -> dict[str, object]:
    total = src_snap + route_km + dst_snap if route_found else None
    return {
        "src": src,
        "dst": dst,
        "distance_km": direct_km,
        "candidate_local_radius": local,
        "candidate_sparse_review": review,
        "straight_marine_fraction": straight_water,
        "salishseacast_route_found": route_found,
        "salishseacast_water_route_km": route_km if route_found else None,
        "salishseacast_src_snap_km": src_snap,
        "salishseacast_dst_snap_km": dst_snap,
        "salishseacast_total_route_proxy_km": total,
        "salishseacast_total_detour_ratio": total / direct_km if total is not None else None,
    }


def _rows() -> list[dict[str, object]]:
    return [
        _row("A", "B", local=True, review=False, straight_water=1.0),
        _row("B", "C", local=True, review=False, straight_water=0.2),
        _row("C", "D", local=True, review=False, straight_water=0.0),
        _row(
            "A",
            "D",
            local=False,
            review=True,
            straight_water=1.0,
            direct_km=30.0,
            route_km=35.0,
        ),
    ]


def test_primary_keeps_all_routed_local_edges_including_zero_straight_water() -> None:
    primary, review = select_real_graph_v0_edges(_rows())
    pairs = {(row["src"], row["dst"]) for row in primary}
    assert pairs == {("A", "B"), ("B", "C"), ("C", "D")}
    assert len(review) == 1


def test_zero_straight_water_no_longer_controls_adjacency() -> None:
    primary, _ = select_real_graph_v0_edges(_rows())
    zero = next(row for row in primary if row["src"] == "C" and row["dst"] == "D")
    assert zero["straight_marine_fraction"] == 0.0
    assert zero["v0_edge_status"] == "primary_local_curved_water_route_audited"
    assert zero["v0_primary"] is True


def test_sparse_review_edge_never_repairs_primary_connectivity() -> None:
    primary, review = select_real_graph_v0_edges(_rows())
    assert all(row["candidate_local_radius"] for row in primary)
    assert review[0]["src"] == "A"
    assert review[0]["dst"] == "D"
    assert review[0]["v0_primary"] is False


def test_missing_route_is_not_silently_promoted() -> None:
    rows = _rows() + [
        _row(
            "A",
            "C",
            local=True,
            review=False,
            straight_water=1.0,
            route_found=False,
        )
    ]
    primary, _ = select_real_graph_v0_edges(rows)
    assert ("A", "C") not in {(row["src"], row["dst"]) for row in primary}


def test_audit_reports_revised_primary_topology_and_open_connectivity_weight() -> None:
    _, summary = build_real_graph_v0_audit(_sites(), _rows())
    assert summary["primary_edges"] == 3
    assert summary["straight_zero_water_edges_retained"] == 1
    assert summary["sampling_isolation_review_only_edges"] == 1
    assert summary["primary_graph"]["components"] == 1
    assert summary["primary_graph"]["isolated_sites"] == []
    assert "rejected" in summary["rejected_rule"].lower()
    assert any("Connectivity weight remains OPEN" in note for note in summary["notes"])
