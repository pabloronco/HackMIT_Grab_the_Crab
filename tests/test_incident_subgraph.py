from __future__ import annotations

from adaptive_response.incident_subgraph import (
    audit_all_incident_seeds,
    extract_incident_subgraph,
)


def _edge(src: str, dst: str, distance: float, route: float | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "src": src,
        "dst": dst,
        "distance_km": distance,
    }
    if route is not None:
        row["salishseacast_total_route_proxy_km"] = route
    return row


def test_extraction_never_crosses_component_to_hit_preferred_minimum() -> None:
    site_ids = ["A", "B", "C", "D", "E"]
    edges = [
        _edge("A", "B", 1.0),
        _edge("B", "C", 1.0),
        _edge("D", "E", 1.0),
    ]
    selected, sub_edges, summary = extract_incident_subgraph(
        site_ids,
        edges,
        seed_site_id="A",
        max_sites=5,
        preferred_min_sites=4,
    )
    assert set(selected) == {"A", "B", "C"}
    assert len(sub_edges) == 2
    assert summary["below_preferred_min"] is True
    assert summary["never_crossed_component"] is True


def test_large_component_is_capped_by_static_shortest_path_order() -> None:
    site_ids = ["A", "B", "C", "D", "E"]
    edges = [
        _edge("A", "B", 1.0, route=1.0),
        _edge("B", "C", 1.0, route=1.0),
        _edge("A", "D", 1.0, route=10.0),
        _edge("D", "E", 1.0, route=1.0),
    ]
    selected, _, summary = extract_incident_subgraph(
        site_ids,
        edges,
        seed_site_id="A",
        max_sites=3,
        preferred_min_sites=2,
    )
    assert selected == ["A", "B", "C"]
    assert summary["capped_by_max_sites"] is True


def test_isolated_seed_stays_isolated_instead_of_inventing_edges() -> None:
    site_ids = ["A", "B", "C"]
    edges = [_edge("A", "B", 1.0)]
    selected, sub_edges, summary = extract_incident_subgraph(
        site_ids,
        edges,
        seed_site_id="C",
        max_sites=3,
        preferred_min_sites=2,
    )
    assert selected == ["C"]
    assert sub_edges == []
    assert summary["source_component_sites"] == 1
    assert summary["below_preferred_min"] is True


def test_all_seed_audit_uses_topology_only_and_reports_size_distribution() -> None:
    site_ids = ["A", "B", "C", "D"]
    edges = [
        _edge("A", "B", 1.0),
        _edge("B", "C", 1.0),
    ]
    audit = audit_all_incident_seeds(
        site_ids,
        edges,
        max_sites=3,
        preferred_min_sites=2,
    )
    assert audit["seeds_in_preferred_size_range"] == 3
    assert audit["seeds_below_preferred_min"] == 1
    assert audit["selected_size_histogram"] == {"1": 1, "3": 3}
    assert any("topology/static geography only" in note for note in audit["notes"])
