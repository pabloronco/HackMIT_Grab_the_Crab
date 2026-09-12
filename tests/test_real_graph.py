from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptive_response.real_graph import (
    build_knn_edges,
    build_radius_edges,
    connected_components,
    diagnose_real_graph_geometry,
    haversine_km,
    load_real_sites,
)


def _sites() -> list[dict[str, object]]:
    return [
        {"site_id": "1", "latitude": 48.000, "longitude": -122.000},
        {"site_id": "2", "latitude": 48.010, "longitude": -122.000},
        {"site_id": "3", "latitude": 48.020, "longitude": -122.000},
        {"site_id": "4", "latitude": 48.200, "longitude": -122.000},
    ]


def test_haversine_zero_and_one_degree_latitude() -> None:
    assert haversine_km(48.0, -122.0, 48.0, -122.0) == pytest.approx(0.0)
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.2, rel=0.01)


def test_union_knn_is_deterministic_and_undirected() -> None:
    edges = build_knn_edges(_sites(), 1)
    pairs = {(edge["src"], edge["dst"]) for edge in edges}
    assert pairs == {("1", "2"), ("2", "3"), ("3", "4")}
    assert all(edge["src"] < edge["dst"] for edge in edges)


def test_radius_graph_can_remain_disconnected() -> None:
    sites = _sites()
    edges = build_radius_edges(sites, 2.0)
    components = connected_components([str(site["site_id"]) for site in sites], edges)
    assert len(components) == 2
    assert sorted(len(component) for component in components) == [1, 3]


def test_diagnostic_reports_candidate_without_freezing_connectivity() -> None:
    summary, candidate = diagnose_real_graph_geometry(_sites(), k_values=(1, 2, 3), radius_values_km=(2.0, 30.0))
    assert summary["sites"] == 4
    assert summary["pairwise_pairs"] == 6
    assert summary["diagnostic_candidate"]["not_frozen"] is True
    assert summary["diagnostic_candidate"]["construction"] == "union_knn_k3"
    assert candidate
    assert "Connectivity weight remains OPEN" in summary["notes"][-1]


def test_load_real_sites_rejects_duplicate_site_ids(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_id", "latitude", "longitude"])
        writer.writeheader()
        writer.writerow({"site_id": "1", "latitude": 48.0, "longitude": -122.0})
        writer.writerow({"site_id": "1", "latitude": 48.1, "longitude": -122.1})

    with pytest.raises(ValueError, match="Duplicate site_id"):
        load_real_sites(path)
