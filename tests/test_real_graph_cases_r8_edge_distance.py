from __future__ import annotations

import csv

from adaptive_response.rl.real_graph_cases import (
    REAL_GRAPH_EDGES_CSV,
    build_real_incident_case,
    eligible_incident_seed_sites,
)


def _distance_km_by_pair() -> dict[frozenset[str], float]:
    with REAL_GRAPH_EDGES_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {frozenset((row["src"], row["dst"])): float(row["distance_km"]) for row in rows}


def _route_proxy_km_by_pair() -> dict[frozenset[str], float]:
    with REAL_GRAPH_EDGES_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        proxy = row.get("salishseacast_total_route_proxy_km")
        if proxy not in (None, ""):
            out[frozenset((row["src"], row["dst"]))] = float(proxy)
    return out


def test_edge_distance_uses_direct_distance_km_not_the_route_proxy() -> None:
    """R8 correction (review 2026-09-15): Edge.distance (the GraphState edge-
    distance feature the planner sees) must be the direct distance_km, never
    salishseacast_total_route_proxy_km, even when the proxy is available and
    numerically much larger (real data: edge 108-128 has distance_km ~14.2
    but a route proxy of ~113.2 - if the proxy silently won by presence, this
    test would fail loudly, not go unnoticed)."""

    distance_km = _distance_km_by_pair()
    seed_site_id = eligible_incident_seed_sites()[0]
    case = build_real_incident_case(seed_site_id, budget=18, rl_seed=0)

    assert case.incident.edges, "expected at least one edge in the incident subgraph"
    checked = 0
    for edge in case.incident.edges:
        pair = frozenset((edge.src, edge.dst))
        expected = distance_km.get(pair)
        if expected is None:
            continue  # edge not in the raw CSV lookup (shouldn't happen, but don't hide it as a false pass)
        assert edge.distance == expected, f"edge {edge.src}-{edge.dst}: distance={edge.distance}, expected distance_km={expected}"
        checked += 1
    assert checked == len(case.incident.edges), "every incident edge should be resolvable against the raw CSV"


def test_route_proxy_is_preserved_as_travel_cost_not_dropped() -> None:
    route_proxy = _route_proxy_km_by_pair()
    seed_site_id = eligible_incident_seed_sites()[0]
    case = build_real_incident_case(seed_site_id, budget=18, rl_seed=0)

    edges_with_known_proxy = [e for e in case.incident.edges if frozenset((e.src, e.dst)) in route_proxy]
    assert edges_with_known_proxy, "expected at least one incident edge with a known route proxy value to check"
    for edge in edges_with_known_proxy:
        assert edge.travel_cost == route_proxy[frozenset((edge.src, edge.dst))]
