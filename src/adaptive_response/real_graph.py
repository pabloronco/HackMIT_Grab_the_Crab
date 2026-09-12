from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 latitude/longitude points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def load_real_sites(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "latitude", "longitude"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Real-site table missing fields: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in reader:
            site_id = str(row.get("site_id") or "").strip()
            if not site_id:
                raise ValueError("Real-site table contains blank site_id")
            if site_id in seen:
                raise ValueError(f"Duplicate site_id={site_id}")
            seen.add(site_id)
            rows.append(
                {
                    "site_id": site_id,
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
    if len(rows) < 2:
        raise ValueError("At least two real sites are required for graph diagnostics")
    return rows


def _pairwise_distances(sites: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    for i, a in enumerate(sites):
        for b in sites[i + 1 :]:
            key = tuple(sorted((str(a["site_id"]), str(b["site_id"]))))
            distances[key] = haversine_km(
                float(a["latitude"]),
                float(a["longitude"]),
                float(b["latitude"]),
                float(b["longitude"]),
            )
    return distances


def build_knn_edges(
    sites: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Build an undirected union-kNN geographic candidate graph.

    An edge is present when either endpoint ranks the other among its k nearest
    sites. This is a geometry diagnostic, not an ecological dispersal claim.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= len(sites):
        raise ValueError("k must be smaller than the number of sites")

    distances = _pairwise_distances(sites)
    ids = [str(site["site_id"]) for site in sites]
    by_site: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (a, b), distance in distances.items():
        by_site[a].append((distance, b))
        by_site[b].append((distance, a))

    selected: set[tuple[str, str]] = set()
    for site_id in ids:
        nearest = sorted(by_site[site_id], key=lambda item: (item[0], item[1]))[:k]
        for _, neighbor in nearest:
            selected.add(tuple(sorted((site_id, neighbor))))

    return [
        {"src": a, "dst": b, "distance_km": distances[(a, b)]}
        for a, b in sorted(selected)
    ]


def build_radius_edges(
    sites: list[dict[str, Any]],
    radius_km: float,
) -> list[dict[str, Any]]:
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    distances = _pairwise_distances(sites)
    return [
        {"src": a, "dst": b, "distance_km": distance}
        for (a, b), distance in sorted(distances.items())
        if distance <= radius_km
    ]


def connected_components(
    site_ids: list[str],
    edges: list[dict[str, Any]],
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {site_id: set() for site_id in site_ids}
    for edge in edges:
        src = str(edge["src"])
        dst = str(edge["dst"])
        if src not in adjacency or dst not in adjacency:
            raise ValueError("Edge references unknown site")
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    remaining = set(site_ids)
    components: list[list[str]] = []
    while remaining:
        root = min(remaining)
        queue: deque[str] = deque([root])
        remaining.remove(root)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values))


def graph_summary(
    sites: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = [str(site["site_id"]) for site in sites]
    degree = {site_id: 0 for site_id in ids}
    for edge in edges:
        degree[str(edge["src"])] += 1
        degree[str(edge["dst"])] += 1
    components = connected_components(ids, edges)
    distances = [float(edge["distance_km"]) for edge in edges]
    degree_values = list(degree.values())
    return {
        "edges": len(edges),
        "components": len(components),
        "largest_component_sites": len(components[0]) if components else 0,
        "isolated_sites": sorted(site_id for site_id, value in degree.items() if value == 0),
        "degree_min": min(degree_values) if degree_values else None,
        "degree_median": median(degree_values) if degree_values else None,
        "degree_max": max(degree_values) if degree_values else None,
        "edge_distance_min_km": min(distances) if distances else None,
        "edge_distance_median_km": median(distances) if distances else None,
        "edge_distance_max_km": max(distances) if distances else None,
        "component_sizes": [len(component) for component in components],
    }


def diagnose_real_graph_geometry(
    sites: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5),
    radius_values_km: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 50.0),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    distances = _pairwise_distances(sites)
    by_site: dict[str, list[float]] = defaultdict(list)
    for (a, b), distance in distances.items():
        by_site[a].append(distance)
        by_site[b].append(distance)
    nearest = [min(values) for values in by_site.values()]

    knn: dict[str, Any] = {}
    for k in k_values:
        if k >= len(sites):
            continue
        edges = build_knn_edges(sites, k)
        knn[str(k)] = graph_summary(sites, edges)

    radius: dict[str, Any] = {}
    for value in radius_values_km:
        edges = build_radius_edges(sites, value)
        radius[f"{value:g}"] = graph_summary(sites, edges)

    candidate_k = 3 if len(sites) > 3 else 1
    candidate_edges = build_knn_edges(sites, candidate_k)
    longest_candidate_edges = sorted(
        candidate_edges,
        key=lambda edge: float(edge["distance_km"]),
        reverse=True,
    )[:10]

    summary = {
        "sites": len(sites),
        "pairwise_pairs": len(distances),
        "nearest_neighbor_distance_min_km": min(nearest),
        "nearest_neighbor_distance_median_km": median(nearest),
        "nearest_neighbor_distance_max_km": max(nearest),
        "latitude_min": min(float(site["latitude"]) for site in sites),
        "latitude_max": max(float(site["latitude"]) for site in sites),
        "longitude_min": min(float(site["longitude"]) for site in sites),
        "longitude_max": max(float(site["longitude"]) for site in sites),
        "knn_union": knn,
        "radius_graphs_km": radius,
        "diagnostic_candidate": {
            "construction": f"union_knn_k{candidate_k}",
            "not_frozen": True,
            "summary": graph_summary(sites, candidate_edges),
            "longest_edges": longest_candidate_edges,
        },
        "notes": [
            "Distances are great-circle geometry, not coastal-water-path distances.",
            "kNN/radius graphs are topology diagnostics only; they are not yet ecological connectivity claims.",
            "Connectivity weight remains OPEN and must be separated from geometric distance.",
        ],
    }
    return summary, candidate_edges
