from __future__ import annotations

import csv
import heapq
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _edge_weight_km(row: dict[str, Any]) -> float:
    """Return a static geographic/pathway cost for incident-local ranking.

    Prefer the audited curved-water total-route proxy when present; otherwise
    fall back to direct geographic distance. This weight is used only to rank
    observable/static proximity around the initial detection. It is NOT an
    ecological dispersal probability or planner reward.
    """
    value = row.get("salishseacast_total_route_proxy_km")
    if value not in (None, ""):
        weight = float(value)
    else:
        weight = float(row["distance_km"])
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("incident-subgraph edge weight must be finite and non-negative")
    return weight


def load_graph_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"src", "dst", "distance_km"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Graph edge table missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def _adjacency(
    site_ids: list[str],
    edges: list[dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {site_id: [] for site_id in site_ids}
    for row in edges:
        src = str(row["src"])
        dst = str(row["dst"])
        if src not in adjacency or dst not in adjacency:
            raise ValueError(f"Edge references unknown site: {src}--{dst}")
        weight = _edge_weight_km(row)
        adjacency[src].append((dst, weight))
        adjacency[dst].append((src, weight))
    for site_id in adjacency:
        adjacency[site_id].sort(key=lambda item: (item[1], item[0]))
    return adjacency


def _reachable_component(
    seed_site_id: str,
    adjacency: dict[str, list[tuple[str, float]]],
) -> set[str]:
    seen = {seed_site_id}
    queue: deque[str] = deque([seed_site_id])
    while queue:
        node = queue.popleft()
        for neighbor, _ in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _dijkstra_order(
    seed_site_id: str,
    adjacency: dict[str, list[tuple[str, float]]],
) -> tuple[list[str], dict[str, float]]:
    best: dict[str, float] = {seed_site_id: 0.0}
    settled: list[str] = []
    queue: list[tuple[float, str]] = [(0.0, seed_site_id)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != best.get(node):
            continue
        settled.append(node)
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return settled, best


def extract_incident_subgraph(
    site_ids: list[str],
    edges: list[dict[str, Any]],
    *,
    seed_site_id: str,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Extract a t0-observable incident-local subgraph around the first detection.

    Rules:
    - the seed is the known initial detection site;
    - only the already-frozen/static source graph is visible;
    - future detections, historical outcomes and hidden synthetic truth are never inputs;
    - never cross a disconnected source-graph component merely to hit a target size;
    - when the reachable component exceeds ``max_sites``, keep the closest sites by
      shortest-path cost using static route/distance metadata.

    ``preferred_min_sites`` is diagnostic only. A smaller real component is kept
    smaller rather than repaired with invented connectivity.
    """
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("site_ids must be unique")
    if seed_site_id not in set(site_ids):
        raise ValueError(f"Unknown seed site: {seed_site_id}")
    if max_sites < 1:
        raise ValueError("max_sites must be >= 1")
    if preferred_min_sites < 1:
        raise ValueError("preferred_min_sites must be >= 1")
    if preferred_min_sites > max_sites:
        raise ValueError("preferred_min_sites cannot exceed max_sites")

    adjacency = _adjacency(site_ids, edges)
    component = _reachable_component(seed_site_id, adjacency)
    order, distances = _dijkstra_order(seed_site_id, adjacency)
    reachable_order = [site_id for site_id in order if site_id in component]

    selected = reachable_order[:max_sites]
    selected_set = set(selected)
    sub_edges = [
        dict(row)
        for row in edges
        if str(row["src"]) in selected_set and str(row["dst"]) in selected_set
    ]

    summary = {
        "seed_site_id": seed_site_id,
        "source_component_sites": len(component),
        "selected_sites": len(selected),
        "selected_site_ids": selected,
        "selected_edges": len(sub_edges),
        "preferred_min_sites": preferred_min_sites,
        "max_sites": max_sites,
        "below_preferred_min": len(selected) < preferred_min_sites,
        "capped_by_max_sites": len(component) > max_sites,
        "never_crossed_component": True,
        "max_selected_shortest_path_km": max((distances[site] for site in selected), default=0.0),
        "selection_inputs": [
            "initial_detection_site",
            "frozen_primary_adjacency",
            "static_edge_route_or_distance_metadata",
        ],
        "forbidden_inputs": [
            "future_detections",
            "historical_future_outcomes",
            "hidden_synthetic_truth",
        ],
        "notes": [
            "Preferred 12-24 graph size is an implementation target, not a reason to invent cross-component edges.",
            "If a real connected component is smaller than the preferred minimum, the incident subgraph remains smaller.",
            "Shortest-path ranking uses static geography/pathway metadata only and is not a learned policy or ecological probability.",
        ],
    }
    return selected, sub_edges, summary


def audit_all_incident_seeds(
    site_ids: list[str],
    edges: list[dict[str, Any]],
    *,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
) -> dict[str, Any]:
    """Evaluate incident-subgraph size for every possible initial-detection site."""
    rows: list[dict[str, Any]] = []
    for seed in sorted(site_ids):
        _, _, summary = extract_incident_subgraph(
            site_ids,
            edges,
            seed_site_id=seed,
            max_sites=max_sites,
            preferred_min_sites=preferred_min_sites,
        )
        rows.append(
            {
                "seed_site_id": seed,
                "source_component_sites": summary["source_component_sites"],
                "selected_sites": summary["selected_sites"],
                "below_preferred_min": summary["below_preferred_min"],
                "capped_by_max_sites": summary["capped_by_max_sites"],
            }
        )

    size_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        size_counts[int(row["selected_sites"])] += 1

    return {
        "sites": len(site_ids),
        "max_sites": max_sites,
        "preferred_min_sites": preferred_min_sites,
        "seeds_in_preferred_size_range": sum(
            preferred_min_sites <= int(row["selected_sites"]) <= max_sites for row in rows
        ),
        "seeds_below_preferred_min": sum(bool(row["below_preferred_min"]) for row in rows),
        "seeds_capped_by_max": sum(bool(row["capped_by_max_sites"]) for row in rows),
        "selected_size_histogram": {str(size): count for size, count in sorted(size_counts.items())},
        "seed_rows": rows,
        "notes": [
            "This audit uses topology/static geography only; it does not inspect future biological outcomes.",
            "A demo seed should ultimately come from the chosen incident/replay framing, not from whichever seed later produces the nicest biological result.",
        ],
    }
