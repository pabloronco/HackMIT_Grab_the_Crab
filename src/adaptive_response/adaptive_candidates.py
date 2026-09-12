from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from .real_graph import graph_summary, haversine_km


def _pairwise_geometry(
    sites: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, list[tuple[float, str]]],
    dict[str, dict[str, int]],
]:
    ids = [str(site["site_id"]) for site in sites]
    if len(ids) != len(set(ids)):
        raise ValueError("site ids must be unique")
    if len(ids) < 2:
        raise ValueError("at least two sites are required")

    by_id = {str(site["site_id"]): site for site in sites}
    distances: dict[tuple[str, str], float] = {}
    neighbours: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for i, src in enumerate(ids):
        a = by_id[src]
        for dst in ids[i + 1 :]:
            b = by_id[dst]
            distance = haversine_km(
                float(a["latitude"]),
                float(a["longitude"]),
                float(b["latitude"]),
                float(b["longitude"]),
            )
            key = tuple(sorted((src, dst)))
            distances[key] = distance
            neighbours[src].append((distance, dst))
            neighbours[dst].append((distance, src))

    ranks: dict[str, dict[str, int]] = {}
    for site_id in ids:
        ordered = sorted(neighbours[site_id], key=lambda item: (item[0], item[1]))
        neighbours[site_id] = ordered
        ranks[site_id] = {
            neighbour: index + 1 for index, (_, neighbour) in enumerate(ordered)
        }
    return distances, neighbours, ranks


def build_adaptive_candidate_pool(
    sites: list[dict[str, Any]],
    *,
    local_radius_km: float = 20.0,
    sparse_degree_threshold: int = 2,
    sparse_review_k: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a variable-degree candidate pool without forcing uniform k.

    Tier 1 contains every site pair within ``local_radius_km``. Tier 2 adds a
    small number of nearest-neighbour *review candidates* only for sites whose
    Tier-1 degree is below ``sparse_degree_threshold``. Tier-2 edges are not
    accepted ecological connections; they exist so sparse monitoring coverage
    does not silently erase potentially relevant neighbours from review.

    This separates sampling isolation from ecological isolation. No edge is
    accepted or rejected by this function.
    """
    if local_radius_km <= 0:
        raise ValueError("local_radius_km must be positive")
    if sparse_degree_threshold < 0:
        raise ValueError("sparse_degree_threshold cannot be negative")
    if sparse_review_k < 0:
        raise ValueError("sparse_review_k cannot be negative")
    if sparse_review_k >= len(sites):
        raise ValueError("sparse_review_k must be smaller than number of sites")

    distances, neighbours, ranks = _pairwise_geometry(sites)
    ids = [str(site["site_id"]) for site in sites]

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    local_degree = {site_id: 0 for site_id in ids}

    # Tier 1: symmetric radius-based candidate generation. This does not force
    # every site to have the same number of neighbours.
    for (src, dst), distance in sorted(distances.items()):
        if distance <= local_radius_km:
            selected[(src, dst)] = {
                "candidate_local_radius": True,
                "candidate_sparse_review": False,
                "sparse_review_requested_by": set(),
            }
            local_degree[src] += 1
            local_degree[dst] += 1

    sparse_sites = {
        site_id for site_id, degree in local_degree.items() if degree < sparse_degree_threshold
    }

    # Tier 2: expose, but do not accept, a few nearest alternatives for sites
    # poorly represented by the monitoring network inside the local radius.
    if sparse_review_k:
        for site_id in sorted(sparse_sites):
            for _, neighbour in neighbours[site_id][:sparse_review_k]:
                key = tuple(sorted((site_id, neighbour)))
                record = selected.setdefault(
                    key,
                    {
                        "candidate_local_radius": False,
                        "candidate_sparse_review": True,
                        "sparse_review_requested_by": set(),
                    },
                )
                record["candidate_sparse_review"] = True
                record["sparse_review_requested_by"].add(site_id)

    rows: list[dict[str, Any]] = []
    for src, dst in sorted(selected):
        metadata = selected[(src, dst)]
        distance = distances[(src, dst)]
        requested_by = sorted(metadata["sparse_review_requested_by"])
        tier = (
            "local_radius+sampling_isolation_review"
            if metadata["candidate_local_radius"] and metadata["candidate_sparse_review"]
            else "local_radius"
            if metadata["candidate_local_radius"]
            else "sampling_isolation_review"
        )
        rows.append(
            {
                "src": src,
                "dst": dst,
                "distance_km": distance,
                "src_rank": ranks[src][dst],
                "dst_rank": ranks[dst][src],
                "candidate_tier": tier,
                "candidate_local_radius": bool(metadata["candidate_local_radius"]),
                "candidate_sparse_review": bool(metadata["candidate_sparse_review"]),
                "sparse_review_requested_by": "|".join(requested_by),
                "src_local_degree": local_degree[src],
                "dst_local_degree": local_degree[dst],
                "src_sampling_sparse": src in sparse_sites,
                "dst_sampling_sparse": dst in sparse_sites,
            }
        )

    local_rows = [row for row in rows if bool(row["candidate_local_radius"])]
    review_only_rows = [
        row
        for row in rows
        if bool(row["candidate_sparse_review"])
        and not bool(row["candidate_local_radius"])
    ]
    local_summary = graph_summary(sites, local_rows)
    candidate_summary = graph_summary(sites, rows)
    local_degrees = list(local_degree.values())

    summary = {
        "sites": len(sites),
        "local_radius_km": local_radius_km,
        "sparse_degree_threshold": sparse_degree_threshold,
        "sparse_review_k": sparse_review_k,
        "local_radius_edges": len(local_rows),
        "sampling_isolation_review_only_edges": len(review_only_rows),
        "candidate_edges_total": len(rows),
        "sampling_sparse_sites": len(sparse_sites),
        "sampling_sparse_site_ids": sorted(sparse_sites),
        "local_degree_min": min(local_degrees),
        "local_degree_median": median(local_degrees),
        "local_degree_max": max(local_degrees),
        "local_radius_graph": local_summary,
        "candidate_pool_graph": candidate_summary,
        "notes": [
            "Local radius controls candidate generation, not ecological edge acceptance.",
            "Sparse review candidates protect against confusing sparse monitoring coverage with ecological isolation.",
            "Sparse review edges require independent water/coastal plausibility review before any final graph decision.",
            "No minimum ecological degree is imposed; the final graph may contain low-degree or isolated nodes.",
        ],
    }
    return rows, summary


def compare_adaptive_candidate_radii(
    sites: list[dict[str, Any]],
    *,
    radii_km: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0),
    sparse_degree_threshold: int = 2,
    sparse_review_k: int = 2,
) -> dict[str, dict[str, Any]]:
    """Compare candidate-pool sensitivity before freezing a radius choice."""
    result: dict[str, dict[str, Any]] = {}
    for radius in radii_km:
        _, summary = build_adaptive_candidate_pool(
            sites,
            local_radius_km=radius,
            sparse_degree_threshold=sparse_degree_threshold,
            sparse_review_k=sparse_review_k,
        )
        result[f"{radius:g}"] = summary
    return result
