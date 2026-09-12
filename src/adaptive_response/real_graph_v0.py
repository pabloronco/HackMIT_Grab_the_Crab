from __future__ import annotations

from statistics import median
from typing import Any

from .real_graph import graph_summary


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_real_graph_v0_edges(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the current transparent v0 adjacency from audited route candidates.

    CURRENT DEFAULT, not ecological truth:
    - primary adjacency contains every <=20 km local-radius candidate for which a
      curved SalishSeaCast wet-grid route was found;
    - sparse-monitoring review-only candidates never force connectivity;
    - straight-line open-water sampling is retained only as diagnostic metadata
      and never withholds an otherwise local routed edge;
    - route distance, detour ratio and snap distance remain diagnostics rather
      than hard ecological thresholds.

    Returns ``(primary, review_only)``.
    """
    primary: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []

    for row in rows:
        candidate = dict(row)
        is_local = _as_bool(candidate.get("candidate_local_radius"))
        is_review = _as_bool(candidate.get("candidate_sparse_review"))
        route_found = _as_bool(candidate.get("salishseacast_route_found"))

        if not is_local:
            if is_review:
                candidate["v0_edge_status"] = "sampling_isolation_review_only"
                candidate["v0_primary"] = False
                review_only.append(candidate)
            continue

        if not route_found:
            # Keep this explicit even though the current audited pool found routes
            # for every local edge. A missing route would remain unresolved rather
            # than being silently promoted into primary adjacency.
            continue

        candidate["v0_edge_status"] = "primary_local_curved_water_route_audited"
        candidate["v0_primary"] = True
        primary.append(candidate)

    return primary, review_only


def build_real_graph_v0_audit(
    sites: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary, review_only = select_real_graph_v0_edges(rows)
    primary_summary = graph_summary(sites, primary)

    straight_zero_primary = [
        row
        for row in primary
        if float(row.get("straight_marine_fraction", 0.0)) <= 0.0
    ]
    total_routes = [
        float(row["salishseacast_total_route_proxy_km"])
        for row in primary
        if row.get("salishseacast_total_route_proxy_km") not in (None, "")
    ]
    detours = [
        float(row["salishseacast_total_detour_ratio"])
        for row in primary
        if row.get("salishseacast_total_detour_ratio") not in (None, "")
    ]
    max_snaps = [
        max(
            float(row["salishseacast_src_snap_km"]),
            float(row["salishseacast_dst_snap_km"]),
        )
        for row in primary
    ]

    summary = {
        "status": "CURRENT_DEFAULT_PRIMARY_ADJACENCY_NOT_ECOLOGICAL_TRUTH",
        "sites": len(sites),
        "primary_edges": len(primary),
        "sampling_isolation_review_only_edges": len(review_only),
        "primary_graph": primary_summary,
        "straight_zero_water_edges_retained": len(straight_zero_primary),
        "route_diagnostics": {
            "total_route_proxy_km_median": median(total_routes) if total_routes else None,
            "total_detour_ratio_median": median(detours) if detours else None,
            "edges_with_endpoint_snap_gt_1km": sum(value > 1.0 for value in max_snaps),
            "edges_with_endpoint_snap_gt_2km": sum(value > 2.0 for value in max_snaps),
        },
        "rejected_rule": (
            "straight_marine_fraction == 0 => withhold edge; rejected because all audited "
            "zero-straight-water local edges had curved wet-grid routes"
        ),
        "notes": [
            "The <=20 km direct-radius rule is a candidate-generation design default, not an ecological dispersal threshold.",
            "Straight-line open-water sampling is diagnostic only and no longer controls primary adjacency.",
            "Curved SalishSeaCast route existence is used as a geometry sanity check; route length and detour ratio are not hard ecological thresholds.",
            "Large site-to-grid snap distances lower confidence in route geometry for some pocket-estuary sites, so route metrics remain uncertainty/context metadata.",
            "Sparse-monitoring review-only edges never repair connectivity automatically.",
            "The primary graph may be disconnected and may contain isolated sites.",
            "Connectivity weight remains OPEN; this v0 freezes a primary adjacency default only, not dispersal strength.",
            "Retain route-threshold and alternative-radius topologies for robustness/OOD sensitivity tests.",
        ],
    }
    return primary, summary
