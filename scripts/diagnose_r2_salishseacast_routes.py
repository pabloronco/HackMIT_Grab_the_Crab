from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

from adaptive_response.adaptive_water_audit import load_adaptive_candidate_edges
from adaptive_response.water_plausibility import load_real_site_coordinates
from adaptive_response.water_route import (
    annotate_edges_with_water_routes,
    load_salishseacast_grid,
)


REAL_SITES = Path("data/processed/r2_real_sites.csv")
ADAPTIVE_WATER = Path("data/processed/r2_adaptive_water_candidates.csv")
GRID_CACHE = Path("data/cache/salishseacast/ubcSSn2DMeshMaskV21-08.csv")
OUT_CSV = Path("data/processed/r2_salishseacast_water_routes.csv")
OUT_JSON = Path("data/processed/r2_salishseacast_water_routes_audit.json")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not ADAPTIVE_WATER.is_file():
        raise SystemExit(
            f"Missing {ADAPTIVE_WATER}. Run diagnose_r2_adaptive_water.py first."
        )

    print("=== R2 SALISHSEACAST CURVED WATER-ROUTE AUDIT ===")
    print("Purpose: replace straight-line thinking with a water-only shortest-path diagnostic.")
    print("Source: SalishSeaCast NEMO v21-08 T-grid + explicit land/water mask.")
    print("Important: this is geometry only — NOT currents, travel time, or dispersal probability.")
    print()

    if not GRID_CACHE.is_file():
        print("SalishSeaCast grid cache not found; downloading static grid once...")
        print(f"Cache target: {GRID_CACHE}")

    grid = load_salishseacast_grid(GRID_CACHE)
    edges = load_adaptive_candidate_edges(ADAPTIVE_WATER)
    site_coordinates = load_real_site_coordinates(REAL_SITES)
    annotated, snaps = annotate_edges_with_water_routes(
        grid,
        edges,
        site_coordinates,
        max_route_km=250.0,
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in annotated for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)

    snap_distances = [snap.distance_km for snap in snaps.values()]
    routes_found = [row for row in annotated if _as_bool(row["salishseacast_route_found"])]
    routes_missing = [row for row in annotated if not _as_bool(row["salishseacast_route_found"])]
    local = [row for row in annotated if _as_bool(row.get("candidate_local_radius"))]
    zero_straight_local = [
        row
        for row in local
        if math.isclose(float(row.get("straight_marine_fraction", 0.0)), 0.0, abs_tol=1e-12)
    ]
    zero_straight_with_route = [
        row for row in zero_straight_local if _as_bool(row["salishseacast_route_found"])
    ]
    review_only = [
        row
        for row in annotated
        if _as_bool(row.get("candidate_sparse_review"))
        and not _as_bool(row.get("candidate_local_radius"))
    ]

    route_lengths = [float(row["salishseacast_water_route_km"]) for row in routes_found]
    detours = [
        float(row["salishseacast_detour_ratio"])
        for row in routes_found
        if row.get("salishseacast_detour_ratio") not in (None, "")
    ]

    summary = {
        "grid_shape": list(grid.shape),
        "grid_wet_cells": int(grid.wet.sum()),
        "sites_snapped": len(snaps),
        "snap_distance_km_min": min(snap_distances),
        "snap_distance_km_median": median(snap_distances),
        "snap_distance_km_max": max(snap_distances),
        "sites_snap_gt_0_5km": sum(value > 0.5 for value in snap_distances),
        "sites_snap_gt_1km": sum(value > 1.0 for value in snap_distances),
        "candidate_edges": len(annotated),
        "routes_found": len(routes_found),
        "routes_missing": len(routes_missing),
        "route_km_median": median(route_lengths) if route_lengths else None,
        "detour_ratio_median": median(detours) if detours else None,
        "local_edges": len(local),
        "zero_straight_water_local_edges": len(zero_straight_local),
        "zero_straight_water_local_edges_with_curved_route": len(zero_straight_with_route),
        "review_only_edges": len(review_only),
        "notes": [
            "SalishSeaCast wet-cell shortest paths allow curved water routes around land barriers.",
            "The route is a static geometry proxy; it does not model currents, directionality, travel time, or green-crab dispersal probability.",
            "Snap distance must be inspected because some pocket-estuary monitoring sites may be finer-scale than the model grid.",
            "Straight-line zero-water edges must not be declared ecologically impossible if a plausible curved wet-grid route exists.",
            "connectivity_weight remains OPEN after this audit.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Site-to-grid snap distance (km): "
        f"min={summary['snap_distance_km_min']:.2f} | "
        f"median={summary['snap_distance_km_median']:.2f} | "
        f"max={summary['snap_distance_km_max']:.2f} | "
        f">0.5km={summary['sites_snap_gt_0_5km']} | >1km={summary['sites_snap_gt_1km']}"
    )
    print(
        f"candidate edges={len(annotated)} | curved water route found={len(routes_found)} | "
        f"no route={len(routes_missing)}"
    )
    if route_lengths:
        print(
            f"water-route distance median={median(route_lengths):.2f}km | "
            f"detour-ratio median={median(detours):.2f}"
        )
    print()
    print(
        "Critical check on the old straight-line filter: "
        f"zero-water local edges={len(zero_straight_local)} | "
        f"with curved SalishSeaCast route={len(zero_straight_with_route)}"
    )
    for row in sorted(
        zero_straight_with_route,
        key=lambda item: float(item["salishseacast_detour_ratio"] or math.inf),
    ):
        print(
            f"  {row['src']} -- {row['dst']}: direct={float(row['distance_km']):.2f}km | "
            f"route={float(row['salishseacast_water_route_km']):.2f}km | "
            f"detour={float(row['salishseacast_detour_ratio']):.2f} | "
            f"old straight-water={float(row['straight_marine_fraction']):.2f}"
        )

    if routes_missing:
        print()
        print("Candidates with no wet-grid route found (diagnostic, not automatic rejection):")
        for row in sorted(routes_missing, key=lambda item: float(item["distance_km"]))[:15]:
            print(
                f"  {row['src']} -- {row['dst']}: d={float(row['distance_km']):.2f}km | "
                f"local={_as_bool(row.get('candidate_local_radius'))} | "
                f"src_snap={float(row['salishseacast_src_snap_km']):.2f}km | "
                f"dst_snap={float(row['salishseacast_dst_snap_km']):.2f}km"
            )

    print()
    print("DECISION GATE:")
    print("  Re-open the old 'straight-water == 0 => withhold' rule if curved routes rescue edges.")
    print("  Do NOT turn water-route distance into connectivity probability.")
    print("  Next choose graph v0 using water-route evidence + snap-quality diagnostics.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
