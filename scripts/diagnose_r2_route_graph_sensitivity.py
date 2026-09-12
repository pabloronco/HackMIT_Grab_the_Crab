from __future__ import annotations

import json
from pathlib import Path

from adaptive_response.real_graph import load_real_sites
from adaptive_response.route_graph_sensitivity import (
    compare_route_distance_topologies,
    load_water_route_edges,
)


REAL_SITES = Path("data/processed/r2_real_sites.csv")
ROUTES = Path("data/processed/r2_salishseacast_water_routes.csv")
OUT_JSON = Path("data/processed/r2_route_graph_sensitivity.json")


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not ROUTES.is_file():
        raise SystemExit(
            f"Missing {ROUTES}. Run diagnose_r2_salishseacast_routes.py first."
        )

    sites = load_real_sites(REAL_SITES)
    rows = load_water_route_edges(ROUTES)
    summary = compare_route_distance_topologies(sites, rows)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== R2 REFINED CURVED WATER-ROUTE GRAPH SENSITIVITY ===")
    print("Route proxy = wet-grid path + both site-to-grid snap distances.")
    print("A* uses safe diagonal water moves to reduce grid staircase inflation.")
    print("Sparse-review-only edges remain excluded from all primary variants.")
    print()

    for name, stats in summary["topology_variants"].items():
        print(
            f"{name}: edges={stats['edges']} | components={stats['components']} | "
            f"largest={stats['largest_component_sites']}/{len(sites)} | "
            f"isolated={len(stats['isolated_sites'])} | "
            f"degree min/med/max={stats['degree_min']}/{stats['degree_median']}/{stats['degree_max']}"
        )

    snap = summary["site_snap"]
    print()
    print(
        "Site-to-SalishSeaCast snap quality: "
        f"median={snap['distance_km_median']:.2f}km | max={snap['distance_km_max']:.2f}km | "
        f">0.5km={snap['sites_gt_0_5km']} | >1km={snap['sites_gt_1km']} | >2km={snap['sites_gt_2km']}"
    )
    print("Worst site snaps:")
    for row in snap["worst_sites"]:
        print(f"  site {row['site_id']}: snap={float(row['snap_km']):.2f}km")

    route = summary["local_route"]
    print()
    print(
        "Local-edge route diagnostics: "
        f"edges={route['edges']} | grid-route median={route['grid_route_km_median']:.2f}km | "
        f"total-route proxy median={route['total_route_proxy_km_median']:.2f}km | "
        f"total detour median={route['total_detour_ratio_median']:.2f} | "
        f"detour>2={route['total_detour_ratio_gt_2']} | "
        f">3={route['total_detour_ratio_gt_3']} | >5={route['total_detour_ratio_gt_5']}"
    )
    print(
        "Edges touching coarse snaps: "
        f">1km endpoint={route['edges_with_endpoint_snap_gt_1km']} | "
        f">2km endpoint={route['edges_with_endpoint_snap_gt_2km']}"
    )
    print("Most extreme local detours:")
    for row in route["most_extreme_detours"]:
        print(
            f"  {row['src']} -- {row['dst']}: direct={row['direct_km']:.2f}km | "
            f"grid={row['grid_route_km']:.2f}km | total={row['total_route_proxy_km']:.2f}km | "
            f"detour={row['total_detour_ratio']:.2f} | "
            f"max-snap={row['max_endpoint_snap_km']:.2f}km"
        )

    print()
    print("DECISION GATE:")
    print("  Do not freeze a route threshold from connectivity convenience alone.")
    print("  Endpoint snap distance is now included in the route proxy, but large snaps still lower confidence.")
    print("  Safe diagonal moves reduce grid artefact; remaining route values are still geometric proxies, not dispersal probabilities.")
    print("  connectivity_weight remains OPEN.")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
