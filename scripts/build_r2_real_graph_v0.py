from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.real_graph import load_real_sites
from adaptive_response.real_graph_v0 import build_real_graph_v0_audit
from adaptive_response.route_graph_sensitivity import load_water_route_edges


REAL_SITES = Path("data/processed/r2_real_sites.csv")
ROUTES = Path("data/processed/r2_salishseacast_water_routes.csv")
OUT_EDGES = Path("data/processed/r2_real_graph_v0_edges.csv")
OUT_AUDIT = Path("data/processed/r2_real_graph_v0_audit.json")


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not ROUTES.is_file():
        raise SystemExit(
            f"Missing {ROUTES}. Run diagnose_r2_salishseacast_routes.py first."
        )

    sites = load_real_sites(REAL_SITES)
    rows = load_water_route_edges(ROUTES)
    edges, summary = build_real_graph_v0_audit(sites, rows)

    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in edges for key in row})
    with OUT_EDGES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

    OUT_AUDIT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    graph = summary["primary_graph"]
    route = summary["route_diagnostics"]

    print("=== R2 REVISED REAL GRAPH V0 BUILD ===")
    print("STATUS: CURRENT DEFAULT PRIMARY ADJACENCY, not ecological ground truth.")
    print("Rule: <=20 km local candidates + audited curved water-route existence; no hard route-distance threshold.")
    print("Sparse-review-only edges never force connectivity.")
    print()
    print(
        f"sites={summary['sites']} | primary edges={summary['primary_edges']} | "
        f"review-only edges excluded={summary['sampling_isolation_review_only_edges']}"
    )
    print(
        f"Primary topology: components={graph['components']} | "
        f"largest={graph['largest_component_sites']}/{summary['sites']} | "
        f"isolated={len(graph['isolated_sites'])} {graph['isolated_sites']} | "
        f"degree min/med/max={graph['degree_min']}/{graph['degree_median']}/{graph['degree_max']}"
    )
    print(
        "Route context: "
        f"total-route median={route['total_route_proxy_km_median']:.2f}km | "
        f"total-detour median={route['total_detour_ratio_median']:.2f} | "
        f"edges touching snap>1km={route['edges_with_endpoint_snap_gt_1km']} | "
        f">2km={route['edges_with_endpoint_snap_gt_2km']}"
    )
    print(
        "Old straight-line rule check: "
        f"straight-zero-water edges retained={summary['straight_zero_water_edges_retained']}"
    )
    print()
    print("DECISION:")
    print("  REJECTED: straight-water == 0 => withhold edge.")
    print("  CURRENT DEFAULT: retain all routed <=20 km local candidates in primary adjacency.")
    print("  Route length/detour stay metadata + topology-sensitivity inputs, not ecological thresholds.")
    print("  connectivity_weight remains OPEN.")
    print("  Next: incident-subgraph extraction, then observation-model/q work.")
    print(f"Wrote {OUT_EDGES}")
    print(f"Wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()
