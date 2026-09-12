from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.real_graph import diagnose_real_graph_geometry, load_real_sites


REAL_SITES = Path("data/processed/r2_real_sites.csv")
OUT_JSON = Path("data/processed/r2_graph_geometry_audit.json")
OUT_EDGES = Path("data/processed/r2_graph_knn3_candidate_edges.csv")


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.2f}"


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}; build R2 real sites first.")

    sites = load_real_sites(REAL_SITES)
    summary, candidate_edges = diagnose_real_graph_geometry(sites)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_EDGES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src", "dst", "distance_km"])
        writer.writeheader()
        writer.writerows(candidate_edges)

    print("=== R2 REAL GRAPH GEOMETRY DIAGNOSTIC ===")
    print(f"sites={summary['sites']} | pairwise pairs={summary['pairwise_pairs']}")
    print(
        "Nearest-neighbor distance (km): "
        f"min={summary['nearest_neighbor_distance_min_km']:.2f} | "
        f"median={summary['nearest_neighbor_distance_median_km']:.2f} | "
        f"max={summary['nearest_neighbor_distance_max_km']:.2f}"
    )
    print()
    print("Union-kNN candidate graphs:")
    for k, stats in summary["knn_union"].items():
        print(
            f"  k={k}: edges={stats['edges']} | components={stats['components']} | "
            f"largest={stats['largest_component_sites']}/{summary['sites']} | "
            f"degree min/med/max={_fmt(stats['degree_min'])}/"
            f"{_fmt(stats['degree_median'])}/{_fmt(stats['degree_max'])} | "
            f"edge dist med/max={_fmt(stats['edge_distance_median_km'])}/"
            f"{_fmt(stats['edge_distance_max_km'])} km"
        )
    print()
    print("Radius candidate graphs:")
    for radius, stats in summary["radius_graphs_km"].items():
        print(
            f"  r<={radius} km: edges={stats['edges']} | components={stats['components']} | "
            f"largest={stats['largest_component_sites']}/{summary['sites']} | "
            f"isolated={len(stats['isolated_sites'])}"
        )
    print()
    candidate = summary["diagnostic_candidate"]
    stats = candidate["summary"]
    print(
        f"Diagnostic candidate {candidate['construction']}: edges={stats['edges']} | "
        f"components={stats['components']} | largest={stats['largest_component_sites']}/{summary['sites']}"
    )
    print("Longest candidate edges:")
    for edge in candidate["longest_edges"]:
        print(f"  {edge['src']} -- {edge['dst']}: {float(edge['distance_km']):.2f} km")

    print()
    print("GATE:")
    print("  Geometry is real; adjacency is still a DESIGN CHOICE.")
    print("  Great-circle distance is not the same as coastal-water-path connectivity.")
    print("  Do not freeze connectivity_weight from this diagnostic alone.")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_EDGES} (candidate only, not frozen)")


if __name__ == "__main__":
    main()
