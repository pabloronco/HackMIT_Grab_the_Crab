from __future__ import annotations

import json
from pathlib import Path

from adaptive_response.graph_filter_sensitivity import (
    compare_water_filter_topologies,
    load_adaptive_water_candidates,
)
from adaptive_response.real_graph import load_real_sites


REAL_SITES = Path("data/processed/r2_real_sites.csv")
ADAPTIVE_WATER = Path("data/processed/r2_adaptive_water_candidates.csv")
OUT_JSON = Path("data/processed/r2_graph_filter_sensitivity.json")


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not ADAPTIVE_WATER.is_file():
        raise SystemExit(
            f"Missing {ADAPTIVE_WATER}. Run diagnose_r2_adaptive_water.py first."
        )

    sites = load_real_sites(REAL_SITES)
    rows = load_adaptive_water_candidates(ADAPTIVE_WATER)
    summary = compare_water_filter_topologies(sites, rows)

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== R2 GRAPH FILTER SENSITIVITY ===")
    print("Primary question: how much does topology change if we treat straight-water support more conservatively?")
    print("Sparse-review-only edges are excluded from ALL primary variants.")
    print()

    for name, stats in summary["topology_variants"].items():
        print(
            f"{name}: edges={stats['edges']} | components={stats['components']} | "
            f"largest={stats['largest_component_sites']}/{len(sites)} | "
            f"isolated={len(stats['isolated_sites'])} | "
            f"degree min/med/max={stats['degree_min']}/{stats['degree_median']}/{stats['degree_max']}"
        )

    review = summary["sampling_isolation_review_only"]
    print()
    print(
        "Sparse-review-only diagnostics: "
        f"edges={review['edges']} | water>0={review['water_gt_0']} | "
        f"water>=0.4={review['water_ge_0_4']} | water>=0.6={review['water_ge_0_6']} | "
        f"all-water={review['all_sampled_water']} | >35km={review['over_35km']}"
    )
    print()
    print("GATE:")
    print("  Do not pick a water threshold because it makes the graph connected.")
    print("  Water thresholds here are sensitivity probes, not ecological constants.")
    print("  Review-only edges never repair connectivity automatically.")
    print("  Next decision: choose a transparent primary graph v0 plus alternative topology variants for robustness tests.")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
