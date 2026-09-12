from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.edge_plausibility import build_edge_plausibility_audit


REAL_SITES = Path("data/processed/r2_real_sites.csv")
OUT_CSV = Path("data/processed/r2_edge_plausibility_candidates.csv")
OUT_JSON = Path("data/processed/r2_edge_plausibility_audit.json")


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(
            f"Missing {REAL_SITES}. Build R2 real sites before edge plausibility audit."
        )

    print("=== R2 ADAPTIVE EDGE PLAUSIBILITY DIAGNOSTIC ===")
    print("Candidate pool: union of each site's 5 nearest geographic neighbours.")
    print("Important: candidates are NOT accepted ecological edges yet.")
    print()

    edges, summary = build_edge_plausibility_audit(REAL_SITES, k_max=5)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in edges for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"sites={summary['sites']} | candidate edges={summary['edges']} | "
        f"mutual-kNN={summary['mutual_knn_edges']}"
    )
    print(
        "Candidate edge distance (km): "
        f"min={summary['distance_km_min']:.2f} | "
        f"median={summary['distance_km_median']:.2f} | "
        f"max={summary['distance_km_max']:.2f}"
    )
    print(
        "Max local-distance ratio: "
        f"median={summary['local_distance_ratio_median']:.2f} | "
        f"max={summary['local_distance_ratio_max']:.2f}"
    )
    print(
        "ShoreZone mapping-context overlap among candidate edges: "
        f"same region={summary['same_shorezone_region_edges']}/{summary['edges']} | "
        f"same area={summary['same_shorezone_area_edges']}/{summary['edges']} | "
        f"same SHORENAME={summary['same_shorename_edges']}/{summary['edges']}"
    )
    print(
        "ShoreZone fields available from full SZLine layer: "
        f"{summary['shorezone_metadata_fields']}"
    )

    print()
    print("Most suspicious-by-geometry candidates (diagnostic only):")
    suspicious = sorted(
        edges,
        key=lambda row: (
            float(row["max_local_distance_ratio"]),
            float(row["distance_km"]),
        ),
        reverse=True,
    )[:12]
    for row in suspicious:
        print(
            f"  {row['src']} -- {row['dst']}: "
            f"d={float(row['distance_km']):.2f}km | "
            f"ranks={row['src_rank']}/{row['dst_rank']} | "
            f"mutual={row['mutual_knn']} | "
            f"local-ratio={float(row['max_local_distance_ratio']):.2f} | "
            f"same-area={row.get('same_shorezone_area')} | "
            f"shore={row.get('src_shorename')} -> {row.get('dst_shorename')}"
        )

    print()
    print("GATE:")
    print("  Do NOT threshold these diagnostics into final edges yet.")
    print("  REGION/AREA are ShoreZone mapping units, not ecological basins.")
    print("  Next decision: combine these clues with explicit water/coastal plausibility.")
    print("  Variable node degree is allowed; the graph need not be globally connected.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
