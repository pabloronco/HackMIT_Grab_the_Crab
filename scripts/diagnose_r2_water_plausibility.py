from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.water_plausibility import (
    annotate_open_saltwater_support,
    load_candidate_edges,
    load_real_site_coordinates,
    summarize_water_support,
)


REAL_SITES = Path("data/processed/r2_real_sites.csv")
CANDIDATES = Path("data/processed/r2_edge_plausibility_candidates.csv")
OUT_CSV = Path("data/processed/r2_edge_water_plausibility.csv")
OUT_JSON = Path("data/processed/r2_edge_water_plausibility_audit.json")


def main() -> None:
    for path in (REAL_SITES, CANDIDATES):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; complete the preceding R2 graph diagnostics first.")

    print("=== R2 OPEN-SALTWATER EDGE SANITY CHECK ===")
    print("Source: Washington DNR HYDRO Water Bodies (Open Saltwater, code 116)")
    print("Method: sample 5 interior points along each straight candidate edge.")
    print("Important: this is NOT a water-route or ecological-connectivity model.")
    print()

    coordinates = load_real_site_coordinates(REAL_SITES)
    candidates = load_candidate_edges(CANDIDATES)
    edges = annotate_open_saltwater_support(
        candidates,
        coordinates,
        samples_per_edge=5,
        max_workers=8,
    )
    summary = summarize_water_support(edges)
    summary.update(
        {
            "source": "Washington DNR HYDRO Water Bodies (FP)",
            "open_saltwater_code": 116,
            "samples_per_edge": 5,
            "notes": [
                "Open-saltwater sampling is a coarse major-land-barrier sanity check only.",
                "A low fraction can reflect land, narrow estuarine geometry, or source-map resolution.",
                "A fraction of 1.0 does not prove dispersal connectivity or hydrodynamic linkage.",
                "Do not threshold this diagnostic into final ecological edges without combining other evidence.",
            ],
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in edges for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"candidate edges={summary['edges']} | all-water samples="
        f"{summary['all_samples_open_saltwater_edges']} | mixed="
        f"{summary['mixed_open_saltwater_edges']} | none="
        f"{summary['no_samples_open_saltwater_edges']}"
    )
    print(
        "Straight-line open-saltwater fraction: "
        f"min={summary['straight_marine_fraction_min']:.2f} | "
        f"median={summary['straight_marine_fraction_median']:.2f} | "
        f"max={summary['straight_marine_fraction_max']:.2f}"
    )

    print()
    print("Lowest straight-water-support candidates (diagnostic only):")
    suspicious = sorted(
        edges,
        key=lambda row: (
            float(row["straight_marine_fraction"]),
            -float(row["max_local_distance_ratio"]),
            -float(row["distance_km"]),
        ),
    )[:15]
    for row in suspicious:
        print(
            f"  {row['src']} -- {row['dst']}: "
            f"water={float(row['straight_marine_fraction']):.2f} "
            f"pattern={row['straight_marine_pattern']} | "
            f"d={float(row['distance_km']):.2f}km | "
            f"mutual={row.get('mutual_knn')} | "
            f"local-ratio={float(row['max_local_distance_ratio']):.2f} | "
            f"same-area={row.get('same_shorezone_area')}"
        )

    print()
    print("GATE:")
    print("  This check can flag likely major straight-line land barriers.")
    print("  It cannot prove a true coastal route, current pathway, or dispersal edge.")
    print("  Do not auto-accept/reject edges from this field alone.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
