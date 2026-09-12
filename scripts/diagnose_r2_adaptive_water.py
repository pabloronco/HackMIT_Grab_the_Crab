from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.adaptive_water_audit import (
    annotate_adaptive_candidates_with_water,
    load_adaptive_candidate_edges,
    summarize_adaptive_water_audit,
)
from adaptive_response.water_plausibility import load_real_site_coordinates


REAL_SITES = Path("data/processed/r2_real_sites.csv")
CANDIDATES = Path("data/processed/r2_adaptive_candidate_edges.csv")
OUT_CSV = Path("data/processed/r2_adaptive_water_candidates.csv")
OUT_JSON = Path("data/processed/r2_adaptive_water_audit.json")


def main() -> None:
    for path in (REAL_SITES, CANDIDATES):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; run the previous R2 diagnostics first.")

    coordinates = load_real_site_coordinates(REAL_SITES)
    edges = load_adaptive_candidate_edges(CANDIDATES)

    print("=== R2 ADAPTIVE-CANDIDATE WATER AUDIT ===")
    print("Reference candidate pool: local radius <=20 km + sparse-monitoring review edges.")
    print("Important: 20 km is a design default for candidate generation, not an ecological threshold.")
    print("Water support remains diagnostic only; no edge is accepted/rejected automatically.")
    print()

    annotated = annotate_adaptive_candidates_with_water(
        edges,
        coordinates,
        samples_per_edge=5,
        max_workers=8,
    )
    summary = summarize_adaptive_water_audit(annotated)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in annotated for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotated)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    local = summary["local_radius"]
    review = summary["sampling_isolation_review_only"]
    print(
        "Local-radius candidates: "
        f"edges={local['edges']} | all-water={local['all_water']} | "
        f"mixed={local['mixed_water']} | none={local['no_water']} | "
        f"median water={local['water_fraction_median']:.2f}"
    )
    print(
        "Sparse-review-only candidates: "
        f"edges={review['edges']} | all-water={review['all_water']} | "
        f"mixed={review['mixed_water']} | none={review['no_water']} | "
        f"median water={review['water_fraction_median']:.2f}"
    )

    print()
    print("Sparse-site review candidates (best water-supported first; diagnostic only):")
    for site_id, candidates in summary["sparse_site_review"].items():
        print(f"  site {site_id}:")
        for item in candidates:
            local_flag = "local+review" if item["candidate_local_radius"] else "review-only"
            print(
                f"    -> {item['neighbor']}: d={item['distance_km']:.2f}km | "
                f"water={item['straight_marine_fraction']:.2f} "
                f"pattern={item['straight_marine_pattern']} | {local_flag}"
            )

    print()
    print("INTERPRETATION GATE:")
    print("  This checks whether sampling-sparse sites have any geometrically plausible marine candidate at all.")
    print("  It does NOT force sparse sites to have an ecological edge.")
    print("  Zero straight-water support is a red flag, not an automatic reject.")
    print("  If review-only edges remain weak, we keep the site sparse/isolated rather than invent connectivity.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
