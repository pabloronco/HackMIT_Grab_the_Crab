from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.adaptive_candidates import (
    build_adaptive_candidate_pool,
    compare_adaptive_candidate_radii,
)
from adaptive_response.real_graph import load_real_sites


REAL_SITES = Path("data/processed/r2_real_sites.csv")
OUT_CSV = Path("data/processed/r2_adaptive_candidate_edges.csv")
OUT_JSON = Path("data/processed/r2_adaptive_candidate_audit.json")
REFERENCE_RADIUS_KM = 20.0


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}; build the R2 real-site table first.")

    sites = load_real_sites(REAL_SITES)
    comparison = compare_adaptive_candidate_radii(sites)
    edges, reference = build_adaptive_candidate_pool(
        sites,
        local_radius_km=REFERENCE_RADIUS_KM,
        sparse_degree_threshold=2,
        sparse_review_k=2,
    )

    payload = {
        "reference_candidate_radius_km": REFERENCE_RADIUS_KM,
        "reference_not_frozen": True,
        "radius_sensitivity": comparison,
        "reference_candidate_pool": reference,
    }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in edges for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== R2 ADAPTIVE VARIABLE-DEGREE CANDIDATE DIAGNOSTIC ===")
    print("No fixed k: local-radius candidates + sparse-monitoring review candidates.")
    print("Important: review candidates are NOT accepted ecological edges.")
    print()
    print("Radius sensitivity (sparse degree <2 gets up to 2 review neighbours):")
    for radius in (10.0, 15.0, 20.0, 25.0, 30.0):
        summary = comparison[f"{radius:g}"]
        print(
            f"  r<={radius:.0f}km: local_edges={summary['local_radius_edges']} | "
            f"sparse_sites={summary['sampling_sparse_sites']} | "
            f"review_only_edges={summary['sampling_isolation_review_only_edges']} | "
            f"candidate_total={summary['candidate_edges_total']} | "
            f"local_components={summary['local_radius_graph']['components']}"
        )

    print()
    print(f"Reference diagnostic candidate pool: r<={REFERENCE_RADIUS_KM:.0f}km (NOT frozen)")
    print(
        f"sites={reference['sites']} | local edges={reference['local_radius_edges']} | "
        f"review-only edges={reference['sampling_isolation_review_only_edges']} | "
        f"total candidates={reference['candidate_edges_total']}"
    )
    print(
        "Local degree min/median/max: "
        f"{reference['local_degree_min']}/"
        f"{reference['local_degree_median']}/"
        f"{reference['local_degree_max']}"
    )
    print(
        f"Sampling-sparse sites (<{reference['sparse_degree_threshold']} local neighbours): "
        f"{reference['sampling_sparse_site_ids']}"
    )

    review_only = [
        row
        for row in edges
        if bool(row["candidate_sparse_review"])
        and not bool(row["candidate_local_radius"])
    ]
    print()
    print("Sampling-isolation review-only candidates:")
    if not review_only:
        print("  none")
    else:
        for row in sorted(review_only, key=lambda value: float(value["distance_km"])):
            print(
                f"  {row['src']} -- {row['dst']}: "
                f"d={float(row['distance_km']):.2f}km | "
                f"ranks={row['src_rank']}/{row['dst_rank']} | "
                f"requested_by={row['sparse_review_requested_by']} | "
                f"local_degree={row['src_local_degree']}/{row['dst_local_degree']}"
            )

    print()
    print("GATE:")
    print("  Do not choose the radius only because it makes the graph connected.")
    print("  Sparse review candidates compensate for monitoring sparsity only at the review stage.")
    print("  No final edge gets a guaranteed minimum degree.")
    print("  Next step: run water/coastal plausibility on this adaptive pool before freezing graph v0.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
