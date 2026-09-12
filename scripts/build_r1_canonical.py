from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_response.canonical_data import (
    build_canonical_site_visits,
    write_canonical_csv,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R1 canonical real site-month dataset.")
    parser.add_argument(
        "--cama",
        type=Path,
        default=Path("data/raw/wsg_cama/all_abundance_wide_format.csv"),
    )
    parser.add_argument(
        "--effort",
        type=Path,
        default=Path("data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv"),
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/canonical_site_visit.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/processed/canonical_site_visit_summary.json"),
    )
    args = parser.parse_args()

    rows, summary = build_canonical_site_visits(args.cama, args.effort, args.coords)
    write_canonical_csv(rows, args.out)
    write_summary(summary, args.summary_out)

    coverage = summary["effort_coverage_fraction"]
    coverage_text = "n/a" if coverage is None else f"{100.0 * coverage:.1f}%"
    print("=== R1 CANONICAL REAL DATASET ===")
    print(
        f"rows={summary['rows']} | sites={summary['sites']} | "
        f"years={','.join(str(year) for year in summary['years'])}"
    )
    print(
        f"detections={summary['detections']} | zero_detections={summary['zero_detections']}"
    )
    print(
        f"effort present={summary['effort_present_rows']} | "
        f"missing={summary['effort_missing_rows']} | coverage={coverage_text}"
    )
    if summary["missing_effort_keys"]:
        print(f"missing effort keys={summary['missing_effort_keys']}")
    print(f"coordinate missing rows={summary['coordinate_missing_rows']}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")
    print()
    print("R1 contract:")
    print("  Missing effort remains null + flagged; no imputation.")
    print("  PAMA outcomes are excluded from green-crab evidence.")
    print("  Temperature/ShoreZone enrichment comes next, after core table validation.")


if __name__ == "__main__":
    main()
