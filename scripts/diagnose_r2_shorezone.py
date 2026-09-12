from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.shorezone import audit_shorezone_matches


CANONICAL = Path("data/processed/canonical_site_visit.csv")
OUT_CSV = Path("data/processed/r2_shorezone_site_matches.csv")
OUT_JSON = Path("data/processed/r2_shorezone_audit.json")


def main() -> None:
    if not CANONICAL.is_file():
        raise SystemExit(
            f"Missing {CANONICAL}. Build R1 first with scripts/build_r1_canonical.py."
        )

    print("=== R2 SHOREZONE MATCH DIAGNOSTIC ===")
    print("Source: Washington DNR production ArcGIS ShoreZone service")
    print("Diagnostic only: nearest shoreline assignment is NOT frozen yet.")
    print()

    matches, summary = audit_shorezone_matches(CANONICAL)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in matches for key in row})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"sites={summary['sites']} | matched within {summary['search_radius_m']}m="
        f"{summary['matched_within_search_radius']} | no match="
        f"{summary['no_match_within_search_radius']}"
    )
    if summary["nearest_distance_median_m"] is not None:
        print(
            "Nearest ShoreZone substrate-line distance (m): "
            f"min={summary['nearest_distance_min_m']:.1f} | "
            f"median={summary['nearest_distance_median_m']:.1f} | "
            f"max={summary['nearest_distance_max_m']:.1f}"
        )
    print("Coverage by candidate distance threshold:")
    for threshold, count in summary["distance_threshold_counts"].items():
        print(f"  <= {threshold} m: {count}/{summary['sites']}")
    print(
        "Sites with source-coordinate deviation >100m across years: "
        f"{summary['sites_with_coordinate_deviation_over_100m']}"
    )
    print("Nonmissing ShoreZone fields:")
    for field, count in summary["field_nonmissing"].items():
        print(f"  {field}: {count}/{summary['sites']}")
    print("Ambiguous thematic values for selected UNIT_ID:")
    for field, count in summary["field_ambiguous_sites"].items():
        print(f"  {field}: {count}")

    print()
    print("Closest/farthest site examples:")
    matched = [row for row in matches if row.get("shorezone_distance_m") is not None]
    matched.sort(key=lambda row: float(row["shorezone_distance_m"]))
    examples = matched[:5] + (matched[-5:] if len(matched) > 5 else [])
    seen: set[str] = set()
    for row in examples:
        site_id = str(row["site_id"])
        if site_id in seen:
            continue
        seen.add(site_id)
        print(
            f"  site={site_id} dist={float(row['shorezone_distance_m']):.1f}m "
            f"unit={row.get('shorezone_unit_id')} substrate={row.get('substrate')} "
            f"shoreline={row.get('shoreline_type')} exposure={row.get('exposure')} "
            f"eelgrass={row.get('eelgrass')} salt_marsh={row.get('salt_marsh')}"
        )

    print()
    print("GATE:")
    print("  Do not merge ShoreZone into canonical data from this run alone.")
    print("  First inspect distance coverage, coordinate stability and thematic ambiguity.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
