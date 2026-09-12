from __future__ import annotations

import json
from pathlib import Path

from adaptive_response.q_identifiability import (
    load_canonical_observations,
    summarize_q_identifiability,
)


CANONICAL = Path("data/processed/canonical_site_visit.csv")
OUT_JSON = Path("data/processed/r3_q_identifiability_audit.json")


def main() -> None:
    if not CANONICAL.is_file():
        raise SystemExit(f"Missing {CANONICAL}. Run the R1 canonical builder first.")

    rows = load_canonical_observations(CANONICAL)
    summary = summarize_q_identifiability(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    effort = summary["effort"]
    outcomes = summary["outcomes_with_known_effort"]
    temporal = summary["temporal_repetition"]

    print("=== R3 DETECTABILITY-q IDENTIFIABILITY AUDIT ===")
    print("Question: can the current real monthly table identify a per-effort detectability q?")
    print()
    print(
        f"rows={summary['rows']} | sites={summary['sites']} | site-years={summary['site_years']} | "
        f"duplicate site-year-month keys={summary['duplicate_site_year_month_keys']}"
    )
    print(
        f"effort known={effort['known_rows']} | missing={effort['missing_rows']} | "
        f"coverage={effort['coverage_fraction']:.1%} | "
        f"trap sets min/median/max={effort['trap_sets_min']}/{effort['trap_sets_median']}/{effort['trap_sets_max']}"
    )
    print(
        f"known-effort outcomes: detections={outcomes['detections']} | "
        f"non-detections={outcomes['non_detections']} | "
        f"observed detection fraction={outcomes['observed_detection_fraction']:.3%}"
    )
    print()
    print("Observed outcomes by exact trap-set effort (DESCRIPTIVE ONLY; NOT q estimates):")
    for _, stats in effort["by_exact_trap_sets"].items():
        print(
            f"  e={stats['trap_sets']:g}: rows={stats['rows']} | detections={stats['detections']} | "
            f"non-detections={stats['non_detections']} | observed detection fraction={stats['observed_detection_fraction']:.3%}"
        )
    print()
    print("Temporal repetition diagnostic:")
    print(
        f"  site-years with >=2 sampled months={temporal['site_years_with_2plus_months']} | "
        f"site-years with any detection={temporal['site_years_with_any_detection']}"
    )
    print(
        f"  known-effort months after first detection in same site-year={temporal['known_effort_months_after_first_detection']} | "
        f"detections={temporal['detections_after_first_detection']} | "
        f"non-detections={temporal['non_detections_after_first_detection']}"
    )
    print()
    print("DECISION:")
    print("  q is NOT identifiable from this table alone.")
    print("  Monthly repeated sampling is not the same thing as same-occasion detection replicates or proven occupancy closure.")
    print("  Keep P(no detection | occupied,e,q)=(1-q)^e.")
    print("  Source q from defensible repeated-detection/protocol literature or richer data, then run sensitivity analysis.")
    print("  Do NOT estimate q from raw detection fraction by effort.")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
