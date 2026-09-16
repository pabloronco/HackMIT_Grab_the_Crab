from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.incident_subgraph import load_graph_edges
from adaptive_response.real_multisite_evidence import observed_positive_footprint_report
from adaptive_response.simulator_criticism import read_canonical_rows


CANONICAL = Path("data/processed/canonical_site_visit.csv")
REAL_SITES_RECEIPT = Path("reports/milestones/r2_real_graph_v0/real_sites_v0.csv")
PRIMARY_EDGES = Path("data/processed/r2_real_graph_v0_edges.csv")
RECEIPT_EDGES = Path("reports/milestones/r2_real_graph_v0/real_graph_v0_edges.csv")
OUT = Path("reports/milestones/r9_real_multisite_evidence/observed_positive_footprint.json")


def _edge_path() -> Path:
    if PRIMARY_EDGES.is_file():
        return PRIMARY_EDGES
    if RECEIPT_EDGES.is_file():
        return RECEIPT_EDGES
    raise SystemExit("Missing frozen R2 graph edge table.")


def _site_ids() -> list[str]:
    if not REAL_SITES_RECEIPT.is_file():
        raise SystemExit(f"Missing {REAL_SITES_RECEIPT}")
    with REAL_SITES_RECEIPT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [str(row["site_id"]).strip() for row in rows]


def main() -> None:
    if not CANONICAL.is_file():
        raise SystemExit(
            f"Missing {CANONICAL}; reproduce the processed monitoring table first. "
            "Do not substitute look-alike external data."
        )

    canonical_rows = read_canonical_rows(CANONICAL)
    site_ids = _site_ids()
    edges = load_graph_edges(_edge_path())
    report = observed_positive_footprint_report(canonical_rows, site_ids, edges)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print("=== R9 REAL MULTI-SITE OBSERVED-POSITIVE EVIDENCE AUDIT ===")
    print("STATUS: descriptive reality check; observed positives are NOT latent occupancy.")
    print(f"years={report['years']}")
    print(f"sites={report['site_count']} | graph components={report['component_count']}")
    print()
    print("NETWORK OBSERVED-POSITIVE FOOTPRINT")
    for row in report["network"]["timeline"]:
        print(
            f"  {row['year']}: positive sites={row['observed_positive_site_count']} | "
            f"new={row['new_observed_positive_site_count']} | "
            f"cumulative={row['cumulative_observed_positive_site_count']}"
        )
    print()
    print("COMPONENT THRESHOLD SUMMARY")
    for threshold, count in report["components_reaching_cumulative_threshold"].items():
        print(f"  components reaching >= {threshold} cumulative observed-positive sites: {count}")
    print()
    print("COMPONENTS WITH ANY OBSERVED POSITIVES")
    for component in report["components"]:
        if component["ever_observed_positive_site_count"] == 0:
            continue
        crossings = component["threshold_crossings"]
        crossing_text = ", ".join(
            f">={threshold}: {entry['first_year_reached'] if entry['reached'] else 'no'}"
            for threshold, entry in crossings.items()
        )
        print(
            f"  {component['component_id']} sites={component['site_count']} | "
            f"ever positive={component['ever_observed_positive_site_count']} | {crossing_text}"
        )
    print()
    print("INTERPRETATION GATE")
    print("  - non-detection is never converted to true absence;")
    print("  - cumulative observed-positive footprint is not proof of spread rate or true extent;")
    print("  - graph components are a spatial audit unit, not asserted biological episodes;")
    print("  - do not retune synthetic worlds from this audit after seeing planner results.")
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
