from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.incident_subgraph import (
    audit_all_incident_seeds,
    extract_incident_subgraph,
    load_graph_edges,
)
from adaptive_response.real_graph import load_real_sites


REAL_SITES = Path("data/processed/r2_real_sites.csv")
GRAPH_EDGES = Path("data/processed/r2_real_graph_v0_edges.csv")
OUT_JSON = Path("data/processed/r2_incident_subgraph_audit.json")
OUT_CSV = Path("data/processed/r2_incident_subgraph_seed_audit.csv")

PREFERRED_MIN_SITES = 12
MAX_SITES = 20


def main() -> None:
    if not REAL_SITES.is_file():
        raise SystemExit(f"Missing {REAL_SITES}")
    if not GRAPH_EDGES.is_file():
        raise SystemExit(
            f"Missing {GRAPH_EDGES}. Run build_r2_real_graph_v0.py first."
        )

    sites = load_real_sites(REAL_SITES)
    site_ids = [str(site["site_id"]) for site in sites]
    edges = load_graph_edges(GRAPH_EDGES)
    audit = audit_all_incident_seeds(
        site_ids,
        edges,
        max_sites=MAX_SITES,
        preferred_min_sites=PREFERRED_MIN_SITES,
    )

    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed_site_id",
            "source_component_sites",
            "selected_sites",
            "below_preferred_min",
            "capped_by_max_sites",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit["seed_rows"])

    print("=== R2 INCIDENT-SUBGRAPH AUDIT ===")
    print("Rule: initial detection + frozen primary adjacency + static route/distance metadata only.")
    print("Forbidden: future detections, replay outcomes, hidden synthetic truth.")
    print("Disconnected components are NEVER bridged just to reach 12-20 sites.")
    print()
    print(
        f"source sites={audit['sites']} | preferred={PREFERRED_MIN_SITES}-{MAX_SITES} | "
        f"seeds in preferred range={audit['seeds_in_preferred_size_range']} | "
        f"below preferred min={audit['seeds_below_preferred_min']} | "
        f"capped by max={audit['seeds_capped_by_max']}"
    )
    print(f"Selected-size histogram by possible initial seed: {audit['selected_size_histogram']}")

    groups: dict[int, list[str]] = {}
    for row in audit["seed_rows"]:
        groups.setdefault(int(row["selected_sites"]), []).append(str(row["seed_site_id"]))

    print()
    print("Seeds grouped by incident-subgraph size:")
    for size in sorted(groups, reverse=True):
        print(f"  size {size}: {groups[size]}")

    preferred_rows = [
        row
        for row in audit["seed_rows"]
        if PREFERRED_MIN_SITES <= int(row["selected_sites"]) <= MAX_SITES
    ]
    if preferred_rows:
        example_seed = str(preferred_rows[0]["seed_site_id"])
        selected, sub_edges, summary = extract_incident_subgraph(
            site_ids,
            edges,
            seed_site_id=example_seed,
            max_sites=MAX_SITES,
            preferred_min_sites=PREFERRED_MIN_SITES,
        )
        print()
        print("Topology-only example (NOT a chosen demo incident):")
        print(
            f"  seed={example_seed} | sites={len(selected)} | edges={len(sub_edges)} | "
            f"component={summary['source_component_sites']}"
        )
        print(f"  site ids={selected}")

    print()
    print("DECISION GATE:")
    print("  Preferred graph size is a target, not a license to invent connectivity.")
    print("  If historical/demo t0 lies in a smaller component, keep the smaller real subgraph.")
    print("  Do not choose the final incident seed using future biological outcomes.")
    print("  Next: observation-model / detectability-q work once this extraction rule is accepted.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
