from __future__ import annotations

import json
from pathlib import Path

from adaptive_response.simulator_criticism import (
    build_real_incident_context,
    simulator_criticism_report,
)
from adaptive_response.world_models import default_world_model_split


CANONICAL = Path("data/processed/canonical_site_visit.csv")
REAL_SITES = Path("data/processed/r2_real_sites.csv")
PRIMARY_EDGES = Path("data/processed/r2_real_graph_v0_edges.csv")
RECEIPT_EDGES = Path("reports/milestones/r2_real_graph_v0/real_graph_v0_edges.csv")
OUT = Path("reports/milestones/r5_simulator_criticism/simulator_criticism.json")


def _existing_edges_path() -> Path:
    if PRIMARY_EDGES.is_file():
        return PRIMARY_EDGES
    if RECEIPT_EDGES.is_file():
        return RECEIPT_EDGES
    raise SystemExit(
        "Missing R2 graph edge table. Expected data/processed/r2_real_graph_v0_edges.csv "
        "or the versioned R2 milestone receipt."
    )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    for path in (CANONICAL, REAL_SITES):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; reproduce R1/R2 processed data first.")

    bundle = build_real_incident_context(
        CANONICAL,
        REAL_SITES,
        _existing_edges_path(),
        holdout_years=2,
        max_sites=20,
        preferred_min_sites=12,
    )
    split = default_world_model_split()
    models = (*split["train"], *split["ood_model_holdout"])
    report = simulator_criticism_report(
        bundle,
        models,
        draws_per_family=250,
        seed_start=10_000,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    context = report["incident_context"]
    real = report["observed_positive_pattern_all_years"]
    reality = report["observed_positive_pattern_reality_years"]
    effort = report["real_effort"]

    print("=== R5 REAL-DATA SIMULATOR CRITICISM ===")
    print("STATUS: reality check / simulator criticism, NOT field validation.")
    print()
    print("INCIDENT CONTEXT")
    print(
        f"  topology-only seed={context['topology_only_seed']} | "
        f"sites={context['sites']} | edges={context['edges']}"
    )
    print(
        "  seed selection uses static graph/topology only; "
        "future detections and hidden truth are forbidden."
    )
    print()
    print("REAL DATA SPLIT")
    print(f"  calibration years={report['real_data_split']['calibration_years']}")
    print(f"  held-out reality years={report['real_data_split']['reality_check_years']}")
    print(
        f"  effort known={effort['known_rows']} missing={effort['missing_rows']} | "
        f"min/median/p90/max={_fmt(effort['min'])}/{_fmt(effort['median'])}/"
        f"{_fmt(effort['p90'])}/{_fmt(effort['max'])}"
    )
    print()
    print("OBSERVED POSITIVE-SITE PATTERN (descriptive; NOT true occupancy)")
    print(
        f"  all years: positive sites={real['occupied_count']}/{context['sites']} | "
        f"components={real['occupied_components']} | "
        f"neighbor fraction={_fmt(real['occupied_with_neighbor_fraction'])} | "
        f"median NN km={_fmt(real['median_nearest_neighbor_km'])}"
    )
    print(
        f"  held-out years: positive sites={reality['occupied_count']}/{context['sites']} | "
        f"components={reality['occupied_components']} | "
        f"neighbor fraction={_fmt(reality['occupied_with_neighbor_fraction'])} | "
        f"median NN km={_fmt(reality['median_nearest_neighbor_km'])}"
    )
    print()
    print("HABITAT PROXY")
    for habitat, score in report["habitat_proxy"]["scores"].items():
        support = report["habitat_proxy"]["support_site_years"].get(habitat)
        counts = report["habitat_proxy"]["positive_over_total_site_years"][habitat]
        print(
            f"  {habitat}: proxy={score:.4f} | "
            f"positive site-years={counts['positive']}/{counts['total']} | support={support}"
        )
    print("  NOTE: this is an observation-derived calibration proxy, not latent suitability.")
    print()
    print("SIMULATOR FAMILIES (250 seeded draws each)")
    for family in report["families"]:
        counts = family["occupied_count"]
        cohesion = family["occupied_with_neighbor_fraction"]
        spacing = family["median_nearest_neighbor_km"]
        warnings = family["warnings"] or ["none"]
        print(f"  {family['family_id']}")
        print(
            f"    unique patterns={family['unique_occupancy_patterns']} | "
            f"occupied count p10/median/p90="
            f"{_fmt(counts['p10'])}/{_fmt(counts['median'])}/{_fmt(counts['p90'])}"
        )
        print(
            f"    cohesion p10/median/p90="
            f"{_fmt(cohesion['p10'])}/{_fmt(cohesion['median'])}/{_fmt(cohesion['p90'])} | "
            f"NN km p10/median/p90="
            f"{_fmt(spacing['p10'])}/{_fmt(spacing['median'])}/{_fmt(spacing['p90'])}"
        )
        print(f"    criticism warnings={warnings}")
    print()
    print("INTERPRETATION GATE")
    print("  Warnings are prompts for review, not automatic fitting or rejection.")
    print("  Observed detections are not complete occupancy truth because detection is imperfect.")
    print("  Do not tune simulator parameters to make a planner win.")
    print("  Next decision: keep/adjust model ranges, then freeze benchmark contract.")
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
