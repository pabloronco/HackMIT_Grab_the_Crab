"""R8 corrected formal benchmark (docs/R8_DEMU_BENCHMARK_REVIEW.md,
configs/benchmark_protocol_r8.json): Frontier vs spatial-joint Information
Gain vs GNN+RL on the exact FROZEN 180-case manifest
(reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json),
with real monitoring-site coordinates/habitat (no graph-layout or neutral
placeholders) and the corrected, budget-using baselines:

  - Frontier: existing site heuristic, effort 6 when budget permits,
    otherwise the largest allowed level;
  - Information Gain: (site, effort) jointly, maximizes ABSOLUTE expected
    entropy reduction (IG-per-effort only a tie-break);
  - RL: unchanged action contract.

Reports mean+dispersion per group (id_test / ood_model_test / ood_q_low /
ood_q_high), plus the mandatory secondary fields (effort_spent,
detections_found, num_rounds, wall_clock_seconds), visible failure cases, and
the R8 claim guardrail. Validation cases (60) are intentionally excluded -
reserved for training/checkpoint selection, not folded into test averages.

Usage:
    python scripts/run_spatial_benchmark_r8.py --rl-checkpoint runs/.../final.pt --out reports/r8_benchmark_seed0.csv
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from adaptive_response import FrontierPlanner, InformationGainPlanner
from adaptive_response.rl import (
    RLSpatialPlannerAdapter,
    load_policy_checkpoint,
    run_spatial_benchmark_suite,
    write_spatial_benchmark_csv,
)
from adaptive_response.rl.r8_manifest_cases import FORMAL_REPORTING_SPLITS, load_formal_benchmark_cases

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/r8_benchmark.csv")
    parser.add_argument("--draws-per-model", type=int, default=60)
    parser.add_argument("--effort-levels", type=int, nargs="+", default=[1, 3, 6])
    return parser.parse_args()


def summarize(rows, *, group: str, planner_name: str) -> dict:
    subset = [r for r in rows if r.group == group and r.planner_name == planner_name]
    if not subset:
        return {}
    def agg(field):
        values = [getattr(r, field) for r in subset]
        return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)
    missed_m, missed_s = agg("missed_occupied_fraction")
    cov_m, cov_s = agg("occupied_site_coverage")
    unc_m, _ = agg("final_global_uncertainty")
    effort_m, effort_s = agg("effort_spent")
    det_m, _ = agg("detections_found")
    rounds_m, _ = agg("num_rounds")
    wall_m, _ = agg("wall_clock_seconds")
    return {
        "n": len(subset),
        "missed_fraction_mean": missed_m, "missed_fraction_stdev": missed_s,
        "coverage_mean": cov_m, "coverage_stdev": cov_s,
        "final_uncertainty_mean": unc_m,
        "effort_spent_mean": effort_m, "effort_spent_stdev": effort_s,
        "detections_found_mean": det_m,
        "num_rounds_mean": rounds_m,
        "wall_clock_seconds_total": sum(getattr(r, "wall_clock_seconds") for r in subset),
    }


def main() -> None:
    args = parse_args()

    loaded = load_policy_checkpoint(args.rl_checkpoint)
    print(f"Loaded RL checkpoint: {args.rl_checkpoint} (update_idx={loaded.update_idx}, arch={loaded.architecture})")

    planners = {
        "frontier": FrontierPlanner(max_sites=1, effort_levels=tuple(args.effort_levels)),
        "information_gain": InformationGainPlanner(
            max_sites=1, require_spatial_belief=True, effort_levels=tuple(args.effort_levels)
        ),
        "gnn_rl": RLSpatialPlannerAdapter(loaded.policy),
    }

    print(f"Loading frozen R8 manifest, formal reporting splits: {FORMAL_REPORTING_SPLITS}")
    cases = load_formal_benchmark_cases(draws_per_model=args.draws_per_model)
    print(f"Loaded {len(cases)} frozen formal cases across groups: {sorted(set(c.group for c in cases))}")

    start = time.time()
    rows = run_spatial_benchmark_suite(planners, cases)
    elapsed = time.time() - start
    print(f"Ran {len(rows)} (planner, case) episodes in {elapsed:.1f}s ({elapsed / max(len(rows), 1):.3f}s/episode).")

    out_path = REPO_ROOT / args.out
    write_spatial_benchmark_csv(rows, out_path)
    print(f"Wrote raw rows to {out_path}")

    print("\n=== SUMMARY (mean +/- stdev; NOT a ranking, NOT a real-world claim) ===")
    for group in FORMAL_REPORTING_SPLITS:
        print(f"\n[{group}]")
        for planner_name in planners:
            s = summarize(rows, group=group, planner_name=planner_name)
            if not s:
                continue
            print(
                f"  {planner_name:16s} n={s['n']:3d}  "
                f"missed_frac={s['missed_fraction_mean']:.3f}+-{s['missed_fraction_stdev']:.3f}  "
                f"coverage={s['coverage_mean']:.3f}+-{s['coverage_stdev']:.3f}  "
                f"final_uncertainty={s['final_uncertainty_mean']:.3f}  "
                f"effort_spent={s['effort_spent_mean']:.2f}+-{s['effort_spent_stdev']:.2f}  "
                f"detections={s['detections_found_mean']:.2f}  "
                f"rounds={s['num_rounds_mean']:.2f}  "
                f"wall_clock_total={s['wall_clock_seconds_total']:.1f}s"
            )

    worst = sorted(rows, key=lambda r: -r.missed_occupied_fraction)[:5]
    print("\n=== Visible failure cases (highest missed_occupied_fraction) ===")
    for r in worst:
        print(f"  {r.planner_name:16s} {r.case_label:40s} group={r.group:16s} "
              f"missed_frac={r.missed_occupied_fraction:.3f} occupied_total={r.occupied_sites_total} "
              f"rounds={r.num_rounds} effort={r.effort_spent}")

    print(
        "\nClaim guardrail: final results support performance on the frozen synthetic "
        "sequential-decision benchmark and robustness to tested model/q assumptions only; "
        "they do not prove field effectiveness or superiority to operators. ID = matched-model-class "
        "benchmark, not ecological validation. Not a generic 'OOD robustness' claim: no formal OOD "
        "topology or ecological-parameter OOD exists yet."
    )


if __name__ == "__main__":
    main()
