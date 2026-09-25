"""R11 same-budget benchmark: Dynamic Delimitation vs Grab the Crab GNN/RL
(configs/benchmark_protocol_r11_dynamic.json).

Reuses the exact FROZEN R8 case manifest
(reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json) -
same incidents, same seeds, same hidden worlds - but at the R11 resource
contract (budget=18, max_rounds=18 instead of 6), so a small-effort policy
has enough decision windows to spend its full budget. This does NOT modify
configs/benchmark_protocol_r8.json or any R8 report.

Both planners receive identical field-effort budget and identical raw field
observations. Dynamic Delimitation never sees belief/uncertainty/q/possible
worlds (see dynamic_delimitation_planner.py); GNN/RL keeps its full
evidence-aware GraphState.

Primary reporting metric for R11 is confirmed_unique_positive_sites
(= occupied_sites_total - occupied_sites_missed, both already computed
evaluator-side/post-reveal in SpatialBenchmarkRow) - NOT detections_found,
which can overcount repeat captures at the same site.

Usage:
    python scripts/run_r11_dynamic_benchmark.py --rl-checkpoint runs/r11_seed0_.../final.pt \
        --splits validation --out reports/r11_benchmark_seed0_validation.csv
    python scripts/run_r11_dynamic_benchmark.py --rl-checkpoint runs/r11_seed0_.../final.pt \
        --splits id_test ood_model_test ood_q_low ood_q_high --out reports/r11_benchmark_seed0.csv
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from adaptive_response import DynamicDelimitationMax6Planner
from adaptive_response.rl import (
    RLSpatialPlannerAdapter,
    load_policy_checkpoint,
    run_spatial_benchmark_suite,
    write_spatial_benchmark_csv,
)
from adaptive_response.rl.r8_manifest_cases import load_formal_benchmark_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
R11_MAX_ROUNDS = 18  # configs/benchmark_protocol_r11_dynamic.json frozen_resource_contract.max_windows
R11_BUDGET = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-checkpoint", type=str, required=True)
    parser.add_argument(
        "--splits", type=str, nargs="+",
        default=["id_test", "ood_model_test", "ood_q_low", "ood_q_high"],
        help="Manifest splits to load. Use --splits validation for the pre-freeze check.",
    )
    parser.add_argument("--out", type=str, default="reports/r11_benchmark.csv")
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

    confirmed_unique = [r.occupied_sites_total - r.occupied_sites_missed for r in subset]
    missed_m, missed_s = agg("missed_occupied_fraction")
    cov_m, cov_s = agg("occupied_site_coverage")
    effort_m, effort_s = agg("effort_spent")
    rounds_m, rounds_s = agg("num_rounds")
    return {
        "n": len(subset),
        "confirmed_unique_positive_sites_mean": statistics.mean(confirmed_unique),
        "confirmed_unique_positive_sites_stdev": statistics.pstdev(confirmed_unique) if len(confirmed_unique) > 1 else 0.0,
        "missed_fraction_mean": missed_m, "missed_fraction_stdev": missed_s,
        "coverage_mean": cov_m, "coverage_stdev": cov_s,
        "effort_spent_mean": effort_m, "effort_spent_stdev": effort_s,
        "num_windows_mean": rounds_m, "num_windows_stdev": rounds_s,
        "wall_clock_seconds_total": sum(r.wall_clock_seconds for r in subset),
    }


def main() -> None:
    args = parse_args()

    loaded = load_policy_checkpoint(args.rl_checkpoint)
    print(f"Loaded RL checkpoint: {args.rl_checkpoint} (update_idx={loaded.update_idx})")

    planners = {
        "dynamic_delimitation": DynamicDelimitationMax6Planner(effort_levels=tuple(args.effort_levels)),
        "gnn_rl": RLSpatialPlannerAdapter(loaded.policy),
    }

    print(f"Loading frozen R8 manifest cases, splits={args.splits}, at R11 contract (budget={R11_BUDGET}, max_rounds={R11_MAX_ROUNDS})...")
    cases = load_formal_benchmark_cases(
        splits=tuple(args.splits),
        budget=R11_BUDGET,
        max_rounds=R11_MAX_ROUNDS,
        draws_per_model=args.draws_per_model,
    )
    print(f"Loaded {len(cases)} cases.")

    t0 = time.time()
    rows = run_spatial_benchmark_suite(planners, cases)
    print(f"Ran {len(rows)} planner-case rows in {time.time() - t0:.1f}s.")

    write_spatial_benchmark_csv(rows, args.out)
    print(f"Wrote {args.out}")

    groups = sorted({c.group for c in cases})
    for group in groups:
        print(f"\n=== {group} ===")
        for planner_name in planners:
            summary = summarize(rows, group=group, planner_name=planner_name)
            if not summary:
                continue
            print(
                f"  {planner_name:22s} n={summary['n']:3d} "
                f"confirmed_unique_positive_sites={summary['confirmed_unique_positive_sites_mean']:.3f}"
                f"(+/-{summary['confirmed_unique_positive_sites_stdev']:.3f}) "
                f"missed_frac={summary['missed_fraction_mean']:.3f} "
                f"coverage={summary['coverage_mean']:.3f} "
                f"effort={summary['effort_spent_mean']:.2f} "
                f"windows={summary['num_windows_mean']:.2f}"
            )


if __name__ == "__main__":
    main()
