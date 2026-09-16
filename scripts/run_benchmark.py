"""Planner-agnostic benchmark CLI (engineering block, 2026-09-12).

Runs FrontierPlanner and, optionally, a loaded RL checkpoint against the same
frozen set of held-out incidents/seeds/budgets, writing raw per-(planner,case)
facts to CSV. No ranking metric is computed here - see
docs/DATA_FIRST_VALIDATION_FREEZE.md Section 8: evaluation metrics are not
yet frozen, and the primary competitive baseline is Information Gain (not yet
implemented; drop it into the `planners` dict below once it exists, the
runner needs no changes).

Usage:
    python scripts/run_benchmark.py --num-cases 20 --seed 999 --out benchmark.csv
    python scripts/run_benchmark.py --num-cases 20 --seed 999 \
        --rl-checkpoint runs/some_run/final.pt --out benchmark.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_response import FrontierPlanner
from adaptive_response.rl import (
    IncidentSamplerConfig,
    RLPlannerAdapter,
    load_policy_checkpoint,
    make_benchmark_cases,
    run_benchmark_suite,
    write_benchmark_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=999_999, help="Independent of any training seed.")
    parser.add_argument("--min-sites", type=int, default=12)
    parser.add_argument("--max-sites", type=int, default=24)
    parser.add_argument("--min-budget", type=int, default=20)
    parser.add_argument("--max-budget", type=int, default=40)
    parser.add_argument("--rl-checkpoint", type=str, default=None, help="Path to a checkpoint saved by train_gnn_policy.py; omit to benchmark Frontier alone.")
    parser.add_argument("--out", type=str, default="benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler_config = IncidentSamplerConfig(
        min_sites=args.min_sites,
        max_sites=args.max_sites,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )
    cases = make_benchmark_cases(num_cases=args.num_cases, seed=args.seed, sampler_config=sampler_config)

    planners = {"frontier": FrontierPlanner()}
    if args.rl_checkpoint:
        loaded = load_policy_checkpoint(args.rl_checkpoint)
        planners["rl"] = RLPlannerAdapter(loaded.policy)
        print(
            f"Loaded RL checkpoint from {args.rl_checkpoint} "
            f"(update_idx={loaded.update_idx}, architecture={loaded.architecture})"
        )
    # Information Gain slot: once Pablo/Fede's InformationGainPlanner exists,
    # add it here, e.g. planners["information_gain"] = InformationGainPlanner(...)

    print(f"Running {len(planners)} planner(s) on {len(cases)} held-out case(s)...")
    rows = run_benchmark_suite(planners, cases)

    out_path = REPO_ROOT / args.out
    write_benchmark_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
