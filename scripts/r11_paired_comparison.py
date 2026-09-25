"""Paired, case-wise comparison of Grab the Crab GNN/RL against Dynamic
Delimitation on the frozen R11 benchmark (reports/r11_benchmark_seed{0,1,2}.csv),
same statistical method as scripts/r8_paired_comparison.py.

Primary metric: confirmed_unique_positive_sites (occupied_sites_total -
occupied_sites_missed), NOT raw detections_found - see
configs/benchmark_protocol_r11_dynamic.json effort_to_parity.parity_is_measured_on.

delta(case) = confirmed_unique(GNN, case) - confirmed_unique(Dynamic, case)
  positive delta -> GNN found MORE unique positive sites than Dynamic (GNN better)
  negative delta -> GNN found FEWER (Dynamic better)

Dynamic Delimitation is deterministic given a case, so its value is read once
per case (identical across all 3 seed CSVs by construction - this script
asserts that, rather than assuming it). GNN varies by seed, so two views are
reported, exactly like r8_paired_comparison.py:
  1. Per-seed: bootstrap over cases, once per RL seed.
  2. Pooled: bootstrap resampling cases (not case-seed rows independently),
     pooling all 3 seeds' deltas for each resampled case.

Usage:
    python scripts/r11_paired_comparison.py \
        --gnn-csv reports/r11_benchmark_seed0.csv reports/r11_benchmark_seed1.csv reports/r11_benchmark_seed2.csv \
        --out docs/R11_PAIRED_COMPARISON.md
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 10_000
CI_LOW, CI_HIGH = 0.025, 0.975


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confirmed_unique(row: dict) -> int:
    return int(row["occupied_sites_total"]) - int(row["occupied_sites_missed"])


def index_confirmed_unique(rows: list[dict], planner_name: str) -> dict[str, int]:
    return {
        r["case_label"]: confirmed_unique(r)
        for r in rows
        if r["planner_name"] == planner_name
    }


def group_of(rows: list[dict], planner_name: str) -> dict[str, str]:
    return {r["case_label"]: r["group"] for r in rows if r["planner_name"] == planner_name}


def bootstrap_ci(deltas: list[float], *, n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    mean = statistics.mean(deltas)
    n = len(deltas)
    resampled_means = []
    for _ in range(n_bootstrap):
        sample = [deltas[random.randrange(n)] for _ in range(n)]
        resampled_means.append(statistics.mean(sample))
    resampled_means.sort()
    lo = resampled_means[int(CI_LOW * n_bootstrap)]
    hi = resampled_means[int(CI_HIGH * n_bootstrap)]
    return mean, lo, hi


def win_tie_lose(deltas: list[float]) -> tuple[float, float, float]:
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    losses = sum(1 for d in deltas if d < 0)
    return wins / n * 100, ties / n * 100, losses / n * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-csv", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=str, default="docs/R11_PAIRED_COMPARISON.md")
    args = parser.parse_args()

    random.seed(0)

    seed_paths = [Path(p) for p in args.gnn_csv]
    all_rows = [read_rows(p) for p in seed_paths]

    dynamic_by_case = [index_confirmed_unique(rows, "dynamic_delimitation") for rows in all_rows]
    for i in range(1, len(dynamic_by_case)):
        if dynamic_by_case[i] != dynamic_by_case[0]:
            raise ValueError(
                f"{seed_paths[i]} has different Dynamic Delimitation results than "
                f"{seed_paths[0]} for the same cases - Dynamic must be deterministic. "
                "This indicates a bug, not real variance."
            )
    dynamic = dynamic_by_case[0]
    groups = group_of(all_rows[0], "dynamic_delimitation")

    gnn_by_seed = [index_confirmed_unique(rows, "gnn_rl") for rows in all_rows]
    case_ids = sorted(dynamic)
    for i, gnn in enumerate(gnn_by_seed):
        missing = set(case_ids) - set(gnn)
        if missing:
            raise ValueError(f"{seed_paths[i]} is missing gnn_rl rows for cases: {sorted(missing)[:5]}...")

    lines = ["# R11 paired comparison: Grab the Crab GNN/RL vs Dynamic Delimitation", ""]
    lines.append(
        "Metric: confirmed_unique_positive_sites (delta = GNN - Dynamic). "
        "Positive means GNN found more distinct occupied sites than the deterministic "
        "reactive baseline, at the same 18-unit field-effort budget."
    )
    lines.append("")

    # Per-seed view
    lines.append("## Per-seed")
    lines.append("")
    lines.append("| Seed | n | mean delta | 95% CI | win% | tie% | lose% |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, gnn in enumerate(gnn_by_seed):
        deltas = [gnn[c] - dynamic[c] for c in case_ids]
        mean, lo, hi = bootstrap_ci(deltas)
        w, t, l = win_tie_lose(deltas)
        lines.append(
            f"| seed{i} | {len(deltas)} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{w:.1f}% | {t:.1f}% | {l:.1f}% |"
        )
    lines.append("")

    # Pooled view (resample cases, pool all 3 seeds' deltas per resampled case)
    lines.append("## Pooled (case-clustered bootstrap across all 3 seeds)")
    lines.append("")
    pooled_deltas_by_case: dict[str, list[float]] = defaultdict(list)
    for gnn in gnn_by_seed:
        for c in case_ids:
            pooled_deltas_by_case[c].append(gnn[c] - dynamic[c])

    flat_deltas = [d for c in case_ids for d in pooled_deltas_by_case[c]]
    pooled_mean = statistics.mean(flat_deltas)
    resampled_means = []
    for _ in range(N_BOOTSTRAP):
        sampled_cases = [case_ids[random.randrange(len(case_ids))] for _ in range(len(case_ids))]
        sampled_deltas = [d for c in sampled_cases for d in pooled_deltas_by_case[c]]
        resampled_means.append(statistics.mean(sampled_deltas))
    resampled_means.sort()
    lo = resampled_means[int(CI_LOW * N_BOOTSTRAP)]
    hi = resampled_means[int(CI_HIGH * N_BOOTSTRAP)]
    p_gnn_better = sum(1 for m in resampled_means if m > 0) / N_BOOTSTRAP * 100
    w, t, l = win_tie_lose(flat_deltas)
    lines.append(f"- n = {len(case_ids)} cases x {len(gnn_by_seed)} seeds = {len(flat_deltas)} rows")
    lines.append(f"- mean delta = {pooled_mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    lines.append(f"- P(GNN mean > 0) (bootstrap confidence GNN is better on average) = {p_gnn_better:.1f}%")
    lines.append(f"- win/tie/lose (row-level, GNN vs Dynamic) = {w:.1f}% / {t:.1f}% / {l:.1f}%")
    lines.append("")

    # Per-group breakdown (pooled across seeds)
    lines.append("## Per-group breakdown (pooled across seeds)")
    lines.append("")
    lines.append("| Group | n cases | Dynamic mean | GNN mean (pooled) | mean delta |")
    lines.append("|---|---|---|---|---|")
    for group in sorted(set(groups.values())):
        group_cases = [c for c in case_ids if groups[c] == group]
        dyn_mean = statistics.mean(dynamic[c] for c in group_cases)
        gnn_vals = [gnn[c] for gnn in gnn_by_seed for c in group_cases]
        gnn_mean = statistics.mean(gnn_vals)
        group_deltas = [d for c in group_cases for d in pooled_deltas_by_case[c]]
        group_delta_mean = statistics.mean(group_deltas)
        lines.append(
            f"| {group} | {len(group_cases)} | {dyn_mean:.3f} | {gnn_mean:.3f} | {group_delta_mean:+.3f} |"
        )
    lines.append("")

    lines.append(
        "**Claim guardrail** (configs/benchmark_protocol_r11_dynamic.json): these results hold "
        "only on the frozen real-data-constrained synthetic incidents above. GNN training was "
        "time-boxed to 1 hour per seed with no hyperparameter search - the seed variance visible "
        "here is real and reported, not selected around."
    )

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
