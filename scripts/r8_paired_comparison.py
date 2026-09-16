"""Paired, case-wise comparison of GNN+RL against each baseline on the
frozen R8 manifest, with 95% bootstrap confidence intervals on the delta in
missed_occupied_fraction.

This is a reporting-only analysis: it does not change the policy, reward,
or benchmark protocol. Requested alongside the R8 corrected rerun (review
2026-09-15) because the group-level mean +/- stdev in
docs/R8_BENCHMARK_REPORT.md doesn't show whether RL wins/loses/ties on a
per-case basis, which the paired delta does.

delta(case) = missed_fraction(baseline, case) - missed_fraction(RL, case)
  positive delta -> RL missed LESS than the baseline on that case (RL better)
  negative delta -> RL missed MORE than the baseline on that case (RL worse)

Two views are reported, because RL varies by training seed but Frontier and
Information Gain are deterministic given a case (see
docs/R8_BENCHMARK_REPORT.md's table note):
  1. Per-seed: bootstrap over the 180 cases, once per RL seed.
  2. Pooled: bootstrap over the 180 cases, resampling cases (not
     case-seed rows independently) and pooling all 3 seeds' deltas for
     each resampled case - this respects the paired/clustered structure
     instead of pretending 540 case-seed rows are independent.

Usage:
    python scripts/r8_paired_comparison.py \
        --rl-csv reports/r8_benchmark_seed0.csv reports/r8_benchmark_seed1.csv reports/r8_benchmark_seed2.csv \
        --out reports/r8_paired_comparison.md
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


def index_missed_fraction(rows: list[dict], planner_name: str) -> dict[str, float]:
    return {
        r["case_label"]: float(r["missed_occupied_fraction"])
        for r in rows
        if r["planner_name"] == planner_name
    }


def case_groups(rows: list[dict]) -> dict[str, str]:
    return {r["case_label"]: r["group"] for r in rows}


def bootstrap_ci(values: list[float], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via case-level resampling with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(resample))
    means.sort()
    return (
        statistics.mean(values),
        means[int(CI_LOW * n_bootstrap)],
        means[int(CI_HIGH * n_bootstrap) - 1],
    )


def clustered_bootstrap_ci(
    deltas_by_case_then_seed: dict[str, list[float]], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0
) -> tuple[float, float, float]:
    """Case-clustered bootstrap: resample CASES with replacement; each
    resampled case contributes all of its seeds' deltas together (not
    resampled independently), so within-case seed correlation is preserved
    instead of being pseudo-replicated away."""
    rng = random.Random(seed)
    case_ids = list(deltas_by_case_then_seed)
    n = len(case_ids)
    all_deltas_flat = [d for deltas in deltas_by_case_then_seed.values() for d in deltas]
    means = []
    for _ in range(n_bootstrap):
        resampled_cases = [case_ids[rng.randrange(n)] for _ in range(n)]
        pooled = [d for c in resampled_cases for d in deltas_by_case_then_seed[c]]
        means.append(statistics.mean(pooled))
    means.sort()
    return (
        statistics.mean(all_deltas_flat),
        means[int(CI_LOW * n_bootstrap)],
        means[int(CI_HIGH * n_bootstrap) - 1],
    )


def paired_deltas(baseline_missed: dict[str, float], rl_missed: dict[str, float]) -> dict[str, float]:
    common = sorted(set(baseline_missed) & set(rl_missed))
    return {c: baseline_missed[c] - rl_missed[c] for c in common}


def format_ci(mean: float, lo: float, hi: float) -> str:
    verdict = "RL better (CI excludes 0, positive)" if lo > 0 else (
        "RL worse (CI excludes 0, negative)" if hi < 0 else "not distinguishable from 0"
    )
    return f"mean_delta={mean:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-csv", nargs=3, required=True, help="Exactly 3 CSVs, one per RL seed, in seed 0/1/2 order.")
    parser.add_argument("--out", type=str, default="reports/r8_paired_comparison.md")
    args = parser.parse_args()

    seed_rows = [read_rows(Path(p)) for p in args.rl_csv]
    groups = case_groups(seed_rows[0])
    all_groups = sorted(set(groups.values()))

    frontier_missed = index_missed_fraction(seed_rows[0], "frontier")
    ig_missed = index_missed_fraction(seed_rows[0], "information_gain")
    # sanity: frontier/IG must be identical (deterministic) across the 3 seed CSVs.
    for i, rows in enumerate(seed_rows[1:], start=1):
        assert index_missed_fraction(rows, "frontier") == frontier_missed, f"frontier differs in seed-{i} CSV - not deterministic as expected"
        assert index_missed_fraction(rows, "information_gain") == ig_missed, f"information_gain differs in seed-{i} CSV - not deterministic as expected"

    rl_missed_by_seed = [index_missed_fraction(rows, "gnn_rl") for rows in seed_rows]

    lines = ["# R8 paired case-wise comparison (bootstrap 95% CI on missed_occupied_fraction delta)", ""]
    lines.append(
        "delta = baseline_missed_fraction - rl_missed_fraction; positive means RL missed less "
        "(RL better on that case). Reporting-only analysis, does not change policy/reward/protocol."
    )

    for baseline_name, baseline_missed in (("frontier", frontier_missed), ("information_gain", ig_missed)):
        lines.append(f"\n## {baseline_name} <-> gnn_rl")

        lines.append("\n### Per-seed (bootstrap over 180 cases, one CI per RL training seed)")
        for group in ["ALL"] + all_groups:
            lines.append(f"\n**{group}**")
            for seed_idx, rl_missed in enumerate(rl_missed_by_seed):
                deltas = paired_deltas(baseline_missed, rl_missed)
                if group != "ALL":
                    deltas = {c: d for c, d in deltas.items() if groups[c] == group}
                values = list(deltas.values())
                mean, lo, hi = bootstrap_ci(values, seed=seed_idx)
                lines.append(f"- seed {seed_idx} (n={len(values)}): {format_ci(mean, lo, hi)}")

        lines.append("\n### Pooled across 3 seeds (case-clustered bootstrap)")
        for group in ["ALL"] + all_groups:
            per_case: dict[str, list[float]] = defaultdict(list)
            for rl_missed in rl_missed_by_seed:
                deltas = paired_deltas(baseline_missed, rl_missed)
                for c, d in deltas.items():
                    if group != "ALL" and groups[c] != group:
                        continue
                    per_case[c].append(d)
            if not per_case:
                continue
            mean, lo, hi = clustered_bootstrap_ci(per_case, seed=42)
            n_cases = len(per_case)
            n_obs = sum(len(v) for v in per_case.values())
            lines.append(f"- **{group}** (n_cases={n_cases}, n_case_seed_obs={n_obs}): {format_ci(mean, lo, hi)}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
