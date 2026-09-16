"""Three preregistered-ablation diagnostics for the R8 corrected retrain
(review 2026-09-15). These are observation-only: they must not, and do not,
feed back into changing the policy, reward, architecture, or training
length for this rerun. They exist to tell the team whether a *separate*,
future preregistered ablation is scientifically justified - not to justify
one here.

1. Validation missed-fraction vs. training update, from each seed's
   eval.csv - is the curve still clearly improving at update 300, or has it
   plateaued?
2. Average magnitude/contribution of each reward term (dense
   mean_uncertainty_reduction, dense new_detections, terminal
   missed_occupied_fraction) from each seed's decisions.jsonl - the R7/R8
   reward contract's own watch-item: is the terminal term drowned out by
   the dense per-round terms?
3. Action statistics from decisions.jsonl: effort level {1,3,6} frequency,
   same-episode site revisit rate, distinct sites visited per episode.

Usage:
    python scripts/r8_training_diagnostics.py \
        --run-dir runs/r8_fixed_seed0_<timestamp> \
        --run-dir runs/r8_fixed_seed1_<timestamp> \
        --run-dir runs/r8_fixed_seed2_<timestamp> \
        --out reports/r8_training_diagnostics.md
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_eval_csv(run_dir: Path) -> list[dict]:
    path = run_dir / "eval.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_decisions_jsonl(run_dir: Path) -> list[dict]:
    path = run_dir / "decisions.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def diagnostic_1_plateau_check(eval_rows: list[dict]) -> dict:
    updates = [int(r["update"]) for r in eval_rows]
    missed = [float(r["rl_missed_fraction"]) for r in eval_rows]
    n = len(missed)
    tail_n = max(min(10, n // 3), 2)
    tail_updates = updates[-tail_n:]
    tail_missed = missed[-tail_n:]

    mean_x = statistics.mean(tail_updates)
    mean_y = statistics.mean(tail_missed)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(tail_updates, tail_missed))
    var_x = sum((x - mean_x) ** 2 for x in tail_updates)
    slope = cov / var_x if var_x > 0 else 0.0

    early_half = missed[: n // 2] if n >= 4 else missed[:1]
    late_half = missed[n // 2 :] if n >= 4 else missed[-1:]
    improvement_first_to_second_half = statistics.mean(early_half) - statistics.mean(late_half)

    still_improving = slope < -1e-4  # missed_fraction meaningfully decreasing with more updates
    return {
        "n_eval_points": n,
        "first_update": updates[0] if updates else None,
        "last_update": updates[-1] if updates else None,
        "tail_slope_per_update": slope,
        "tail_window": tail_n,
        "missed_fraction_first_half_mean": statistics.mean(early_half),
        "missed_fraction_second_half_mean": statistics.mean(late_half),
        "improvement_first_to_second_half": improvement_first_to_second_half,
        "last_3_missed_fraction": missed[-3:],
        "still_improving_at_last_update": still_improving,
    }


def diagnostic_2_reward_term_magnitude(records: list[dict]) -> dict:
    episodes: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        episodes[r["episode_index"]].append(r)

    dense_episode_sums: dict[str, list[float]] = defaultdict(list)
    dense_round_values: dict[str, list[float]] = defaultdict(list)
    terminal_values: list[float] = []

    for rounds in episodes.values():
        rounds_sorted = sorted(rounds, key=lambda r: r["round_index"])
        ep_sum: dict[str, float] = defaultdict(float)
        for r in rounds_sorted:
            for k, v in r["reward_components"].items():
                if k == "missed_occupied_fraction":
                    continue
                ep_sum[k] += v
                dense_round_values[k].append(v)
        for k, v in ep_sum.items():
            dense_episode_sums[k].append(v)

        terminal_round = rounds_sorted[-1]
        if "missed_occupied_fraction" in terminal_round["reward_components"]:
            terminal_values.append(terminal_round["reward_components"]["missed_occupied_fraction"])

    def summarize(values: list[float]) -> dict:
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "mean_abs": statistics.mean(abs(v) for v in values) if values else 0.0,
            "n": len(values),
        }

    dense_episode_total_abs = sum(
        summarize(v)["mean_abs"] for v in dense_episode_sums.values()
    )
    terminal_abs = summarize(terminal_values)["mean_abs"]

    return {
        "per_round_dense_terms": {k: summarize(v) for k, v in dense_round_values.items()},
        "per_episode_summed_dense_terms": {k: summarize(v) for k, v in dense_episode_sums.items()},
        "terminal_term": summarize(terminal_values),
        "dense_episode_total_mean_abs": dense_episode_total_abs,
        "terminal_vs_dense_episode_ratio": (
            terminal_abs / dense_episode_total_abs if dense_episode_total_abs > 0 else float("inf")
        ),
    }


def diagnostic_3_action_statistics(records: list[dict]) -> dict:
    effort_counter: Counter = Counter()
    episodes: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        effort_counter[r["effort_units"]] += 1
        episodes[r["episode_index"]].append(r)

    total_rounds = 0
    total_revisits = 0
    distinct_site_counts = []
    for rounds in episodes.values():
        rounds_sorted = sorted(rounds, key=lambda r: r["round_index"])
        seen: set[str] = set()
        revisits = 0
        for r in rounds_sorted:
            if r["site_id"] in seen:
                revisits += 1
            seen.add(r["site_id"])
        total_rounds += len(rounds_sorted)
        total_revisits += revisits
        distinct_site_counts.append(len(seen))

    total_actions = sum(effort_counter.values())
    return {
        "effort_level_frequency": {
            str(level): {"count": count, "fraction": count / total_actions if total_actions else 0.0}
            for level, count in sorted(effort_counter.items())
        },
        "revisit_rate": total_revisits / total_rounds if total_rounds else 0.0,
        "mean_distinct_sites_per_episode": statistics.mean(distinct_site_counts) if distinct_site_counts else 0.0,
        "n_episodes": len(episodes),
        "n_rounds": total_rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, dest="run_dirs")
    parser.add_argument("--out", type=str, default="reports/r8_training_diagnostics.md")
    args = parser.parse_args()

    lines = [
        "# R8 training diagnostics (observation-only, no policy/reward/protocol change)",
        "",
        "Requested alongside the R8 corrected rerun to decide whether a *separately "
        "preregistered* future ablation is scientifically justified. Not used to modify "
        "this rerun.",
    ]

    triggered_any = False

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        seed_label = run_dir.name

        lines.append(f"\n## {seed_label}")

        eval_rows = read_eval_csv(run_dir)
        d1 = diagnostic_1_plateau_check(eval_rows)
        lines.append("\n### 1. Validation curve plateau check")
        lines.append(f"- eval points: {d1['n_eval_points']} (update {d1['first_update']} -> {d1['last_update']})")
        lines.append(f"- last 3 rl_missed_fraction: {[round(v, 4) for v in d1['last_3_missed_fraction']]}")
        lines.append(
            f"- tail linear slope (last {d1['tail_window']} points): "
            f"{d1['tail_slope_per_update']:+.6f} missed_fraction/update"
        )
        lines.append(
            f"- first-half mean {d1['missed_fraction_first_half_mean']:.4f} -> "
            f"second-half mean {d1['missed_fraction_second_half_mean']:.4f} "
            f"(improvement {d1['improvement_first_to_second_half']:+.4f})"
        )
        if d1["still_improving_at_last_update"]:
            lines.append("- **FLAG: still meaningfully improving at the last update — report before further training.**")
            triggered_any = True
        else:
            lines.append("- Read as plateaued (tail slope not meaningfully negative).")

        records = read_decisions_jsonl(run_dir)
        d2 = diagnostic_2_reward_term_magnitude(records)
        lines.append("\n### 2. Reward term magnitude (dense vs. terminal)")
        lines.append("Per-round dense terms (fires every round):")
        for k, v in d2["per_round_dense_terms"].items():
            lines.append(f"  - {k}: mean={v['mean']:+.4f}, mean_abs={v['mean_abs']:.4f} (n={v['n']} rounds)")
        lines.append("Per-episode SUMMED dense terms (comparable scale to the one terminal term):")
        for k, v in d2["per_episode_summed_dense_terms"].items():
            lines.append(f"  - {k}: mean={v['mean']:+.4f}, mean_abs={v['mean_abs']:.4f} (n={v['n']} episodes)")
        lines.append(
            f"Terminal term (missed_occupied_fraction, fires once/episode): "
            f"mean={d2['terminal_term']['mean']:+.4f}, mean_abs={d2['terminal_term']['mean_abs']:.4f} "
            f"(n={d2['terminal_term']['n']} episodes)"
        )
        ratio = d2["terminal_vs_dense_episode_ratio"]
        lines.append(f"terminal_mean_abs / summed_dense_mean_abs = {ratio:.3f}")
        if ratio < 0.2:
            lines.append(
                "- **FLAG: terminal term looks drowned out by dense terms (ratio < 0.2) — "
                "the R7/R8 reward contract's preregistered watch-item — report before further training.**"
            )
            triggered_any = True
        else:
            lines.append("- Not clearly drowned out by this ratio.")

        d3 = diagnostic_3_action_statistics(records)
        lines.append("\n### 3. Action statistics")
        lines.append(f"- episodes: {d3['n_episodes']}, rounds: {d3['n_rounds']}")
        for level, stats in d3["effort_level_frequency"].items():
            lines.append(f"  - effort={level}: {stats['count']} ({stats['fraction']:.1%})")
        lines.append(f"- same-episode site revisit rate: {d3['revisit_rate']:.1%}")
        lines.append(f"- mean distinct sites visited per episode: {d3['mean_distinct_sites_per_episode']:.2f}")

    lines.append(f"\n## Overall: {'AT LEAST ONE FLAG TRIGGERED — see above, report before any further training.' if triggered_any else 'no flags triggered.'}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
