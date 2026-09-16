"""Quick comparison of eval.csv across one or more runs/<name>_<timestamp>/ dirs.

Usage:
    python scripts/compare_runs.py runs/checkpointB_v0_unattended_* runs/checkpointB_v1_* runs/checkpointB_v2_*

Prints, per run: number of eval points, latest RL/Frontier missed_fraction,
best RL missed_fraction seen so far, and a simple trend correlation over the
last N eval points so a plateau vs. still-improving run can be told apart at
a glance instead of eyeballing a CSV.
"""

import csv
import statistics
import sys
from pathlib import Path


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def summarize(run_dir: Path, recent_window: int = 20) -> None:
    eval_path = run_dir / "eval.csv"
    if not eval_path.exists():
        print(f"{run_dir.name}: no eval.csv yet")
        return

    rows = list(csv.DictReader(eval_path.open()))
    if not rows:
        print(f"{run_dir.name}: eval.csv is empty")
        return

    updates = [int(r["update"]) for r in rows]
    rl = [float(r["rl_missed_fraction"]) for r in rows]
    fr = [float(r["frontier_missed_fraction"]) for r in rows]
    rl_rounds = [float(r["rl_mean_rounds"]) for r in rows]

    recent_updates = updates[-recent_window:]
    recent_rl = rl[-recent_window:]
    trend = pearson([float(u) for u in recent_updates], recent_rl)

    print(f"=== {run_dir.name} ===")
    print(f"  eval points: {len(rows)} (up to update {updates[-1]})")
    print(f"  frontier missed_fraction: {fr[-1]:.4f} (constant)")
    print(f"  RL missed_fraction: latest={rl[-1]:.4f}  best={min(rl):.4f} (at update {updates[rl.index(min(rl))]})")
    if len(rl) >= 10:
        print(
            f"  RL missed_fraction first10 avg={statistics.mean(rl[:10]):.4f}  "
            f"last10 avg={statistics.mean(rl[-10:]):.4f}"
        )
    print(f"  RL mean_rounds: latest={rl_rounds[-1]:.1f}  (frontier={fr and rows[-1]['frontier_mean_rounds']})")
    print(f"  trend correlation (update, missed_fraction) over last {min(recent_window, len(rows))} points: {trend:.3f}")
    gap = rl[-1] - fr[-1]
    print(f"  gap to frontier (latest): {gap:+.4f} ({'RL AHEAD' if gap < 0 else 'RL behind'})")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    for pattern in sys.argv[1:]:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if path.is_dir():
                summarize(path)


if __name__ == "__main__":
    main()
