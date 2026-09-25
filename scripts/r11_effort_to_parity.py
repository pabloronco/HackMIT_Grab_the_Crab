"""R11 Effort-to-Parity: how much extra field effort would Dynamic
Delimitation need to match Grab the Crab GNN/RL's confirmed-unique-positive-
sites result at the original 18-unit budget?
(configs/benchmark_protocol_r11_dynamic.json effort_to_parity)

For each frozen formal case and each GNN seed:
  target = GNN confirmed_unique_positive_sites at effort=18 (from
           reports/r11_benchmark_seed{N}.csv)
  Continue ONLY Dynamic Delimitation - same rules, same fixed effort=6,
  same information firewall, unchanged - at budget in {18, 24, 30, 36}
  (each a fresh run of the same deterministic planner on the same case/seed,
  which is equivalent to one continuous run since Dynamic's first 3 rounds
  are identical regardless of the eventual budget cap).

  E*_dyn = min{E in {18,24,30,36} : confirmed_unique(Dynamic, E) >= target}
  ExtraEffort = E*_dyn - 18 (0 if Dynamic already >= target at E=18)
  EffortOverhead% = ExtraEffort / 18 * 100
  parity_reached = False, "not reached within 2x", if no such E <= 36 exists.

Usage:
    python scripts/r11_effort_to_parity.py \
        --gnn-csv reports/r11_benchmark_seed0.csv reports/r11_benchmark_seed1.csv reports/r11_benchmark_seed2.csv \
        --out-csv reports/r11_effort_to_parity.csv --out-md docs/R11_EFFORT_TO_PARITY.md
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from adaptive_response import DynamicDelimitationMax6Planner
from adaptive_response.rl.r8_manifest_cases import (
    FORMAL_REPORTING_SPLITS,
    load_frozen_manifest,
    manifest_case_to_benchmark_case,
)
from adaptive_response.rl.spatial_benchmark import run_spatial_planner_case

PARITY_STEPS = (18, 24, 30, 36)  # frozen_resource_contract / effort_to_parity.continuation_effort_steps


def confirmed_unique_from_row(row) -> int:
    return row.occupied_sites_total - row.occupied_sites_missed


_SUPPORT_FAILURE_MESSAGE = "zero probability under every spatial/q hypothesis"


def dynamic_confirmed_at_each_budget(spec: dict) -> dict[int, int | None]:
    """None marks a budget level (and, by construction, every higher one -
    Dynamic is deterministic, so a higher-budget run replays the exact same
    early rounds) hit by a genuine finite-ensemble support failure: the
    frozen case's belief-ensemble draws were sized/validated for the
    original max_rounds=6/budget=18 R8 contract, not for continuing Dynamic
    into rounds 4-6 here. This is the same real, documented phenomenon
    scripts/train_spatial_gnn_policy.py retries during TRAINING (freshly
    resampled, non-frozen episodes) but - per that module's own docstring -
    a frozen case must never be silently resampled. So this is recorded and
    reported, not hidden or retried."""
    planner = DynamicDelimitationMax6Planner()
    result: dict[int, int | None] = {}
    failed = False
    for budget in PARITY_STEPS:
        if failed:
            result[budget] = None
            continue
        case = manifest_case_to_benchmark_case(
            spec, budget=budget, max_rounds=budget // 6,
        )
        try:
            row = run_spatial_planner_case(planner, case)
        except ValueError as exc:
            if _SUPPORT_FAILURE_MESSAGE not in str(exc):
                raise
            failed = True
            result[budget] = None
            continue
        result[budget] = confirmed_unique_from_row(row)
    return result


def read_gnn_targets(paths: list[Path]) -> list[dict[str, int]]:
    """One dict per seed: case_label -> confirmed_unique_positive_sites at effort 18."""
    targets = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_case = {
            r["case_label"]: int(r["occupied_sites_total"]) - int(r["occupied_sites_missed"])
            for r in rows if r["planner_name"] == "gnn_rl"
        }
        targets.append(by_case)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-csv", type=str, nargs="+", required=True)
    parser.add_argument("--out-csv", type=str, default="reports/r11_effort_to_parity.csv")
    parser.add_argument("--out-md", type=str, default="docs/R11_EFFORT_TO_PARITY.md")
    args = parser.parse_args()

    manifest = load_frozen_manifest()
    specs = {c["case_id"]: c for c in manifest["cases"] if c["split"] in FORMAL_REPORTING_SPLITS}
    print(f"Loaded {len(specs)} formal case specs.")

    gnn_targets = read_gnn_targets([Path(p) for p in args.gnn_csv])
    seed_names = [f"seed{i}" for i in range(len(gnn_targets))]

    t0 = time.time()
    dynamic_by_case: dict[str, dict[int, int]] = {}
    for i, (case_id, spec) in enumerate(specs.items()):
        dynamic_by_case[case_id] = dynamic_confirmed_at_each_budget(spec)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(specs)} cases continued ({time.time() - t0:.1f}s elapsed)")
    print(f"Continued Dynamic Delimitation on all {len(specs)} cases in {time.time() - t0:.1f}s.")

    parity_rows: list[dict] = []
    for case_id, spec in specs.items():
        dyn = dynamic_by_case[case_id]
        for seed_name, targets in zip(seed_names, gnn_targets):
            target = targets[case_id]
            dyn18 = dyn[18]
            if dyn18 is None:
                # Support failure at the very first checkpoint (budget=18):
                # should not happen (this exact case/budget already ran
                # cleanly in run_r11_dynamic_benchmark.py), but handled
                # rather than assumed away.
                parity_rows.append({
                    "case_id": case_id, "group": spec["split"], "seed": seed_name,
                    "gnn_confirmed_unique_at_18": target,
                    "dynamic_confirmed_unique_at_18": "", "dynamic_confirmed_unique_at_24": "",
                    "dynamic_confirmed_unique_at_30": "", "dynamic_confirmed_unique_at_36": "",
                    "status_at_18": "support_failure_at_18",
                    "parity_effort": "", "extra_effort_to_parity": "", "effort_overhead_pct": "",
                    "parity_reached": False,
                })
                continue
            if dyn18 >= target:
                status = "dynamic_ahead_or_tied_at_18"
                e_star = 18
            else:
                e_star = next((b for b in PARITY_STEPS if dyn[b] is not None and dyn[b] >= target), None)
                status = "gnn_ahead_at_18"
                if e_star is None and any(dyn[b] is None for b in PARITY_STEPS):
                    status = "gnn_ahead_at_18_support_failure_before_36"
            extra_effort = (e_star - 18) if e_star is not None else None
            overhead_pct = (extra_effort / 18 * 100) if extra_effort is not None else None
            parity_rows.append({
                "case_id": case_id, "group": spec["split"], "seed": seed_name,
                "gnn_confirmed_unique_at_18": target,
                "dynamic_confirmed_unique_at_18": dyn18,
                "dynamic_confirmed_unique_at_24": dyn[24] if dyn[24] is not None else "",
                "dynamic_confirmed_unique_at_30": dyn[30] if dyn[30] is not None else "",
                "dynamic_confirmed_unique_at_36": dyn[36] if dyn[36] is not None else "",
                "status_at_18": status,
                "parity_effort": e_star if e_star is not None else "",
                "extra_effort_to_parity": extra_effort if extra_effort is not None else "",
                "effort_overhead_pct": f"{overhead_pct:.1f}" if overhead_pct is not None else "",
                "parity_reached": e_star is not None,
            })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(parity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(parity_rows)
    print(f"Wrote {args.out_csv}")

    # Aggregate report
    lines = ["# R11 Effort-to-Parity: Dynamic Delimitation vs Grab the Crab GNN/RL", ""]
    lines.append(
        "How much MORE field effort (beyond the initial 18-unit budget, in e6 blocks) "
        "would Dynamic Delimitation need to match the GNN/RL policy's confirmed-unique-"
        "positive-sites result at effort 18? Parity target and continuation both use "
        "confirmed_unique_positive_sites, not raw detection-event counts."
    )
    lines.append("")
    lines.append(f"Cap: {max(PARITY_STEPS)} (2x initial budget). Steps: {PARITY_STEPS}.")
    lines.append("")

    gnn_ahead_statuses = {"gnn_ahead_at_18", "gnn_ahead_at_18_support_failure_before_36"}

    for seed_name in seed_names:
        seed_rows = [r for r in parity_rows if r["seed"] == seed_name]
        n = len(seed_rows)
        n_support_failure_18 = sum(1 for r in seed_rows if r["status_at_18"] == "support_failure_at_18")
        n_gnn_ahead = sum(1 for r in seed_rows if r["status_at_18"] in gnn_ahead_statuses)
        n_tied_or_dyn_ahead = n - n_gnn_ahead - n_support_failure_18
        n_tied = sum(
            1 for r in seed_rows
            if r["status_at_18"] == "dynamic_ahead_or_tied_at_18"
            and r["dynamic_confirmed_unique_at_18"] == r["gnn_confirmed_unique_at_18"]
        )
        n_dyn_strictly_ahead = n_tied_or_dyn_ahead - n_tied

        gnn_ahead_rows = [r for r in seed_rows if r["status_at_18"] in gnn_ahead_statuses]
        reached = [r for r in gnn_ahead_rows if r["parity_reached"]]
        not_reached = [r for r in gnn_ahead_rows if not r["parity_reached"]]
        n_support_failure_before_36 = sum(
            1 for r in gnn_ahead_rows if r["status_at_18"] == "gnn_ahead_at_18_support_failure_before_36"
        )
        extra_efforts = [r["extra_effort_to_parity"] for r in reached]

        lines.append(f"## {seed_name}")
        lines.append("")
        lines.append(f"- n cases = {n}")
        if n_support_failure_18:
            lines.append(
                f"- excluded, finite-ensemble support failure already at effort 18: {n_support_failure_18} "
                "(see docs/R11_EFFORT_TO_PARITY.md caveat below)"
            )
        lines.append(f"- GNN ahead at effort 18: {n_gnn_ahead} ({n_gnn_ahead / n * 100:.1f}%)")
        lines.append(f"- Tie at effort 18: {n_tied} ({n_tied / n * 100:.1f}%)")
        lines.append(f"- Dynamic ahead at effort 18: {n_dyn_strictly_ahead} ({n_dyn_strictly_ahead / n * 100:.1f}%)")
        lines.append("")
        if gnn_ahead_rows:
            lines.append(f"Among the {len(gnn_ahead_rows)} cases where GNN was ahead at effort 18:")
            if extra_efforts:
                lines.append(f"- median extra effort to parity = {statistics.median(extra_efforts)}")
                lines.append(f"- mean extra effort to parity = {statistics.mean(extra_efforts):.2f}")
            for step in (24, 30, 36):
                n_by_step = sum(1 for r in reached if r["parity_effort"] <= step)
                lines.append(f"- parity reached by e{step}: {n_by_step}/{len(gnn_ahead_rows)} ({n_by_step / len(gnn_ahead_rows) * 100:.1f}%)")
            lines.append(
                f"- not reached by e36: {len(not_reached)}/{len(gnn_ahead_rows)} "
                f"({len(not_reached) / len(gnn_ahead_rows) * 100:.1f}%) - "
                "\"Did not reach parity within 2x the initial effort budget.\""
            )
            if n_support_failure_before_36:
                lines.append(
                    f"  - of which {n_support_failure_before_36} hit a finite-ensemble support "
                    "failure before e36 rather than a genuine non-parity (the belief-ensemble "
                    "draws for these frozen cases were sized for the original max_rounds=6 "
                    "contract, not for continuing 4-6 rounds deep)"
                )
        else:
            lines.append("GNN was never ahead of Dynamic at effort 18 for this seed - no parity gap to close.")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
