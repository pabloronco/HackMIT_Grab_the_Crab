"""R7 required learned-policy benchmark (docs/DEMU_HANDOFF_R7.md): Frontier vs
spatial-joint Information Gain vs GNN+RL on exactly the same cases, seeds,
budget, horizon and observable information, across:

  - id_test: real-graph held-out topologies, train world-model families (A/B/C)
  - ood_model_E: same held-out topologies, hidden simulator uses the
    structurally-held-out FragmentedPatchyWorldModel (E) - belief support is
    still only ever built from A/B/C, so this genuinely tests robustness to a
    generative process the planner was never told to consider
  - ood_q_low / ood_q_high: simulator q_true = 0.02 / 0.35 (configs/q_protocol_r7.json
    ood_q_mismatch), belief support unchanged at {0.05, 0.10, 0.20}

Per the handoff's explicit claim guardrail: reports mean + dispersion per
group, does not produce a winner-only score, and does not claim real-world
optimality - only performance on this formalized synthetic decision task and
robustness to the tested simulator/q assumptions.

Usage:
    python scripts/run_spatial_benchmark_r7.py --rl-checkpoint runs/.../final.pt --out reports/r7_benchmark.csv
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from adaptive_response import FrontierPlanner, InformationGainPlanner
from adaptive_response.spatial_belief import QHypothesis
from adaptive_response.world_models import (
    FragmentedPatchyWorldModel,
    WorldModelContext,
    default_world_model_split,
    sample_ecological_hypotheses,
)
from adaptive_response.rl import (
    RLSpatialPlannerAdapter,
    build_real_incident_case,
    eligible_incident_seed_sites,
    load_policy_checkpoint,
    run_spatial_benchmark_suite,
    write_spatial_benchmark_csv,
)
from adaptive_response.rl.spatial_benchmark import SpatialBenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[1]
Q_BELIEF_VALUES = (0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/r7_benchmark.csv")
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--incident-max-sites", type=int, default=20)
    parser.add_argument("--incident-preferred-min-sites", type=int, default=12)
    parser.add_argument("--draws-per-model", type=int, default=60)
    parser.add_argument("--held-out-seed-count", type=int, default=6, help="Must match the training run's --eval-holdout-seeds so ID/OOD cases stay unseen by the policy during training.")
    parser.add_argument("--cases-per-held-out-site", type=int, default=5)
    parser.add_argument("--effort-levels", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def _hypotheses_for(sites, edges, seed_site_id, *, train_models, draws_per_model, seed):
    context = WorldModelContext(tuple(sites), tuple(edges), seed_site_id)
    return sample_ecological_hypotheses(train_models, context, draws_per_model=draws_per_model, seed=seed)


def build_cases(args, rng: np.random.Generator) -> list[SpatialBenchmarkCase]:
    all_seed_sites = sorted(
        eligible_incident_seed_sites(
            max_sites=args.incident_max_sites, preferred_min_sites=args.incident_preferred_min_sites,
        )
    )
    held_out = all_seed_sites[: args.held_out_seed_count]
    train_models = default_world_model_split()["train"]
    q_hypotheses = [QHypothesis(q) for q in Q_BELIEF_VALUES]

    cases: list[SpatialBenchmarkCase] = []
    for seed_site_id in held_out:
        base_case = build_real_incident_case(
            seed_site_id, budget=args.budget,
            max_sites=args.incident_max_sites, preferred_min_sites=args.incident_preferred_min_sites,
            rl_seed=0, layout_seed=hash(seed_site_id) % (2**31 - 1),
        )
        sites, edges = base_case.incident.sites, base_case.incident.edges
        hypotheses = tuple(
            _hypotheses_for(
                sites, edges, seed_site_id, train_models=train_models,
                draws_per_model=args.draws_per_model, seed=int(rng.integers(0, 2**31 - 1)),
            )
        )

        for i in range(args.cases_per_held_out_site):
            rl_seed = int(rng.integers(0, 2**31 - 1))
            incident = build_real_incident_case(
                seed_site_id, budget=args.budget,
                max_sites=args.incident_max_sites, preferred_min_sites=args.incident_preferred_min_sites,
                rl_seed=rl_seed, layout_seed=hash(seed_site_id) % (2**31 - 1),
            ).incident

            # --- id_test: world model drawn from the same train families the belief already assumes ---
            world_model = train_models[int(rng.integers(0, len(train_models)))]
            q_true = float(Q_BELIEF_VALUES[int(rng.integers(0, len(Q_BELIEF_VALUES)))])
            cases.append(SpatialBenchmarkCase(
                label=f"id_test/{seed_site_id}/{i}", incident=incident, world_model=world_model,
                q_true=q_true, ecological_hypotheses=hypotheses, q_hypotheses=tuple(q_hypotheses),
                max_rounds=args.max_rounds, seed=rl_seed, group="id_test",
            ))

            # --- ood_model_E: hidden simulator is the structurally held-out family ---
            cases.append(SpatialBenchmarkCase(
                label=f"ood_model_E/{seed_site_id}/{i}", incident=incident, world_model=FragmentedPatchyWorldModel(),
                q_true=float(Q_BELIEF_VALUES[int(rng.integers(0, len(Q_BELIEF_VALUES)))]),
                ecological_hypotheses=hypotheses, q_hypotheses=tuple(q_hypotheses),
                max_rounds=args.max_rounds, seed=rl_seed, group="ood_model_E",
            ))

            # --- ood_q_low / ood_q_high: q mismatch, belief support unchanged ---
            world_model_q = train_models[int(rng.integers(0, len(train_models)))]
            cases.append(SpatialBenchmarkCase(
                label=f"ood_q_low/{seed_site_id}/{i}", incident=incident, world_model=world_model_q,
                q_true=0.02, ecological_hypotheses=hypotheses, q_hypotheses=tuple(q_hypotheses),
                max_rounds=args.max_rounds, seed=rl_seed, group="ood_q_low",
            ))
            cases.append(SpatialBenchmarkCase(
                label=f"ood_q_high/{seed_site_id}/{i}", incident=incident, world_model=world_model_q,
                q_true=0.35, ecological_hypotheses=hypotheses, q_hypotheses=tuple(q_hypotheses),
                max_rounds=args.max_rounds, seed=rl_seed, group="ood_q_high",
            ))
    return cases


def summarize(rows, *, group: str, planner_name: str) -> dict:
    subset = [r for r in rows if r.group == group and r.planner_name == planner_name]
    if not subset:
        return {}
    missed = [r.missed_occupied_fraction for r in subset]
    coverage = [r.occupied_site_coverage for r in subset]
    uncertainty = [r.final_global_uncertainty for r in subset]
    return {
        "n": len(subset),
        "missed_fraction_mean": statistics.mean(missed),
        "missed_fraction_stdev": statistics.pstdev(missed) if len(missed) > 1 else 0.0,
        "coverage_mean": statistics.mean(coverage),
        "coverage_stdev": statistics.pstdev(coverage) if len(coverage) > 1 else 0.0,
        "final_uncertainty_mean": statistics.mean(uncertainty),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    loaded = load_policy_checkpoint(args.rl_checkpoint)
    print(f"Loaded RL checkpoint: {args.rl_checkpoint} (update_idx={loaded.update_idx}, arch={loaded.architecture})")

    planners = {
        # effort_per_site=1, max_sites=1: Frontier has no per-effort search
        # (unlike the InformationGainPlanner effort_levels extension below), so
        # it is pinned to the smallest action-contract effort level rather than
        # left at its own defaults (effort_per_site=1, max_sites=None), which
        # would let it pick many sites in a single round - violating "same
        # action feasibility ... for every planner" (benchmark_protocol_r7.json
        # comparison_rule). This is a real, disclosed asymmetry: Frontier here
        # only ever uses effort=1, never 3 or 6.
        "frontier": FrontierPlanner(effort_per_site=min(args.effort_levels), max_sites=1),
        "information_gain": InformationGainPlanner(
            max_sites=1, require_spatial_belief=True, effort_levels=tuple(args.effort_levels)
        ),
        "gnn_rl": RLSpatialPlannerAdapter(loaded.policy),
    }

    cases = build_cases(args, rng)
    print(f"Built {len(cases)} cases across groups: "
          f"{sorted(set(c.group for c in cases))}")

    start = time.time()
    rows = run_spatial_benchmark_suite(planners, cases)
    elapsed = time.time() - start
    print(f"Ran {len(rows)} (planner, case) episodes in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.3f}s/episode).")

    out_path = REPO_ROOT / args.out
    write_spatial_benchmark_csv(rows, out_path)
    print(f"Wrote raw rows to {out_path}")

    print("\n=== SUMMARY (mean +/- stdev; NOT a ranking, NOT a real-world claim) ===")
    for group in ("id_test", "ood_model_E", "ood_q_low", "ood_q_high"):
        print(f"\n[{group}]")
        for planner_name in planners:
            s = summarize(rows, group=group, planner_name=planner_name)
            if not s:
                continue
            print(
                f"  {planner_name:16s} n={s['n']:3d}  "
                f"missed_frac={s['missed_fraction_mean']:.3f}+-{s['missed_fraction_stdev']:.3f}  "
                f"coverage={s['coverage_mean']:.3f}+-{s['coverage_stdev']:.3f}  "
                f"final_uncertainty={s['final_uncertainty_mean']:.3f}"
            )

    worst = sorted(rows, key=lambda r: -r.missed_occupied_fraction)[:5]
    print("\n=== Visible failure cases (highest missed_occupied_fraction) ===")
    for r in worst:
        print(f"  {r.planner_name:16s} {r.case_label:28s} group={r.group:12s} "
              f"missed_frac={r.missed_occupied_fraction:.3f} occupied_total={r.occupied_sites_total} "
              f"rounds={r.num_rounds} effort={r.effort_spent}")

    print(
        "\nClaim guardrail: results demonstrate performance on this formalized synthetic "
        "decision task and robustness to the tested simulator/q assumptions only - not "
        "proven field effectiveness, and this is not a winner-only score."
    )


if __name__ == "__main__":
    main()
