"""R7 training entry point (docs/DEMU_HANDOFF_R7.md): trains the R7
`SiteEffortRoundPolicy` actor-critic against the real monitoring graph
topology through `SpatialAdaptiveMissionLoop` - spatial belief + uncertain q
-> GraphState -> planner -> MissionAction -> hidden simulator -> observation
-> posterior -> replan - with the ACKed (site, effort) action contract
(effort in {1,3,6}, budget 18, horizon 6 by default).

Cases are drawn from the real 49-site/118-edge graph
(reports/milestones/r2_real_graph_v0/real_graph_v0_edges.csv), not the older
toy `sample_incident` generator. Site x/y and habitat_score are placeholders
(deterministic graph layout / neutral constant) because the raw Dryad
coordinate/habitat CSVs are not fetchable from this environment - see
adaptive_response.rl.real_graph_cases module docstring. This does not affect
the graph TOPOLOGY, which is real.

Usage:
    python scripts/train_spatial_gnn_policy.py --run-name my_run --max-updates 50
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from adaptive_response import FrontierPlanner, InformationGainPlanner
from adaptive_response.spatial_belief import QHypothesis
from adaptive_response.world_models import default_world_model_split, sample_ecological_hypotheses
from adaptive_response.rl import (
    ActorCriticTrainer,
    JsonlDecisionLogger,
    RLSpatialPlannerAdapter,
    SiteEffortPolicyArchitectureConfig,
    SpatialRewardConfig,
    TrainerConfig,
    build_real_incident_case,
    eligible_incident_seed_sites,
    run_spatial_episode,
    run_spatial_planner_case,
    save_policy_checkpoint,
)
from adaptive_response.rl.spatial_benchmark import SpatialBenchmarkCase

REPO_ROOT = Path(__file__).resolve().parents[1]

# q_protocol_r7.json benchmark_design_support (FROZEN benchmark scenario
# support, not an ecological estimate).
Q_BELIEF_VALUES = (0.05, 0.10, 0.20)
Q_TRAIN_TRUE_VALUES = (0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--effort-levels", type=int, nargs="+", default=[1, 3, 6])

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=None)

    parser.add_argument("--mean-uncertainty-reduction-weight", type=float, default=2.0)
    parser.add_argument("--new-detection-weight", type=float, default=0.5)
    parser.add_argument("--missed-occupied-fraction-weight", type=float, default=-2.0)

    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--incident-max-sites", type=int, default=20)
    parser.add_argument("--incident-preferred-min-sites", type=int, default=12)
    parser.add_argument("--draws-per-model", type=int, default=60, help="Ecological-hypothesis samples per train world-model family per episode.")
    parser.add_argument("--eval-holdout-seeds", type=int, default=6, help="Number of eligible seed sites reserved for periodic eval, held out from training draws.")

    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--eval-every-updates", type=int, default=5)
    parser.add_argument("--checkpoint-every-updates", type=int, default=10)
    parser.add_argument("--decision-log", action="store_true")
    return parser.parse_args()


def make_run_dir(log_dir: str, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / log_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_case_ingredients(seed_site_id: str, *, args, rl_rng: np.random.Generator):
    """One training episode's incident + world model + q_true + belief support."""

    train_models = default_world_model_split()["train"]
    world_model = train_models[int(rl_rng.integers(0, len(train_models)))]
    q_true = float(Q_TRAIN_TRUE_VALUES[int(rl_rng.integers(0, len(Q_TRAIN_TRUE_VALUES)))])

    case = build_real_incident_case(
        seed_site_id,
        budget=args.budget,
        max_sites=args.incident_max_sites,
        preferred_min_sites=args.incident_preferred_min_sites,
        rl_seed=int(rl_rng.integers(0, 2**31 - 1)),
    )
    context_sites = tuple(case.incident.sites)
    context_edges = tuple(case.incident.edges)
    from adaptive_response.world_models import WorldModelContext

    context = WorldModelContext(context_sites, context_edges, seed_site_id)
    hypotheses = sample_ecological_hypotheses(
        train_models, context, draws_per_model=args.draws_per_model,
        seed=int(rl_rng.integers(0, 2**31 - 1)),
    )
    q_hypotheses = [QHypothesis(q) for q in Q_BELIEF_VALUES]
    return case.incident, world_model, q_true, hypotheses, q_hypotheses


def run_eval(policy, eval_seed_sites: list[str], *, args, rng: np.random.Generator) -> dict:
    rl_adapter = RLSpatialPlannerAdapter(policy)
    # R8 baseline-fairness correction (docs/R8_DEMU_BENCHMARK_REVIEW.md,
    # configs/benchmark_protocol_r8.json): Frontier uses the standard-event
    # effort (largest feasible level) instead of a fixed small effort_per_site,
    # and Information Gain ranks by absolute expected entropy reduction
    # (IG-per-effort only a tie-break) - both corrected so this periodic eval
    # is not misleading during retraining, same rules as the formal benchmark.
    frontier = FrontierPlanner(max_sites=1, effort_levels=tuple(args.effort_levels))
    ig = InformationGainPlanner(max_sites=1, require_spatial_belief=True, effort_levels=tuple(args.effort_levels))

    rows_by_planner: dict[str, list] = {"rl": [], "frontier": [], "information_gain": []}
    for seed_site_id in eval_seed_sites:
        incident, world_model, q_true, hypotheses, q_hypotheses = build_case_ingredients(
            seed_site_id, args=args, rl_rng=rng,
        )
        case = SpatialBenchmarkCase(
            label=f"eval/{seed_site_id}", incident=incident, world_model=world_model, q_true=q_true,
            ecological_hypotheses=tuple(hypotheses), q_hypotheses=tuple(q_hypotheses),
            max_rounds=args.max_rounds, seed=incident.seed, group="id_eval",
        )
        rows_by_planner["rl"].append(run_spatial_planner_case(rl_adapter, case))
        rows_by_planner["frontier"].append(run_spatial_planner_case(frontier, case))
        rows_by_planner["information_gain"].append(run_spatial_planner_case(ig, case))

    result = {}
    for name, rows in rows_by_planner.items():
        result[f"{name}_missed_fraction"] = float(np.mean([r.missed_occupied_fraction for r in rows]))
        result[f"{name}_coverage"] = float(np.mean([r.occupied_site_coverage for r in rows]))
        result[f"{name}_final_uncertainty"] = float(np.mean([r.final_global_uncertainty for r in rows]))
    return result


# R8 correction (review 2026-09-15): renamed from the earlier
# "zero-likelihood"/"numerical edge case" framing. The actual cause is not a
# floating-point/numerical issue: a FINITE Monte Carlo sample of ecological
# hypotheses (draws_per_model per train family) does not exhaustively cover
# every possible occupancy pattern, so the hidden simulator's realized
# observation can fall outside the support of the sampled hypothesis set.
# That is a finite-ensemble support failure - precise language matters here
# because "numerical edge case" reads as something perturbation/precision
# tuning could fix, which it cannot; only a larger/different hypothesis
# sample changes the support.
_FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT = 0


def _run_episode_with_retry(
    policy, train_seed_sites: list[str], *, args, rl_rng: np.random.Generator,
    reward_config, decision_logger, episode_index: int, max_attempts: int = 5,
):
    """Resample and retry on a rare, disclosed finite-ensemble support failure.

    A finite Monte Carlo sample of ecological hypotheses (draws_per_model per
    train family) does not exhaustively cover every possible occupancy
    pattern. If the hidden simulator's realized world/observation happens to
    fall outside the support of every sampled hypothesis,
    SpatialBeliefEngine.update() raises ValueError("...zero probability
    under every spatial/q hypothesis..."). This is a real, occasionally-hit
    finite-ensemble support failure (observed once in ~14,000+ training
    rounds during the R8 retrain), not a bug in the episode itself and not a
    generic numerical edge case - retrying with a freshly resampled
    incident/world/hypotheses is safe here because training episodes are
    randomly drawn each time anyway (unlike frozen benchmark cases, where
    this must never be silently retried - see
    scripts/run_spatial_benchmark_r8.py, which surfaces it instead).
    Occurrence count is tracked in
    _FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT for reporting.
    """

    global _FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT
    for attempt in range(max_attempts):
        seed_site_id = train_seed_sites[int(rl_rng.integers(0, len(train_seed_sites)))]
        incident, world_model, q_true, hypotheses, q_hypotheses = build_case_ingredients(
            seed_site_id, args=args, rl_rng=rl_rng,
        )
        try:
            return run_spatial_episode(
                policy, incident, world_model=world_model, q_true=q_true,
                ecological_hypotheses=hypotheses, q_hypotheses=q_hypotheses,
                max_rounds=args.max_rounds, reward_config=reward_config,
                seed=int(rl_rng.integers(0, 2**31 - 1)),
                decision_logger=decision_logger, episode_index=episode_index,
            )
        except ValueError as exc:
            if "zero probability under every spatial/q hypothesis" not in str(exc):
                raise
            _FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT += 1
            print(
                f"    [warn] finite-ensemble support failure on {seed_site_id!r} "
                f"(attempt {attempt + 1}/{max_attempts}), resampling and retrying. "
                f"Total occurrences this run: {_FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT}."
            )
    raise RuntimeError(
        f"Gave up after {max_attempts} finite-ensemble support failure retries in a row - "
        "this is no longer a rare edge case, stop and investigate."
    )


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    run_dir = make_run_dir(args.log_dir, args.run_name)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run directory: {run_dir}")

    torch.manual_seed(args.seed)
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(999_999)

    all_seed_sites = eligible_incident_seed_sites(
        max_sites=args.incident_max_sites, preferred_min_sites=args.incident_preferred_min_sites,
    )
    all_seed_sites_sorted = sorted(all_seed_sites)
    eval_seed_sites = all_seed_sites_sorted[: args.eval_holdout_seeds]
    train_seed_sites = all_seed_sites_sorted[args.eval_holdout_seeds :]
    print(
        f"Real graph v0: {len(all_seed_sites_sorted)} eligible incident seeds "
        f"({args.incident_preferred_min_sites}-{args.incident_max_sites} sites); "
        f"{len(train_seed_sites)} for training, {len(eval_seed_sites)} held out for eval."
    )

    architecture = SiteEffortPolicyArchitectureConfig(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, effort_levels=tuple(args.effort_levels),
    )
    policy = architecture.build()
    trainer = ActorCriticTrainer(
        policy,
        TrainerConfig(
            lr=args.lr, gamma=args.gamma, value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
        ),
    )
    reward_config = SpatialRewardConfig(
        mean_uncertainty_reduction_weight=args.mean_uncertainty_reduction_weight,
        new_detection_weight=args.new_detection_weight,
        missed_occupied_fraction_weight=args.missed_occupied_fraction_weight,
    )

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"
    metrics_fields = [
        "update", "elapsed_s", "loss", "policy_loss", "value_loss", "entropy",
        "mean_return", "mean_advantage", "mean_rounds", "mean_detections",
        "mean_effort", "mean_missed_fraction",
    ]
    eval_fields = [
        "update", "elapsed_s",
        "rl_missed_fraction", "rl_coverage", "rl_final_uncertainty",
        "frontier_missed_fraction", "frontier_coverage", "frontier_final_uncertainty",
        "information_gain_missed_fraction", "information_gain_coverage", "information_gain_final_uncertainty",
    ]
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(metrics_fields)
    with eval_path.open("w", newline="") as f:
        csv.writer(f).writerow(eval_fields)

    decision_logger = JsonlDecisionLogger(run_dir / "decisions.jsonl") if args.decision_log else None
    episode_counter = 0

    start_time = time.time()
    max_seconds = args.max_hours * 3600.0
    update_idx = 0
    stop_requested = False

    def handle_sigint(signum, frame) -> None:
        nonlocal stop_requested
        print("\nInterrupt received; finishing current update then saving and exiting.")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_requested:
            if args.max_updates is not None and update_idx >= args.max_updates:
                break
            if args.max_updates is None and (time.time() - start_time) >= max_seconds:
                break

            rollouts = []
            for _ in range(args.episodes_per_update):
                rollouts.append(
                    _run_episode_with_retry(
                        policy, train_seed_sites, args=args, rl_rng=train_rng,
                        reward_config=reward_config, decision_logger=decision_logger,
                        episode_index=episode_counter,
                    )
                )
                episode_counter += 1
            stats = trainer.update(rollouts)
            update_idx += 1
            elapsed = time.time() - start_time

            mean_rounds = float(np.mean([r.num_rounds for r in rollouts]))
            mean_detections = float(np.mean([r.detections_found for r in rollouts]))
            mean_effort = float(np.mean([r.effort_spent for r in rollouts]))
            mean_missed_fraction = float(np.mean([r.metrics.missed_occupied_fraction for r in rollouts]))

            with metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    update_idx, f"{elapsed:.1f}", stats.loss, stats.policy_loss,
                    stats.value_loss, stats.entropy, stats.mean_episode_return,
                    stats.mean_advantage, mean_rounds, mean_detections, mean_effort,
                    mean_missed_fraction,
                ])
            print(
                f"[update {update_idx:4d} | {elapsed/3600:5.2f}h] "
                f"return={stats.mean_episode_return:+7.3f} missed_frac={mean_missed_fraction:.3f} "
                f"loss={stats.loss:+7.4f} entropy={stats.entropy:.3f} rounds={mean_rounds:.1f}"
            )

            if update_idx % args.eval_every_updates == 0:
                eval_stats = run_eval(policy, eval_seed_sites, args=args, rng=eval_rng)
                with eval_path.open("a", newline="") as f:
                    csv.writer(f).writerow([update_idx, f"{elapsed:.1f}"] + [eval_stats[k] for k in eval_fields[2:]])
                print(
                    f"    EVAL missed_frac: RL={eval_stats['rl_missed_fraction']:.3f} "
                    f"Frontier={eval_stats['frontier_missed_fraction']:.3f} "
                    f"InfoGain={eval_stats['information_gain_missed_fraction']:.3f}"
                )

            if update_idx % args.checkpoint_every_updates == 0:
                save_policy_checkpoint(
                    run_dir / f"checkpoint_{update_idx}.pt", policy, architecture=architecture,
                    update_idx=update_idx, optimizer=trainer.optimizer,
                    extra={"run_name": args.run_name, "seed": args.seed},
                )
                save_policy_checkpoint(
                    run_dir / "latest.pt", policy, architecture=architecture,
                    update_idx=update_idx, optimizer=trainer.optimizer,
                    extra={"run_name": args.run_name, "seed": args.seed},
                )
    finally:
        save_policy_checkpoint(
            run_dir / "final.pt", policy, architecture=architecture,
            update_idx=update_idx, optimizer=trainer.optimizer,
            extra={"run_name": args.run_name, "seed": args.seed},
        )
        print(f"Saved final checkpoint at update {update_idx} to {run_dir / 'final.pt'}")
        print(f"Finite-ensemble support failure retry count this run: {_FINITE_ENSEMBLE_SUPPORT_FAILURE_RETRY_COUNT}")
        if decision_logger is not None:
            decision_logger.close()


if __name__ == "__main__":
    main()
