"""Checkpoint-B-territory training entry point (PROPOSED action space/reward,
see docs/DECISION_LOG.md "GNN/RL backbone v0" and the follow-up training
entry). Trains the RoundPolicy actor-critic against the M1 toy Environment
with randomized incidents, logging metrics/checkpoints and periodically
comparing against FrontierPlanner on a fixed held-out eval set so progress
(or a plateau) is visible without waiting for the run to finish.

Usage:
    python scripts/train_gnn_policy.py --run-name my_run --max-hours 7.5

Resume is not implemented for v0: a run is self-contained. Stop any time with
Ctrl+C -- a final checkpoint is saved on interrupt.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from adaptive_response import FrontierPlanner
from adaptive_response.rl import (
    ActorCriticTrainer,
    GNNActorCritic,
    IncidentSamplerConfig,
    RewardConfig,
    RLPlannerAdapter,
    RoundPolicy,
    TrainerConfig,
    run_episode,
    run_planner_episode,
    sample_incident,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--effort-per-pick", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-final", type=float, default=0.01, help="If different from --entropy-coef, linearly decayed to over --entropy-coef-decay-updates.")
    parser.add_argument("--entropy-coef-decay-updates", type=int, default=0, help="0 disables decay (entropy_coef stays constant).")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads(); leave unset for torch's default. Set low (e.g. 2) when running several training processes concurrently on one CPU.")

    parser.add_argument("--uncertainty-reduction-weight", type=float, default=2.0, help="RewardConfig alpha: per-round reward for reducing mean belief uncertainty.")
    parser.add_argument("--detection-weight", type=float, default=1.0, help="RewardConfig beta: per-round reward per detection.")
    parser.add_argument("--effort-cost-weight", type=float, default=0.02, help="RewardConfig lambda: per-round cost per effort unit spent.")
    parser.add_argument("--missed-extent-weight", type=float, default=1.0, help="RewardConfig eta: terminal penalty per truly-occupied site never detected.")

    parser.add_argument("--min-sites", type=int, default=12)
    parser.add_argument("--max-sites", type=int, default=24)
    parser.add_argument("--min-budget", type=int, default=20)
    parser.add_argument("--max-budget", type=int, default=40)

    parser.add_argument("--max-hours", type=float, default=7.5)
    parser.add_argument("--max-updates", type=int, default=None, help="Overrides --max-hours if set.")
    parser.add_argument("--eval-every-updates", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-every-updates", type=int, default=20)
    return parser.parse_args()


def make_run_dir(log_dir: str, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / log_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(path: Path, policy: RoundPolicy, trainer: ActorCriticTrainer, update_idx: int) -> None:
    torch.save(
        {
            "update_idx": update_idx,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        },
        path,
    )


def run_eval(policy: RoundPolicy, eval_incidents: list, eval_seeds: list[int]) -> dict:
    rl_adapter = RLPlannerAdapter(policy)
    frontier = FrontierPlanner()

    rl_missed, rl_effort, rl_rounds = [], [], []
    frontier_missed, frontier_effort, frontier_rounds = [], [], []

    for incident, seed in zip(eval_incidents, eval_seeds):
        rl_metrics = run_planner_episode(rl_adapter, incident, seed=seed)
        frontier_metrics = run_planner_episode(frontier, incident, seed=seed)

        rl_missed.append(rl_metrics.occupied_sites_missed / max(rl_metrics.occupied_sites_total, 1))
        rl_effort.append(rl_metrics.effort_spent)
        rl_rounds.append(rl_metrics.num_rounds)
        frontier_missed.append(
            frontier_metrics.occupied_sites_missed / max(frontier_metrics.occupied_sites_total, 1)
        )
        frontier_effort.append(frontier_metrics.effort_spent)
        frontier_rounds.append(frontier_metrics.num_rounds)

    return {
        "rl_missed_fraction": float(np.mean(rl_missed)),
        "rl_mean_rounds": float(np.mean(rl_rounds)),
        "frontier_missed_fraction": float(np.mean(frontier_missed)),
        "frontier_mean_rounds": float(np.mean(frontier_rounds)),
    }


def main() -> None:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    run_dir = make_run_dir(args.log_dir, args.run_name)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run directory: {run_dir}")

    torch.manual_seed(args.seed)
    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(999_999)  # fixed, independent of --seed and of training draws

    sampler_config = IncidentSamplerConfig(
        min_sites=args.min_sites,
        max_sites=args.max_sites,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
    )
    eval_incidents = [sample_incident(eval_rng, sampler_config) for _ in range(args.eval_episodes)]
    eval_seeds = [int(eval_rng.integers(0, 2**31 - 1)) for _ in range(args.eval_episodes)]

    backbone = GNNActorCritic(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    policy = RoundPolicy(backbone, hidden_dim=args.hidden_dim, effort_per_pick=args.effort_per_pick)
    trainer = ActorCriticTrainer(
        policy,
        TrainerConfig(
            lr=args.lr,
            gamma=args.gamma,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            entropy_coef_final=args.entropy_coef_final,
            entropy_coef_decay_updates=args.entropy_coef_decay_updates,
            max_grad_norm=args.max_grad_norm,
        ),
    )
    reward_config = RewardConfig(
        uncertainty_reduction_weight=args.uncertainty_reduction_weight,
        detection_weight=args.detection_weight,
        effort_cost_weight=args.effort_cost_weight,
        missed_extent_weight=args.missed_extent_weight,
    )

    metrics_path = run_dir / "metrics.csv"
    eval_path = run_dir / "eval.csv"
    metrics_fields = [
        "update", "elapsed_s", "loss", "policy_loss", "value_loss", "entropy",
        "entropy_coef_used", "mean_return", "mean_advantage", "mean_rounds",
        "mean_detections", "mean_effort", "mean_missed_fraction",
    ]
    eval_fields = [
        "update", "elapsed_s", "rl_missed_fraction", "rl_mean_rounds",
        "frontier_missed_fraction", "frontier_mean_rounds",
    ]
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(metrics_fields)
    with eval_path.open("w", newline="") as f:
        csv.writer(f).writerow(eval_fields)

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

            rollouts = [
                run_episode(
                    policy,
                    sample_incident(train_rng, sampler_config),
                    reward_config=reward_config,
                    seed=int(train_rng.integers(0, 2**31 - 1)),
                )
                for _ in range(args.episodes_per_update)
            ]
            stats = trainer.update(rollouts)
            update_idx += 1
            elapsed = time.time() - start_time

            mean_rounds = float(np.mean([r.num_rounds for r in rollouts]))
            mean_detections = float(np.mean([r.detections_found for r in rollouts]))
            mean_effort = float(np.mean([r.effort_spent for r in rollouts]))
            missed_fractions = [
                r.occupied_sites_missed / max(r.occupied_sites_total, 1) for r in rollouts
            ]
            mean_missed_fraction = float(np.mean(missed_fractions))

            with metrics_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    update_idx, f"{elapsed:.1f}", stats.loss, stats.policy_loss,
                    stats.value_loss, stats.entropy, stats.entropy_coef_used,
                    stats.mean_episode_return, stats.mean_advantage, mean_rounds,
                    mean_detections, mean_effort, mean_missed_fraction,
                ])

            print(
                f"[update {update_idx:5d} | {elapsed/3600:5.2f}h] "
                f"return={stats.mean_episode_return:+7.3f} "
                f"missed_frac={mean_missed_fraction:.3f} "
                f"loss={stats.loss:+7.4f} entropy={stats.entropy:.3f} "
                f"(coef={stats.entropy_coef_used:.4f})"
            )

            if update_idx % args.eval_every_updates == 0:
                eval_stats = run_eval(policy, eval_incidents, eval_seeds)
                with eval_path.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        update_idx, f"{elapsed:.1f}", eval_stats["rl_missed_fraction"],
                        eval_stats["rl_mean_rounds"], eval_stats["frontier_missed_fraction"],
                        eval_stats["frontier_mean_rounds"],
                    ])
                print(
                    f"    EVAL: RL missed_frac={eval_stats['rl_missed_fraction']:.3f} "
                    f"(rounds={eval_stats['rl_mean_rounds']:.1f}) vs "
                    f"Frontier missed_frac={eval_stats['frontier_missed_fraction']:.3f} "
                    f"(rounds={eval_stats['frontier_mean_rounds']:.1f})"
                )

            if update_idx % args.checkpoint_every_updates == 0:
                save_checkpoint(run_dir / f"checkpoint_{update_idx}.pt", policy, trainer, update_idx)
                save_checkpoint(run_dir / "latest.pt", policy, trainer, update_idx)
    finally:
        save_checkpoint(run_dir / "final.pt", policy, trainer, update_idx)
        print(f"Saved final checkpoint at update {update_idx} to {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
