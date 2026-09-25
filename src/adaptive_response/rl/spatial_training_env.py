from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..environment import Environment
from ..models import IncidentConfig, MissionAction
from ..spatial_belief import EcologicalHypothesis, QHypothesis
from ..spatial_mission_loop import SpatialAdaptiveMissionLoop
from ..world_models import WorldModel
from .decision_logger import JsonlDecisionLogger, SpatialRoundLogRecord
from .reward import (
    SpatialRewardConfig,
    spatial_round_reward_components,
    spatial_terminal_reward_components,
)
from .site_effort_policy import SiteEffortRoundPolicy
from .spatial_metrics import SpatialPrimaryMetrics, compute_spatial_primary_metrics


@dataclass
class SpatialEpisodeRollout:
    """Everything the trainer/reporter needs from one full R7 episode."""

    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    num_rounds: int = 0
    detections_found: int = 0
    effort_spent: int = 0
    metrics: SpatialPrimaryMetrics | None = None


class _StashingPlanner:
    """Adapts `SiteEffortRoundPolicy` to the `Planner` protocol.

    Same reasoning as the older `training_env._StashingPlanner`:
    `SpatialAdaptiveMissionLoop.execute_pending()` internally calls
    `plan_next()` again for the *next* round as soon as the current one isn't
    done, so the caller must read `last_decision` immediately after its own
    explicit `plan_next()` call and before calling `execute_pending()` -
    otherwise the lookahead call silently overwrites it with the wrong
    round's decision.
    """

    def __init__(self, policy: SiteEffortRoundPolicy, *, deterministic: bool) -> None:
        self._policy = policy
        self._deterministic = deterministic
        self.last_decision = None

    def plan(self, graph_state, remaining_budget, constraints) -> MissionAction:
        del constraints  # policy only ever sees GraphState; q/spatial belief stay unused here.
        self.last_decision = self._policy.act(
            graph_state, remaining_budget, deterministic=self._deterministic
        )
        return self.last_decision.mission


def run_spatial_episode(
    policy: SiteEffortRoundPolicy,
    incident_config: IncidentConfig,
    *,
    world_model: WorldModel,
    q_true: float,
    ecological_hypotheses: Sequence[EcologicalHypothesis],
    q_hypotheses: Sequence[QHypothesis],
    max_rounds: int,
    reward_config: SpatialRewardConfig | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    decision_logger: JsonlDecisionLogger | None = None,
    episode_index: int = 0,
) -> SpatialEpisodeRollout:
    """Run one R7 formal incident end to end: spatial belief + uncertain q ->
    GraphState -> planner -> MissionAction -> hidden simulator -> observation
    -> posterior -> replan, stopping at whichever comes first: `max_rounds`
    (the action-contract horizon) or budget exhaustion.

    Uses the real `SpatialAdaptiveMissionLoop` (R6/R7 integration path), not
    the older toy `AdaptiveMissionLoop` - the policy never receives anything
    but `GraphState` (see `_StashingPlanner.plan`), and `HiddenWorld`/`q_true`
    are only ever touched here, after `loop.reveal()`/`force_complete()`, for
    the terminal reward term and reporting metrics.
    """

    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")

    cfg = reward_config or SpatialRewardConfig()
    environment = Environment(incident_config, world_model=world_model, q_true=q_true)
    adapter = _StashingPlanner(policy, deterministic=deterministic)
    loop = SpatialAdaptiveMissionLoop(
        environment, adapter, ecological_hypotheses, q_hypotheses
    )
    loop.reset(seed=seed)

    rollout = SpatialEpisodeRollout()

    while True:
        loop.plan_next()
        decision = adapter.last_decision
        budget_before = loop.current_public_state.remaining_budget
        transition = loop.execute_pending()

        components = spatial_round_reward_components(
            belief_before=transition.belief_before,
            belief_after=transition.belief_after,
            new_detections=int(transition.simulator_metrics["detections"]),
            config=cfg,
        )
        reward = sum(components.values())

        rollout.log_probs.append(decision.log_prob)
        rollout.values.append(decision.value)
        rollout.entropies.append(decision.entropy)
        rollout.num_rounds += 1
        rollout.detections_found += int(transition.simulator_metrics["detections"])
        rollout.effort_spent += int(transition.simulator_metrics["effort_spent"])

        horizon_reached = rollout.num_rounds >= max_rounds
        planner_stopped = bool(getattr(transition, "planner_stopped", False))
        episode_over = transition.done or horizon_reached or planner_stopped
        if episode_over and not transition.done and not planner_stopped:
            loop.force_complete()

        if episode_over:
            hidden_world = loop.reveal()
            metrics = compute_spatial_primary_metrics(
                sites=transition.public_state_after.sites,
                hidden_world=hidden_world,
                final_belief=transition.belief_after,
            )
            terminal_components = spatial_terminal_reward_components(
                missed_occupied_fraction=metrics.missed_occupied_fraction,
                config=cfg,
            )
            components = {**components, **terminal_components}
            reward += sum(terminal_components.values())
            rollout.metrics = metrics
            rollout.rewards.append(reward)
            if decision_logger is not None:
                _log_round(
                    decision_logger, episode_index, rollout.num_rounds - 1,
                    decision, transition, components, reward,
                    budget_before=budget_before, done=True,
                )
            break

        rollout.rewards.append(reward)
        if decision_logger is not None:
            _log_round(
                decision_logger, episode_index, rollout.num_rounds - 1,
                decision, transition, components, reward,
                budget_before=budget_before, done=False,
            )

    return rollout


def _log_round(
    logger: JsonlDecisionLogger,
    episode_index: int,
    round_index: int,
    decision,
    transition,
    reward_components: dict[str, float],
    reward_total: float,
    *,
    budget_before: int,
    done: bool,
) -> None:
    detection = any(o.detection for o in transition.observations.observations)
    record = SpatialRoundLogRecord(
        episode_index=episode_index,
        round_index=round_index,
        budget_before=budget_before,
        budget_after=transition.public_state_after.remaining_budget,
        site_id=decision.site_id,
        effort_units=decision.effort_units,
        node_ids=decision.node_ids,
        effort_levels=decision.effort_levels,
        node_logits=decision.node_logits,
        eligible_mask=decision.eligible_mask,
        value_estimate=float(decision.value.item()),
        entropy=float(decision.entropy.item()),
        log_prob=float(decision.log_prob.item()),
        reward_components=dict(reward_components),
        reward_total=float(reward_total),
        detection=detection,
        done=done,
    )
    logger.log_round(record)
