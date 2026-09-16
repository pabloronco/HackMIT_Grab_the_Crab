from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..environment import Environment
from ..mission_loop import AdaptiveMissionLoop
from ..models import IncidentConfig, MissionAction
from .decision_logger import JsonlDecisionLogger, RoundLogRecord
from .reward import RewardConfig, round_reward_components, terminal_missed_extent_penalty
from .round_policy import RoundPolicy


@dataclass
class EpisodeRollout:
    """Everything the trainer needs from one full episode, one round-decision at a time."""

    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    num_rounds: int = 0
    num_picks_total: int = 0
    detections_found: int = 0
    effort_spent: int = 0
    occupied_sites_missed: int = 0
    occupied_sites_total: int = 0


class _StashingPlanner:
    """Adapts `RoundPolicy` to the `Planner` protocol for `AdaptiveMissionLoop`.

    The loop's `plan_next()` only returns a `MissionAction` - the richer
    `RoundDecision` (log_prob/value/entropy) needed for the policy gradient
    update has nowhere to travel through that interface, so this adapter
    stashes the last decision on itself for the caller to read back. The
    caller MUST read `last_decision` immediately after its own `plan_next()`
    call and before calling `execute_pending()`: that method internally
    triggers another `plan_next()` for the following round as soon as the
    current one isn't done, which would silently overwrite `last_decision`
    with the wrong round's decision otherwise.
    """

    def __init__(self, policy: RoundPolicy, *, deterministic: bool) -> None:
        self._policy = policy
        self._deterministic = deterministic
        self.last_decision = None

    def plan(self, graph_state, remaining_budget, constraints) -> MissionAction:
        del constraints
        self.last_decision = self._policy.act(
            graph_state, remaining_budget, deterministic=self._deterministic
        )
        return self.last_decision.mission


def run_episode(
    policy: RoundPolicy,
    incident_config: IncidentConfig,
    *,
    reward_config: RewardConfig | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    decision_logger: JsonlDecisionLogger | None = None,
    episode_index: int = 0,
) -> EpisodeRollout:
    """Run one full incident from reset to budget exhaustion under `policy`.

    Uses the shared `AdaptiveMissionLoop` (M4) rather than driving `Environment`
    directly, so this gets the real plan -> execute -> Bayes -> replan cycle
    and the proper reveal gate for free instead of re-deriving them. `reveal()`
    is only reachable after the loop's own state machine reaches COMPLETE, and
    only this training/evaluator code calls it - `policy.act` only ever
    receives a `GraphState`, never the loop or the environment.

    `decision_logger`/`episode_index` are opt-in (engineering block,
    2026-09-12): when a logger is passed, one `RoundLogRecord` is written per
    round with the raw logits/entropy/value/reward-components/chosen actions.
    Omitting it (the default) leaves every existing caller's behavior and
    timing unchanged.
    """

    cfg = reward_config or RewardConfig()
    env = Environment(incident_config)
    prior = {site.id: 0.5 for site in incident_config.sites}
    adapter = _StashingPlanner(policy, deterministic=deterministic)
    loop = AdaptiveMissionLoop(env, adapter, prior_by_site=prior)
    loop.reset(seed=seed)

    rollout = EpisodeRollout()

    while True:
        # IMPORTANT: capture the decision right after plan_next(), not after
        # execute_pending()/run_round(). execute_pending() internally calls
        # plan_next() again for the *next* round as soon as the current one
        # isn't done (see AdaptiveMissionLoop.execute_pending), which would
        # silently overwrite adapter.last_decision with the wrong round's
        # decision before we get a chance to read it back - misattributing
        # this round's reward to the next round's log_prob/value/entropy.
        loop.plan_next()
        decision = adapter.last_decision
        budget_before = loop.current_public_state.remaining_budget
        transition = loop.execute_pending()
        metrics = transition.simulator_metrics

        components = round_reward_components(
            belief_before=transition.belief_before,
            belief_after=transition.belief_after,
            detections_this_round=int(metrics["detections"]),
            effort_spent_this_round=int(metrics["effort_spent"]),
            config=cfg,
        )
        reward = sum(components.values())

        rollout.log_probs.append(decision.log_prob)
        rollout.values.append(decision.value)
        rollout.entropies.append(decision.entropy)
        rollout.num_rounds += 1
        rollout.num_picks_total += decision.num_picks
        rollout.detections_found += int(metrics["detections"])
        rollout.effort_spent += int(metrics["effort_spent"])

        if transition.done:
            hidden_world = loop.reveal()
            terminal_penalty = terminal_missed_extent_penalty(
                public_state=transition.public_state_after,
                hidden_world=hidden_world,
                config=cfg,
            )
            components["terminal_missed_extent"] = terminal_penalty
            reward += terminal_penalty
            rollout.occupied_sites_total = sum(hidden_world.occupied_by_site.values())
            rollout.occupied_sites_missed = sum(
                1
                for site in transition.public_state_after.sites
                if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
            )
            rollout.rewards.append(reward)
            if decision_logger is not None:
                _log_round(
                    decision_logger, episode_index, rollout.num_rounds - 1,
                    decision, transition, components, reward,
                    budget_before=budget_before, done=True,
                )
            # Returning here (rather than looping once more to let a
            # `while phase is not COMPLETE` condition catch it) matters:
            # reveal() just moved the loop's phase from COMPLETE to REVEALED,
            # so that condition would misfire into one extra invalid round.
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
    record = RoundLogRecord(
        episode_index=episode_index,
        round_index=round_index,
        budget_before=budget_before,
        budget_after=transition.public_state_after.remaining_budget,
        site_ids_picked=tuple(a.site_id for a in decision.mission.allocations),
        num_picks=decision.num_picks,
        node_ids=decision.node_ids,
        node_logits=decision.node_logits,
        eligible_mask=decision.eligible_mask,
        value_estimate=float(decision.value.item()),
        entropy=float(decision.entropy.item()),
        log_prob=float(decision.log_prob.item()),
        reward_components=dict(reward_components),
        reward_total=float(reward_total),
        detections_this_round=int(transition.simulator_metrics["detections"]),
        done=done,
    )
    logger.log_round(record)
