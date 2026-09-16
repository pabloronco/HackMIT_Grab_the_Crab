from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .round_policy import RoundPolicy
from .training_env import EpisodeRollout

# On-policy actor-critic (REINFORCE + learned baseline), explicitly one of
# the two candidates the Technical Specification names ("Candidate RL:
# PPO/actor-critic. Non e FROZEN finche piccoli benchmark non mostrano
# stabilita"). Deliberately not full clipped-PPO: with a single gradient
# step per freshly-collected rollout batch (no multi-epoch replay of stale
# rollouts), PPO's importance-sampling ratio/clip has nothing to correct for,
# so the simpler, easier-to-get-right formulation is used instead. Revisit if
# training proves unstable.


@dataclass(frozen=True)
class TrainerConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    # Optional linear decay of entropy_coef down to entropy_coef_final over
    # entropy_coef_decay_updates update() calls, then held constant.
    # CURRENT DEFAULT off (entropy_coef_decay_updates=0): added after
    # observing (2026-09-11 Decision Log entries) that a larger
    # missed_extent_weight plateaus entropy around 8-13 nats for thousands of
    # updates instead of converging like the original weighting did - a
    # bigger, noisier terminal-reward-driven gradient signal plus a constant
    # entropy bonus can outweigh the pressure to commit to a confident policy.
    # Decaying the bonus's weight is the standard lever for "explore early,
    # commit later" without changing the reward itself.
    entropy_coef_final: float = 0.01
    entropy_coef_decay_updates: int = 0
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class UpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_episode_return: float
    mean_advantage: float
    entropy_coef_used: float


class ActorCriticTrainer:
    def __init__(self, policy: RoundPolicy, config: TrainerConfig | None = None) -> None:
        self.policy = policy
        self.config = config or TrainerConfig()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr)
        self._update_count = 0

    def current_entropy_coef(self) -> float:
        cfg = self.config
        if cfg.entropy_coef_decay_updates <= 0:
            return cfg.entropy_coef
        progress = min(self._update_count / cfg.entropy_coef_decay_updates, 1.0)
        return cfg.entropy_coef + progress * (cfg.entropy_coef_final - cfg.entropy_coef)

    def discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, rollouts: list[EpisodeRollout]) -> UpdateStats:
        if not rollouts:
            raise ValueError("update() requires at least one episode rollout.")

        log_probs = torch.stack([lp for r in rollouts for lp in r.log_probs])
        values = torch.stack([v for r in rollouts for v in r.values])
        entropies = torch.stack([e for r in rollouts for e in r.entropies])
        returns = torch.cat([self.discounted_returns(r.rewards) for r in rollouts])

        # Batch-normalize returns before they become the critic's regression
        # target. Without this, the absolute scale of RewardConfig's weights
        # (e.g. a large missed_extent_weight relative to the per-round dense
        # terms) directly sets the scale of value_loss: a freshly-initialized
        # critic hasn't learned that scale yet, so value_loss can dominate the
        # combined loss and destabilize early training (observed directly: a
        # 4x-larger missed_extent_weight alone took loss from single digits to
        # the hundreds in a smoke test). Normalizing decouples "how we relatively
        # weight reward terms" from "how large gradients are," so RewardConfig
        # tuning stays about behavior, not about re-deriving a stable lr/coef
        # every time.
        if returns.numel() > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        advantages = (returns - values).detach()
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        entropy_coef = self.current_entropy_coef()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_bonus = entropies.mean()
        loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - entropy_coef * entropy_bonus
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self._update_count += 1

        mean_episode_return = sum(sum(r.rewards) for r in rollouts) / len(rollouts)

        return UpdateStats(
            loss=float(loss.item()),
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy_bonus.item()),
            mean_episode_return=float(mean_episode_return),
            mean_advantage=float(advantages.mean().item()),
            entropy_coef_used=float(entropy_coef),
        )
