from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from ..models import GraphState, MissionAction, MissionAllocation
from .backbone import GNNActorCritic
from .layers import make_mlp
from .tensor_adapter import graph_state_to_tensors

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class RoundDecision:
    """One round's action plus everything needed for a policy-gradient update.

    `log_prob` and `entropy` are sums over every autoregressive pick made this
    round (including the final STOP pick): the round's `MissionAction` is
    treated as one factorized joint action, credited with one round-level
    reward, which is the simplest defensible choice given the team hasn't
    frozen finer-grained credit assignment (Checkpoint B item).
    """

    mission: MissionAction
    log_prob: torch.Tensor  # scalar, requires_grad during training rollout
    value: torch.Tensor  # scalar, critic estimate at round start
    entropy: torch.Tensor  # scalar
    num_picks: int


class RoundPolicy(nn.Module):
    """Autoregressive per-round site selection on top of the Checkpoint-A backbone.

    PROPOSED CURRENT DEFAULT (Checkpoint B territory, not frozen): implements
    exactly the action-space pattern already sketched in
    docs (Technical Specification, "Action space - current default"):
    select a site, assign a fixed effort block, update the round's local
    feasibility bookkeeping, repeat until the round budget/mask is exhausted
    or the policy picks STOP. `effort_per_pick=1` matches the granularity
    FrontierPlanner already uses by default, rather than inventing a new one.

    The backbone's node embeddings/graph context/value are computed exactly
    once per round (the underlying GraphState is fixed until the mission is
    executed); only the local "already picked this round" mask changes
    between picks, so this stays cheap regardless of how many picks happen.

    STOP is masked out on the very first pick of a round so a round can never
    produce an empty MissionAction (Environment rejects zero-effort actions).
    """

    def __init__(
        self,
        backbone: GNNActorCritic,
        *,
        hidden_dim: int = 64,
        effort_per_pick: int = 1,
        max_picks_per_round: int | None = None,
    ) -> None:
        super().__init__()
        if effort_per_pick <= 0:
            raise ValueError("effort_per_pick must be positive.")
        self.backbone = backbone
        self.effort_per_pick = effort_per_pick
        self.max_picks_per_round = max_picks_per_round
        self.stop_head = make_mlp(hidden_dim, hidden_dim, 1)

    def act(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        *,
        deterministic: bool = False,
    ) -> RoundDecision:
        if remaining_budget < self.effort_per_pick:
            raise ValueError(
                "RoundPolicy.act called with insufficient budget for even one "
                "pick; the caller should treat this as a terminal state."
            )

        tensors = graph_state_to_tensors(graph_state)
        node_embeddings = self.backbone.encoder(tensors)
        graph_context = self.backbone.global_context(node_embeddings, tensors.global_features)
        node_logits = self.backbone.actor(node_embeddings, graph_context)
        value = self.backbone.critic(graph_context)
        stop_logit = self.stop_head(graph_context).squeeze(-1)

        picked = torch.zeros(tensors.num_nodes, dtype=torch.bool)
        allocations: list[MissionAllocation] = []
        round_budget_left = remaining_budget
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        max_picks = self.max_picks_per_round or tensors.num_nodes
        for pick_index in range(max_picks):
            if round_budget_left < self.effort_per_pick:
                break
            feasible = tensors.feasibility_mask & ~picked
            if not bool(feasible.any()):
                break

            masked_node_logits = node_logits.masked_fill(~feasible, _NEG_INF)
            can_stop = pick_index > 0
            stop_component = stop_logit if can_stop else torch.tensor(_NEG_INF)
            combined_logits = torch.cat([masked_node_logits, stop_component.unsqueeze(0)])
            dist = Categorical(logits=combined_logits)

            choice = combined_logits.argmax() if deterministic else dist.sample()
            log_probs.append(dist.log_prob(choice))
            entropies.append(dist.entropy())

            if int(choice.item()) == tensors.num_nodes:
                break  # STOP

            node_index = int(choice.item())
            allocations.append(
                MissionAllocation(
                    site_id=tensors.node_ids[node_index], effort_units=self.effort_per_pick
                )
            )
            picked[node_index] = True
            round_budget_left -= self.effort_per_pick

        if not allocations:
            raise RuntimeError(
                "RoundPolicy produced no allocations despite STOP being masked "
                "on the first pick; this indicates a bug, not a valid terminal "
                "state (the caller already guarantees remaining_budget >= "
                "effort_per_pick and feasibility_mask has at least one True "
                "entry whenever budget remains)."
            )

        mission = MissionAction(
            allocations=tuple(allocations),
            total_cost=len(allocations) * self.effort_per_pick,
            diagnostics={"planner": "rl_round_policy", "num_picks": len(allocations)},
        )
        total_log_prob = torch.stack(log_probs).sum()
        total_entropy = torch.stack(entropies).sum()
        return RoundDecision(
            mission=mission,
            log_prob=total_log_prob,
            value=value,
            entropy=total_entropy,
            num_picks=len(allocations),
        )
