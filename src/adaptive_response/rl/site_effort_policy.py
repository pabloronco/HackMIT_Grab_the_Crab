from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from ..models import GraphState, MissionAction, MissionAllocation
from .backbone import GNNActorCritic, SiteEffortActorHead
from .tensor_adapter import graph_state_to_tensors

_NEG_INF = float("-inf")

DEFAULT_EFFORT_LEVELS: tuple[int, ...] = (1, 3, 6)


@dataclass(frozen=True)
class SiteEffortDecision:
    """One round's (site, effort) pick plus everything needed for a policy update.

    R7 action contract: exactly one allocation per round (see
    docs/DEMU_HANDOFF_R7.md, configs/benchmark_protocol_r7.json). `node_logits`
    and `eligible_mask` are [N][K] (K = number of effort levels), index-aligned
    with `node_ids` and with `effort_levels`, so a later reader can zip them
    together - same intent as the engineering-block `eligible_mask` addition
    to the older autoregressive `RoundDecision`.
    """

    mission: MissionAction
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    site_id: str
    effort_units: int
    node_ids: tuple[str, ...] = ()
    effort_levels: tuple[int, ...] = ()
    node_logits: tuple[tuple[float, ...], ...] = ()
    eligible_mask: tuple[tuple[bool, ...], ...] = ()
    # R12 additions (defaulted, so older callers/tests are unaffected): the flat
    # [N*K] index of the chosen (site, effort) so PPO can re-evaluate it under
    # updated parameters, plus action-probability diagnostics.
    action_index: int = -1
    action_probability: float = float("nan")
    max_action_probability: float = float("nan")


class SiteEffortRoundPolicy(nn.Module):
    """R7 action-contract policy: pick exactly one (site, effort) pair per round.

    Reuses the Checkpoint-A GNN backbone (encoder/global-context/critic)
    unchanged; only the actor head differs from the older autoregressive
    `RoundPolicy` (SiteEffortActorHead scores site x effort-level jointly
    instead of scoring sites one at a time with a learned STOP token). There
    is no STOP action here: the R7 contract has no discretionary early stop -
    an episode ends when the caller's round/horizon cap or the environment's
    budget is exhausted, not by a policy choice.
    """

    def __init__(
        self,
        backbone: GNNActorCritic,
        *,
        hidden_dim: int = 64,
        effort_levels: tuple[int, ...] = DEFAULT_EFFORT_LEVELS,
    ) -> None:
        super().__init__()
        if not effort_levels or any(e <= 0 for e in effort_levels):
            raise ValueError("effort_levels must be a non-empty sequence of positive ints.")
        if len(set(effort_levels)) != len(effort_levels):
            raise ValueError("effort_levels must not contain duplicates.")
        self.backbone = backbone
        self.effort_levels = tuple(sorted(effort_levels))
        self.actor = SiteEffortActorHead(
            hidden_dim=hidden_dim, num_effort_levels=len(self.effort_levels)
        )

    def act(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        *,
        deterministic: bool = False,
    ) -> SiteEffortDecision:
        min_effort = self.effort_levels[0]
        if remaining_budget < min_effort:
            raise ValueError(
                "SiteEffortRoundPolicy.act called with insufficient budget for "
                "even the smallest effort level; the caller should treat this "
                "as a terminal state."
            )

        tensors = graph_state_to_tensors(graph_state)
        node_embeddings = self.backbone.encoder(tensors)
        graph_context = self.backbone.global_context(node_embeddings, tensors.global_features)
        node_logits = self.actor(node_embeddings, graph_context)  # [N, K]
        value = self.backbone.critic(graph_context)

        effort_tensor = torch.tensor(self.effort_levels, dtype=torch.long)
        effort_feasible = effort_tensor <= remaining_budget  # [K]
        eligible = tensors.feasibility_mask.unsqueeze(-1) & effort_feasible.unsqueeze(0)  # [N, K]
        if not bool(eligible.any()):
            raise RuntimeError(
                "SiteEffortRoundPolicy found no eligible (site, effort) pair despite "
                "sufficient remaining_budget for the smallest effort level; this "
                "indicates a bug (feasibility_mask should have at least one True "
                "entry whenever budget remains)."
            )

        masked_logits = node_logits.masked_fill(~eligible, _NEG_INF)
        flat_logits = masked_logits.reshape(-1)
        dist = Categorical(logits=flat_logits)
        choice = flat_logits.argmax() if deterministic else dist.sample()
        log_prob = dist.log_prob(choice)
        entropy = dist.entropy()

        num_levels = len(self.effort_levels)
        flat_index = int(choice.item())
        node_index, effort_index = divmod(flat_index, num_levels)
        site_id = tensors.node_ids[node_index]
        effort_units = self.effort_levels[effort_index]

        mission = MissionAction(
            allocations=(MissionAllocation(site_id=site_id, effort_units=effort_units),),
            total_cost=effort_units,
            diagnostics={"planner": "rl_site_effort_policy", "num_picks": 1},
        )
        return SiteEffortDecision(
            mission=mission,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            site_id=site_id,
            effort_units=effort_units,
            node_ids=tensors.node_ids,
            effort_levels=self.effort_levels,
            node_logits=tuple(tuple(row) for row in node_logits.detach().tolist()),
            eligible_mask=tuple(tuple(bool(v) for v in row) for row in eligible.tolist()),
        )
