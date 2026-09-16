from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..graph_state import EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES
from .layers import MessagePassingLayer, make_mlp
from .tensor_adapter import GraphTensors

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class ActorCriticOutput:
    """Forward-pass output. Deliberately stops short of an action/MissionAction:

    action-space semantics (effort units, per-site caps, masking beyond raw
    feasibility) are Checkpoint B decisions, not something this module should
    pre-empt.
    """

    node_ids: tuple[str, ...]
    node_embeddings: torch.Tensor  # [N, hidden_dim]
    graph_context: torch.Tensor  # [hidden_dim]
    node_logits: torch.Tensor  # [N], raw (pre-mask) per-node policy score
    masked_node_logits: torch.Tensor  # [N], infeasible nodes set to -inf
    value: torch.Tensor  # scalar, critic estimate for this graph state

    def masked_action_distribution(self) -> torch.distributions.Categorical:
        """Convenience for sanity checks / rollout code: a masked categorical
        over feasible nodes. Not part of the frozen action-space design.
        """

        if torch.isneginf(self.masked_node_logits).all():
            raise ValueError(
                "No feasible node to act on (feasibility_mask is all False); "
                "the caller should have already treated this as a terminal "
                "state instead of querying the policy."
            )
        return torch.distributions.Categorical(logits=self.masked_node_logits)


class GraphEncoder(nn.Module):
    """GraphState tensors -> per-node embeddings via K rounds of message passing.

    Graph-size agnostic by construction: every layer is a shared per-node/
    per-edge function, so the same weights apply whether N is 12 or 24.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = len(NODE_FEATURE_NAMES),
        edge_feature_dim: int = len(EDGE_FEATURE_NAMES),
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(hidden_dim, edge_feature_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tensors: GraphTensors) -> torch.Tensor:
        h = self.input_proj(tensors.node_features)
        for layer in self.layers:
            h = layer(h, tensors.edge_index, tensors.edge_features)
        return h


class GlobalContextEncoder(nn.Module):
    """Combine mean-pooled node embeddings with encoded global features.

    Mean pooling (rather than e.g. flattening) is what keeps this permutation-
    invariant and graph-size agnostic, matching the node encoder.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        global_feature_dim: int = len(GLOBAL_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        self.global_proj = nn.Linear(global_feature_dim, hidden_dim)
        self.combine = make_mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_embeddings: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        pooled = node_embeddings.mean(dim=0)
        encoded_global = self.global_proj(global_features)
        return self.norm(self.combine(torch.cat([pooled, encoded_global], dim=-1)))


class ActorHead(nn.Module):
    """Shared per-node scorer: one logit per node, any N.

    A fixed-size output layer here would silently break graph-size agnosticism,
    so this must stay a function applied identically to every node row rather
    than a Linear(hidden_dim * N -> N).
    """

    def __init__(self, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score = make_mlp(2 * hidden_dim, hidden_dim, 1)

    def forward(
        self, node_embeddings: torch.Tensor, graph_context: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = node_embeddings.shape[0]
        context_per_node = graph_context.unsqueeze(0).expand(num_nodes, -1)
        logits = self.score(torch.cat([node_embeddings, context_per_node], dim=-1))
        return logits.squeeze(-1)

    @staticmethod
    def apply_feasibility_mask(
        logits: torch.Tensor, feasibility_mask: torch.Tensor
    ) -> torch.Tensor:
        return logits.masked_fill(~feasibility_mask, _NEG_INF)


class CriticHead(nn.Module):
    """Graph-level value estimate from the pooled/global context vector."""

    def __init__(self, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.value = make_mlp(hidden_dim, hidden_dim, 1)

    def forward(self, graph_context: torch.Tensor) -> torch.Tensor:
        return self.value(graph_context).squeeze(-1)


class GNNActorCritic(nn.Module):
    """Full Checkpoint-A backbone: GraphState tensors -> logits + value.

    Small by design (default hidden_dim=64, num_layers=2): the brief asks for
    robust and fast to train, not a large model. hidden_dim/num_layers are
    left as constructor args precisely so this can be tuned later without
    touching the encoder/head wiring.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = len(NODE_FEATURE_NAMES),
        edge_feature_dim: int = len(EDGE_FEATURE_NAMES),
        global_feature_dim: int = len(GLOBAL_FEATURE_NAMES),
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.global_context = GlobalContextEncoder(
            hidden_dim=hidden_dim, global_feature_dim=global_feature_dim
        )
        self.actor = ActorHead(hidden_dim=hidden_dim)
        self.critic = CriticHead(hidden_dim=hidden_dim)

    def forward(self, tensors: GraphTensors) -> ActorCriticOutput:
        if tensors.num_nodes == 0:
            raise ValueError("GraphTensors must contain at least one node.")

        node_embeddings = self.encoder(tensors)
        graph_context = self.global_context(node_embeddings, tensors.global_features)
        node_logits = self.actor(node_embeddings, graph_context)
        masked_node_logits = ActorHead.apply_feasibility_mask(
            node_logits, tensors.feasibility_mask
        )
        value = self.critic(graph_context)

        return ActorCriticOutput(
            node_ids=tensors.node_ids,
            node_embeddings=node_embeddings,
            graph_context=graph_context,
            node_logits=node_logits,
            masked_node_logits=masked_node_logits,
            value=value,
        )
