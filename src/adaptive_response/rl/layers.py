from __future__ import annotations

import torch
from torch import nn


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_hidden_layers: int = 1,
) -> nn.Sequential:
    """Small ReLU MLP factory shared by every head/layer in this backbone.

    Kept deliberately plain (Linear + ReLU, no dropout/batchnorm) per the
    Checkpoint A brief: small, robust, fast to train, nothing sophisticated
    for its own sake.
    """

    if num_hidden_layers < 0:
        raise ValueError("num_hidden_layers cannot be negative.")

    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def segment_mean(
    values: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Mean of `values` rows grouped by `index`, for segments with no rows -> 0.

    Plain-PyTorch replacement for `torch_scatter.scatter_mean` so this
    backbone has no dependency on PyTorch Geometric / torch-scatter (those
    require platform-matched prebuilt wheels that are a common source of
    Windows/CI breakage; index_add_ is pure PyTorch and graph-size agnostic).
    """

    if values.shape[0] != index.shape[0]:
        raise ValueError("values and index must have the same length.")

    feature_dim = values.shape[1] if values.dim() > 1 else 1
    values_2d = values if values.dim() > 1 else values.unsqueeze(-1)

    summed = torch.zeros(
        (num_segments, feature_dim), dtype=values.dtype, device=values.device
    )
    summed.index_add_(0, index, values_2d)

    counts = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(-1)

    result = summed / counts
    return result if values.dim() > 1 else result.squeeze(-1)


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing with a residual update.

    message_ij = MLP([h_i ; h_j ; edge_ij])   for each directed edge i -> j
    aggregated_j = mean over incoming messages at node j
    h_j_new = LayerNorm(h_j + MLP([h_j ; aggregated_j]))

    `edge_index` is expected in the GraphState/PyG convention: row 0 is the
    source node index, row 1 is the destination node index, and the exporter
    already emits both directions for an undirected graph, so this layer does
    not need to special-case direction.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = make_mlp(2 * node_dim + edge_dim, hidden_dim, hidden_dim)
        self.update_mlp = make_mlp(node_dim + hidden_dim, hidden_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,  # [N, node_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_features: torch.Tensor,  # [E, edge_dim]
    ) -> torch.Tensor:
        num_nodes = node_embeddings.shape[0]
        if edge_index.shape[1] == 0:
            aggregated = torch.zeros(
                (num_nodes, self.hidden_dim),
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
        else:
            src, dst = edge_index[0], edge_index[1]
            messages = self.message_mlp(
                torch.cat(
                    [node_embeddings[src], node_embeddings[dst], edge_features], dim=-1
                )
            )
            aggregated = segment_mean(messages, dst, num_nodes)

        update = self.update_mlp(torch.cat([node_embeddings, aggregated], dim=-1))
        return self.norm(node_embeddings + update)
