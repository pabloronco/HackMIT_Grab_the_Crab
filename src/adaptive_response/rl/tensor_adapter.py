from __future__ import annotations

from dataclasses import dataclass

import torch

from ..graph_state import EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES
from ..models import GraphState

# CURRENT DEFAULT (this module only, not a shared contract): which named
# features get a log1p transform before entering the network. GraphState
# itself stays in raw, interpretable units (see graph_state.py); scaling is
# explicitly the learned planner's responsibility per docs/INTERFACES_V0.md.
#
# Rationale: belief/uncertainty/frontier are already bounded in [0, 1] and are
# left untouched. observed_effort/detections/remaining_budget/round are
# non-negative counts that can grow arbitrarily large across a multi-round
# episode with a large budget; log1p keeps them on a comparable scale to the
# bounded features without needing a running-statistics normalizer (which
# would add hidden state and break determinism across graph sizes/episodes).
_NODE_LOG1P_FEATURES = frozenset({"observed_effort", "detections"})
_EDGE_LOG1P_FEATURES = frozenset({"distance"})
_GLOBAL_LOG1P_FEATURES = frozenset({"remaining_budget", "round"})


def _log1p_mask(names: tuple[str, ...], selected: frozenset[str]) -> torch.Tensor:
    return torch.tensor([name in selected for name in names], dtype=torch.bool)


_NODE_LOG1P_MASK = _log1p_mask(NODE_FEATURE_NAMES, _NODE_LOG1P_FEATURES)
_EDGE_LOG1P_MASK = _log1p_mask(EDGE_FEATURE_NAMES, _EDGE_LOG1P_FEATURES)
_GLOBAL_LOG1P_MASK = _log1p_mask(GLOBAL_FEATURE_NAMES, _GLOBAL_LOG1P_FEATURES)


def _apply_log1p(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply log1p column-wise where `mask` is True, elementwise where 1-D."""

    if values.numel() == 0:
        return values
    out = values.clone()
    if out.dim() == 1:
        out[mask] = torch.log1p(out[mask].clamp(min=0.0))
    else:
        out[:, mask] = torch.log1p(out[:, mask].clamp(min=0.0))
    return out


@dataclass(frozen=True)
class GraphTensors:
    """Tensor view of a `GraphState`, index-aligned exactly like the source.

    `node_ids` is carried through unchanged so any per-node output (policy
    logits, embeddings) can be mapped back to the real `site_id` later,
    without the network ever needing to know site identity.
    """

    node_ids: tuple[str, ...]
    node_features: torch.Tensor  # [N, F] float32, normalized
    edge_index: torch.Tensor  # [2, E] int64
    edge_features: torch.Tensor  # [E, Fe] float32, normalized
    global_features: torch.Tensor  # [Fg] float32, normalized
    feasibility_mask: torch.Tensor  # [N] bool

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def graph_state_to_tensors(
    graph_state: GraphState,
    *,
    device: torch.device | str | None = None,
) -> GraphTensors:
    """Convert an observable `GraphState` into normalized tensors.

    This function's only input is `GraphState` (never `PublicState` or
    `HiddenWorld`): there is structurally no path for hidden occupancy to
    reach the network through this adapter.
    """

    if len(graph_state.node_features) != len(graph_state.node_ids):
        raise ValueError("GraphState node_ids/node_features are misaligned.")
    if any(len(row) != len(NODE_FEATURE_NAMES) for row in graph_state.node_features):
        raise ValueError(
            f"Expected {len(NODE_FEATURE_NAMES)} node features "
            f"({NODE_FEATURE_NAMES}); got a differently-shaped row."
        )
    if len(graph_state.edge_index) != 2:
        raise ValueError("GraphState.edge_index must have shape [2, E].")
    if len(graph_state.global_features) != len(GLOBAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(GLOBAL_FEATURE_NAMES)} global features "
            f"({GLOBAL_FEATURE_NAMES})."
        )

    node_features = torch.tensor(graph_state.node_features, dtype=torch.float32)
    node_features = _apply_log1p(node_features, _NODE_LOG1P_MASK)

    src, dst = graph_state.edge_index
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.int64)
    if edge_index.numel() and (
        int(edge_index.max()) >= len(graph_state.node_ids) or int(edge_index.min()) < 0
    ):
        raise ValueError("edge_index contains an index outside node_ids range.")

    if graph_state.edge_features:
        edge_features = torch.tensor(graph_state.edge_features, dtype=torch.float32)
    else:
        edge_features = torch.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32)
    edge_features = _apply_log1p(edge_features, _EDGE_LOG1P_MASK)

    global_features = torch.tensor(graph_state.global_features, dtype=torch.float32)
    global_features = _apply_log1p(global_features, _GLOBAL_LOG1P_MASK)

    feasibility_mask = torch.tensor(graph_state.feasibility_mask, dtype=torch.bool)

    tensors = GraphTensors(
        node_ids=graph_state.node_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
        feasibility_mask=feasibility_mask,
    )
    if device is not None:
        tensors = GraphTensors(
            node_ids=tensors.node_ids,
            node_features=tensors.node_features.to(device),
            edge_index=tensors.edge_index.to(device),
            edge_features=tensors.edge_features.to(device),
            global_features=tensors.global_features.to(device),
            feasibility_mask=tensors.feasibility_mask.to(device),
        )
    return tensors
