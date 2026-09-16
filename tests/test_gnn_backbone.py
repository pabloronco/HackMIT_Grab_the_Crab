"""Checkpoint A: GraphState -> tensors -> GNN -> logits/value backbone tests.

Requires torch (see pyproject.toml's "rl" extra). Skipped entirely, rather
than failed, when torch is not installed so teammates working on the
Bayes/simulator/UI side are never blocked by an ML dependency they don't need.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from adaptive_response import (  # noqa: E402
    BeliefEngine,
    Edge,
    Environment,
    GraphStateExporter,
    IncidentConfig,
    Site,
)
from adaptive_response.models import HiddenWorld  # noqa: E402
from adaptive_response.rl import GNNActorCritic, graph_state_to_tensors  # noqa: E402
from adaptive_response.rl.tensor_adapter import GraphTensors  # noqa: E402


def make_graph_state(num_sites: int, *, seed: int = 20260910):
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
            q_model=0.25,
        )
        for i in range(num_sites)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(num_sites - 1)
    ]
    config = IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_00",
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=seed,
        world_model_id="toy_graph_cluster_m1",
    )
    env = Environment(config)
    public = env.reset(seed=seed)
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    return GraphStateExporter().export(public, belief)


@pytest.fixture(scope="module")
def model() -> "GNNActorCritic":
    torch.manual_seed(0)
    return GNNActorCritic(hidden_dim=32, num_layers=2)


def test_adapter_rejects_misaligned_graph_state() -> None:
    graph = make_graph_state(6)
    bad = graph.__class__(
        node_ids=graph.node_ids,
        node_features=graph.node_features[:-1],  # drop a row on purpose
        edge_index=graph.edge_index,
        edge_features=graph.edge_features,
        global_features=graph.global_features,
        feasibility_mask=graph.feasibility_mask,
    )
    with pytest.raises(ValueError, match="misaligned"):
        graph_state_to_tensors(bad)


def test_adapter_only_accepts_graph_state_no_hidden_world_path() -> None:
    """Structural no-leakage check: the adapter's signature only accepts
    GraphState, and the module never imports HiddenWorld at all, so there is
    no code path through which latent occupancy could reach the network.
    """

    from adaptive_response.rl import tensor_adapter

    sig = inspect.signature(graph_state_to_tensors)
    param_types = [p.annotation for p in sig.parameters.values()]
    assert "GraphState" in [getattr(t, "__name__", str(t)) for t in param_types]
    assert not hasattr(tensor_adapter, "HiddenWorld")
    assert HiddenWorld.__name__ not in dir(tensor_adapter)


@pytest.mark.parametrize("num_sites", [4, 12, 20, 24])
def test_forward_pass_shapes_for_variable_graph_size(
    model: "GNNActorCritic", num_sites: int
) -> None:
    graph = make_graph_state(num_sites)
    tensors = graph_state_to_tensors(graph)

    with torch.no_grad():
        output = model(tensors)

    assert output.node_ids == graph.node_ids
    assert output.node_embeddings.shape == (num_sites, 32)
    assert output.graph_context.shape == (32,)
    assert output.node_logits.shape == (num_sites,)
    assert output.masked_node_logits.shape == (num_sites,)
    assert output.value.shape == ()
    assert torch.isfinite(output.node_logits).all()
    assert torch.isfinite(output.value)


def test_same_model_instance_handles_multiple_graph_sizes_in_a_row(
    model: "GNNActorCritic",
) -> None:
    """Same weights, back-to-back different N: proves there is no hidden
    per-graph-size parameter (e.g. a Linear sized to a specific N) anywhere.
    """

    for num_sites in (12, 20, 24, 12):
        tensors = graph_state_to_tensors(make_graph_state(num_sites))
        with torch.no_grad():
            output = model(tensors)
        assert output.node_logits.shape == (num_sites,)


def test_feasibility_mask_forces_infeasible_nodes_to_neg_inf(
    model: "GNNActorCritic",
) -> None:
    graph = make_graph_state(6)
    tensors = graph_state_to_tensors(graph)
    forced_infeasible = GraphTensors(
        node_ids=tensors.node_ids,
        node_features=tensors.node_features,
        edge_index=tensors.edge_index,
        edge_features=tensors.edge_features,
        global_features=tensors.global_features,
        feasibility_mask=torch.tensor(
            [True, False, True, False, False, False]
        ),
    )

    with torch.no_grad():
        output = model(forced_infeasible)

    assert torch.isneginf(output.masked_node_logits[[1, 3, 4, 5]]).all()
    assert torch.isfinite(output.masked_node_logits[[0, 2]]).all()

    dist = output.masked_action_distribution()
    assert dist.probs[1] == pytest.approx(0.0)
    assert dist.probs[3] == pytest.approx(0.0)


def test_all_infeasible_raises_instead_of_producing_nan(
    model: "GNNActorCritic",
) -> None:
    graph = make_graph_state(4)
    tensors = graph_state_to_tensors(graph)
    exhausted = GraphTensors(
        node_ids=tensors.node_ids,
        node_features=tensors.node_features,
        edge_index=tensors.edge_index,
        edge_features=tensors.edge_features,
        global_features=tensors.global_features,
        feasibility_mask=torch.zeros(4, dtype=torch.bool),
    )

    with torch.no_grad():
        output = model(exhausted)

    with pytest.raises(ValueError, match="No feasible node"):
        output.masked_action_distribution()


def test_no_q_or_hidden_signal_embedded_in_node_features() -> None:
    """Mirrors tests/test_graph_state.py's leakage contract at the tensor
    boundary: q stays out of node_features here too.
    """

    from adaptive_response.graph_state import NODE_FEATURE_NAMES

    assert "q" not in NODE_FEATURE_NAMES
    graph = make_graph_state(6)
    tensors = graph_state_to_tensors(graph)
    assert tensors.node_features.shape[1] == len(NODE_FEATURE_NAMES)


def test_parameter_count_is_reported_and_small(model: "GNNActorCritic") -> None:
    param_count = sum(p.numel() for p in model.parameters())
    # Sanity ceiling, not a frozen budget: catches an accidental blow-up
    # (e.g. a fixed-N layer) far more than it constrains legitimate tuning.
    assert 0 < param_count < 200_000
