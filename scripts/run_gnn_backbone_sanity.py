"""Checkpoint A sanity run: GraphState -> tensors -> GNN -> logits + value.

Requires the "rl" extra (torch). Mirrors the style of
scripts/run_graph_state_sanity.py: build a toy incident, show the evidence ->
belief -> GraphState transformation, then show what the learned backbone does
with that GraphState, without ever touching HiddenWorld.
"""

import torch

from adaptive_response import (
    BeliefEngine,
    Edge,
    Environment,
    GraphStateExporter,
    IncidentConfig,
    Observation,
    ObservationBatch,
    Site,
)
from adaptive_response.rl import GNNActorCritic, graph_state_to_tensors


def build_incident() -> IncidentConfig:
    sites = [
        Site(
            id=f"site_{i:02d}",
            x=float(i),
            y=0.0,
            habitat_score=(i % 5) / 4,
            q_model=0.25,
        )
        for i in range(7)
    ]
    edges = [
        Edge(
            src=f"site_{i:02d}",
            dst=f"site_{i + 1:02d}",
            distance=1.0,
            connectivity_weight=1.0,
        )
        for i in range(6)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_03",
        budget=30,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def build_graph_of_size(num_sites: int, seed: int):
    sites = [
        Site(id=f"n{i:02d}", x=float(i), y=0.0, habitat_score=0.5, q_model=0.25)
        for i in range(num_sites)
    ]
    edges = [
        Edge(src=f"n{i:02d}", dst=f"n{i + 1:02d}", distance=1.0, connectivity_weight=1.0)
        for i in range(num_sites - 1)
    ]
    config = IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="n00",
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


def main() -> None:
    torch.manual_seed(0)
    model = GNNActorCritic(hidden_dim=64, num_layers=2)
    param_count = sum(p.numel() for p in model.parameters())

    print("=== GNN/RL BACKBONE SANITY RUN (Checkpoint A) ===")
    print(f"Architecture: hidden_dim=64, num_layers=2, params={param_count:,}")
    print()

    env = Environment(build_incident())
    public = env.reset()
    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    exporter = GraphStateExporter()

    graph_before = exporter.export(public, belief)
    evidence = Observation(site_id="site_04", effort=10, detection=False, round=1)
    belief_after = BeliefEngine.update(
        belief,
        ObservationBatch(observations=(evidence,), round=1, total_effort=10),
        {site.id: 0.25 for site in public.sites},
    )
    graph_after = exporter.export(public, belief_after)

    tensors_before = graph_state_to_tensors(graph_before)
    tensors_after = graph_state_to_tensors(graph_after)

    with torch.no_grad():
        out_before = model(tensors_before)
        out_after = model(tensors_after)

    print(f"node_ids: {graph_before.node_ids}")
    print(f"node_embeddings shape: {tuple(out_before.node_embeddings.shape)}")
    print(f"graph_context shape: {tuple(out_before.graph_context.shape)}")
    print(f"value (before evidence): {out_before.value.item():.4f}")
    print(f"value (after evidence):  {out_after.value.item():.4f}")
    print()

    print("Per-node policy logits BEFORE field evidence (untrained weights):")
    for site_id, logit in zip(out_before.node_ids, out_before.node_logits.tolist()):
        print(f"  {site_id}: logit={logit:+.4f}")
    print()

    print("FIELD EVIDENCE: site_04 -> 0 detections / 10 checks")
    print()

    print("Per-node policy logits AFTER field evidence (same untrained weights):")
    for site_id, logit in zip(out_after.node_ids, out_after.node_logits.tolist()):
        print(f"  {site_id}: logit={logit:+.4f}")
    print()
    print(
        "Note: logits differ before/after only because node_features (belief) "
        "changed; the network weights are identical and untrained. This "
        "demonstrates the GraphState -> tensor -> GNN path is live end to end, "
        "not that the untrained policy has learned anything yet."
    )
    print()

    dist = out_before.masked_action_distribution()
    print("Masked action distribution (feasibility-only mask, untrained):")
    for site_id, prob in zip(out_before.node_ids, dist.probs.tolist()):
        print(f"  {site_id}: p={prob:.4f}")
    print()

    print("Variable graph-size check, same model instance:")
    for num_sites in (12, 20, 24):
        graph = build_graph_of_size(num_sites, seed=20260910 + num_sites)
        tensors = graph_state_to_tensors(graph)
        with torch.no_grad():
            output = model(tensors)
        print(
            f"  N={num_sites:2d}: node_logits shape={tuple(output.node_logits.shape)}, "
            f"value={output.value.item():+.4f}"
        )
    print()

    print("HiddenWorld is never constructed or referenced in this script's model path.")
    print("Only GraphState (public, Bayes-derived) feeds graph_state_to_tensors().")


if __name__ == "__main__":
    main()
