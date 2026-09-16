from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..models import Edge, IncidentConfig, Site

# PROPOSED CURRENT DEFAULT (training-only, not an ecological fact): parameter
# ranges used to sample training incidents. Mirrors the "domain randomization"
# targets already listed in the Data/Simulator/Validation Spec (graph size,
# budget, q, habitat), scoped to what the current M1 toy Environment supports
# (a single connected-ish generator family). Widening to real world-model
# families is Pablo/Fede's simulator work, not something this sampler
# pre-empts.


@dataclass(frozen=True)
class IncidentSamplerConfig:
    min_sites: int = 12
    max_sites: int = 24
    min_budget: int = 20
    max_budget: int = 40
    min_teams: int = 1
    max_teams: int = 3
    q_low: float = 0.15
    q_high: float = 0.45
    extra_edge_fraction: float = 0.15  # beyond a spanning tree, for topology variety


def sample_incident(rng: np.random.Generator, config: IncidentSamplerConfig | None = None) -> IncidentConfig:
    """Sample a random IncidentConfig within the current toy environment's assumptions.

    Topology: a random spanning tree (guarantees connectivity, so the frontier/
    belief semantics stay meaningful) plus a small fraction of extra edges for
    non-chain structure. Chains only (as in the M2.5/M3 sanity scripts) would
    under-exercise message passing across branching topologies.
    """

    cfg = config or IncidentSamplerConfig()
    num_sites = int(rng.integers(cfg.min_sites, cfg.max_sites + 1))

    # Random recursive attachment tree: node i (i>=1) attaches to a uniformly
    # random earlier node. Simple, dependency-free, and guarantees a connected
    # graph, avoiding any dependency on a specific networkx random-tree API
    # version.
    graph = nx.Graph()
    graph.add_node(0)
    for i in range(1, num_sites):
        parent = int(rng.integers(0, i))
        graph.add_edge(parent, i)

    num_extra_edges = int(cfg.extra_edge_fraction * num_sites)
    non_edges = list(nx.non_edges(graph))
    if non_edges and num_extra_edges:
        chosen = rng.choice(len(non_edges), size=min(num_extra_edges, len(non_edges)), replace=False)
        for idx in np.atleast_1d(chosen):
            graph.add_edge(*non_edges[int(idx)])

    site_ids = [f"site_{i:02d}" for i in range(num_sites)]
    habitat_scores = rng.uniform(0.0, 1.0, size=num_sites)
    q_values = rng.uniform(cfg.q_low, cfg.q_high, size=num_sites)

    sites = [
        Site(
            id=site_ids[i],
            x=float(i),
            y=0.0,
            habitat_score=float(habitat_scores[i]),
            q_model=float(q_values[i]),
        )
        for i in range(num_sites)
    ]
    edges = [
        Edge(
            src=site_ids[u],
            dst=site_ids[v],
            distance=1.0,
            connectivity_weight=1.0,
        )
        for u, v in graph.edges()
    ]

    initial_detection = site_ids[int(rng.integers(0, num_sites))]
    budget = int(rng.integers(cfg.min_budget, cfg.max_budget + 1))
    teams = int(rng.integers(cfg.min_teams, cfg.max_teams + 1))
    seed = int(rng.integers(0, 2**31 - 1))

    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection=initial_detection,
        budget=budget,
        teams=teams,
        protocol="binary_detection",
        seed=seed,
        world_model_id="toy_graph_cluster_m1",
    )
