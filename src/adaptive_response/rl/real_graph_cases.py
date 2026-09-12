from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..incident_subgraph import audit_all_incident_seeds, extract_incident_subgraph, load_graph_edges
from ..models import Edge, IncidentConfig, Site

# R7 benchmark cases must come from the real monitoring graph, not the toy
# `sample_incident` generator (docs/DEMU_HANDOFF_R7.md engineering integration
# requirements). The exact frozen OOD topology case manifest is explicitly
# still open ("still_open_after_handoff" in configs/benchmark_protocol_r7.json)
# - this module is a provisional, real-topology case sampler to unblock
# integration/smoke/serious-training work now, not a claim of being the
# team's final frozen case manifest.
#
# IMPORTANT LIMITATION (real, not a design choice): per-site latitude/longitude
# and Crab Team habitat calibration live only in the raw Dryad-hosted CSVs
# (data/raw/, see data/raw/README.md), which this environment cannot download
# (Dryad returns HTTP 401/403 to the automated fetch - confirmed by attempting
# scripts/fetch_r0_data.py). Per that same README's own instruction ("do not
# substitute similar-looking files"), site x/y here are a deterministic graph
# layout (not real geography) and habitat_score is a neutral constant (not a
# calibrated value) - both clearly placeholders. The GRAPH TOPOLOGY itself
# (site ids, which sites are adjacent) is real and unchanged: it is read
# directly from the committed R2 milestone receipt
# reports/milestones/r2_real_graph_v0/real_graph_v0_edges.csv plus the
# isolated-site list from real_graph_v0_audit.json, both frozen artifacts of
# the ecology team's own R2 work, not something generated here.

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_GRAPH_EDGES_CSV = REPO_ROOT / "reports" / "milestones" / "r2_real_graph_v0" / "real_graph_v0_edges.csv"
ISOLATED_SITE_IDS = ("219", "367", "74")  # from real_graph_v0_audit.json primary_graph.isolated_sites


@dataclass(frozen=True)
class RealGraphCase:
    label: str
    seed_site_id: str
    incident: IncidentConfig
    below_preferred_min: bool


def load_real_site_ids_and_edges(
    edges_csv: Path = REAL_GRAPH_EDGES_CSV,
) -> tuple[list[str], list[dict]]:
    edge_rows = load_graph_edges(edges_csv)
    site_ids = sorted(
        {str(row["src"]) for row in edge_rows}
        | {str(row["dst"]) for row in edge_rows}
        | set(ISOLATED_SITE_IDS)
    )
    return site_ids, edge_rows


def eligible_incident_seed_sites(
    *,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
    edges_csv: Path = REAL_GRAPH_EDGES_CSV,
) -> list[str]:
    """Seed sites whose incident subgraph lands in [preferred_min_sites, max_sites].

    Matches the "46/49 possible seeds naturally produce 14-17-site incident
    graphs" finding reported by the ecology team - recomputed here directly
    from the real committed graph rather than trusted from a description.
    """

    site_ids, edge_rows = load_real_site_ids_and_edges(edges_csv)
    audit = audit_all_incident_seeds(
        site_ids, edge_rows, max_sites=max_sites, preferred_min_sites=preferred_min_sites
    )
    return [
        row["seed_site_id"]
        for row in audit["seed_rows"]
        if preferred_min_sites <= row["selected_sites"] <= max_sites
    ]


def build_real_incident_case(
    seed_site_id: str,
    *,
    budget: int,
    teams: int = 1,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
    rl_seed: int,
    edges_csv: Path = REAL_GRAPH_EDGES_CSV,
    layout_seed: int = 0,
) -> RealGraphCase:
    """Build one IncidentConfig from the real graph topology around `seed_site_id`.

    See module docstring: topology (site ids, adjacency) is real; x/y is a
    deterministic spring-layout placeholder and habitat_score is a neutral
    constant, both because real coordinates/habitat calibration are not
    fetchable in this environment. `rl_seed` seeds the Environment/world-model
    sampling, not the topology extraction (which is a static, deterministic
    audit of the frozen graph, independent of any RL seed).
    """

    site_ids, edge_rows = load_real_site_ids_and_edges(edges_csv)
    selected_ids, selected_edges, summary = extract_incident_subgraph(
        site_ids, edge_rows,
        seed_site_id=seed_site_id, max_sites=max_sites, preferred_min_sites=preferred_min_sites,
    )

    graph = nx.Graph()
    graph.add_nodes_from(selected_ids)
    for row in selected_edges:
        graph.add_edge(str(row["src"]), str(row["dst"]))
    if graph.number_of_edges() > 0:
        positions = nx.spring_layout(graph, seed=layout_seed)
    else:
        positions = {selected_ids[0]: (0.0, 0.0)}

    sites = [
        Site(
            id=site_id,
            x=float(positions[site_id][0]),
            y=float(positions[site_id][1]),
            habitat_score=0.5,
            q_model={"status": "OPEN", "semantics": "effective_protocol_detectability"},
        )
        for site_id in selected_ids
    ]

    def _edge_weight(row: dict) -> float:
        value = row.get("salishseacast_total_route_proxy_km")
        return float(value) if value not in (None, "") else float(row["distance_km"])

    edges = [
        Edge(
            src=str(row["src"]),
            dst=str(row["dst"]),
            distance=_edge_weight(row),
        )
        for row in selected_edges
    ]

    incident = IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection=seed_site_id,
        budget=budget,
        teams=teams,
        protocol="r7_real_graph_v0",
        seed=rl_seed,
        world_model_id=None,  # filled in by the caller-selected WorldModel family
    )
    return RealGraphCase(
        label=f"real_graph_v0/{seed_site_id}",
        seed_site_id=seed_site_id,
        incident=incident,
        below_preferred_min=bool(summary["below_preferred_min"]),
    )
