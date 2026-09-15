from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from ..incident_subgraph import audit_all_incident_seeds, extract_incident_subgraph, load_graph_edges
from ..models import Edge, IncidentConfig, Site
from ..simulator_criticism import _habitat_score, _local_xy_km, read_real_sites

# R8 real-site context (docs/R8_DEMU_BENCHMARK_REVIEW.md, configs/real_site_context_r8.json):
# formal training/evaluation must use the versioned monitoring-authoritative
# coordinates and the frozen R5 habitat proxy. Graph-layout coordinates and a
# neutral habitat placeholder (this module's R7 provisional behavior) are
# forbidden for the final benchmark. Topology (site ids, adjacency) was
# already real in R7 and is unchanged here - only x/y and habitat_score move
# from placeholders to the versioned real values.

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_GRAPH_EDGES_CSV = REPO_ROOT / "reports" / "milestones" / "r2_real_graph_v0" / "real_graph_v0_edges.csv"
REAL_SITES_CSV = REPO_ROOT / "reports" / "milestones" / "r2_real_graph_v0" / "real_sites_v0.csv"
REAL_SITE_CONTEXT_R8_JSON = REPO_ROOT / "configs" / "real_site_context_r8.json"
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


def _load_habitat_proxy_scores(context_json: Path = REAL_SITE_CONTEXT_R8_JSON) -> dict[str, float]:
    context = json.loads(context_json.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in context["habitat_proxy"]["scores"].items()}


def _real_site_rows(sites_csv: Path = REAL_SITES_CSV) -> dict[str, dict]:
    return {row["site_id"]: row for row in read_real_sites(sites_csv)}


def build_real_incident_case(
    seed_site_id: str,
    *,
    budget: int,
    teams: int = 1,
    max_sites: int = 20,
    preferred_min_sites: int = 12,
    rl_seed: int,
    edges_csv: Path = REAL_GRAPH_EDGES_CSV,
    sites_csv: Path = REAL_SITES_CSV,
    context_json: Path = REAL_SITE_CONTEXT_R8_JSON,
) -> RealGraphCase:
    """Build one IncidentConfig from the real graph topology + real site context.

    Topology (site ids, adjacency) and now x/y (real lat/lon, locally
    projected to km around the incident's own site median) and habitat_score
    (frozen R5 habitat proxy by crabteam_habitat class) are all real, versioned
    values - see module docstring. `rl_seed` seeds the Environment/world-model
    sampling, not the topology/context extraction (both are static and
    deterministic given the frozen R2/R8 receipts).
    """

    site_ids, edge_rows = load_real_site_ids_and_edges(edges_csv)
    selected_ids, selected_edges, summary = extract_incident_subgraph(
        site_ids, edge_rows,
        seed_site_id=seed_site_id, max_sites=max_sites, preferred_min_sites=preferred_min_sites,
    )

    real_rows = _real_site_rows(sites_csv)
    habitat_scores = _load_habitat_proxy_scores(context_json)
    missing = [site_id for site_id in selected_ids if site_id not in real_rows]
    if missing:
        raise ValueError(f"real_sites_v0.csv is missing site ids: {missing}")

    lat0 = median(real_rows[site_id]["latitude"] for site_id in selected_ids)
    lon0 = median(real_rows[site_id]["longitude"] for site_id in selected_ids)

    sites = []
    for site_id in selected_ids:
        row = real_rows[site_id]
        x, y = _local_xy_km(row["latitude"], row["longitude"], lat0=lat0, lon0=lon0)
        sites.append(
            Site(
                id=site_id,
                x=x,
                y=y,
                habitat_score=_habitat_score(row["crabteam_habitat"], habitat_scores),
                q_model={"status": "OPEN", "semantics": "effective_protocol_detectability"},
            )
        )

    def _route_proxy_km(row: dict) -> float | None:
        value = row.get("salishseacast_total_route_proxy_km")
        return float(value) if value not in (None, "") else None

    # R8 correction (review 2026-09-15): distance_km is the direct edge
    # distance and is what Edge.distance / the GraphState edge-distance
    # feature must carry - it is what the planner (including the GNN) sees.
    # salishseacast_total_route_proxy_km is a real navigable-route distance
    # but is audit/context metadata only; it must not silently replace the
    # direct-distance graph input. Kept on Edge.travel_cost, which no
    # GraphState feature currently reads (graph_state.py builds edge
    # features from (distance, connectivity_weight) only) - so this
    # preserves the value without it leaking into the planner graph.
    edges = [
        Edge(
            src=str(row["src"]),
            dst=str(row["dst"]),
            distance=float(row["distance_km"]),
            travel_cost=_route_proxy_km(row),
        )
        for row in selected_edges
    ]

    incident = IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection=seed_site_id,
        budget=budget,
        teams=teams,
        protocol="r8_real_graph_v0",
        seed=rl_seed,
        world_model_id=None,  # filled in by the caller-selected WorldModel family
    )
    return RealGraphCase(
        label=f"real_graph_v0/{seed_site_id}",
        seed_site_id=seed_site_id,
        incident=incident,
        below_preferred_min=bool(summary["below_preferred_min"]),
    )
