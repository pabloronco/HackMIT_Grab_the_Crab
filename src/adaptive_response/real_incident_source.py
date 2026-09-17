from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .incident_subgraph import audit_all_incident_seeds, extract_incident_subgraph, load_graph_edges
from .models import Edge, IncidentConfig, Site
from .simulator_criticism import _habitat_score, _local_xy_km, read_real_sites

# Product/UI real-incident source.
#
# rl/real_graph_cases.py already implements exactly this (frozen R8 real-site
# x/y + habitat proxy, distance_km-not-route-proxy edge fix - see its module
# docstring), but it lives under adaptive_response.rl, whose __init__.py
# eagerly imports the GNN/torch training stack ("importing adaptive_response
# must never require torch - only `from adaptive_response.rl import ...`
# pays that cost"). A product web app has no business requiring torch, so
# this module re-derives the same real-incident construction from the
# underlying core (non-rl) helpers instead of importing through rl/.
#
# Keep this in lockstep with rl/real_graph_cases.py's build_real_incident_case
# if that function's real-data handling changes; consider factoring both out
# of adaptive_response.rl in a dedicated follow-up so this duplication goes
# away (flagged in the UI-porting checkpoint report rather than done here).

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_GRAPH_EDGES_CSV = REPO_ROOT / "reports" / "milestones" / "r2_real_graph_v0" / "real_graph_v0_edges.csv"
REAL_SITES_CSV = REPO_ROOT / "reports" / "milestones" / "r2_real_graph_v0" / "real_sites_v0.csv"
REAL_SITE_CONTEXT_R8_JSON = REPO_ROOT / "configs" / "real_site_context_r8.json"
ISOLATED_SITE_IDS = ("219", "367", "74")  # from real_graph_v0_audit.json primary_graph.isolated_sites


@dataclass(frozen=True)
class RealIncident:
    label: str
    seed_site_id: str
    incident: IncidentConfig
    below_preferred_min: bool


def _load_real_site_ids_and_edges(edges_csv: Path) -> tuple[list[str], list[dict]]:
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
    site_ids, edge_rows = _load_real_site_ids_and_edges(edges_csv)
    audit = audit_all_incident_seeds(
        site_ids, edge_rows, max_sites=max_sites, preferred_min_sites=preferred_min_sites
    )
    return [
        row["seed_site_id"]
        for row in audit["seed_rows"]
        if preferred_min_sites <= row["selected_sites"] <= max_sites
    ]


def _load_habitat_proxy_scores(context_json: Path) -> dict[str, float]:
    context = json.loads(context_json.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in context["habitat_proxy"]["scores"].items()}


def build_real_incident(
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
) -> RealIncident:
    site_ids, edge_rows = _load_real_site_ids_and_edges(edges_csv)
    selected_ids, selected_edges, summary = extract_incident_subgraph(
        site_ids, edge_rows,
        seed_site_id=seed_site_id, max_sites=max_sites, preferred_min_sites=preferred_min_sites,
    )

    real_rows = {row["site_id"]: row for row in read_real_sites(sites_csv)}
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
        world_model_id=None,
    )
    return RealIncident(
        label=f"real_graph_v0/{seed_site_id}",
        seed_site_id=seed_site_id,
        incident=incident,
        below_preferred_min=bool(summary["below_preferred_min"]),
    )
