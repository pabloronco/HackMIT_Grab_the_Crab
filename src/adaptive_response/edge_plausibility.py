from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .real_graph import haversine_km, load_real_sites


SHOREZONE_SZLINE_URL = (
    "https://gis.dnr.wa.gov/site3/rest/services/"
    "Aquatics/AQ_Environment/MapServer/46"
)


def parse_phy_ident(value: str | None) -> dict[str, str | None]:
    """Parse ShoreZone PHY_IDENT into documented mapping components.

    ShoreZone documents PHY_IDENT as Region/Area/PhysicalUnit/Subunit.
    These are mapping identifiers, not ecological connectivity labels.
    """
    text = str(value or "").strip()
    parts = text.split("/") if text else []
    if len(parts) != 4 or any(not part for part in parts):
        return {
            "phy_ident": text or None,
            "region": None,
            "area": None,
            "physical_unit": None,
            "subunit": None,
        }
    return {
        "phy_ident": text,
        "region": parts[0],
        "area": parts[1],
        "physical_unit": parts[2],
        "subunit": parts[3],
    }


def build_ranked_candidate_edges(
    sites: list[dict[str, Any]],
    *,
    k_max: int = 5,
) -> list[dict[str, Any]]:
    """Build a candidate edge pool with local-distance diagnostics.

    The candidate pool is the union of each site's k_max geographic nearest
    neighbours. No candidate is accepted as an ecological edge here.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if k_max >= len(sites):
        raise ValueError("k_max must be smaller than number of sites")

    ids = [str(site["site_id"]) for site in sites]
    if len(ids) != len(set(ids)):
        raise ValueError("site ids must be unique")

    site_by_id = {str(site["site_id"]): site for site in sites}
    by_site: dict[str, list[tuple[float, str]]] = defaultdict(list)
    distances: dict[tuple[str, str], float] = {}

    for i, src in enumerate(ids):
        a = site_by_id[src]
        for dst in ids[i + 1 :]:
            b = site_by_id[dst]
            distance = haversine_km(
                float(a["latitude"]),
                float(a["longitude"]),
                float(b["latitude"]),
                float(b["longitude"]),
            )
            key = tuple(sorted((src, dst)))
            distances[key] = distance
            by_site[src].append((distance, dst))
            by_site[dst].append((distance, src))

    rank: dict[str, dict[str, int]] = {}
    nearest_distance: dict[str, float] = {}
    for site_id in ids:
        ordered = sorted(by_site[site_id], key=lambda item: (item[0], item[1]))
        rank[site_id] = {neighbor: index + 1 for index, (_, neighbor) in enumerate(ordered)}
        nearest_distance[site_id] = ordered[0][0]

    selected: set[tuple[str, str]] = set()
    for site_id in ids:
        for neighbor, neighbor_rank in rank[site_id].items():
            if neighbor_rank <= k_max:
                selected.add(tuple(sorted((site_id, neighbor))))

    rows: list[dict[str, Any]] = []
    for src, dst in sorted(selected):
        distance = distances[(src, dst)]
        src_rank = rank[src][dst]
        dst_rank = rank[dst][src]
        rows.append(
            {
                "src": src,
                "dst": dst,
                "distance_km": distance,
                "src_rank": src_rank,
                "dst_rank": dst_rank,
                "mutual_knn": src_rank <= k_max and dst_rank <= k_max,
                "src_local_distance_ratio": distance / nearest_distance[src],
                "dst_local_distance_ratio": distance / nearest_distance[dst],
                "max_local_distance_ratio": max(
                    distance / nearest_distance[src],
                    distance / nearest_distance[dst],
                ),
            }
        )
    return rows


def load_site_shorezone_units(real_sites_path: Path) -> dict[str, int | None]:
    with real_sites_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "shorezone_unit_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Real-site table missing fields: {sorted(missing)}")
        result: dict[str, int | None] = {}
        for row in reader:
            site_id = str(row.get("site_id") or "").strip()
            text = str(row.get("shorezone_unit_id") or "").strip()
            result[site_id] = int(float(text)) if text else None
    return result


def _fetch_json(url: str, params: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "adaptive-first-response/0.1 edge plausibility audit"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected ArcGIS response type")
    if "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    return payload


def fetch_shorezone_mapping_metadata(
    unit_ids: list[int],
    *,
    fetch_json=_fetch_json,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Fetch mapping identifiers for matched ShoreZone units.

    This intentionally retrieves mapping/context fields only. REGION/AREA from
    PHY_IDENT may help diagnose candidate edges but must not be interpreted as
    hydrodynamic or dispersal truth.
    """
    unique_ids = sorted(set(int(value) for value in unit_ids))
    if not unique_ids:
        return {}, []

    metadata = fetch_json(SHOREZONE_SZLINE_URL, {"f": "json"})
    available = {
        str(field.get("name"))
        for field in metadata.get("fields", [])
        if isinstance(field, dict) and field.get("name")
    }
    desired = [
        field
        for field in ("UNIT_ID", "PHY_IDENT", "SHORENAME", "UNIT_TYPE")
        if field in available
    ]
    if "UNIT_ID" not in desired:
        raise RuntimeError("ShoreZone SZLine metadata does not expose UNIT_ID")

    ids_sql = ",".join(str(value) for value in unique_ids)
    payload = fetch_json(
        f"{SHOREZONE_SZLINE_URL}/query",
        {
            "f": "json",
            "where": f"UNIT_ID IN ({ids_sql})",
            "outFields": ",".join(desired),
            "returnGeometry": "false",
        },
    )

    values: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for feature in payload.get("features", []):
        attrs = feature.get("attributes") or {}
        unit_id = attrs.get("UNIT_ID")
        if unit_id is None:
            continue
        unit = int(unit_id)
        for field in desired:
            if field == "UNIT_ID":
                continue
            value = attrs.get(field)
            if value not in (None, ""):
                values[unit][field].add(str(value))

    result: dict[int, dict[str, Any]] = {}
    for unit_id in unique_ids:
        row: dict[str, Any] = {"UNIT_ID": unit_id}
        for field in desired:
            if field == "UNIT_ID":
                continue
            distinct = sorted(values[unit_id].get(field, set()))
            row[field] = "|".join(distinct) if distinct else None
            row[f"{field}_ambiguous"] = len(distinct) > 1
        phy = parse_phy_ident(row.get("PHY_IDENT"))
        row.update(
            {
                "shorezone_region": phy["region"],
                "shorezone_area": phy["area"],
                "shorezone_physical_unit": phy["physical_unit"],
                "shorezone_subunit": phy["subunit"],
            }
        )
        result[unit_id] = row
    return result, desired


def annotate_candidate_edges(
    edges: list[dict[str, Any]],
    site_units: dict[str, int | None],
    unit_metadata: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for edge in edges:
        row = dict(edge)
        src = str(row["src"])
        dst = str(row["dst"])
        src_unit = site_units.get(src)
        dst_unit = site_units.get(dst)
        src_meta = unit_metadata.get(src_unit, {}) if src_unit is not None else {}
        dst_meta = unit_metadata.get(dst_unit, {}) if dst_unit is not None else {}

        src_region = src_meta.get("shorezone_region")
        dst_region = dst_meta.get("shorezone_region")
        src_area = src_meta.get("shorezone_area")
        dst_area = dst_meta.get("shorezone_area")
        src_shore = src_meta.get("SHORENAME")
        dst_shore = dst_meta.get("SHORENAME")

        row.update(
            {
                "src_unit_id": src_unit,
                "dst_unit_id": dst_unit,
                "src_phy_ident": src_meta.get("PHY_IDENT"),
                "dst_phy_ident": dst_meta.get("PHY_IDENT"),
                "src_shorezone_region": src_region,
                "dst_shorezone_region": dst_region,
                "src_shorezone_area": src_area,
                "dst_shorezone_area": dst_area,
                "src_shorename": src_shore,
                "dst_shorename": dst_shore,
                "same_shorezone_region": (
                    src_region is not None and dst_region is not None and src_region == dst_region
                ),
                "same_shorezone_area": (
                    src_region is not None
                    and dst_region is not None
                    and src_area is not None
                    and dst_area is not None
                    and src_region == dst_region
                    and src_area == dst_area
                ),
                "same_shorename": (
                    src_shore is not None and dst_shore is not None and src_shore == dst_shore
                ),
            }
        )
        annotated.append(row)
    return annotated


def summarize_edge_candidates(edges: list[dict[str, Any]]) -> dict[str, Any]:
    if not edges:
        return {"edges": 0}
    distances = [float(edge["distance_km"]) for edge in edges]
    ratios = [float(edge["max_local_distance_ratio"]) for edge in edges]
    return {
        "edges": len(edges),
        "mutual_knn_edges": sum(bool(edge.get("mutual_knn")) for edge in edges),
        "same_shorezone_region_edges": sum(
            bool(edge.get("same_shorezone_region")) for edge in edges
        ),
        "same_shorezone_area_edges": sum(
            bool(edge.get("same_shorezone_area")) for edge in edges
        ),
        "same_shorename_edges": sum(bool(edge.get("same_shorename")) for edge in edges),
        "distance_km_min": min(distances),
        "distance_km_median": median(distances),
        "distance_km_max": max(distances),
        "local_distance_ratio_median": median(ratios),
        "local_distance_ratio_max": max(ratios),
    }


def build_edge_plausibility_audit(
    real_sites_path: Path,
    *,
    k_max: int = 5,
    fetch_json=_fetch_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sites = load_real_sites(real_sites_path)
    candidates = build_ranked_candidate_edges(sites, k_max=k_max)
    site_units = load_site_shorezone_units(real_sites_path)
    unit_ids = [value for value in site_units.values() if value is not None]
    unit_metadata, fields = fetch_shorezone_mapping_metadata(
        [int(value) for value in unit_ids], fetch_json=fetch_json
    )
    annotated = annotate_candidate_edges(candidates, site_units, unit_metadata)
    summary = summarize_edge_candidates(annotated)
    summary.update(
        {
            "sites": len(sites),
            "k_max_candidate_pool": k_max,
            "shorezone_metadata_fields": fields,
            "notes": [
                "Candidate edges are generated geographically, then diagnosed; none are accepted yet.",
                "Mutual-kNN and local-distance ratios are geometry diagnostics, not ecological truth.",
                "ShoreZone REGION/AREA are mapping identifiers parsed from PHY_IDENT, not hydrodynamic basins.",
                "SHORENAME, when available, is context only and does not by itself prove dispersal connectivity.",
                "Final graph may have variable degree and may be disconnected; global connectivity is not required.",
            ],
        }
    )
    return annotated, summary
