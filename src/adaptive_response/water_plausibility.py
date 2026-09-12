from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Callable


DNR_WATER_BODIES_QUERY_URL = (
    "https://gis.dnr.wa.gov/site2/rest/services/"
    "Public_Forest_Practices/WADNR_PUBLIC_FP_Hydro/MapServer/1/query"
)


def _fetch_json(url: str, params: dict[str, Any], *, timeout: float = 20.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "adaptive-first-response/0.1 water plausibility audit"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected ArcGIS response type")
    if "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    return payload


def sample_interior_segment_points(
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
    *,
    samples: int = 5,
) -> list[tuple[float, float]]:
    """Return evenly spaced interior points, excluding shoreline endpoints."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    points: list[tuple[float, float]] = []
    for index in range(1, samples + 1):
        t = index / (samples + 1)
        lat = src_lat + t * (dst_lat - src_lat)
        lon = src_lon + t * (dst_lon - src_lon)
        points.append((lat, lon))
    return points


def point_is_open_saltwater(
    latitude: float,
    longitude: float,
    *,
    fetch_json: Callable[..., dict[str, Any]] = _fetch_json,
) -> bool:
    """Check whether a point intersects DNR HYDRO 'Open Saltwater' polygon code 116.

    This is a coarse land-barrier sanity check only. It is not a water-routing,
    current, hydrodynamic, or dispersal model.
    """
    payload = fetch_json(
        DNR_WATER_BODIES_QUERY_URL,
        {
            "f": "json",
            "where": "WB_CART_FTR_CD=116",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "returnCountOnly": "true",
        },
    )
    return int(payload.get("count", 0)) > 0


def load_real_site_coordinates(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "latitude", "longitude"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Real-site table missing fields: {sorted(missing)}")
        result: dict[str, tuple[float, float]] = {}
        for row in reader:
            site_id = str(row.get("site_id") or "").strip()
            if not site_id:
                raise ValueError("Real-site table contains blank site_id")
            if site_id in result:
                raise ValueError(f"Duplicate site_id={site_id}")
            result[site_id] = (float(row["latitude"]), float(row["longitude"]))
    return result


def load_candidate_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"src", "dst", "distance_km", "max_local_distance_ratio"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidate edge table missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def annotate_open_saltwater_support(
    edges: list[dict[str, Any]],
    site_coordinates: dict[str, tuple[float, float]],
    *,
    samples_per_edge: int = 5,
    checker: Callable[[float, float], bool] | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Add a coarse straight-line marine-support diagnostic to candidate edges.

    Five interior points are sampled by default. A value of 1.0 means all sampled
    interior points lie inside DNR HYDRO polygons labelled Open Saltwater. A lower
    value may indicate a land barrier, a narrow/estuarine feature not represented
    by this layer, or another map-resolution issue. It must not be used as an
    automatic ecological-edge acceptance rule.
    """
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be >= 1")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    point_jobs: list[tuple[int, int, float, float]] = []
    for edge_index, edge in enumerate(edges):
        src = str(edge["src"])
        dst = str(edge["dst"])
        if src not in site_coordinates or dst not in site_coordinates:
            raise ValueError(f"Candidate edge references unknown site: {src}--{dst}")
        src_lat, src_lon = site_coordinates[src]
        dst_lat, dst_lon = site_coordinates[dst]
        for sample_index, (lat, lon) in enumerate(
            sample_interior_segment_points(
                src_lat,
                src_lon,
                dst_lat,
                dst_lon,
                samples=samples_per_edge,
            )
        ):
            point_jobs.append((edge_index, sample_index, lat, lon))

    results: dict[tuple[int, int], bool] = {}

    if checker is not None:
        for edge_index, sample_index, lat, lon in point_jobs:
            results[(edge_index, sample_index)] = bool(checker(lat, lon))
    else:
        def run(job: tuple[int, int, float, float]) -> tuple[int, int, bool]:
            edge_index, sample_index, lat, lon = job
            return edge_index, sample_index, point_is_open_saltwater(lat, lon)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for edge_index, sample_index, is_water in pool.map(run, point_jobs):
                results[(edge_index, sample_index)] = is_water

    annotated: list[dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        flags = [
            bool(results[(edge_index, sample_index)])
            for sample_index in range(samples_per_edge)
        ]
        hits = sum(flags)
        row = dict(edge)
        row.update(
            {
                "marine_sample_count": samples_per_edge,
                "marine_sample_hits": hits,
                "straight_marine_fraction": hits / samples_per_edge,
                "straight_marine_all": hits == samples_per_edge,
                "straight_marine_none": hits == 0,
                "straight_marine_pattern": "".join("1" if flag else "0" for flag in flags),
            }
        )
        annotated.append(row)
    return annotated


def summarize_water_support(edges: list[dict[str, Any]]) -> dict[str, Any]:
    if not edges:
        return {"edges": 0}
    fractions = [float(row["straight_marine_fraction"]) for row in edges]
    return {
        "edges": len(edges),
        "all_samples_open_saltwater_edges": sum(bool(row["straight_marine_all"]) for row in edges),
        "mixed_open_saltwater_edges": sum(
            not bool(row["straight_marine_all"]) and not bool(row["straight_marine_none"])
            for row in edges
        ),
        "no_samples_open_saltwater_edges": sum(bool(row["straight_marine_none"]) for row in edges),
        "straight_marine_fraction_min": min(fractions),
        "straight_marine_fraction_median": median(fractions),
        "straight_marine_fraction_max": max(fractions),
    }
