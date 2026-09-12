from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


SHOREZONE_BASE_URL = (
    "https://gis.dnr.wa.gov/site3/rest/services/"
    "Aquatics/AQ_Environment/MapServer"
)

# Production DNR ShoreZone layers. These are explicit because the layer ids are
# part of the public ArcGIS service contract we are auditing.
SHOREZONE_LAYERS = {
    "shoreline_type": {"id": 53, "field": "BC_NAME"},
    "substrate": {"id": 52, "field": "SUBNAME"},
    "exposure": {"id": 50, "field": "EXP_CALC"},
    "eelgrass": {"id": 37, "field": "EELGRASS"},
    "salt_marsh": {"id": 44, "field": "SALTMARSH"},
}


def _fetch_json(url: str, params: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "adaptive-first-response/0.1 ShoreZone audit"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"ArcGIS service error: {payload['error']}")
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected ArcGIS response type")
    return payload


def _xy_m(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Local equirectangular approximation; adequate for sub-km matching diagnostics."""
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def _point_segment_distance_m(
    lon0: float,
    lat0: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = _xy_m(a[0], a[1], lon0, lat0)
    bx, by = _xy_m(b[0], b[1], lon0, lat0)
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(ax, ay)
    t = -(ax * dx + ay * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(px, py)


def distance_to_paths_m(
    lon: float,
    lat: float,
    paths: Iterable[Iterable[Iterable[float]]],
) -> float:
    best = math.inf
    for path in paths:
        points = [tuple(map(float, point[:2])) for point in path]
        if len(points) == 1:
            x, y = _xy_m(points[0][0], points[0][1], lon, lat)
            best = min(best, math.hypot(x, y))
            continue
        for a, b in zip(points, points[1:]):
            best = min(best, _point_segment_distance_m(lon, lat, a, b))
    return best


def load_site_coordinate_summary(canonical_path: Path) -> list[dict[str, Any]]:
    """Collapse site-month rows to one diagnostic point per site without hiding drift.

    We use the median coordinate only for the ShoreZone *audit*. The final graph
    should retain the source coordinates and document any material site movement.
    """
    by_site: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with canonical_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "latitude", "longitude"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical table missing fields: {sorted(missing)}")
        for row in reader:
            site_id = str(row["site_id"]).strip()
            by_site[site_id].append((float(row["latitude"]), float(row["longitude"])))

    result: list[dict[str, Any]] = []
    for site_id, coords in sorted(by_site.items(), key=lambda item: item[0]):
        lat = median(value[0] for value in coords)
        lon = median(value[1] for value in coords)
        max_spread_m = 0.0
        for c_lat, c_lon in coords:
            x, y = _xy_m(c_lon, c_lat, lon, lat)
            max_spread_m = max(max_spread_m, math.hypot(x, y))
        result.append(
            {
                "site_id": site_id,
                "latitude": lat,
                "longitude": lon,
                "coordinate_records": len(coords),
                "max_coordinate_deviation_m": max_spread_m,
            }
        )
    return result


def query_nearest_substrate_unit(
    latitude: float,
    longitude: float,
    *,
    search_radius_m: int = 2_000,
    fetch_json=_fetch_json,
) -> dict[str, Any] | None:
    layer_id = SHOREZONE_LAYERS["substrate"]["id"]
    payload = fetch_json(
        f"{SHOREZONE_BASE_URL}/{layer_id}/query",
        {
            "f": "json",
            "where": "1=1",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "distance": search_radius_m,
            "units": "esriSRUnit_Meter",
            "outFields": "OBJECTID,UNIT_ID,SUBNAME",
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    features = payload.get("features", [])
    candidates: list[dict[str, Any]] = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        paths = geometry.get("paths") or []
        if not paths:
            continue
        distance_m = distance_to_paths_m(longitude, latitude, paths)
        attrs = feature.get("attributes") or {}
        if attrs.get("UNIT_ID") is None:
            continue
        candidates.append(
            {
                "unit_id": int(attrs["UNIT_ID"]),
                "substrate": attrs.get("SUBNAME"),
                "distance_m": distance_m,
                "object_id": attrs.get("OBJECTID"),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["distance_m"])


def query_unit_attributes(
    unit_ids: Iterable[int],
    *,
    fetch_json=_fetch_json,
) -> dict[int, dict[str, Any]]:
    """Fetch decoded ShoreZone thematic fields for already matched UNIT_IDs."""
    unique_ids = sorted(set(int(value) for value in unit_ids))
    result: dict[int, dict[str, Any]] = {unit_id: {} for unit_id in unique_ids}
    if not unique_ids:
        return result

    ids_sql = ",".join(str(value) for value in unique_ids)
    for name, spec in SHOREZONE_LAYERS.items():
        if name == "substrate":
            continue
        field = str(spec["field"])
        payload = fetch_json(
            f"{SHOREZONE_BASE_URL}/{spec['id']}/query",
            {
                "f": "json",
                "where": f"UNIT_ID IN ({ids_sql})",
                "outFields": f"UNIT_ID,{field}",
                "returnGeometry": "false",
            },
        )
        values: dict[int, set[str]] = defaultdict(set)
        for feature in payload.get("features", []):
            attrs = feature.get("attributes") or {}
            unit_id = attrs.get("UNIT_ID")
            value = attrs.get(field)
            if unit_id is None or value in (None, ""):
                continue
            values[int(unit_id)].add(str(value))
        for unit_id in unique_ids:
            distinct = sorted(values.get(unit_id, set()))
            result[unit_id][name] = "|".join(distinct) if distinct else None
            result[unit_id][f"{name}_ambiguous"] = len(distinct) > 1
    return result


def audit_shorezone_matches(
    canonical_path: Path,
    *,
    search_radius_m: int = 2_000,
    fetch_json=_fetch_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sites = load_site_coordinate_summary(canonical_path)
    matches: list[dict[str, Any]] = []

    for site in sites:
        nearest = query_nearest_substrate_unit(
            site["latitude"],
            site["longitude"],
            search_radius_m=search_radius_m,
            fetch_json=fetch_json,
        )
        record = dict(site)
        if nearest is None:
            record.update(
                {
                    "shorezone_unit_id": None,
                    "shorezone_distance_m": None,
                    "substrate": None,
                }
            )
        else:
            record.update(
                {
                    "shorezone_unit_id": nearest["unit_id"],
                    "shorezone_distance_m": nearest["distance_m"],
                    "substrate": nearest["substrate"],
                }
            )
        matches.append(record)

    unit_ids = [
        int(row["shorezone_unit_id"])
        for row in matches
        if row["shorezone_unit_id"] is not None
    ]
    attributes = query_unit_attributes(unit_ids, fetch_json=fetch_json)
    for row in matches:
        unit_id = row["shorezone_unit_id"]
        if unit_id is not None:
            row.update(attributes.get(int(unit_id), {}))

    distances = [
        float(row["shorezone_distance_m"])
        for row in matches
        if row["shorezone_distance_m"] is not None
    ]
    thresholds = [50, 100, 250, 500, 1000]
    summary = {
        "sites": len(matches),
        "search_radius_m": search_radius_m,
        "matched_within_search_radius": len(distances),
        "no_match_within_search_radius": len(matches) - len(distances),
        "nearest_distance_min_m": min(distances) if distances else None,
        "nearest_distance_median_m": median(distances) if distances else None,
        "nearest_distance_max_m": max(distances) if distances else None,
        "distance_threshold_counts": {
            str(threshold): sum(distance <= threshold for distance in distances)
            for threshold in thresholds
        },
        "sites_with_coordinate_deviation_over_100m": sum(
            float(row["max_coordinate_deviation_m"]) > 100.0 for row in matches
        ),
        "field_nonmissing": {
            field: sum(row.get(field) not in (None, "") for row in matches)
            for field in ["substrate", "shoreline_type", "exposure", "eelgrass", "salt_marsh"]
        },
        "field_ambiguous_sites": {
            field: sum(bool(row.get(f"{field}_ambiguous", False)) for row in matches)
            for field in ["shoreline_type", "exposure", "eelgrass", "salt_marsh"]
        },
    }
    return matches, summary
