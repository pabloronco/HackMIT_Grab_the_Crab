from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable
import urllib.parse
import urllib.request


SHOREZONE_LINE_QUERY_URL = (
    "https://gis.dnr.wa.gov/site3/rest/services/"
    "Aquatics/AQ_Environment/MapServer/46/query"
)


@dataclass(frozen=True, slots=True)
class ShorelineContext:
    """Real DNR ShoreZone shoreline geometry clipped to a lon/lat envelope.

    This is visualization context only. It is not an ecological connectivity model
    and must never be interpreted as dispersal probability or operational travel cost.
    """

    bbox: tuple[float, float, float, float]
    paths: tuple[tuple[tuple[float, float], ...], ...]
    object_count: int
    source_url: str = SHOREZONE_LINE_QUERY_URL

    @property
    def path_count(self) -> int:
        return len(self.paths)


FetchJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _fetch_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "adaptive-first-response/0.1 coastline-render"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected ShoreZone response type.")
    if "error" in payload:
        raise RuntimeError(f"ShoreZone ArcGIS error: {payload['error']}")
    return payload


def fetch_shoreline_context(
    bbox: tuple[float, float, float, float],
    *,
    cache_path: Path | None = None,
    timeout: float = 30.0,
    fetch_json: FetchJson = _fetch_json,
    batch_size: int = 500,
) -> ShorelineContext:
    """Fetch real ShoreZone line geometry for `bbox`, reusing an exact-bbox cache.

    The ArcGIS service has a record limit, so object ids are requested first and
    geometries are then fetched in deterministic batches. If no cache exists and the
    service cannot be reached, this function fails explicitly rather than drawing a
    fabricated coastline.
    """
    _validate_bbox(bbox)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if cache_path is not None and cache_path.is_file():
        cached = _load_cache(cache_path)
        if _same_bbox(cached.bbox, bbox):
            return cached

    envelope = ",".join(f"{value:.8f}" for value in bbox)
    id_payload = fetch_json(
        SHOREZONE_LINE_QUERY_URL,
        {
            "f": "json",
            "where": "1=1",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
        },
        timeout,
    )
    object_ids = sorted(int(value) for value in (id_payload.get("objectIds") or []))
    if not object_ids:
        raise RuntimeError("ShoreZone returned no shoreline features for the graph envelope.")

    paths: list[tuple[tuple[float, float], ...]] = []
    for start in range(0, len(object_ids), batch_size):
        batch = object_ids[start : start + batch_size]
        payload = fetch_json(
            SHOREZONE_LINE_QUERY_URL,
            {
                "f": "json",
                "objectIds": ",".join(str(value) for value in batch),
                "outFields": "OBJECTID",
                "returnGeometry": "true",
                "returnTrueCurves": "false",
                "outSR": 4326,
            },
            timeout,
        )
        for feature in payload.get("features") or []:
            geometry = feature.get("geometry") or {}
            for raw_path in geometry.get("paths") or []:
                path = tuple(
                    (float(point[0]), float(point[1]))
                    for point in raw_path
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                )
                if len(path) >= 2:
                    paths.append(path)

    if not paths:
        raise RuntimeError("ShoreZone features were returned but no polyline geometry was usable.")

    context = ShorelineContext(
        bbox=tuple(float(value) for value in bbox),
        paths=tuple(paths),
        object_count=len(object_ids),
    )
    if cache_path is not None:
        _write_cache(cache_path, context)
    return context


def _write_cache(path: Path, context: ShorelineContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_url": context.source_url,
        "bbox": list(context.bbox),
        "object_count": context.object_count,
        "paths": [
            [[lon, lat] for lon, lat in polyline]
            for polyline in context.paths
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _load_cache(path: Path) -> ShorelineContext:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = tuple(
        tuple((float(point[0]), float(point[1])) for point in raw_path)
        for raw_path in payload["paths"]
    )
    return ShorelineContext(
        bbox=tuple(float(value) for value in payload["bbox"]),
        paths=paths,
        object_count=int(payload["object_count"]),
        source_url=str(payload.get("source_url") or SHOREZONE_LINE_QUERY_URL),
    )


def _same_bbox(a: Iterable[float], b: Iterable[float]) -> bool:
    return all(abs(float(x) - float(y)) <= 1e-9 for x, y in zip(a, b))


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat).")
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox bounds must be strictly increasing.")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("bbox longitude is outside [-180, 180].")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("bbox latitude is outside [-90, 90].")
