from __future__ import annotations

import csv
import heapq
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .real_graph import haversine_km


SALISHSEACAST_MESH_CSV_URL = (
    "https://salishsea.eos.ubc.ca/erddap/griddap/"
    "ubcSSn2DMeshMaskV21-08.csv"
)
GRID_Y = 898
GRID_X = 398


@dataclass(frozen=True)
class GridSnap:
    y: int
    x: int
    latitude: float
    longitude: float
    distance_km: float


@dataclass
class WaterGrid:
    latitude: np.ndarray
    longitude: np.ndarray
    wet: np.ndarray

    def __post_init__(self) -> None:
        if self.latitude.shape != self.longitude.shape or self.latitude.shape != self.wet.shape:
            raise ValueError("latitude, longitude and wet arrays must have identical shapes")
        if self.latitude.ndim != 2:
            raise ValueError("water grid arrays must be 2D")
        self.wet = self.wet.astype(bool, copy=False)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.latitude.shape[0]), int(self.latitude.shape[1])


def _download_salishseacast_csv(cache_path: Path, *, timeout: float = 180.0) -> None:
    """Download one cached static SalishSeaCast T-grid land/water mask extract."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    query = (
        f"glamt[0][0:1:{GRID_Y - 1}][0:1:{GRID_X - 1}],"
        f"gphit[0][0:1:{GRID_Y - 1}][0:1:{GRID_X - 1}],"
        f"tmaskutil[0][0:1:{GRID_Y - 1}][0:1:{GRID_X - 1}]"
    )
    url = f"{SALISHSEACAST_MESH_CSV_URL}?{urllib.parse.quote(query, safe='[],=:')}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "adaptive-first-response/0.1 SalishSeaCast route audit"},
    )
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_salishseacast_grid(cache_path: Path) -> WaterGrid:
    """Load/download SalishSeaCast T-grid geometry and explicit land/water mask."""
    if not cache_path.is_file():
        _download_salishseacast_csv(cache_path)

    latitude = np.full((GRID_Y, GRID_X), np.nan, dtype=np.float64)
    longitude = np.full((GRID_Y, GRID_X), np.nan, dtype=np.float64)
    wet = np.zeros((GRID_Y, GRID_X), dtype=bool)
    parsed = 0

    with cache_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gridY", "gridX", "glamt", "gphit", "tmaskutil"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"SalishSeaCast CSV missing fields: {sorted(missing)}")
        for row in reader:
            try:
                y = int(float(row["gridY"]))
                x = int(float(row["gridX"]))
                lon = float(row["glamt"])
                lat = float(row["gphit"])
                water_flag = int(float(row["tmaskutil"]))
            except (TypeError, ValueError):
                continue
            if not (0 <= y < GRID_Y and 0 <= x < GRID_X):
                continue
            latitude[y, x] = lat
            longitude[y, x] = lon
            wet[y, x] = water_flag == 1
            parsed += 1

    expected = GRID_Y * GRID_X
    if parsed != expected:
        raise ValueError(f"Expected {expected} SalishSeaCast grid cells, parsed {parsed}")

    wet &= np.isfinite(latitude) & np.isfinite(longitude)
    return WaterGrid(latitude=latitude, longitude=longitude, wet=wet)


def snap_to_nearest_wet_cell(grid: WaterGrid, latitude: float, longitude: float) -> GridSnap:
    """Snap a monitoring site to the nearest wet SalishSeaCast T-grid cell."""
    wet_y, wet_x = np.nonzero(grid.wet)
    if len(wet_y) == 0:
        raise ValueError("water grid contains no wet cells")
    wet_lat = grid.latitude[wet_y, wet_x]
    wet_lon = grid.longitude[wet_y, wet_x]

    lat_scale = math.cos(math.radians(float(latitude)))
    dlat = wet_lat - float(latitude)
    dlon = (wet_lon - float(longitude)) * lat_scale
    index = int(np.argmin(dlat * dlat + dlon * dlon))
    y = int(wet_y[index])
    x = int(wet_x[index])
    cell_lat = float(grid.latitude[y, x])
    cell_lon = float(grid.longitude[y, x])
    return GridSnap(
        y=y,
        x=x,
        latitude=cell_lat,
        longitude=cell_lon,
        distance_km=haversine_km(latitude, longitude, cell_lat, cell_lon),
    )


def _neighbor_cells(grid: WaterGrid, y: int, x: int):
    """Yield orthogonal and safe diagonal wet neighbours.

    Diagonals reduce staircase inflation on the curvilinear grid. To avoid
    cutting through a land corner, a diagonal is allowed only when both
    orthogonal side cells are also wet.
    """
    height, width = grid.shape
    for dy, dx in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        ny = y + dy
        nx = x + dx
        if not (0 <= ny < height and 0 <= nx < width and bool(grid.wet[ny, nx])):
            continue
        if dy != 0 and dx != 0:
            side_a = (y + dy, x)
            side_b = (y, x + dx)
            if not (
                0 <= side_a[0] < height
                and 0 <= side_a[1] < width
                and 0 <= side_b[0] < height
                and 0 <= side_b[1] < width
                and bool(grid.wet[side_a])
                and bool(grid.wet[side_b])
            ):
                continue
        yield ny, nx


def shortest_water_route_km(
    grid: WaterGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_route_km: float = 250.0,
) -> float | None:
    """A* shortest path constrained to wet SalishSeaCast T-grid cells.

    This is a geometric water-route proxy only. It contains no currents,
    directionality, travel time or species-dispersal physics.
    """
    sy, sx = start
    gy, gx = goal
    if not bool(grid.wet[sy, sx]) or not bool(grid.wet[gy, gx]):
        return None
    if start == goal:
        return 0.0

    goal_lat = float(grid.latitude[gy, gx])
    goal_lon = float(grid.longitude[gy, gx])

    def heuristic(y: int, x: int) -> float:
        return haversine_km(
            float(grid.latitude[y, x]),
            float(grid.longitude[y, x]),
            goal_lat,
            goal_lon,
        )

    best: dict[tuple[int, int], float] = {start: 0.0}
    queue: list[tuple[float, float, int, int]] = [(heuristic(sy, sx), 0.0, sy, sx)]

    while queue:
        _, distance, y, x = heapq.heappop(queue)
        if distance != best.get((y, x)):
            continue
        if distance > max_route_km:
            continue
        if (y, x) == goal:
            return distance
        lat = float(grid.latitude[y, x])
        lon = float(grid.longitude[y, x])
        for ny, nx in _neighbor_cells(grid, y, x):
            step = haversine_km(
                lat,
                lon,
                float(grid.latitude[ny, nx]),
                float(grid.longitude[ny, nx]),
            )
            new_distance = distance + step
            if new_distance > max_route_km:
                continue
            key = (ny, nx)
            if new_distance < best.get(key, math.inf):
                best[key] = new_distance
                heapq.heappush(
                    queue,
                    (new_distance + heuristic(ny, nx), new_distance, ny, nx),
                )
    return None


def annotate_edges_with_water_routes(
    grid: WaterGrid,
    edges: list[dict[str, Any]],
    site_coordinates: dict[str, tuple[float, float]],
    *,
    max_route_km: float = 250.0,
) -> tuple[list[dict[str, Any]], dict[str, GridSnap]]:
    """Annotate candidate edges with curved water-only path diagnostics.

    ``salishseacast_water_route_km`` is the route between snapped wet cells.
    ``salishseacast_total_route_proxy_km`` adds both endpoint snap distances so
    the reported proxy does not silently ignore the site-to-grid displacement.
    """
    relevant_sites = sorted(
        {str(row["src"]) for row in edges} | {str(row["dst"]) for row in edges}
    )
    snaps: dict[str, GridSnap] = {}
    for site_id in relevant_sites:
        if site_id not in site_coordinates:
            raise ValueError(f"Missing coordinates for site {site_id}")
        lat, lon = site_coordinates[site_id]
        snaps[site_id] = snap_to_nearest_wet_cell(grid, lat, lon)

    annotated: list[dict[str, Any]] = []
    route_cache: dict[tuple[tuple[int, int], tuple[int, int]], float | None] = {}
    for row in edges:
        src = str(row["src"])
        dst = str(row["dst"])
        src_snap = snaps[src]
        dst_snap = snaps[dst]
        a = (src_snap.y, src_snap.x)
        b = (dst_snap.y, dst_snap.x)
        key = tuple(sorted((a, b)))
        if key not in route_cache:
            route_cache[key] = shortest_water_route_km(
                grid,
                a,
                b,
                max_route_km=max_route_km,
            )
        grid_route = route_cache[key]
        direct = float(row["distance_km"])
        total_route_proxy = (
            grid_route + src_snap.distance_km + dst_snap.distance_km
            if grid_route is not None
            else None
        )
        out = dict(row)
        out.update(
            {
                "salishseacast_src_snap_km": src_snap.distance_km,
                "salishseacast_dst_snap_km": dst_snap.distance_km,
                "salishseacast_route_found": grid_route is not None,
                "salishseacast_water_route_km": grid_route,
                "salishseacast_detour_ratio": (
                    grid_route / direct if grid_route is not None and direct > 0.0 else None
                ),
                "salishseacast_total_route_proxy_km": total_route_proxy,
                "salishseacast_total_detour_ratio": (
                    total_route_proxy / direct
                    if total_route_proxy is not None and direct > 0.0
                    else None
                ),
            }
        )
        annotated.append(out)
    return annotated, snaps
