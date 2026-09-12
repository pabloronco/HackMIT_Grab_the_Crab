from __future__ import annotations

import numpy as np

from adaptive_response.water_route import (
    WaterGrid,
    annotate_edges_with_water_routes,
    shortest_water_route_km,
    snap_to_nearest_wet_cell,
)


def _grid(wet: list[list[int]]) -> WaterGrid:
    wet_array = np.asarray(wet, dtype=bool)
    height, width = wet_array.shape
    latitude = np.zeros((height, width), dtype=float)
    longitude = np.zeros((height, width), dtype=float)
    for y in range(height):
        for x in range(width):
            latitude[y, x] = 48.0 + 0.01 * y
            longitude[y, x] = -122.0 + 0.01 * x
    return WaterGrid(latitude=latitude, longitude=longitude, wet=wet_array)


def test_shortest_water_route_can_bend_around_land() -> None:
    grid = _grid(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    route = shortest_water_route_km(grid, (1, 0), (1, 4), max_route_km=20.0)
    assert route is not None
    direct = 4 * 0.01 * 111.0
    assert route > direct


def test_shortest_water_route_returns_none_when_components_disconnected() -> None:
    grid = _grid(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]
    )
    assert shortest_water_route_km(grid, (1, 0), (1, 2), max_route_km=50.0) is None


def test_diagonal_water_step_reduces_staircase_inflation() -> None:
    grid = _grid(
        [
            [1, 1],
            [1, 1],
        ]
    )
    diagonal = shortest_water_route_km(grid, (0, 0), (1, 1), max_route_km=10.0)
    orthogonal = (
        shortest_water_route_km(grid, (0, 0), (0, 1), max_route_km=10.0)
        + shortest_water_route_km(grid, (0, 1), (1, 1), max_route_km=10.0)
    )
    assert diagonal is not None
    assert diagonal < orthogonal


def test_diagonal_does_not_cut_through_land_corner() -> None:
    grid = _grid(
        [
            [1, 0],
            [0, 1],
        ]
    )
    assert shortest_water_route_km(grid, (0, 0), (1, 1), max_route_km=10.0) is None


def test_snap_uses_nearest_wet_cell_not_nearest_land_cell() -> None:
    grid = _grid(
        [
            [0, 1],
            [1, 1],
        ]
    )
    snap = snap_to_nearest_wet_cell(grid, 48.0, -122.0)
    assert (snap.y, snap.x) in {(0, 1), (1, 0)}
    assert snap.distance_km > 0.0


def test_annotation_reports_grid_and_endpoint_corrected_route() -> None:
    grid = _grid(
        [
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    coords = {
        "A": (48.0005, -122.0005),
        "B": (48.0005, -121.9795),
    }
    edges = [{"src": "A", "dst": "B", "distance_km": 1.5}]
    rows, snaps = annotate_edges_with_water_routes(grid, edges, coords, max_route_km=20.0)
    row = rows[0]
    assert set(snaps) == {"A", "B"}
    assert row["salishseacast_route_found"] is True
    grid_route = float(row["salishseacast_water_route_km"])
    total_route = float(row["salishseacast_total_route_proxy_km"])
    assert grid_route > 0.0
    assert total_route == grid_route + snaps["A"].distance_km + snaps["B"].distance_km
    assert float(row["salishseacast_total_detour_ratio"]) > 0.0
