from __future__ import annotations

import csv
from pathlib import Path

from adaptive_response.shorezone import (
    distance_to_paths_m,
    load_site_coordinate_summary,
    query_nearest_substrate_unit,
    query_unit_attributes,
)


def test_distance_to_paths_is_zero_on_line() -> None:
    distance = distance_to_paths_m(
        -122.5,
        48.0,
        [[[-122.6, 48.0], [-122.4, 48.0]]],
    )
    assert distance < 0.01


def test_load_site_coordinate_summary_surfaces_coordinate_drift(tmp_path: Path) -> None:
    path = tmp_path / "canonical.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_id", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(
            [
                {"site_id": "1", "latitude": 48.0, "longitude": -122.5},
                {"site_id": "1", "latitude": 48.001, "longitude": -122.5},
                {"site_id": "2", "latitude": 47.5, "longitude": -123.0},
            ]
        )

    rows = load_site_coordinate_summary(path)

    site1 = next(row for row in rows if row["site_id"] == "1")
    assert site1["coordinate_records"] == 2
    assert site1["max_coordinate_deviation_m"] > 50


def test_query_nearest_substrate_unit_selects_nearest_geometry() -> None:
    def fake_fetch(url: str, params: dict[str, object]) -> dict[str, object]:
        return {
            "features": [
                {
                    "attributes": {"OBJECTID": 1, "UNIT_ID": 10, "SUBNAME": "Sand"},
                    "geometry": {"paths": [[[-122.51, 48.01], [-122.49, 48.01]]]},
                },
                {
                    "attributes": {"OBJECTID": 2, "UNIT_ID": 20, "SUBNAME": "Mud"},
                    "geometry": {"paths": [[[-122.51, 48.001], [-122.49, 48.001]]]},
                },
            ]
        }

    match = query_nearest_substrate_unit(48.0, -122.5, fetch_json=fake_fetch)

    assert match is not None
    assert match["unit_id"] == 20
    assert match["substrate"] == "Mud"


def test_query_unit_attributes_preserves_ambiguity() -> None:
    responses = {
        53: [
            {"attributes": {"UNIT_ID": 10, "BC_NAME": "Mud flat"}},
            {"attributes": {"UNIT_ID": 20, "BC_NAME": "Channel"}},
        ],
        50: [
            {"attributes": {"UNIT_ID": 10, "EXP_CALC": "P"}},
            {"attributes": {"UNIT_ID": 20, "EXP_CALC": "SP"}},
        ],
        37: [
            {"attributes": {"UNIT_ID": 10, "EELGRASS": "PATCHY"}},
            {"attributes": {"UNIT_ID": 20, "EELGRASS": "ABSENT"}},
        ],
        44: [
            {"attributes": {"UNIT_ID": 10, "SALTMARSH": "ABSENT"}},
            {"attributes": {"UNIT_ID": 20, "SALTMARSH": "PATCHY"}},
            {"attributes": {"UNIT_ID": 20, "SALTMARSH": "CONTINUOUS"}},
        ],
    }

    def fake_fetch(url: str, params: dict[str, object]) -> dict[str, object]:
        layer_id = int(url.split("/")[-2])
        return {"features": responses[layer_id]}

    attrs = query_unit_attributes([10, 20], fetch_json=fake_fetch)

    assert attrs[10]["shoreline_type"] == "Mud flat"
    assert attrs[10]["eelgrass"] == "PATCHY"
    assert attrs[20]["salt_marsh"] == "CONTINUOUS|PATCHY"
    assert attrs[20]["salt_marsh_ambiguous"] is True
