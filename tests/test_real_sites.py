from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptive_response.real_sites import build_real_site_table


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_rows() -> list[dict[str, object]]:
    return [
        {"site_id": "1", "latitude": 48.0, "longitude": -122.0, "habitat": "Lagoon"},
        {"site_id": "1", "latitude": 48.0, "longitude": -122.0, "habitat": "Lagoon"},
        {"site_id": "2", "latitude": 48.1, "longitude": -122.1, "habitat": "Tideflat"},
    ]


def _shorezone_rows() -> list[dict[str, object]]:
    return [
        {
            "site_id": "1",
            "shorezone_unit_id": 10,
            "shorezone_distance_m": 12.5,
            "substrate": "sand",
            "shoreline_type": "Sand flat",
            "exposure": "PROTECTED",
            "eelgrass": "CONTINUOUS",
            "salt_marsh": "ABSENT",
        },
        {
            "site_id": "2",
            "shorezone_unit_id": 20,
            "shorezone_distance_m": 150.0,
            "substrate": "mud and fines",
            "shoreline_type": "Mud flat",
            "exposure": "SEMI-PROTECTED",
            "eelgrass": "ABSENT",
            "salt_marsh": "PATCHY",
        },
    ]


def test_real_site_table_accepts_near_match_and_rejects_far_match(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    shorezone = tmp_path / "shorezone.csv"
    _write_csv(canonical, ["site_id", "latitude", "longitude", "habitat"], _canonical_rows())
    _write_csv(
        shorezone,
        [
            "site_id",
            "shorezone_unit_id",
            "shorezone_distance_m",
            "substrate",
            "shoreline_type",
            "exposure",
            "eelgrass",
            "salt_marsh",
        ],
        _shorezone_rows(),
    )

    rows, summary = build_real_site_table(canonical, shorezone, shorezone_acceptance_m=100.0)
    by_id = {row["site_id"]: row for row in rows}

    assert summary["shorezone_accepted_sites"] == 1
    assert summary["shorezone_rejected_site_ids"] == ["2"]
    assert by_id["1"]["shorezone_accepted"] is True
    assert by_id["1"]["substrate"] == "sand"
    assert by_id["2"]["shorezone_accepted"] is False
    assert by_id["2"]["substrate"] == ""
    assert by_id["2"]["shorezone_distance_m"] == 150.0


def test_real_site_table_surfaces_habitat_ambiguity(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    shorezone = tmp_path / "shorezone.csv"
    rows = _canonical_rows()
    rows[1]["habitat"] = "Channel"
    _write_csv(canonical, ["site_id", "latitude", "longitude", "habitat"], rows)
    _write_csv(
        shorezone,
        [
            "site_id",
            "shorezone_unit_id",
            "shorezone_distance_m",
            "substrate",
            "shoreline_type",
            "exposure",
            "eelgrass",
            "salt_marsh",
        ],
        _shorezone_rows(),
    )

    result, summary = build_real_site_table(canonical, shorezone)
    site1 = next(row for row in result if row["site_id"] == "1")

    assert site1["crabteam_habitat_ambiguous"] is True
    assert site1["crabteam_habitat"] == "Channel|Lagoon"
    assert summary["crabteam_habitat_ambiguous_site_ids"] == ["1"]


def test_real_site_table_requires_one_shorezone_row_per_site(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    shorezone = tmp_path / "shorezone.csv"
    _write_csv(canonical, ["site_id", "latitude", "longitude", "habitat"], _canonical_rows())
    _write_csv(
        shorezone,
        [
            "site_id",
            "shorezone_unit_id",
            "shorezone_distance_m",
            "substrate",
            "shoreline_type",
            "exposure",
            "eelgrass",
            "salt_marsh",
        ],
        _shorezone_rows()[:1],
    )

    with pytest.raises(ValueError, match="Missing ShoreZone diagnostic rows"):
        build_real_site_table(canonical, shorezone)


def test_real_site_table_rejects_nonpositive_threshold(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.csv"
    shorezone = tmp_path / "shorezone.csv"
    _write_csv(canonical, ["site_id", "latitude", "longitude", "habitat"], _canonical_rows())
    _write_csv(
        shorezone,
        [
            "site_id",
            "shorezone_unit_id",
            "shorezone_distance_m",
            "substrate",
            "shoreline_type",
            "exposure",
            "eelgrass",
            "salt_marsh",
        ],
        _shorezone_rows(),
    )

    with pytest.raises(ValueError, match="must be positive"):
        build_real_site_table(canonical, shorezone, shorezone_acceptance_m=0)
