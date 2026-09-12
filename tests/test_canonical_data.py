from __future__ import annotations

import csv
from pathlib import Path

import pytest

from adaptive_response.canonical_data import build_canonical_site_visits


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_canonical_builder_preserves_missing_effort_without_imputation(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"
    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [
            {"Year": 2023, "SiteID": 1, "Month": "April", "habtype": "Lagoon", "CAMA": 0},
            {"Year": 2023, "SiteID": 1, "Month": "May", "habtype": "Lagoon", "CAMA": 2},
        ],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets"],
        [{"SiteID": 1, "month": 4, "Year": 2023, "trap.sets": 6}],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [{"SiteID": 1, "Year": 2023, "LatitudeDD": 48.1, "LongitudeDD": -122.5}],
    )

    rows, summary = build_canonical_site_visits(cama, effort, coords)

    assert rows[0]["month"] == 4
    assert rows[0]["trap_sets"] == 6.0
    assert rows[0]["effort_missing"] is False
    assert rows[0]["detected"] is False
    assert rows[1]["month"] == 5
    assert rows[1]["trap_sets"] is None
    assert rows[1]["effort_missing"] is True
    assert rows[1]["detected"] is True
    assert summary["effort_present_rows"] == 1
    assert summary["effort_missing_rows"] == 1
    assert summary["missing_effort_keys"] == [["1", "2023", "5"]]


def test_canonical_builder_rejects_missing_cama_outcome(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"
    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [{"Year": 2023, "SiteID": 1, "Month": "April", "habtype": "Lagoon", "CAMA": ""}],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets"],
        [{"SiteID": 1, "month": 4, "Year": 2023, "trap.sets": 6}],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [{"SiteID": 1, "Year": 2023, "LatitudeDD": 48.1, "LongitudeDD": -122.5}],
    )

    with pytest.raises(ValueError, match="Missing CAMA outcome"):
        build_canonical_site_visits(cama, effort, coords)


def test_canonical_builder_requires_real_coordinates(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"
    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [{"Year": 2023, "SiteID": 1, "Month": "April", "habtype": "Lagoon", "CAMA": 0}],
    )
    _write_csv(effort, ["SiteID", "month", "Year", "trap.sets"], [])
    _write_csv(coords, ["SiteID", "Year", "LatitudeDD", "LongitudeDD"], [])

    with pytest.raises(ValueError, match="Missing coordinates"):
        build_canonical_site_visits(cama, effort, coords)


def test_canonical_builder_never_copies_pama_outcomes(tmp_path: Path) -> None:
    cama = tmp_path / "cama.csv"
    effort = tmp_path / "effort.csv"
    coords = tmp_path / "coords.csv"
    _write_csv(
        cama,
        ["Year", "SiteID", "Month", "habtype", "CAMA"],
        [{"Year": 2023, "SiteID": 1, "Month": "April", "habtype": "Lagoon", "CAMA": 1}],
    )
    _write_csv(
        effort,
        ["SiteID", "month", "Year", "trap.sets", "total.pama", "CPUE"],
        [{"SiteID": 1, "month": 4, "Year": 2023, "trap.sets": 6, "total.pama": 99, "CPUE": 16.5}],
    )
    _write_csv(
        coords,
        ["SiteID", "Year", "LatitudeDD", "LongitudeDD"],
        [{"SiteID": 1, "Year": 2023, "LatitudeDD": 48.1, "LongitudeDD": -122.5}],
    )

    rows, _ = build_canonical_site_visits(cama, effort, coords)
    assert "total.pama" not in rows[0]
    assert "CPUE" not in rows[0]
