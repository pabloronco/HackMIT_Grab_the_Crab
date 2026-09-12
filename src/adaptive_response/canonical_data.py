from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data_audit import (
    _complete_key,
    _count_or_none,
    _float_or_none,
    _key_cama,
    _key_coord,
    _key_effort,
    _norm,
    _norm_integer_like,
    _read_csv,
    _require_columns,
    EXPECTED_CAMA_COLUMNS,
    EXPECTED_COORD_COLUMNS,
    EXPECTED_EFFORT_COLUMNS,
)


CANONICAL_FIELDS = [
    "site_id",
    "year",
    "month",
    "habitat",
    "latitude",
    "longitude",
    "trap_sets",
    "effort_missing",
    "effort_unit",
    "cama_count",
    "detected",
    "source_cama",
    "source_effort",
    "source_coords",
]


def _unique_index(
    rows: list[dict[str, str]],
    keys: list[tuple[str, ...]],
    *,
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    valid = [(key, row) for key, row in zip(keys, rows) if _complete_key(key)]
    counts = Counter(key for key, _ in valid)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} contains duplicate join keys; examples={duplicates[:5]}")
    return {key: row for key, row in valid}


def build_canonical_site_visits(
    cama_path: Path,
    effort_path: Path,
    coords_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the R1 canonical site-month table without imputing missing effort.

    Contract:
    - one row per observed CAMA site-year-month survey;
    - coordinates are required and must join uniquely by site-year;
    - effort is joined by site-year-month when present, otherwise left null and flagged;
    - missing CAMA is rejected rather than interpreted as non-detection;
    - PAMA outcome fields are never copied into the canonical green-crab table.
    """

    cama_rows, cama_cols = _read_csv(cama_path)
    effort_rows, effort_cols = _read_csv(effort_path)
    coord_rows, coord_cols = _read_csv(coords_path)

    _require_columns(cama_path, cama_cols, EXPECTED_CAMA_COLUMNS)
    _require_columns(effort_path, effort_cols, EXPECTED_EFFORT_COLUMNS)
    _require_columns(coords_path, coord_cols, EXPECTED_COORD_COLUMNS)

    cama_keys = [_key_cama(row) for row in cama_rows]
    if any(not _complete_key(key) for key in cama_keys):
        raise ValueError("CAMA table contains incomplete site/year/month join keys")
    cama_counts = Counter(cama_keys)
    duplicate_cama = [key for key, count in cama_counts.items() if count > 1]
    if duplicate_cama:
        raise ValueError(f"CAMA table contains duplicate site-month keys; examples={duplicate_cama[:5]}")

    effort_index = _unique_index(
        effort_rows,
        [_key_effort(row) for row in effort_rows],
        label="effort table",
    )
    coord_index = _unique_index(
        coord_rows,
        [_key_coord(row) for row in coord_rows],
        label="coordinate table",
    )

    canonical: list[dict[str, Any]] = []
    missing_effort_keys: list[tuple[str, str, str]] = []

    for row, key in zip(cama_rows, cama_keys):
        site_id, year, month = key
        cama_count = _count_or_none(row.get("CAMA"))
        if cama_count is None:
            raise ValueError(f"Missing CAMA outcome for key={key}; canonical builder will not impute zero")

        coord_key = (site_id, year)
        coord = coord_index.get(coord_key)
        if coord is None:
            raise ValueError(f"Missing coordinates for CAMA key={key}")
        latitude = _float_or_none(coord.get("LatitudeDD"))
        longitude = _float_or_none(coord.get("LongitudeDD"))
        if latitude is None or longitude is None:
            raise ValueError(f"Missing coordinate value for CAMA key={key}")

        effort = effort_index.get(key)
        trap_sets: float | None = None
        effort_missing = effort is None
        if effort is not None:
            trap_sets = _float_or_none(effort.get("trap.sets"))
            if trap_sets is None:
                effort_missing = True
            elif trap_sets < 0:
                raise ValueError(f"Negative trap.sets for key={key}")
        if effort_missing:
            missing_effort_keys.append(key)

        canonical.append(
            {
                "site_id": site_id,
                "year": int(year),
                "month": int(month),
                "habitat": _norm(row.get("habtype")),
                "latitude": latitude,
                "longitude": longitude,
                "trap_sets": trap_sets,
                "effort_missing": effort_missing,
                "effort_unit": "trap_set",
                "cama_count": cama_count,
                "detected": cama_count > 0,
                "source_cama": cama_path.name,
                "source_effort": effort_path.name if effort is not None else "",
                "source_coords": coords_path.name,
            }
        )

    summary = {
        "rows": len(canonical),
        "sites": len({row["site_id"] for row in canonical}),
        "years": sorted({row["year"] for row in canonical}),
        "detections": sum(bool(row["detected"]) for row in canonical),
        "zero_detections": sum(not bool(row["detected"]) for row in canonical),
        "effort_present_rows": sum(not bool(row["effort_missing"]) for row in canonical),
        "effort_missing_rows": sum(bool(row["effort_missing"]) for row in canonical),
        "effort_coverage_fraction": (
            sum(not bool(row["effort_missing"]) for row in canonical) / len(canonical)
            if canonical
            else None
        ),
        "missing_effort_keys": [list(key) for key in missing_effort_keys],
        "coordinate_missing_rows": 0,
        "provenance": {
            "cama": str(cama_path),
            "effort": str(effort_path),
            "coordinates": str(coords_path),
        },
        "notes": [
            "Missing effort is preserved as null and flagged; it is not imputed.",
            "PAMA outcome columns are not included as green-crab evidence.",
            "Temperature and ShoreZone enrichment are intentionally deferred to later R1/R2 steps.",
        ],
    }
    return canonical, summary


def write_canonical_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
