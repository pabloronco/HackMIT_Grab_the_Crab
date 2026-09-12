from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CAMA_COLUMNS = {"Year", "SiteID", "Month", "habtype", "CAMA"}
EXPECTED_EFFORT_COLUMNS = {"SiteID", "month", "Year", "trap.sets"}
EXPECTED_COORD_COLUMNS = {"SiteID", "Year", "LatitudeDD", "LongitudeDD"}
EXPECTED_TEMP_COLUMNS = {"SiteNum", "year", "mdy", "x"}

MONTH_NAME_TO_NUMBER = {
    "january": "1",
    "jan": "1",
    "february": "2",
    "feb": "2",
    "march": "3",
    "mar": "3",
    "april": "4",
    "apr": "4",
    "may": "5",
    "june": "6",
    "jun": "6",
    "july": "7",
    "jul": "7",
    "august": "8",
    "aug": "8",
    "september": "9",
    "sep": "9",
    "sept": "9",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return list(reader), fieldnames


def _require_columns(path: Path, columns: list[str], expected: set[str]) -> None:
    missing = sorted(expected.difference(columns))
    if missing:
        raise ValueError(f"{path}: missing expected columns: {', '.join(missing)}")


def _norm(value: str | None) -> str:
    return "" if value is None else value.strip()


def _norm_integer_like(value: str | None) -> str:
    """Normalize keys such as `101` and `101.0` without changing text IDs."""

    text = _norm(value)
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def _norm_month(value: str | None) -> str:
    """Normalize supported English month names/abbreviations and numeric months.

    Unknown non-empty labels return an empty string so the audit surfaces them as
    invalid join keys rather than silently treating them as a different month code.
    """

    text = _norm(value)
    if not text:
        return ""

    named = MONTH_NAME_TO_NUMBER.get(text.lower())
    if named is not None:
        return named

    try:
        numeric = float(text)
    except ValueError:
        return ""
    if numeric.is_integer() and 1 <= int(numeric) <= 12:
        return str(int(numeric))
    return ""


def _key_cama(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _norm_integer_like(row.get("SiteID")),
        _norm_integer_like(row.get("Year")),
        _norm_month(row.get("Month")),
    )


def _key_effort(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _norm_integer_like(row.get("SiteID")),
        _norm_integer_like(row.get("Year")),
        _norm_month(row.get("month")),
    )


def _key_coord(row: dict[str, str]) -> tuple[str, str]:
    return (
        _norm_integer_like(row.get("SiteID")),
        _norm_integer_like(row.get("Year")),
    )


def _complete_key(key: tuple[str, ...]) -> bool:
    return all(part != "" for part in key)


def _float_or_none(value: str | None) -> float | None:
    text = _norm(value)
    if text.lower() in {"", "na", "nan", "null", "none"}:
        return None
    return float(text)


def _count_or_none(value: str | None) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    if parsed < 0 or not parsed.is_integer():
        raise ValueError(f"Expected a non-negative integer count, got {value!r}")
    return int(parsed)


def audit(
    cama_path: Path,
    effort_path: Path,
    coords_path: Path,
    temperature_path: Path | None = None,
) -> dict[str, Any]:
    """Audit candidate joins before any ecological model is fitted.

    This function intentionally does not impute missing outcomes, effort,
    coordinates or environmental data. Its purpose is to quantify whether the
    proposed joins are actually supported by the downloaded source tables.
    """

    cama_rows, cama_cols = _read_csv(cama_path)
    effort_rows, effort_cols = _read_csv(effort_path)
    coord_rows, coord_cols = _read_csv(coords_path)

    _require_columns(cama_path, cama_cols, EXPECTED_CAMA_COLUMNS)
    _require_columns(effort_path, effort_cols, EXPECTED_EFFORT_COLUMNS)
    _require_columns(coords_path, coord_cols, EXPECTED_COORD_COLUMNS)

    cama_keys = [_key_cama(row) for row in cama_rows]
    effort_keys = [_key_effort(row) for row in effort_rows]
    coord_keys = [_key_coord(row) for row in coord_rows]

    cama_duplicates = Counter(key for key in cama_keys if _complete_key(key))
    effort_duplicates = Counter(key for key in effort_keys if _complete_key(key))
    coord_duplicates = Counter(key for key in coord_keys if _complete_key(key))

    effort_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row, key in zip(effort_rows, effort_keys):
        if _complete_key(key):
            effort_by_key.setdefault(key, row)

    coord_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row, key in zip(coord_rows, coord_keys):
        if _complete_key(key):
            coord_by_key.setdefault(key, row)

    valid_cama_keys = [key for key in cama_keys if _complete_key(key)]
    matched_effort = [key for key in valid_cama_keys if key in effort_by_key]
    matched_coords = [
        key for key in valid_cama_keys if (key[0], key[1]) in coord_by_key
    ]

    cama_counts = [_count_or_none(row.get("CAMA")) for row in cama_rows]
    observed_cama_counts = [count for count in cama_counts if count is not None]
    detection_rows = sum(count > 0 for count in observed_cama_counts)
    zero_detection_rows = sum(count == 0 for count in observed_cama_counts)

    trap_sets = [
        _float_or_none(effort_by_key[key].get("trap.sets"))
        for key in matched_effort
    ]
    trap_sets = [value for value in trap_sets if value is not None]

    years = sorted(
        {
            _norm_integer_like(row.get("Year"))
            for row in cama_rows
            if _norm(row.get("Year"))
        }
    )
    sites = sorted(
        {
            _norm_integer_like(row.get("SiteID"))
            for row in cama_rows
            if _norm(row.get("SiteID"))
        }
    )
    habitats = Counter(_norm(row.get("habtype")) for row in cama_rows)

    unmatched_effort_keys = [
        key for key in valid_cama_keys if key not in effort_by_key
    ]
    unmatched_coord_keys = [
        key
        for key in valid_cama_keys
        if (key[0], key[1]) not in coord_by_key
    ]

    report: dict[str, Any] = {
        "cama": {
            "rows": len(cama_rows),
            "sites": len(sites),
            "years": years,
            "valid_join_key_rows": len(valid_cama_keys),
            "invalid_join_key_rows": len(cama_rows) - len(valid_cama_keys),
            "duplicate_site_month_keys": sum(
                count - 1 for count in cama_duplicates.values() if count > 1
            ),
            "observed_outcome_rows": len(observed_cama_counts),
            "missing_outcome_rows": len(cama_rows) - len(observed_cama_counts),
            "total_cama": sum(observed_cama_counts),
            "detection_rows": detection_rows,
            "zero_detection_rows": zero_detection_rows,
            "detection_row_fraction": (
                detection_rows / len(observed_cama_counts)
                if observed_cama_counts
                else None
            ),
            "habitat_counts": dict(sorted(habitats.items())),
        },
        "effort_join": {
            "join_key": ["SiteID", "Year", "Month"],
            "source_rows": len(effort_rows),
            "valid_source_key_rows": sum(_complete_key(key) for key in effort_keys),
            "matched_rows": len(matched_effort),
            "coverage_fraction": (
                len(matched_effort) / len(cama_rows) if cama_rows else None
            ),
            "coverage_among_valid_cama_keys": (
                len(matched_effort) / len(valid_cama_keys)
                if valid_cama_keys
                else None
            ),
            "duplicate_effort_keys": sum(
                count - 1 for count in effort_duplicates.values() if count > 1
            ),
            "unmatched_example_keys": unmatched_effort_keys[:20],
            "trap_sets_nonmissing_rows": len(trap_sets),
            "trap_sets_min": min(trap_sets) if trap_sets else None,
            "trap_sets_median": statistics.median(trap_sets) if trap_sets else None,
            "trap_sets_max": max(trap_sets) if trap_sets else None,
        },
        "coordinate_join": {
            "join_key": ["SiteID", "Year"],
            "source_rows": len(coord_rows),
            "valid_source_key_rows": sum(_complete_key(key) for key in coord_keys),
            "matched_rows": len(matched_coords),
            "coverage_fraction": (
                len(matched_coords) / len(cama_rows) if cama_rows else None
            ),
            "coverage_among_valid_cama_keys": (
                len(matched_coords) / len(valid_cama_keys)
                if valid_cama_keys
                else None
            ),
            "duplicate_coordinate_keys": sum(
                count - 1 for count in coord_duplicates.values() if count > 1
            ),
            "unmatched_example_keys": unmatched_coord_keys[:20],
        },
    }

    if temperature_path is not None:
        temperature_rows, temperature_cols = _read_csv(temperature_path)
        _require_columns(temperature_path, temperature_cols, EXPECTED_TEMP_COLUMNS)
        temp_sites = {
            _norm_integer_like(row.get("SiteNum"))
            for row in temperature_rows
            if _norm(row.get("SiteNum"))
        }
        temp_years = {
            _norm_integer_like(row.get("year"))
            for row in temperature_rows
            if _norm(row.get("year"))
        }
        temp_values = [
            value
            for row in temperature_rows
            if (value := _float_or_none(row.get("x"))) is not None
        ]
        cama_site_years = {(key[0], key[1]) for key in valid_cama_keys}
        temp_site_years = {
            (
                _norm_integer_like(row.get("SiteNum")),
                _norm_integer_like(row.get("year")),
            )
            for row in temperature_rows
            if _norm(row.get("SiteNum")) and _norm(row.get("year"))
        }
        overlapping_site_years = cama_site_years.intersection(temp_site_years)
        report["temperature"] = {
            "rows": len(temperature_rows),
            "sites": len(temp_sites),
            "years": sorted(temp_years),
            "nonmissing_temperature_rows": len(temp_values),
            "overlapping_cama_site_years": len(overlapping_site_years),
            "cama_site_year_coverage_fraction": (
                len(overlapping_site_years) / len(cama_site_years)
                if cama_site_years
                else None
            ),
            "min_c": min(temp_values) if temp_values else None,
            "median_c": statistics.median(temp_values) if temp_values else None,
            "max_c": max(temp_values) if temp_values else None,
        }

    return report


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def print_report(report: dict[str, Any]) -> None:
    cama = report["cama"]
    effort = report["effort_join"]
    coords = report["coordinate_join"]

    print("=== R0 REAL-DATA AUDIT ===")
    print(
        f"CAMA table: {cama['rows']} rows | {cama['sites']} sites | "
        f"years={','.join(cama['years'])}"
    )
    print(
        f"CAMA keys: valid={cama['valid_join_key_rows']} | "
        f"invalid={cama['invalid_join_key_rows']} | "
        f"duplicate site-month keys={cama['duplicate_site_month_keys']}"
    )
    print(
        f"Green-crab outcomes: observed={cama['observed_outcome_rows']} | "
        f"missing={cama['missing_outcome_rows']} | total CAMA={cama['total_cama']} | "
        f"detection rows={cama['detection_rows']} | "
        f"zero fraction={pct(1.0 - cama['detection_row_fraction']) if cama['detection_row_fraction'] is not None else 'n/a'}"
    )
    print(
        f"Exact effort join (SiteID,Year,Month): {effort['matched_rows']}/{cama['rows']} "
        f"({pct(effort['coverage_fraction'])}) | duplicate effort keys={effort['duplicate_effort_keys']}"
    )
    print(
        f"Coordinate join (SiteID,Year): {coords['matched_rows']}/{cama['rows']} "
        f"({pct(coords['coverage_fraction'])}) | duplicate coordinate keys={coords['duplicate_coordinate_keys']}"
    )
    if effort["trap_sets_median"] is not None:
        print(
            "Trap sets among matched rows: "
            f"n={effort['trap_sets_nonmissing_rows']}, min={effort['trap_sets_min']}, "
            f"median={effort['trap_sets_median']}, max={effort['trap_sets_max']}"
        )
    print(f"Habitat counts: {cama['habitat_counts']}")

    if "temperature" in report:
        temp = report["temperature"]
        print(
            f"Temperature logger table: {temp['rows']} rows | {temp['sites']} sites | "
            f"years={','.join(temp['years'])} | nonmissing={temp['nonmissing_temperature_rows']}"
        )
        print(
            "Temperature overlap with CAMA site-years: "
            f"{temp['overlapping_cama_site_years']} "
            f"({pct(temp['cama_site_year_coverage_fraction'])})"
        )
        if temp["median_c"] is not None:
            print(
                f"Temperature °C: min={temp['min_c']:.2f}, "
                f"median={temp['median_c']:.2f}, max={temp['max_c']:.2f}"
            )

    print()
    print("Interpretation gate:")
    print("  Month names and numeric months are normalized explicitly for joins.")
    print("  Missing CAMA is NOT treated as zero detection.")
    print("  This script checks data availability and join coverage only.")
    print("  It does NOT validate an ecological model and does NOT estimate q.")
