from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_canonical_observations(path: Path) -> list[dict[str, Any]]:
    """Load the canonical monthly site-level green-crab monitoring table."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "site_id",
            "year",
            "month",
            "trap_sets",
            "effort_missing",
            "cama_count",
            "detected",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical observation table missing fields: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for raw in reader:
            site_id = str(raw.get("site_id") or "").strip()
            if not site_id:
                raise ValueError("Blank site_id in canonical observation table")
            year = int(raw["year"])
            month = int(raw["month"])
            if not 1 <= month <= 12:
                raise ValueError(f"Invalid month={month} for site={site_id}, year={year}")
            effort_missing = _as_bool(raw.get("effort_missing"))
            trap_sets_raw = str(raw.get("trap_sets") or "").strip()
            trap_sets = None if effort_missing or not trap_sets_raw else float(trap_sets_raw)
            if trap_sets is not None and trap_sets < 0:
                raise ValueError("trap_sets cannot be negative")
            rows.append(
                {
                    "site_id": site_id,
                    "year": year,
                    "month": month,
                    "trap_sets": trap_sets,
                    "effort_missing": effort_missing,
                    "cama_count": int(float(raw["cama_count"])),
                    "detected": _as_bool(raw.get("detected")),
                }
            )
    return rows


def _effort_key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def summarize_q_identifiability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether the current monthly table can identify a per-effort detectability q.

    The canonical table has one row per site-year-month, not trap-level outcomes or
    repeated same-occasion detection replicates. Therefore this routine reports
    empirical effort/detection structure but deliberately does NOT fit q.
    """
    if not rows:
        raise ValueError("At least one observation row is required")

    keys = [(str(row["site_id"]), int(row["year"]), int(row["month"])) for row in rows]
    duplicate_keys = [key for key, n in Counter(keys).items() if n > 1]

    effort_known = [row for row in rows if not bool(row["effort_missing"]) and row["trap_sets"] is not None]
    effort_missing = [row for row in rows if row not in effort_known]
    detected_known = [row for row in effort_known if bool(row["detected"])]
    nondetected_known = [row for row in effort_known if not bool(row["detected"])]

    by_effort: dict[str, dict[str, Any]] = {}
    effort_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in effort_known:
        effort_groups[float(row["trap_sets"])].append(row)
    for effort in sorted(effort_groups):
        group = effort_groups[effort]
        detections = sum(bool(row["detected"]) for row in group)
        by_effort[_effort_key(effort)] = {
            "trap_sets": effort,
            "rows": len(group),
            "detections": detections,
            "non_detections": len(group) - detections,
            "observed_detection_fraction": detections / len(group),
        }

    site_year_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        site_year_groups[(str(row["site_id"]), int(row["year"]))].append(row)
    for group in site_year_groups.values():
        group.sort(key=lambda row: int(row["month"]))

    repeated_site_years = [group for group in site_year_groups.values() if len(group) >= 2]
    positive_site_years = [group for group in site_year_groups.values() if any(bool(row["detected"]) for row in group)]

    post_first_detection_rows: list[dict[str, Any]] = []
    for group in positive_site_years:
        first_positive_month = min(int(row["month"]) for row in group if bool(row["detected"]))
        post_first_detection_rows.extend(
            row for row in group if int(row["month"]) > first_positive_month
        )

    known_post_first_detection = [
        row
        for row in post_first_detection_rows
        if not bool(row["effort_missing"]) and row["trap_sets"] is not None
    ]

    effort_values = [float(row["trap_sets"]) for row in effort_known]
    site_counts = Counter(str(row["site_id"]) for row in rows)

    return {
        "rows": len(rows),
        "sites": len(site_counts),
        "site_years": len(site_year_groups),
        "unique_site_year_month_keys": len(set(keys)),
        "duplicate_site_year_month_keys": len(duplicate_keys),
        "same_month_replicates_available": len(duplicate_keys) > 0,
        "effort": {
            "known_rows": len(effort_known),
            "missing_rows": len(effort_missing),
            "coverage_fraction": len(effort_known) / len(rows),
            "trap_sets_min": min(effort_values) if effort_values else None,
            "trap_sets_median": median(effort_values) if effort_values else None,
            "trap_sets_max": max(effort_values) if effort_values else None,
            "by_exact_trap_sets": by_effort,
        },
        "outcomes_with_known_effort": {
            "detections": len(detected_known),
            "non_detections": len(nondetected_known),
            "observed_detection_fraction": (
                len(detected_known) / len(effort_known) if effort_known else None
            ),
        },
        "temporal_repetition": {
            "site_years_with_2plus_months": len(repeated_site_years),
            "site_years_with_any_detection": len(positive_site_years),
            "months_after_first_detection_same_site_year": len(post_first_detection_rows),
            "known_effort_months_after_first_detection": len(known_post_first_detection),
            "detections_after_first_detection": sum(bool(row["detected"]) for row in known_post_first_detection),
            "non_detections_after_first_detection": sum(not bool(row["detected"]) for row in known_post_first_detection),
        },
        "q_identifiability": {
            "per_effort_q_identified_from_current_table": False,
            "reason": (
                "The canonical data are monthly site-level aggregates with no trap-level outcomes or "
                "same-occasion repeated detection replicates. Occupancy/prevalence and detectability are therefore "
                "confounded without additional closure assumptions, repeated-visit data, or external q information."
            ),
            "do_not_do": [
                "Do not equate observed detection fraction at a given effort with q.",
                "Do not fit a universal q by treating all non-detection rows as occupied trials.",
                "Do not assume a full site-year is occupancy-closed merely because the same site was sampled in multiple months.",
            ],
            "current_default": (
                "Keep the evidence model P(no detection | occupied,e,q)=(1-q)^e, but treat q as an externally "
                "constrained uncertain parameter until suitable repeated-detection data or defensible literature/protocol ranges are sourced."
            ),
        },
        "notes": [
            "Observed detection fractions by effort are simulator-criticism / descriptive statistics, not direct detectability estimates.",
            "Rows after a first detection in the same site-year are reported only as a diagnostic; they do not establish occupancy closure.",
            "Missing effort remains excluded from effort-conditioned summaries and is never imputed.",
        ],
    }
