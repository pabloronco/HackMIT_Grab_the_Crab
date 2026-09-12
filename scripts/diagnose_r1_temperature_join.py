from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


CANONICAL = Path("data/processed/canonical_site_visit.csv")
TEMPERATURE = Path("data/raw/wsg_cama/DailyMaxTemperature.csv")


def _norm(value: str | None) -> str:
    return "" if value is None else value.strip()


def _norm_int(value: str | None) -> str:
    text = _norm(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _read(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _parse_month(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    # Diagnostic only: accept common encodings, but never silently infer from
    # an unrecognized string. We report parse failures below.
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d %Y",
        "%B %d %Y",
    ):
        try:
            return datetime.strptime(text, fmt).month
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).month
    except ValueError:
        return None


def main() -> None:
    canonical, canonical_cols = _read(CANONICAL)
    temp, temp_cols = _read(TEMPERATURE)

    required_canonical = {"site_id", "year", "month", "detected"}
    required_temp = {"SiteNum", "year", "mdy", "x"}
    if not required_canonical.issubset(canonical_cols):
        raise RuntimeError(f"Canonical table missing columns: {sorted(required_canonical - set(canonical_cols))}")
    if not required_temp.issubset(temp_cols):
        raise RuntimeError(f"Temperature table missing columns: {sorted(required_temp - set(temp_cols))}")

    canonical_keys = {
        (_norm_int(row["site_id"]), _norm_int(row["year"]), _norm_int(row["month"]))
        for row in canonical
    }
    detection_keys = {
        (_norm_int(row["site_id"]), _norm_int(row["year"]), _norm_int(row["month"]))
        for row in canonical
        if _norm(row.get("detected")).lower() in {"true", "1", "yes"}
    }

    parsed_rows: list[tuple[tuple[str, str, str], float]] = []
    parse_failures: list[str] = []
    bad_temp_values = 0
    date_examples: list[str] = []
    for row in temp:
        raw_date = _norm(row.get("mdy"))
        if raw_date and raw_date not in date_examples and len(date_examples) < 12:
            date_examples.append(raw_date)
        month = _parse_month(raw_date)
        if month is None:
            if raw_date and len(parse_failures) < 20:
                parse_failures.append(raw_date)
            continue
        try:
            value = float(_norm(row.get("x")))
        except ValueError:
            bad_temp_values += 1
            continue
        key = (_norm_int(row.get("SiteNum")), _norm_int(row.get("year")), str(month))
        parsed_rows.append((key, value))

    temp_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, value in parsed_rows:
        temp_by_key[key].append(value)

    temp_keys = set(temp_by_key)
    overlap = canonical_keys & temp_keys
    detection_overlap = detection_keys & temp_keys
    days_per_overlap = [len(temp_by_key[key]) for key in overlap]

    canonical_by_year = Counter(key[1] for key in canonical_keys)
    overlap_by_year = Counter(key[1] for key in overlap)
    canonical_by_month = Counter(key[2] for key in canonical_keys)
    overlap_by_month = Counter(key[2] for key in overlap)

    print("=== R1 TEMPERATURE JOIN DIAGNOSTIC ===")
    print(f"Canonical rows={len(canonical)} | unique site-month keys={len(canonical_keys)}")
    print(f"Temperature rows={len(temp)} | parsed date+temperature rows={len(parsed_rows)}")
    print(f"Raw mdy examples={date_examples}")
    print(f"Date parse failures={len(parse_failures)}" + (f" | examples={parse_failures[:10]}" if parse_failures else ""))
    print(f"Bad temperature values={bad_temp_values}")
    print()
    print(f"Temperature site-month keys={len(temp_keys)}")
    print(
        f"Exact canonical site-year-month overlap={len(overlap)}/{len(canonical_keys)} "
        f"({100 * len(overlap) / len(canonical_keys):.1f}%)"
    )
    print(
        f"Detection rows with temperature context={len(detection_overlap)}/{len(detection_keys)} "
        f"({100 * len(detection_overlap) / len(detection_keys):.1f}% if detections exist)"
        if detection_keys
        else "Detection rows with temperature context=n/a"
    )
    if days_per_overlap:
        ordered = sorted(days_per_overlap)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        print(
            f"Logger days per overlapping site-month: min={min(ordered)} | "
            f"median={median} | max={max(ordered)}"
        )
    print()
    print("Coverage by year (overlap/canonical unique site-month keys):")
    for year in sorted(canonical_by_year, key=int):
        total = canonical_by_year[year]
        matched = overlap_by_year[year]
        print(f"  {year}: {matched}/{total} ({100 * matched / total:.1f}%)")
    print("Coverage by month:")
    for month in sorted(canonical_by_month, key=int):
        total = canonical_by_month[month]
        matched = overlap_by_month[month]
        print(f"  month={month}: {matched}/{total} ({100 * matched / total:.1f}%)")

    missing_examples = sorted(canonical_keys - temp_keys)[:20]
    print()
    print(f"Unmatched canonical key examples={missing_examples}")
    print()
    print("GATE:")
    print("  Diagnostic only. Do NOT impute temperature for uncovered site-months.")
    print("  Do NOT choose monthly aggregation (mean/max/etc.) until coverage and date semantics are inspected.")
    print("  Temperature is optional ecological context, not required evidence for every observation.")


if __name__ == "__main__":
    main()
