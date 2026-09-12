from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

CAMA_PATH = Path("data/raw/wsg_cama/all_abundance_wide_format.csv")
EFFORT_PATH = Path("data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def norm(v):
    if v is None:
        return ""
    t = str(v).strip()
    if not t:
        return ""
    try:
        x = float(t)
        if x.is_integer():
            return str(int(x))
    except ValueError:
        pass
    return t


def unique_sample(values, n=20):
    return sorted({v for v in values if v})[:n]


def main() -> None:
    cama_rows, cama_cols = read_csv(CAMA_PATH)
    effort_rows, effort_cols = read_csv(EFFORT_PATH)

    cama_sites = {norm(r.get("SiteID")) for r in cama_rows if norm(r.get("SiteID"))}
    effort_sites = {norm(r.get("SiteID")) for r in effort_rows if norm(r.get("SiteID"))}
    cama_years = {norm(r.get("Year")) for r in cama_rows if norm(r.get("Year"))}
    effort_years = {norm(r.get("Year")) for r in effort_rows if norm(r.get("Year"))}
    cama_months = {norm(r.get("Month")) for r in cama_rows if norm(r.get("Month"))}
    effort_months = {norm(r.get("month")) for r in effort_rows if norm(r.get("month"))}

    cama_site_year = {(norm(r.get("SiteID")), norm(r.get("Year"))) for r in cama_rows}
    effort_site_year = {(norm(r.get("SiteID")), norm(r.get("Year"))) for r in effort_rows}
    cama_keys = {(norm(r.get("SiteID")), norm(r.get("Year")), norm(r.get("Month"))) for r in cama_rows}
    effort_keys = {(norm(r.get("SiteID")), norm(r.get("Year")), norm(r.get("month"))) for r in effort_rows}

    print("=== R0 EFFORT JOIN DIAGNOSTIC ===")
    print(f"CAMA columns:   {cama_cols}")
    print(f"Effort columns: {effort_cols}")
    print()
    print(f"Rows: CAMA={len(cama_rows)} | effort={len(effort_rows)}")
    print(f"Sites: CAMA={len(cama_sites)} | effort={len(effort_sites)} | overlap={len(cama_sites & effort_sites)}")
    print(f"Years CAMA:   {sorted(cama_years)}")
    print(f"Years effort: {sorted(effort_years)}")
    print(f"Year overlap: {sorted(cama_years & effort_years)}")
    print(f"Months CAMA:   {sorted(cama_months)}")
    print(f"Months effort: {sorted(effort_months)}")
    print(f"Month overlap: {sorted(cama_months & effort_months)}")
    print()
    print(f"Site-year pairs: CAMA={len(cama_site_year)} | effort={len(effort_site_year)} | overlap={len(cama_site_year & effort_site_year)}")
    print(f"Exact site-year-month key overlap: {len(cama_keys & effort_keys)}")
    print()
    print(f"Sample CAMA SiteID values:   {unique_sample(cama_sites)}")
    print(f"Sample effort SiteID values: {unique_sample(effort_sites)}")
    print()

    shared_sy = sorted(cama_site_year & effort_site_year)[:10]
    if shared_sy:
        cama_months_by_sy = defaultdict(set)
        effort_months_by_sy = defaultdict(set)
        for r in cama_rows:
            cama_months_by_sy[(norm(r.get("SiteID")), norm(r.get("Year")))].add(norm(r.get("Month")))
        for r in effort_rows:
            effort_months_by_sy[(norm(r.get("SiteID")), norm(r.get("Year")))].add(norm(r.get("month")))
        print("Shared site-year month comparison (first 10):")
        for sy in shared_sy:
            print(f"  {sy}: CAMA={sorted(cama_months_by_sy[sy])} | effort={sorted(effort_months_by_sy[sy])}")
    else:
        print("No shared site-year pairs. The mismatch is upstream of month.")

    print()
    print("Raw first 5 rows from each table:")
    for label, rows in (("CAMA", cama_rows), ("EFFORT", effort_rows)):
        print(f"[{label}]")
        for row in rows[:5]:
            print(row)

    print()
    print("Diagnostic only: do not change join rules until this output explains the 0% overlap.")


if __name__ == "__main__":
    main()
