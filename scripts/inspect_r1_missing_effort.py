from __future__ import annotations

import csv
from pathlib import Path


CANONICAL_PATH = Path("data/processed/canonical_site_visit.csv")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    if not CANONICAL_PATH.is_file():
        raise SystemExit(
            f"Missing {CANONICAL_PATH}. Run: python scripts/build_r1_canonical.py"
        )

    with CANONICAL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    missing = [row for row in rows if _truthy(row.get("effort_missing"))]
    detections = [row for row in missing if _truthy(row.get("detected"))]

    print("=== R1 MISSING-EFFORT DIAGNOSTIC ===")
    print(f"canonical rows={len(rows)} | missing effort rows={len(missing)}")
    print(f"missing-effort rows with green-crab detection={len(detections)}")
    print()

    if not missing:
        print("No missing-effort rows.")
        return

    print("Rows with missing effort:")
    for row in missing:
        print(
            "  "
            f"site={row.get('site_id')} year={row.get('year')} month={row.get('month')} "
            f"habitat={row.get('habitat')} CAMA={row.get('cama_count')} "
            f"detected={row.get('detected')} lat={row.get('latitude')} lon={row.get('longitude')}"
        )

    print()
    if detections:
        print("GATE: at least one detection row lacks effort. Do not use those rows for q/observation-model fitting until resolved or explicitly excluded.")
    else:
        print("GATE: all missing-effort rows are non-detections. Keep effort null; exclude them from effort-conditioned q fitting rather than imputing 6.")


if __name__ == "__main__":
    main()
