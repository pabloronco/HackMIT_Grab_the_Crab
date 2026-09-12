from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_response.data_audit import audit, print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit first real Crab Team data joins.")
    parser.add_argument(
        "--cama",
        type=Path,
        default=Path("data/raw/wsg_cama/all_abundance_wide_format.csv"),
    )
    parser.add_argument(
        "--effort",
        type=Path,
        default=Path("data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv"),
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv"),
    )
    parser.add_argument(
        "--temperature",
        type=Path,
        default=Path("data/raw/wsg_cama/DailyMaxTemperature.csv"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    for required in (args.cama, args.effort, args.coords):
        if not required.exists():
            raise SystemExit(
                f"Missing {required}. See data/raw/README.md for expected downloads."
            )

    temperature = args.temperature if args.temperature.exists() else None
    report = audit(args.cama, args.effort, args.coords, temperature)
    print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
