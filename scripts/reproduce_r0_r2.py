from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUTS = (
    REPO_ROOT / "data/raw/wsg_cama/all_abundance_wide_format.csv",
    REPO_ROOT / "data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv",
    REPO_ROOT / "data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv",
)

# Minimal dependency chain required to rebuild the accepted R2 graph milestone.
# Diagnostic scripts not needed by the final outputs remain available separately.
STEPS = (
    "scripts/audit_real_data.py",
    "scripts/build_r1_canonical.py",
    "scripts/diagnose_r2_shorezone.py",
    "scripts/build_r2_real_sites.py",
    "scripts/diagnose_r2_adaptive_candidates.py",
    "scripts/diagnose_r2_adaptive_water.py",
    "scripts/diagnose_r2_salishseacast_routes.py",
    "scripts/build_r2_real_graph_v0.py",
    "scripts/diagnose_r2_incident_subgraphs.py",
)


def main() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in RAW_INPUTS if not path.is_file()]
    if missing:
        print("R0-R2 reproduction cannot start: required raw inputs are missing.")
        for path in missing:
            print(f"  - {path}")
        print()
        print("Raw files remain gitignored by design. Acquire/import the documented source files first.")
        raise SystemExit(2)

    print("=== REPRODUCE R0 -> R2 REAL GRAPH MILESTONE ===")
    print("Raw data remain local/gitignored.")
    print("Public ShoreZone/DNR GIS and SalishSeaCast access may be required on first run.")
    print("Each step must succeed before the next one starts.")
    print()

    for index, relative in enumerate(STEPS, start=1):
        script = REPO_ROOT / relative
        if not script.is_file():
            raise SystemExit(f"Missing reproduction step: {relative}")
        print(f"[{index}/{len(STEPS)}] {relative}")
        subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)
        print()

    print("R0-R2 reproduction complete.")
    print("Next publish the compact versioned receipt with:")
    print("  python scripts/publish_r2_real_graph_receipt.py")


if __name__ == "__main__":
    main()
