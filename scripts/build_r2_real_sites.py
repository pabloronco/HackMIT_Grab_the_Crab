from __future__ import annotations

from pathlib import Path

from adaptive_response.real_sites import (
    build_real_site_table,
    write_real_sites_csv,
    write_real_sites_summary,
)


CANONICAL = Path("data/processed/canonical_site_visit.csv")
SHOREZONE = Path("data/processed/r2_shorezone_site_matches.csv")
OUT_CSV = Path("data/processed/r2_real_sites.csv")
OUT_JSON = Path("data/processed/r2_real_sites_summary.json")


def main() -> None:
    for path in (CANONICAL, SHOREZONE):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; complete the previous R1/R2 diagnostic step first.")

    rows, summary = build_real_site_table(CANONICAL, SHOREZONE, shorezone_acceptance_m=100.0)
    write_real_sites_csv(rows, OUT_CSV)
    write_real_sites_summary(summary, OUT_JSON)

    print("=== R2 REAL SITE TABLE ===")
    print(f"sites={summary['sites']}")
    print(
        f"ShoreZone accepted <= {summary['shorezone_acceptance_m']:.0f}m: "
        f"{summary['shorezone_accepted_sites']}/{summary['sites']}"
    )
    print(
        f"ShoreZone rejected: {summary['shorezone_rejected_sites']} | "
        f"sites={summary['shorezone_rejected_site_ids']}"
    )
    print(
        "Crab Team habitat ambiguous across months/years: "
        f"{summary['crabteam_habitat_ambiguous_sites']} | "
        f"sites={summary['crabteam_habitat_ambiguous_site_ids']}"
    )
    print()
    print("R2 contract:")
    print("  Monitoring coordinates remain the graph-node geometry.")
    print("  ShoreZone <=100m is accepted as static habitat context for v0.")
    print("  >100m candidates stay visible but their thematic fields are not trusted in the primary table.")
    print("  The 100m threshold is a design choice, not an ecological fact.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
