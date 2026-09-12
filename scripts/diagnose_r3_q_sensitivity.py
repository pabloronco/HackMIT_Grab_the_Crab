from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_response.q_sensitivity import (
    DEFAULT_EFFORT_PROBES,
    DEFAULT_Q_PROBES,
    build_q_sensitivity_table,
)


OUT_CSV = Path("data/processed/r3_q_sensitivity_probes.csv")
OUT_JSON = Path("data/processed/r3_q_sensitivity_audit.json")


def main() -> None:
    prior = 0.5
    rows = build_q_sensitivity_table(prior=prior)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "Q_NUMERIC_RANGE_STILL_OPEN",
        "q_probes": list(DEFAULT_Q_PROBES),
        "effort_probes": list(DEFAULT_EFFORT_PROBES),
        "prior_for_posterior_demo": prior,
        "probe_semantics": (
            "Broad design sensitivity probes only; not estimates, confidence intervals, "
            "or a claimed biologically plausible green-crab detectability range."
        ),
        "observation_model": "P(no detection | occupied,e,q)=(1-q)^e",
        "notes": [
            "The current monthly real table does not identify occupancy-conditional q.",
            "Crab Team standard monitoring uses a mixed six-trap overnight protocol, so a single scalar q would be an effective protocol-level/per-trap-equivalent abstraction unless trap type is modelled separately.",
            "External per-individual catchability or trap-entry estimates are not interchangeable with site-level occupancy detection probability.",
            "Do not freeze q from this table. Use the probes to quantify consequence, then choose simulation/training ranges with explicit source/assumption labels and OOD q shifts.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== R3 q SENSITIVITY PROBES ===")
    print("STATUS: numeric q range is OPEN.")
    print("The values below are DESIGN SENSITIVITY PROBES, not empirical q estimates.")
    print(f"Prior used only for Bayes illustration: p={prior:.2f}")
    print()
    print("q | effort | P(detect >=1 | occupied) | posterior p after NO detection")
    print("--|--------|--------------------------|-------------------------------")
    for q in DEFAULT_Q_PROBES:
        for row in rows:
            if float(row["q_probe"]) != float(q):
                continue
            effort = int(row["effort"])
            if effort not in DEFAULT_EFFORT_PROBES:
                continue
            print(
                f"{q:0.2f} | {effort:>6d} | "
                f"{float(row['detection_if_occupied']):>24.1%} | "
                f"{float(row['posterior_after_nondetection']):>29.3f}"
            )
        print()

    print("INTERPRETATION GATE:")
    print("  Do not select q because it makes replanning look dramatic.")
    print("  Do not map per-individual catch probability directly onto occupancy-detection q.")
    print("  Keep q uncertainty explicit in simulator/OOD tests.")
    print("  Next decision: choose an effective q treatment for MVP, with sourced evidence vs model assumptions separated.")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
