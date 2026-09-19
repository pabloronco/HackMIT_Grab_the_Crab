from __future__ import annotations

import argparse
import json

from adaptive_response.mission_control import MissionControlSession


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic backend smoke test for interactive Marine judge mode."
    )
    parser.add_argument(
        "--case-id",
        default="incident_001",
        help="Frozen demo case id, e.g. incident_001.",
    )
    parser.add_argument(
        "--effort",
        type=int,
        default=6,
        choices=(1, 3, 6),
        help="Effort used when following Marine's current top recommendation.",
    )
    args = parser.parse_args()

    session = MissionControlSession()
    snap = session.reset(case_id=args.case_id)

    while not snap["can_reveal"]:
        remaining = snap["resources"]["remaining_budget"]
        allowed = [e for e in (1, 3, 6) if e <= remaining]
        effort = args.effort if args.effort in allowed else max(allowed)
        recommendation = snap["global_recommendations"][0]
        snap = session.deploy(
            site_id=recommendation["site_id"],
            effort=effort,
        )

    revealed = session.reveal()
    receipt = {
        "case": revealed["case"],
        "initial_detection": revealed["incident"]["initial_detection"],
        "spent_budget": revealed["resources"]["spent_budget"],
        "marine": revealed["performance"]["marine"],
        "static": revealed["performance"]["static"],
        "you": revealed["performance"]["you"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
