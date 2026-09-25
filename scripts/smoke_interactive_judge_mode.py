from __future__ import annotations

import argparse
import json

from adaptive_response.mission_control import MissionControlSession


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic backend smoke test for the resource-efficiency-aware Marine judge mode. "
            "By default the script follows both the current site and effort recommendation."
        )
    )
    parser.add_argument(
        "--case-id",
        default="incident_097",
        help="Frozen demo case id, e.g. incident_097.",
    )
    parser.add_argument(
        "--effort",
        default="marine",
        choices=("marine", "1", "3", "6"),
        help=(
            "Effort policy for the interactive path. 'marine' follows the adaptive planner's "
            "current recommended effort; 1/3/6 force that effort when affordable."
        ),
    )
    args = parser.parse_args()

    session = MissionControlSession()
    snap = session.reset(case_id=args.case_id)

    while not snap["can_reveal"]:
        recommendation = snap["global_recommendations"][0]
        if args.effort == "marine":
            effort = int(recommendation["recommended_effort"])
        else:
            requested = int(args.effort)
            allowed = [
                effort
                for effort in (1, 3, 6)
                if effort <= snap["resources"]["remaining_budget"]
            ]
            if not allowed:
                raise RuntimeError("No allowed effort fits the remaining budget.")
            effort = requested if requested in allowed else max(allowed)

        snap = session.deploy(
            site_id=recommendation["site_id"],
            effort=effort,
        )

    revealed = session.reveal()
    performance = revealed["performance"]
    receipt = {
        "case": revealed["case"],
        "initial_detection": revealed["incident"]["initial_detection"],
        "response_window": {
            "deployments": revealed["resources"]["round"],
            "mission_horizon": revealed["resources"]["mission_horizon"],
            "effort_spent": revealed["resources"]["spent_budget"],
            "capacity_preserved": revealed["resources"]["capacity_preserved"],
        },
        "marine": performance["marine"],
        "static": performance["static"],
        "you": performance["you"],
        "resource_receipt": performance["resource_receipt"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
