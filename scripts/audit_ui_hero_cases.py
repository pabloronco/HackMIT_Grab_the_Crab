from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adaptive_response.mission_control import MissionControlSession


def _run_follow_marine(session: MissionControlSession, case_id: str) -> dict:
    snap = session.reset(case_id=case_id)
    initial_top_world = (
        snap["top_worlds"]["items"][0]["world_id"]
        if snap["top_worlds"]["items"]
        else None
    )

    field_detections = 0
    mission_changed_rounds = 0
    top_world_turnovers = 0
    max_propagated_abs_delta = 0.0
    min_observation_probability = 1.0
    previous_top_world = initial_top_world

    while not snap["can_reveal"]:
        recommendation = snap["global_recommendations"][0]
        effort = min(6, snap["resources"]["remaining_budget"])
        if effort not in (1, 3, 6):
            raise RuntimeError(
                f"Unexpected remaining budget {snap['resources']['remaining_budget']}."
            )

        snap = session.deploy(
            site_id=recommendation["site_id"],
            effort=effort,
        )

        for obs in snap["last_round"]["observations"]:
            field_detections += int(bool(obs["detection"]))

        mission_changed_rounds += int(bool(snap["mission_changed"]))

        propagated = snap["last_round"]["propagated_belief_changes"]
        if propagated:
            max_propagated_abs_delta = max(
                max_propagated_abs_delta,
                max(abs(float(row["delta"])) for row in propagated),
            )

        stress = snap.get("model_stress")
        if stress is not None:
            min_observation_probability = min(
                min_observation_probability,
                float(stress["observation_probability"]),
            )

        current_top_world = (
            snap["top_worlds"]["items"][0]["world_id"]
            if snap["top_worlds"]["items"]
            else None
        )
        if (
            previous_top_world is not None
            and current_top_world is not None
            and current_top_world != previous_top_world
        ):
            top_world_turnovers += 1
        previous_top_world = current_top_world

    revealed = session.reveal()
    perf = revealed["performance"]
    marine = perf["marine"]
    static = perf["static"]

    marine_sites = list(marine["mission_sites"])
    static_sites = list(static["mission_sites"])
    shared_length = min(len(marine_sites), len(static_sites))
    mission_divergence_count = sum(
        marine_sites[i] != static_sites[i]
        for i in range(shared_length)
    ) + abs(len(marine_sites) - len(static_sites))

    detected_advantage = (
        int(marine["detected_occupied"])
        - int(static["detected_occupied"])
    )

    return {
        "case_id": case_id,
        "initial_detection": revealed["incident"]["initial_detection"],
        "occupied_total": int(marine["occupied_total"]),
        "marine_detected_occupied": int(marine["detected_occupied"]),
        "static_detected_occupied": int(static["detected_occupied"]),
        "marine_minus_static_detected": detected_advantage,
        "marine_mission_sites": marine_sites,
        "static_mission_sites": static_sites,
        "mission_divergence_count": mission_divergence_count,
        "mission_changed_rounds": mission_changed_rounds,
        "field_detections_beyond_initial": field_detections,
        "top_world_turnovers": top_world_turnovers,
        "max_propagated_abs_delta": max_propagated_abs_delta,
        "min_observation_probability": min_observation_probability,
        "hero_positive_candidate": bool(
            detected_advantage > 0
            and mission_divergence_count > 0
            and mission_changed_rounds > 0
            and field_detections > 0
        ),
        "hero_causal_candidate": bool(
            mission_divergence_count > 0
            and mission_changed_rounds > 0
        ),
    }


def _positive_key(row: dict) -> tuple:
    """Predeclared demo-legibility ordering, not a scientific benchmark score."""
    return (
        -int(row["marine_minus_static_detected"]),
        -int(row["mission_divergence_count"]),
        -int(row["mission_changed_rounds"]),
        -int(row["field_detections_beyond_initial"]),
        -float(row["max_propagated_abs_delta"]),
        -int(row["top_world_turnovers"]),
        str(row["case_id"]),
    )


def _causal_key(row: dict) -> tuple:
    return (
        -int(row["mission_divergence_count"]),
        -int(row["mission_changed_rounds"]),
        -float(row["max_propagated_abs_delta"]),
        -int(row["top_world_turnovers"]),
        str(row["case_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the 100 frozen UI cases for demo legibility. "
            "This is an illustrative-case selector, not a formal benchmark."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of frozen cases to inspect for a quick technical run.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of ranked candidate rows to print.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for the full case-level audit CSV.",
    )
    args = parser.parse_args()

    session = MissionControlSession()
    case_rows = session.case_library()["cases"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        case_rows = case_rows[: args.limit]

    rows = []
    for index, case in enumerate(case_rows, start=1):
        case_id = str(case["case_id"])
        row = _run_follow_marine(session, case_id)
        rows.append(row)
        print(
            f"[{index:03d}/{len(case_rows):03d}] {case_id} "
            f"Marine-Static={row['marine_minus_static_detected']:+d} "
            f"divergence={row['mission_divergence_count']} "
            f"mission_changed={row['mission_changed_rounds']} "
            f"field_detections={row['field_detections_beyond_initial']}"
        )

    positive = sorted(
        [row for row in rows if row["hero_positive_candidate"]],
        key=_positive_key,
    )
    causal = sorted(
        [row for row in rows if row["hero_causal_candidate"]],
        key=_causal_key,
    )

    better = sum(row["marine_minus_static_detected"] > 0 for row in rows)
    equal = sum(row["marine_minus_static_detected"] == 0 for row in rows)
    worse = sum(row["marine_minus_static_detected"] < 0 for row in rows)
    divergent = sum(row["mission_divergence_count"] > 0 for row in rows)

    summary = {
        "cases_audited": len(rows),
        "marine_detected_more_than_static": better,
        "marine_equal_static": equal,
        "marine_detected_less_than_static": worse,
        "cases_with_mission_divergence": divergent,
        "hero_positive_candidates": len(positive),
        "hero_causal_candidates": len(causal),
        "selection_semantics": (
            "Illustrative demo legibility only. "
            "Formal performance claims remain governed by frozen R8/R10 analyses."
        ),
    }

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nTOP POSITIVE ILLUSTRATIVE CANDIDATES")
    if positive:
        print(json.dumps(positive[: args.top], indent=2))
    else:
        print("None under the predeclared positive-candidate gate.")

    print("\nTOP CAUSAL-LEGIBILITY CANDIDATES")
    if causal:
        print(json.dumps(causal[: args.top], indent=2))
    else:
        print("None under the predeclared causal-candidate gate.")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_id",
            "initial_detection",
            "occupied_total",
            "marine_detected_occupied",
            "static_detected_occupied",
            "marine_minus_static_detected",
            "marine_mission_sites",
            "static_mission_sites",
            "mission_divergence_count",
            "mission_changed_rounds",
            "field_detections_beyond_initial",
            "top_world_turnovers",
            "max_propagated_abs_delta",
            "min_observation_probability",
            "hero_positive_candidate",
            "hero_causal_candidate",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                serializable = dict(row)
                serializable["marine_mission_sites"] = "|".join(
                    row["marine_mission_sites"]
                )
                serializable["static_mission_sites"] = "|".join(
                    row["static_mission_sites"]
                )
                writer.writerow(serializable)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
