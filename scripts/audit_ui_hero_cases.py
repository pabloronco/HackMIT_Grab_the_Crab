from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adaptive_response.mission_control import MissionControlSession


def _run_follow_marine(session: MissionControlSession, case_id: str) -> dict:
    """Run the interactive track by following Marine's current site + effort advice.

    This audit is for judging-case legibility and RAMP product behavior only.
    It is not a replacement for the frozen R8/R10 benchmark protocols.
    """

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
    followed_efforts: list[int] = []
    followed_sites: list[str] = []

    while not snap["can_reveal"]:
        recommendation = snap["global_recommendations"][0]
        effort = int(recommendation["recommended_effort"])
        site_id = str(recommendation["site_id"])
        followed_efforts.append(effort)
        followed_sites.append(site_id)

        snap = session.deploy(site_id=site_id, effort=effort)

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
    you = perf["you"]
    resource = perf["resource_receipt"]

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
    effort_saved = int(resource["effort_saved_vs_static"])
    same_or_better_detection = detected_advantage >= 0
    resource_win = effort_saved > 0 and same_or_better_detection

    occupied_but_missed = sum(
        row.get("realization") == "occupied_but_missed"
        for row in marine.get("mission_receipts", ())
    )

    return {
        "case_id": case_id,
        "initial_detection": revealed["incident"]["initial_detection"],
        "occupied_total": int(marine["occupied_total"]),
        "marine_detected_occupied": int(marine["detected_occupied"]),
        "static_detected_occupied": int(static["detected_occupied"]),
        "marine_minus_static_detected": detected_advantage,
        "marine_effort_spent": int(marine["effort_spent"]),
        "static_effort_spent": int(static["effort_spent"]),
        "effort_saved_vs_static": effort_saved,
        "marine_capacity_preserved": int(marine["capacity_preserved"]),
        "marine_mission_sites": marine_sites,
        "marine_mission_efforts": list(marine["mission_efforts"]),
        "static_mission_sites": static_sites,
        "followed_sites": followed_sites,
        "followed_efforts": followed_efforts,
        "mission_divergence_count": mission_divergence_count,
        "mission_changed_rounds": mission_changed_rounds,
        "field_detections_beyond_initial": field_detections,
        "top_world_turnovers": top_world_turnovers,
        "max_propagated_abs_delta": max_propagated_abs_delta,
        "min_observation_probability": min_observation_probability,
        "marine_occupied_but_missed_missions": occupied_but_missed,
        "resource_win": resource_win,
        "hero_ramp_candidate": bool(
            resource_win
            and mission_divergence_count > 0
            and mission_changed_rounds > 0
            and field_detections > 0
        ),
        "hero_detection_candidate": bool(
            detected_advantage > 0
            and mission_divergence_count > 0
            and mission_changed_rounds > 0
        ),
        "hero_causal_candidate": bool(
            mission_divergence_count > 0
            and mission_changed_rounds > 0
        ),
        "you_detected_occupied_when_following_marine": int(you["detected_occupied"]),
    }


def _ramp_key(row: dict) -> tuple:
    """Predeclared illustrative ordering; never use as a scientific score."""
    return (
        -int(row["marine_minus_static_detected"]),
        -int(row["effort_saved_vs_static"]),
        -int(row["mission_divergence_count"]),
        -int(row["mission_changed_rounds"]),
        -int(row["field_detections_beyond_initial"]),
        -float(row["max_propagated_abs_delta"]),
        -int(row["top_world_turnovers"]),
        str(row["case_id"]),
    )


def _detection_key(row: dict) -> tuple:
    return (
        -int(row["marine_minus_static_detected"]),
        -int(row["effort_saved_vs_static"]),
        -int(row["mission_changed_rounds"]),
        -int(row["mission_divergence_count"]),
        -float(row["max_propagated_abs_delta"]),
        str(row["case_id"]),
    )


def _causal_key(row: dict) -> tuple:
    return (
        -int(row["mission_changed_rounds"]),
        -int(row["mission_divergence_count"]),
        -int(row["effort_saved_vs_static"]),
        -float(row["max_propagated_abs_delta"]),
        -int(row["top_world_turnovers"]),
        str(row["case_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen UI case library under the RAMP-aware three-deployment "
            "Marine policy. This is an illustrative-case selector, not a formal benchmark."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of frozen cases for a quick technical run.",
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

    rows: list[dict] = []
    for index, case in enumerate(case_rows, start=1):
        case_id = str(case["case_id"])
        row = _run_follow_marine(session, case_id)
        rows.append(row)
        print(
            f"[{index:03d}/{len(case_rows):03d}] {case_id} "
            f"det_delta={row['marine_minus_static_detected']:+d} "
            f"effort_saved={row['effort_saved_vs_static']:+d} "
            f"efforts={row['marine_mission_efforts']} "
            f"replans={row['mission_changed_rounds']}"
        )

    ramp = sorted(
        [row for row in rows if row["hero_ramp_candidate"]],
        key=_ramp_key,
    )
    detection = sorted(
        [row for row in rows if row["hero_detection_candidate"]],
        key=_detection_key,
    )
    causal = sorted(
        [row for row in rows if row["hero_causal_candidate"]],
        key=_causal_key,
    )

    better = sum(row["marine_minus_static_detected"] > 0 for row in rows)
    equal = sum(row["marine_minus_static_detected"] == 0 for row in rows)
    worse = sum(row["marine_minus_static_detected"] < 0 for row in rows)
    resource_wins = sum(bool(row["resource_win"]) for row in rows)
    any_savings = sum(row["effort_saved_vs_static"] > 0 for row in rows)
    divergent = sum(row["mission_divergence_count"] > 0 for row in rows)
    replanned = sum(row["mission_changed_rounds"] > 0 for row in rows)
    unlucky_marine = sum(row["marine_occupied_but_missed_missions"] > 0 for row in rows)

    effort_savings = [int(row["effort_saved_vs_static"]) for row in rows]
    avg_saving = sum(effort_savings) / len(effort_savings) if effort_savings else 0.0

    summary = {
        "cases_audited": len(rows),
        "marine_detected_more_than_static": better,
        "marine_equal_static": equal,
        "marine_detected_less_than_static": worse,
        "marine_saved_effort": any_savings,
        "marine_saved_effort_with_same_or_better_detection": resource_wins,
        "average_effort_saved_vs_static": avg_saving,
        "cases_with_mission_divergence": divergent,
        "cases_with_evidence_caused_replan": replanned,
        "cases_where_marine_surveyed_occupied_but_missed": unlucky_marine,
        "hero_ramp_candidates": len(ramp),
        "hero_detection_candidates": len(detection),
        "hero_causal_candidates": len(causal),
        "selection_semantics": (
            "Illustrative three-deployment demo audit only. Formal performance claims "
            "remain governed by frozen R8/R10 analyses. Do not tune a case after reveal."
        ),
    }

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    print("\nTOP RAMP ILLUSTRATIVE CANDIDATES")
    print(json.dumps(ramp[: args.top], indent=2) if ramp else "None under the frozen RAMP gate.")

    print("\nTOP DETECTION-ADVANTAGE CANDIDATES")
    print(json.dumps(detection[: args.top], indent=2) if detection else "None under the frozen detection gate.")

    print("\nTOP CAUSAL-LEGIBILITY CANDIDATES")
    print(json.dumps(causal[: args.top], indent=2) if causal else "None under the frozen causal gate.")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                serializable = dict(row)
                for key in (
                    "marine_mission_sites",
                    "marine_mission_efforts",
                    "static_mission_sites",
                    "followed_sites",
                    "followed_efforts",
                ):
                    serializable[key] = "|".join(map(str, row[key]))
                writer.writerow(serializable)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
