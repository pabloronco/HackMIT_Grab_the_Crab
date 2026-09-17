from __future__ import annotations

from adaptive_response.mission_control import MissionControlSession


def _run_to_completion(session: MissionControlSession) -> dict:
    snap = session.plan()
    while True:
        snap = session.execute()
        if snap["can_reveal"]:
            return snap


def test_reset_produces_a_real_incident_not_a_toy_scenario() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    assert 12 <= len(snap["nodes"]) <= 20
    assert snap["phase"] == "ready_to_plan"
    assert snap["can_plan"] and not snap["can_execute"] and not snap["can_reveal"]
    # Real site ids are bare monitoring-site ids (e.g. "362"), not the M5 toy
    # scenario's "site_XX" convention.
    for node in snap["nodes"]:
        assert node["id"].isdigit(), node["id"]
    assert snap["incident"]["initial_detection"] in {n["id"] for n in snap["nodes"]}


def test_hidden_truth_is_absent_from_every_snapshot_before_reveal() -> None:
    session = MissionControlSession()
    for node in session.snapshot()["nodes"]:
        assert "true_occupied" not in node

    snap = session.plan()
    for node in snap["nodes"]:
        assert "true_occupied" not in node

    while not snap["can_reveal"]:
        snap = session.execute()
        for node in snap["nodes"]:
            assert "true_occupied" not in node
    assert not snap["revealed"]

    revealed = session.reveal()
    assert revealed["revealed"]
    assert all("true_occupied" in node for node in revealed["nodes"])


def test_full_round_trip_spends_the_whole_budget_and_reaches_reveal() -> None:
    session = MissionControlSession()
    initial_budget = session.snapshot()["resources"]["initial_budget"]

    snap = _run_to_completion(session)
    assert snap["resources"]["remaining_budget"] == 0
    assert snap["resources"]["spent_budget"] == initial_budget
    assert snap["can_reveal"] and not snap["can_plan"] and not snap["can_execute"]

    snap = session.reveal()
    assert snap["revealed"]
    occupied = [n["id"] for n in snap["nodes"] if n["true_occupied"]]
    # The confirmed initial detection is occupied by construction (Environment
    # always keeps it occupied; WorldModel.sample must preserve it too).
    assert snap["incident"]["initial_detection"] in occupied


def test_propagated_belief_changes_never_duplicate_the_surveyed_sites() -> None:
    session = MissionControlSession()
    session.plan()
    snap = session.execute()

    surveyed = {obs["site_id"] for obs in snap["last_round"]["observations"]}
    propagated_ids = {row["site_id"] for row in snap["last_round"]["propagated_belief_changes"]}
    assert surveyed.isdisjoint(propagated_ids)
    # Sorted by descending magnitude of belief change.
    deltas = [abs(row["delta"]) for row in snap["last_round"]["propagated_belief_changes"]]
    assert deltas == sorted(deltas, reverse=True)


def test_reset_with_the_same_seed_is_deterministic() -> None:
    session = MissionControlSession()
    first = session.reset(seed=3)
    second = session.reset(seed=3)

    assert first["incident"]["initial_detection"] == second["incident"]["initial_detection"]
    assert [n["belief"] for n in first["nodes"]] == [n["belief"] for n in second["nodes"]]
    assert [n["id"] for n in first["nodes"]] == [n["id"] for n in second["nodes"]]


def test_mission_diagnostics_reference_a_real_allocated_site() -> None:
    session = MissionControlSession()
    snap = session.plan()

    mission = snap["mission"]
    allocated_ids = {a["site_id"] for a in mission["allocations"]}
    ranked_ids = {row["site_id"] for row in mission["diagnostics"]["ranked_selected_sites"]}
    assert allocated_ids
    assert allocated_ids <= ranked_ids
