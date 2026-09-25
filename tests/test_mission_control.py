from __future__ import annotations

import math

import pytest

from adaptive_response.mission_control import MissionControlSession


def _deploy_top_recommendation(session: MissionControlSession, effort: int = 6) -> dict:
    snap = session.snapshot()
    site_id = snap["global_recommendations"][0]["site_id"]
    return session.deploy(site_id=site_id, effort=effort)


def test_reset_produces_a_real_incident_not_a_toy_scenario() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    assert 12 <= len(snap["nodes"]) <= 20
    assert snap["phase"] == "ready_to_plan"
    assert snap["can_plan"] and not snap["can_execute"] and not snap["can_reveal"]
    for node in snap["nodes"]:
        assert node["id"].isdigit(), node["id"]
    assert snap["incident"]["initial_detection"] in {n["id"] for n in snap["nodes"]}


def test_case_library_freezes_one_hundred_reproducible_demo_cases() -> None:
    session = MissionControlSession()
    library = session.case_library()

    assert library["count"] == 100
    assert len(library["cases"]) == 100
    assert library["cases"][0]["case_id"] == "incident_001"
    assert library["cases"][-1]["case_id"] == "incident_100"
    assert len({row["case_id"] for row in library["cases"]}) == 100


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
        assert snap["incident"]["truth_family"] is None
    assert not snap["revealed"]

    revealed = session.reveal()
    assert revealed["revealed"]
    assert all("true_occupied" in node for node in revealed["nodes"])
    assert revealed["incident"]["truth_family"]


def test_full_round_trip_follows_marine_until_budget_exhausted_and_reaches_reveal() -> None:
    """The response window is budget-driven (up to 18 windows of >=1 effort
    unit each), not a fixed three-deployment cap. Following Marine's own
    recommended effort every round must run until the 18-unit budget is
    fully spent, however many windows that takes."""
    session = MissionControlSession()
    snap = session.snapshot()

    rounds = 0
    while not snap["can_reveal"]:
        rec = snap["global_recommendations"][0]
        snap = session.deploy(site_id=rec["site_id"], effort=int(rec["recommended_effort"]))
        rounds += 1
        assert rounds <= 18  # defensive cap, never a target

    assert snap["resources"]["mission_horizon"] == 18
    assert snap["resources"]["remaining_budget"] == 0
    assert snap["resources"]["spent_budget"] == snap["resources"]["initial_budget"]
    assert snap["can_reveal"] and not snap["can_plan"] and not snap["can_execute"]

    snap = session.reveal()
    assert snap["revealed"]
    occupied = [n["id"] for n in snap["nodes"] if n["true_occupied"]]
    assert snap["incident"]["initial_detection"] in occupied


def test_effort_six_path_uses_exactly_three_windows() -> None:
    """6 + 6 + 6 = 18: budget exhausted in exactly 3 windows."""
    session = MissionControlSession()
    snap = session.snapshot()
    rounds = 0
    while not snap["can_reveal"]:
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=6)
        rounds += 1
        assert rounds <= 18
    assert rounds == 3
    assert snap["resources"]["spent_budget"] == 18
    assert snap["resources"]["remaining_budget"] == 0


def test_effort_three_path_uses_exactly_six_windows() -> None:
    """3 x 6 = 18: budget exhausted in exactly 6 windows."""
    session = MissionControlSession()
    snap = session.snapshot()
    rounds = 0
    while not snap["can_reveal"]:
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=3)
        rounds += 1
        assert rounds <= 18
    assert rounds == 6
    assert snap["resources"]["spent_budget"] == 18
    assert snap["resources"]["remaining_budget"] == 0


def test_mixed_effort_path_terminates_exactly_when_budget_is_exhausted() -> None:
    """3 + 1 + 6 + 1 + 3 + 3 + 1 = 18: a legal mixed trajectory must run
    exactly that many windows, not stop early and not overrun."""
    session = MissionControlSession()
    snap = session.snapshot()
    sequence = (3, 1, 6, 1, 3, 3, 1)
    assert sum(sequence) == 18

    for i, effort in enumerate(sequence):
        assert not snap["can_reveal"], f"reveal unlocked early, before window {i + 1}"
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=effort)

    assert snap["can_reveal"]
    assert snap["resources"]["spent_budget"] == 18
    assert snap["resources"]["remaining_budget"] == 0


def test_propagated_belief_changes_never_duplicate_the_surveyed_sites() -> None:
    session = MissionControlSession()
    session.plan()
    snap = session.execute()

    surveyed = {obs["site_id"] for obs in snap["last_round"]["observations"]}
    propagated_ids = {
        row["site_id"]
        for row in snap["last_round"]["propagated_belief_changes"]
    }
    assert surveyed.isdisjoint(propagated_ids)
    deltas = [
        abs(row["delta"])
        for row in snap["last_round"]["propagated_belief_changes"]
    ]
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
    ranked_ids = {
        row["site_id"]
        for row in mission["diagnostics"]["ranked_selected_sites"]
    }
    assert allocated_ids
    assert allocated_ids <= ranked_ids


def test_recommendations_are_ranked_and_predictive_detection_is_effort_monotone() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    rows = snap["global_recommendations"]

    assert 1 <= len(rows) <= 5
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert snap["incident"]["initial_detection"] not in {row["site_id"] for row in rows}

    for row in rows:
        p1 = row["predictive_detection"]["1"]
        p3 = row["predictive_detection"]["3"]
        p6 = row["predictive_detection"]["6"]
        assert 0.0 <= p1 <= p3 <= p6 <= 1.0


def test_operator_can_override_site_and_effort_without_bypassing_bayes() -> None:
    session = MissionControlSession()
    before = session.snapshot()
    assert len(before["global_recommendations"]) >= 2

    choice = before["global_recommendations"][1]
    snap = session.deploy(site_id=choice["site_id"], effort=3)

    assert snap["resources"]["spent_budget"] == 3
    assert snap["resources"]["remaining_budget"] == before["resources"]["remaining_budget"] - 3
    obs = snap["last_round"]["observations"][0]
    assert obs["site_id"] == choice["site_id"]
    assert obs["effort"] == 3
    assert len(snap["last_round"]["propagated_belief_changes"]) <= 5
    assert snap["model_stress"]["site_id"] == choice["site_id"]
    assert snap["model_stress"]["effort"] == 3


def test_initial_detection_cannot_be_selected_as_new_delimitation_mission() -> None:
    session = MissionControlSession()
    site_id = session.snapshot()["incident"]["initial_detection"]

    with pytest.raises(ValueError, match="initial-detection"):
        session.deploy(site_id=site_id, effort=1)


def test_top_worlds_are_unique_extents_marginalized_over_q() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    bundle = snap["top_worlds"]

    ids = [row["world_id"] for row in bundle["items"]]
    assert len(ids) == len(set(ids))
    assert bundle["unique_world_count"] >= len(ids)
    mass = sum(row["posterior"] for row in bundle["items"]) + bundle["remaining_mass"]
    assert math.isclose(mass, 1.0, abs_tol=1e-9)

    known = {node["id"] for node in snap["nodes"]}
    for row in bundle["items"]:
        assert set(row["occupied_site_ids"]) <= known
        assert row["occupied_count"] == len(row["occupied_site_ids"])


def test_case_site_effort_outcome_is_reproducible_across_resets() -> None:
    session = MissionControlSession()
    snap = session.reset(case_id="incident_004")
    site_id = snap["global_recommendations"][0]["site_id"]

    first = session.deploy(site_id=site_id, effort=3)["last_round"]["observations"][0]
    session.reset(case_id="incident_004")
    second = session.deploy(site_id=site_id, effort=3)["last_round"]["observations"][0]

    assert first["detection"] == second["detection"]
    assert first["belief_before"] == second["belief_before"]
    assert first["belief_after"] == second["belief_after"]


def test_reveal_scores_marine_static_and_human_against_same_hidden_incident() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    while not snap["can_reveal"]:
        affordable = min(6, snap["resources"]["remaining_budget"])
        if affordable not in (1, 3, 6):
            # With the frozen 18-unit budget and effort-6 choices this branch is
            # unreachable; keep the assertion explicit if the contract changes.
            raise AssertionError(f"Unexpected remaining budget {affordable}")
        snap = _deploy_top_recommendation(session, effort=affordable)

    revealed = session.reveal()
    performance = revealed["performance"]
    totals = {
        performance[name]["occupied_total"]
        for name in ("marine", "static", "you")
    }
    assert len(totals) == 1

    for name in ("marine", "static", "you"):
        row = performance[name]
        assert row["detected_occupied"] + row["undetected_occupied"] == row["occupied_total"]
        assert row["curve"][0]["effort"] == 0
        assert row["effort_spent"] == 18
        # Static always precommits 3 x e6 = 18. "You" in this test always
        # forces the largest affordable effort, which also spends 18 in
        # exactly 3 missions. Marine's own comparator now genuinely varies
        # effort by belief (PROBE -> DELIMIT -> CONFIRM), so its mission
        # count is no longer pinned to 3 - only bounded by the budget/effort
        # contract (at least 3, at most 18 windows of >=1 unit each).
        if name in ("static", "you"):
            assert row["missions_completed"] == 3
        else:
            assert 3 <= row["missions_completed"] <= 18
        assert 0 < row["curve"][-1]["effort"] <= 18
        assert 0.0 <= row["curve"][-1]["detected_fraction"] <= 1.0

    assert performance["static"]["effort_spent"] == 18
    assert performance["resource_receipt"]["effort_saved_vs_static"] == (
        performance["static"]["effort_spent"] - performance["marine"]["effort_spent"]
    )


def test_low_effort_judge_path_spends_the_full_budget_across_eighteen_windows() -> None:
    """1 x 18 = 18: choosing the smallest effort every time buys the maximum
    number of decision windows, and the campaign only ends once the full
    18-unit budget is spent - it must NOT stop after three windows leaving
    15 units unused."""
    session = MissionControlSession()
    snap = session.snapshot()

    rounds = 0
    while not snap["can_reveal"]:
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=1)
        rounds += 1
        assert rounds <= 18

    assert rounds == 18
    assert snap["resources"]["spent_budget"] == 18
    assert snap["resources"]["remaining_budget"] == 0
    assert snap["resource_summary"]["capacity_preserved"] == 0


def test_cannot_overspend_remaining_budget() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    site_id = snap["global_recommendations"][0]["site_id"]
    snap = session.deploy(site_id=site_id, effort=6)  # remaining 12
    snap = session.deploy(site_id=snap["global_recommendations"][0]["site_id"], effort=6)  # remaining 6
    snap = session.deploy(site_id=snap["global_recommendations"][0]["site_id"], effort=1)  # remaining 5

    assert snap["resources"]["remaining_budget"] == 5
    with pytest.raises(ValueError, match="exceeds remaining field budget"):
        session.deploy(site_id=snap["global_recommendations"][0]["site_id"], effort=6)


def test_reveal_stays_locked_while_budget_remains() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    assert not snap["can_reveal"]
    with pytest.raises(RuntimeError):
        session.reveal()

    site_id = snap["global_recommendations"][0]["site_id"]
    snap = session.deploy(site_id=site_id, effort=6)  # remaining 12, still active
    assert not snap["can_reveal"]
    with pytest.raises(RuntimeError):
        session.reveal()


def test_hidden_truth_is_absent_from_every_node_before_reveal() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    while not snap["can_reveal"]:
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=6)
        assert all("true_occupied" not in node for node in snap["nodes"])
    assert all("true_occupied" not in node for node in snap["nodes"])

    revealed = session.reveal()
    assert all("true_occupied" in node for node in revealed["nodes"])


def test_same_seed_and_same_action_sequence_is_deterministic() -> None:
    """Same incident + same (site, effort) sequence -> same field outcomes,
    independent of how many windows the trajectory happens to take."""
    sequence = (3, 1, 6, 1, 3, 3, 1)

    def run(seed: int) -> list[tuple[str, int, bool]]:
        session = MissionControlSession()
        snap = session.reset(seed=seed)
        trace = []
        for effort in sequence:
            site_id = snap["global_recommendations"][0]["site_id"]
            snap = session.deploy(site_id=site_id, effort=effort)
            obs = snap["last_round"]["observations"][0]
            trace.append((obs["site_id"], obs["effort"], obs["detection"]))
        return trace

    assert run(seed=7) == run(seed=7)


def test_comparator_tracks_share_the_same_hidden_incident_and_full_budget() -> None:
    """Marine and Static must be evaluated against the identical hidden
    world and must each account for the entire 18-unit budget - a policy
    choosing smaller efforts is not allowed to be truncated at three
    windows while leaving field capacity unaccounted for."""
    session = MissionControlSession()
    snap = session.snapshot()
    while not snap["can_reveal"]:
        rec = snap["global_recommendations"][0]
        snap = session.deploy(site_id=rec["site_id"], effort=int(rec["recommended_effort"]))
    revealed = session.reveal()
    performance = revealed["performance"]

    assert performance["marine"]["occupied_total"] == performance["static"]["occupied_total"]
    assert performance["marine"]["effort_spent"] == 18
    assert performance["static"]["effort_spent"] == 18


def test_mission_updated_flag_means_evidence_changed_counterfactual_next_site() -> None:
    session = MissionControlSession()
    snap = session.reset(case_id="incident_003")
    top = snap["global_recommendations"][0]["site_id"]
    snap = session.deploy(site_id=top, effort=6)

    replan = snap["replan"]
    assert replan["semantics"] == "same_post_survey_public_state_without_new_evidence"

    if replan["from"] is None or replan["to"] is None:
        assert snap["mission_changed"] is False
        return

    from_sig = [
        (row["site_id"], row["effort_units"])
        for row in replan["from"]["allocations"]
    ]
    to_sig = [
        (row["site_id"], row["effort_units"])
        for row in replan["to"]["allocations"]
    ]
    assert snap["mission_changed"] is (from_sig != to_sig)


def test_dashboard_defaults_to_frozen_hero_case_and_exposes_real_map_metadata() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    assert snap["case"]["case_id"] == "incident_079"
    assert len(snap["static_response"]["plan_sites"]) == 3
    assert snap["incident"]["initial_detection"] not in set(
        snap["static_response"]["plan_sites"]
    )

    for node in snap["nodes"]:
        assert isinstance(node["latitude"], float)
        assert isinstance(node["longitude"], float)
        assert -90.0 <= node["latitude"] <= 90.0
        assert -180.0 <= node["longitude"] <= 180.0
        assert node["zone"] == "Salish Sea"
        assert node["habitat_label"]


def test_live_curve_contains_only_observable_progress_and_tracks_budget() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    assert snap["live_curve"] == [
        {
            "round": 0,
            "effort": 0,
            "field_detections": 1,
            "budget_used_fraction": 0.0,
        }
    ]

    top = snap["global_recommendations"][0]["site_id"]
    snap = session.deploy(site_id=top, effort=6)
    point = snap["live_curve"][-1]
    assert point["round"] == 1
    assert point["effort"] == 6
    assert point["field_detections"] >= 1
    assert math.isclose(point["budget_used_fraction"], 6 / 18)
    assert "true_occupied" not in point


def test_resource_aware_recommendation_exposes_site_and_effort_diagnostics() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    top = snap["global_recommendations"][0]

    assert top["recommended_effort"] in (1, 3, 6)
    diag = top["effort_recommendation"]
    assert diag["recommended_effort"] == top["recommended_effort"]
    assert diag["occupancy_band"] in {"exploratory", "delimitation", "confirmation"}
    assert 0.0 <= diag["occupancy_belief"] <= 1.0
    assert diag["rule"] == "probe_delimit_confirm"


def test_all_three_effort_levels_are_genuinely_reachable_actions() -> None:
    """Regression guard for the R11 bug where every recommendation converged
    on e6 regardless of belief (the old rule's fixed detection-power
    retention threshold structurally excluded e1). Uses the frozen default
    demo case (deterministic, real audited trajectory: confirmation -> e6,
    delimitation -> e3, exploratory -> e1) rather than a synthetic scenario,
    so this also pins the exact case the live demo relies on."""
    session = MissionControlSession()
    snap = session.snapshot()  # defaults to the frozen demo case

    seen_bands: dict[str, int] = {}
    while not snap["can_reveal"]:
        rec = snap["global_recommendations"][0]
        diag = rec["effort_recommendation"]
        seen_bands[diag["occupancy_band"]] = int(diag["recommended_effort"])
        snap = session.deploy(site_id=rec["site_id"], effort=int(diag["recommended_effort"]))

    assert seen_bands.get("confirmation") == 6
    assert seen_bands.get("delimitation") == 3
    assert seen_bands.get("exploratory") == 1


def test_q_boundary_pressure_is_separate_from_model_stress() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    qd = snap["q_diagnostics"]
    assert 0.0 <= qd["boundary_pressure"] <= 1.0
    assert "stress" in qd["edge_note"].lower()
    assert snap["model_stress"] is None


def test_environment_layers_never_impute_temperature() -> None:
    session = MissionControlSession()
    snap = session.snapshot()
    temp = snap["environment_layers"]["temperature"]

    observed = sum(node["temperature_median_c"] is not None for node in snap["nodes"])
    assert temp["observed_sites"] == observed
    assert temp["total_sites"] == len(snap["nodes"])
    assert "no imputation" in temp["provenance"].lower()
