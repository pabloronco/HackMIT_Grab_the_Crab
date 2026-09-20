from __future__ import annotations

import math

import pytest

from adaptive_response.mission_control import MissionControlSession


def _run_to_completion(session: MissionControlSession) -> dict:
    snap = session.plan()
    while True:
        snap = session.execute()
        if snap["can_reveal"]:
            return snap


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


def test_full_round_trip_completes_three_deployment_window_and_reaches_reveal() -> None:
    session = MissionControlSession()

    snap = _run_to_completion(session)
    assert snap["resources"]["round"] == 3
    assert snap["resources"]["mission_horizon"] == 3
    assert snap["resources"]["missions_remaining"] == 0
    assert 0 <= snap["resources"]["remaining_budget"] <= snap["resources"]["initial_budget"]
    assert snap["can_reveal"] and not snap["can_plan"] and not snap["can_execute"]

    snap = session.reveal()
    assert snap["revealed"]
    occupied = [n["id"] for n in snap["nodes"] if n["true_occupied"]]
    assert snap["incident"]["initial_detection"] in occupied


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
        assert row["missions_completed"] == 3
        assert 0 < row["curve"][-1]["effort"] <= 18
        assert 0.0 <= row["curve"][-1]["detected_fraction"] <= 1.0

    assert performance["static"]["effort_spent"] == 18
    assert performance["resource_receipt"]["effort_saved_vs_static"] == (
        performance["static"]["effort_spent"] - performance["marine"]["effort_spent"]
    )


def test_low_effort_judge_path_preserves_capacity_after_three_deployments() -> None:
    session = MissionControlSession()
    snap = session.snapshot()

    while not snap["can_reveal"]:
        site_id = snap["global_recommendations"][0]["site_id"]
        snap = session.deploy(site_id=site_id, effort=1)

    assert snap["resources"]["round"] == 3
    assert snap["resources"]["spent_budget"] == 3
    assert snap["resources"]["remaining_budget"] == 15
    assert snap["resource_summary"]["capacity_preserved"] == 15


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

    assert snap["case"]["case_id"] == "incident_097"
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
    assert diag["occupancy_band"] in {"exploratory", "medium", "high"}
    assert 0.0 <= diag["occupancy_belief"] <= 1.0
    assert 0.0 < diag["information_retention_required"] <= 1.0
    assert 0.0 < diag["detection_power_retention_required"] <= 1.0
    assert diag["rule"] == "occupancy_aware_smallest_effort_retaining_information_and_detection_power"


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
