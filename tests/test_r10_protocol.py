import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r10_static_adaptive_protocol.json"
MANIFEST_PATH = (
    ROOT
    / "reports"
    / "milestones"
    / "r8_benchmark_case_manifest"
    / "benchmark_cases.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_r10_protocol_is_frozen_before_results() -> None:
    protocol = _protocol()

    assert protocol["version"] == "r10-v1.1"
    assert protocol["status"] == "FROZEN_BEFORE_RESULTS"
    assert (
        protocol["analysis_role"]
        == "SECONDARY_PAIRED_ANALYSIS_ON_PREVIOUSLY_SEEN_R8_FORMAL_CASES"
    )


def test_r10_reuses_exact_r8_formal_case_counts() -> None:
    protocol = _protocol()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    expected = protocol["case_contract"]["splits"]
    counts = Counter(case["split"] for case in manifest["cases"])

    assert len(manifest["cases"]) == 240
    assert counts["validation"] == 60

    for split, expected_count in expected.items():
        assert counts[split] == expected_count

    assert sum(expected.values()) == 180
    assert protocol["case_contract"]["formal_cases"] == 180
    assert protocol["case_contract"]["excluded_validation_cases"] == 60


def test_r10_static_and_adaptive_differ_only_in_replanning_contract() -> None:
    protocol = _protocol()
    shared = protocol["shared_contract"]
    static = protocol["static_arm"]
    adaptive = protocol["adaptive_arm"]

    assert shared["planner"] == "FrontierPlanner"
    assert shared["total_budget"] == 18
    assert shared["missions"] == 3
    assert shared["effort_per_mission"] == 6
    assert shared["max_sites_per_mission"] == 1
    assert shared["distinct_sites_only"] is True
    assert shared["site_revisit_allowed"] is False

    assert static["posterior_updates_during_execution"] is True
    assert adaptive["posterior_updates_during_execution"] is True

    assert static["future_missions_may_change_after_evidence"] is False
    assert adaptive["future_missions_may_change_after_evidence"] is True

    assert static["planning_time"] == "t0_only"
    assert adaptive["planning_time"] == "before_each_mission"


def test_r10_frontier_ranking_matches_frozen_project_rule() -> None:
    protocol = _protocol()

    assert protocol["shared_contract"]["frontier_ranking"] == [
        "frontier first",
        "higher occupancy belief",
        "higher uncertainty",
        "deterministic site_id tie-break",
    ]


def test_r10_paired_randomness_is_policy_independent() -> None:
    protocol = _protocol()
    randomness = protocol["paired_observation_randomness"]

    assert randomness["scheme"] == "sha256_uniform_v1"
    assert randomness["key_format"] == "r10-v1|{case_id}|{site_id}"
    assert randomness["round_not_in_random_key"] is True
    assert randomness["unoccupied_detection_probability"] == 0.0

    forbidden = set(randomness["planner_forbidden_inputs"])
    assert {
        "hidden occupancy",
        "q_true",
        "paired uniform u",
        "future observations",
    } <= forbidden


def test_r10_primary_delta_and_bootstrap_are_frozen() -> None:
    protocol = _protocol()
    metrics = protocol["metrics"]
    inference = protocol["inference_contract"]

    assert metrics["primary"] == [
        "missed_occupied_fraction",
        "occupied_site_coverage",
    ]
    assert (
        metrics["paired_primary_delta"]["definition"]
        == "static_missed_occupied_fraction - adaptive_missed_occupied_fraction"
    )

    assert inference["bootstrap_replicates"] == 10000
    assert inference["bootstrap_unit"] == "case"
    assert inference["bootstrap_seed"] == 20260917
    assert inference["confidence_interval"] == 0.95
    assert inference["paired_analysis"] is True


def test_r10_post_result_tuning_is_forbidden() -> None:
    protocol = _protocol()
    freeze = protocol["post_result_change_policy"]

    assert freeze["planner_tuning_after_results"] is False
    assert freeze["q_tuning_after_results"] is False
    assert freeze["world_model_tuning_after_results"] is False
    assert freeze["budget_or_effort_tuning_after_results"] is False
    assert freeze["randomness_protocol_tuning_after_results"] is False
    assert freeze["metric_redefinition_after_results"] is False



def test_r10_confirmed_initial_detection_is_preexcluded() -> None:
    protocol = _protocol()
    shared = protocol["shared_contract"]

    assert shared["confirmed_initial_detection_eligible"] is False

    amendment = protocol["pre_result_amendment"]
    assert amendment["status"] == "FROZEN_BEFORE_RESULTS_AMENDMENT"
    assert amendment["changes_randomness_protocol"] is False
    assert amendment["changes_frontier_ranking"] is False
    assert amendment["changes_budget_or_effort"] is False
    assert amendment["changes_metrics"] is False
