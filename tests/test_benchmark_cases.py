import json
from pathlib import Path

from adaptive_response.benchmark_cases import build_benchmark_case_manifest, eligible_incident_seeds


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def manifest() -> dict:
    return build_benchmark_case_manifest(
        load("reports/milestones/r2_real_graph_v0/incident_subgraph_audit.json"),
        load("configs/benchmark_protocol_r5.json"),
        load("configs/benchmark_protocol_r7.json"),
        load("configs/q_protocol_r7.json"),
    )


def test_r8_manifest_has_frozen_recommended_case_counts() -> None:
    result = manifest()
    assert result["case_count"] == 240
    assert result["split_counts"] == {
        "validation": 60,
        "id_test": 90,
        "ood_model_test": 30,
        "ood_q_low": 30,
        "ood_q_high": 30,
    }


def test_manifest_uses_only_preferred_topology_seeds() -> None:
    audit = load("reports/milestones/r2_real_graph_v0/incident_subgraph_audit.json")
    allowed = dict(eligible_incident_seeds(audit))
    result = manifest()
    assert len(allowed) == 46
    assert all(case["incident_seed_site_id"] in allowed for case in result["cases"])
    assert all(12 <= case["incident_site_count"] <= 20 for case in result["cases"])
    assert set(result["incident_topology_size_counts"]).issubset({"14", "15", "17"})


def test_family_holdout_and_q_shift_semantics_are_not_mixed() -> None:
    result = manifest()
    by_split: dict[str, list[dict]] = {}
    for case in result["cases"]:
        by_split.setdefault(case["split"], []).append(case)

    train = {"A_graph_diffusion", "B_spatial_cluster", "C_habitat_driven"}
    assert {case["family_id"] for case in by_split["validation"]} == train
    assert {case["family_id"] for case in by_split["id_test"]} == train
    assert {case["family_id"] for case in by_split["ood_model_test"]} == {"E_fragmented_patchy"}
    assert {case["family_id"] for case in by_split["ood_q_low"]} == train
    assert {case["family_id"] for case in by_split["ood_q_high"]} == train
    assert {case["simulator_q_true"] for case in by_split["ood_q_low"]} == {0.02}
    assert {case["simulator_q_true"] for case in by_split["ood_q_high"]} == {0.35}


def test_manifest_contains_no_truth_action_reward_or_results() -> None:
    result = manifest()
    assert result["contains_hidden_occupancy_truth"] is False
    assert result["contains_planner_results"] is False
    assert result["contains_action_or_reward_contract"] is False
    forbidden = {"occupied_by_site", "hidden_world", "planner_score", "reward", "mission_action"}
    assert all(not forbidden.intersection(case) for case in result["cases"])


def test_manifest_generation_is_deterministic_and_case_ids_unique() -> None:
    first = manifest()
    second = manifest()
    assert first == second
    case_ids = [case["case_id"] for case in first["cases"]]
    assert len(case_ids) == len(set(case_ids))
    assert len({(case["split"], case["world_seed"]) for case in first["cases"]}) == len(first["cases"])
