import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_q_protocol_freezes_scenario_support_without_claiming_calibration() -> None:
    config = _load("configs/q_protocol_r7.json")

    assert config["identifiability"]["current_real_table_directly_identifies_q"] is False
    assert config["identifiability"]["ecological_numeric_q_range_status"] == "OPEN"
    assert config["benchmark_design_support"]["belief_q_values"] == [0.05, 0.10, 0.20]
    assert config["benchmark_design_support"]["simulator_q_true_train_values"] == [0.05, 0.10, 0.20]
    assert [case["simulator_q_true"] for case in config["ood_q_mismatch"]] == [0.02, 0.35]
    assert "not an empirical biological prior" in config["benchmark_design_support"]["selection_semantics"]


def test_q_protocol_keeps_hidden_q_out_of_policy_path() -> None:
    config = _load("configs/q_protocol_r7.json")
    firewall = " ".join(config["information_firewall"])

    assert "q_true must not appear in GraphState" in firewall
    assert "q_true must not appear in planner constraints" in firewall
    assert "q_true must not appear in field-observation metadata" in firewall


def test_r7_benchmark_requires_spatial_information_gain_and_same_cases() -> None:
    config = _load("configs/benchmark_protocol_r7.json")

    assert config["frozen"]["formal_information_gain_mode"] == "spatial_joint"
    assert config["frozen"]["planner_hidden_truth_access"] is False
    assert config["frozen"]["graph_state_schema_change_required"] is False
    assert "same cases" in config["frozen"]["comparison_rule"]
    assert config["ood_q_tests"] == [
        {"q_true": 0.02, "belief_support": [0.05, 0.10, 0.20]},
        {"q_true": 0.35, "belief_support": [0.05, 0.10, 0.20]},
    ]


def test_action_and_reward_remain_explicit_cross_team_ack_items() -> None:
    config = _load("configs/benchmark_protocol_r7.json")

    assert config["recommended_action_contract_for_cross_team_ack"]["status"].startswith("OPEN_CROSS_TEAM")
    assert config["recommended_rl_reward_for_cross_team_ack"]["status"].startswith("OPEN_CROSS_TEAM")
