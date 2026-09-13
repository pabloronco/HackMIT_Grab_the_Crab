import csv
import json
from pathlib import Path

from adaptive_response.benchmark_cases import build_benchmark_case_manifest


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_real_site_context_receipt_has_all_49_monitoring_sites_and_real_coordinates() -> None:
    path = ROOT / "reports/milestones/r2_real_graph_v0/real_sites_v0.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 49
    assert len({row["site_id"] for row in rows}) == 49
    assert all(row["latitude"] and row["longitude"] for row in rows)
    assert {row["crabteam_habitat"] for row in rows} == {"Channel", "Lagoon", "Tideflat"}
    site_138 = next(row for row in rows if row["site_id"] == "138")
    assert site_138["shorezone_accepted"] == "False"


def test_real_site_context_uses_r5_observation_proxy_without_claiming_latent_suitability() -> None:
    context = _json("configs/real_site_context_r8.json")
    scores = context["habitat_proxy"]["scores"]
    assert scores == {
        "Channel": 0.09722222222222222,
        "Lagoon": 0.022727272727272728,
        "Tideflat": 0.06382978723404255,
    }
    assert "not latent habitat suitability" in context["habitat_proxy"]["semantics"]
    assert "placeholder" in context["formal_benchmark_rule"]


def test_r8_protocol_freezes_action_reward_cases_but_blocks_unfair_final_claim() -> None:
    protocol = _json("configs/benchmark_protocol_r8.json")
    action = protocol["frozen_action_contract"]
    assert action["cross_team_ack"] is True
    assert action["effort_levels"] == [1, 3, 6]
    assert action["total_budget"] == 18
    assert action["max_rounds"] == 6
    assert protocol["frozen_rl_reward_contract"]["cross_team_ack"] is True
    assert protocol["frozen_case_manifest"]["total_cases"] == 240
    assert protocol["frozen_case_manifest"]["formal_reporting_cases"] == 180
    fairness = protocol["baseline_fairness_gate"]
    assert fairness["status"] == "BLOCKER_BEFORE_FINAL_PLANNER_CLAIM"
    assert fairness["formal_frontier"]["effort_rule"].startswith("fixed standard-event effort 6")
    assert fairness["formal_information_gain"]["objective"].startswith("maximize absolute expected reduction")


def test_generated_r8_manifest_is_now_frozen_after_runtime_gate() -> None:
    result = build_benchmark_case_manifest(
        _json("reports/milestones/r2_real_graph_v0/incident_subgraph_audit.json"),
        _json("configs/benchmark_protocol_r5.json"),
        _json("configs/benchmark_protocol_r7.json"),
        _json("configs/q_protocol_r7.json"),
    )
    assert result["status"] == "FROZEN_PLANNER_INDEPENDENT_CASE_MANIFEST"
    assert result["case_count_status"] == "FROZEN_AFTER_RUNTIME_GATE"
    assert result["case_count"] == 240
