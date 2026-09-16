"""Engineering block: reward-component breakdown + per-round JSONL logging.

Requires torch (round_policy/training_env do); skipped otherwise.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_response import BeliefEngine  # noqa: E402
from adaptive_response.rl import (  # noqa: E402
    GNNActorCritic,
    IncidentSamplerConfig,
    JsonlDecisionLogger,
    RewardConfig,
    RoundPolicy,
    read_jsonl_records,
    round_reward,
    round_reward_components,
    run_episode,
    sample_incident,
)


def test_reward_components_sum_to_round_reward() -> None:
    """Pins the refactor: round_reward's value must not change when it is
    expressed as the sum of round_reward_components()."""

    prior = {"a": 0.6, "b": 0.3}
    belief_before = BeliefEngine.initialize(prior)
    from adaptive_response import Observation, ObservationBatch

    belief_after = BeliefEngine.update(
        belief_before,
        ObservationBatch(observations=(Observation(site_id="a", effort=5, detection=False, round=1),), round=1, total_effort=5),
        {"a": 0.3, "b": 0.3},
    )

    cfg = RewardConfig(uncertainty_reduction_weight=3.0, detection_weight=2.0, effort_cost_weight=0.05)
    kwargs = dict(
        belief_before=belief_before,
        belief_after=belief_after,
        detections_this_round=2,
        effort_spent_this_round=5,
        config=cfg,
    )

    total = round_reward(**kwargs)
    components = round_reward_components(**kwargs)
    assert total == pytest.approx(sum(components.values()))
    assert set(components) == {"uncertainty_reduction", "detections", "effort_cost"}


def test_jsonl_decision_logger_round_trips(tmp_path) -> None:
    from adaptive_response.rl.decision_logger import RoundLogRecord

    path = tmp_path / "decisions.jsonl"
    with JsonlDecisionLogger(path) as logger:
        logger.log_round(
            RoundLogRecord(
                episode_index=0,
                round_index=0,
                budget_before=30,
                budget_after=29,
                site_ids_picked=("site_00",),
                num_picks=1,
                node_ids=("site_00", "site_01"),
                node_logits=(0.5, -0.3),
                eligible_mask=(True, True),
                value_estimate=0.1,
                entropy=1.2,
                log_prob=-0.7,
                reward_components={"detections": 1.0, "effort_cost": -0.02},
                reward_total=0.98,
                detections_this_round=1,
                done=False,
            )
        )
        logger.log_round(
            RoundLogRecord(
                episode_index=0,
                round_index=1,
                budget_before=29,
                budget_after=0,
                site_ids_picked=("site_01",),
                num_picks=1,
                node_ids=("site_00", "site_01"),
                node_logits=(0.4, -0.1),
                eligible_mask=(True, False),
                value_estimate=0.05,
                entropy=0.9,
                log_prob=-0.5,
                reward_components={"detections": 0.0, "effort_cost": -0.02, "terminal_missed_extent": -1.0},
                reward_total=-1.02,
                detections_this_round=0,
                done=True,
            )
        )

    records = read_jsonl_records(path)
    assert len(records) == 2
    assert records[0]["round_index"] == 0
    assert records[0]["site_ids_picked"] == ["site_00"]
    assert records[0]["eligible_mask"] == [True, True]
    assert records[1]["done"] is True
    assert records[1]["eligible_mask"] == [True, False]
    assert records[1]["reward_components"]["terminal_missed_extent"] == -1.0


def test_run_episode_with_decision_logger_produces_one_record_per_round(tmp_path) -> None:
    torch.manual_seed(0)
    backbone = GNNActorCritic(hidden_dim=16, num_layers=1)
    policy = RoundPolicy(backbone, hidden_dim=16)

    rng = np.random.default_rng(9)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=10, max_sites=10, min_budget=10, max_budget=10))

    log_path = tmp_path / "episode.jsonl"
    with JsonlDecisionLogger(log_path) as logger:
        rollout = run_episode(policy, incident, seed=1, decision_logger=logger, episode_index=3)

    records = read_jsonl_records(log_path)
    assert len(records) == rollout.num_rounds
    assert all(r["episode_index"] == 3 for r in records)
    assert [r["round_index"] for r in records] == list(range(rollout.num_rounds))
    assert records[-1]["done"] is True
    assert all(not r["done"] for r in records[:-1])
    # Node logits/ids/eligibility are present and match the graph size for every round.
    for r in records:
        assert len(r["node_logits"]) == len(r["node_ids"]) == len(r["eligible_mask"]) == 10
        assert any(r["eligible_mask"])  # at least one legal action every round
    # Terminal reward component only appears on the last, done=True record.
    assert "terminal_missed_extent" in records[-1]["reward_components"]
    assert all("terminal_missed_extent" not in r["reward_components"] for r in records[:-1])


def test_run_episode_without_logger_is_unaffected() -> None:
    """The opt-in parameter must not change default behavior at all."""

    torch.manual_seed(0)
    backbone = GNNActorCritic(hidden_dim=16, num_layers=1)
    policy = RoundPolicy(backbone, hidden_dim=16)
    rng = np.random.default_rng(9)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=10, max_sites=10, min_budget=10, max_budget=10))

    rollout = run_episode(policy, incident, seed=1)  # no decision_logger passed
    assert rollout.num_rounds > 0
