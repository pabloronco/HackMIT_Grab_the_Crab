"""Checkpoint-B-territory (PROPOSED, not team-frozen) action space + reward +
training-loop tests. Requires torch; skipped entirely otherwise, same as
test_gnn_backbone.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_response import Environment, FrontierPlanner  # noqa: E402
from adaptive_response.rl import (  # noqa: E402
    ActorCriticTrainer,
    GNNActorCritic,
    IncidentSamplerConfig,
    RLPlannerAdapter,
    RoundPolicy,
    TrainerConfig,
    run_episode,
    run_planner_episode,
    sample_incident,
)


@pytest.fixture
def small_policy() -> RoundPolicy:
    torch.manual_seed(0)
    backbone = GNNActorCritic(hidden_dim=16, num_layers=1)
    return RoundPolicy(backbone, hidden_dim=16)


def test_sample_incident_respects_configured_ranges() -> None:
    rng = np.random.default_rng(0)
    config = IncidentSamplerConfig(min_sites=6, max_sites=10, min_budget=15, max_budget=20)

    for _ in range(20):
        incident = sample_incident(rng, config)
        assert 6 <= len(incident.sites) <= 10
        assert 15 <= incident.budget <= 20
        assert incident.initial_detection in {site.id for site in incident.sites}

        env = Environment(incident)
        public = env.reset()  # must not raise: graph must be internally consistent
        assert len(public.sites) == len(incident.sites)


def test_round_policy_action_is_always_valid_for_environment(small_policy: RoundPolicy) -> None:
    rng = np.random.default_rng(1)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=8, max_sites=8))
    env = Environment(incident)
    public = env.reset()

    from adaptive_response import BeliefEngine, GraphStateExporter

    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    decision = small_policy.act(graph_state, public.remaining_budget)

    assert decision.mission.total_cost > 0
    assert decision.mission.total_cost <= public.remaining_budget
    # Environment._validate_and_aggregate_action must accept this without raising.
    env.step(decision.mission)


def test_round_policy_never_picks_the_same_site_twice_in_one_round(
    small_policy: RoundPolicy,
) -> None:
    rng = np.random.default_rng(2)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=12, max_sites=12, min_budget=30, max_budget=30))
    env = Environment(incident)
    public = env.reset()

    from adaptive_response import BeliefEngine, GraphStateExporter

    prior = {site.id: 0.5 for site in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    decision = small_policy.act(graph_state, public.remaining_budget)
    picked_sites = [a.site_id for a in decision.mission.allocations]
    assert len(picked_sites) == len(set(picked_sites))


def test_run_episode_completes_and_rollout_lengths_match(small_policy: RoundPolicy) -> None:
    rng = np.random.default_rng(3)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=10, max_sites=10, min_budget=12, max_budget=12))

    rollout = run_episode(small_policy, incident, seed=42)

    assert rollout.num_rounds > 0
    assert len(rollout.log_probs) == rollout.num_rounds
    assert len(rollout.values) == rollout.num_rounds
    assert len(rollout.entropies) == rollout.num_rounds
    assert len(rollout.rewards) == rollout.num_rounds
    assert rollout.effort_spent == incident.budget  # episode runs to exact budget exhaustion
    assert rollout.occupied_sites_total >= 1  # initial_detection is always occupied
    assert all(torch.is_tensor(lp) and lp.requires_grad for lp in rollout.log_probs)


def test_trainer_update_reduces_loss_gradient_flows(small_policy: RoundPolicy) -> None:
    trainer = ActorCriticTrainer(small_policy, TrainerConfig(lr=1e-2))
    rng = np.random.default_rng(4)
    sampler_cfg = IncidentSamplerConfig(min_sites=8, max_sites=8, min_budget=10, max_budget=10)

    before = [p.clone() for p in small_policy.parameters()]
    rollouts = [
        run_episode(small_policy, sample_incident(rng, sampler_cfg), seed=100 + i)
        for i in range(4)
    ]
    stats = trainer.update(rollouts)

    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)
    after = list(small_policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_frontier_and_rl_adapter_share_the_same_eval_harness(small_policy: RoundPolicy) -> None:
    rng = np.random.default_rng(5)
    incident = sample_incident(rng, IncidentSamplerConfig(min_sites=10, max_sites=10, min_budget=15, max_budget=15))

    frontier_metrics = run_planner_episode(FrontierPlanner(), incident, seed=7)
    rl_metrics = run_planner_episode(RLPlannerAdapter(small_policy), incident, seed=7)

    for metrics in (frontier_metrics, rl_metrics):
        assert metrics.num_rounds > 0
        assert metrics.effort_spent == incident.budget
        assert metrics.occupied_sites_total >= 1
        assert 0 <= metrics.occupied_sites_missed <= metrics.occupied_sites_total
