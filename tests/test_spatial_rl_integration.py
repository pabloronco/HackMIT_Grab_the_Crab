"""R7 handoff: GNN/RL integration with SpatialAdaptiveMissionLoop under the
ACKed (site, effort) action contract (docs/DEMU_HANDOFF_R7.md).

Covers: deterministic inference, effort-level budget masking, the max_rounds
(horizon) vs. budget-exhaustion termination paths, decision logging,
checkpoint round-trip for the new architecture kind, and structural
hidden-truth-firewall checks extended to the new policy-facing modules.
Requires torch; skipped otherwise.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from adaptive_response import Edge, IncidentConfig, Site  # noqa: E402
from adaptive_response.models import HiddenWorld  # noqa: E402
from adaptive_response.spatial_belief import QHypothesis  # noqa: E402
from adaptive_response.world_models import (  # noqa: E402
    GraphDiffusionWorldModel,
    WorldModelContext,
    sample_ecological_hypotheses,
)
from adaptive_response.rl import (  # noqa: E402
    GNNActorCritic,
    JsonlDecisionLogger,
    SiteEffortPolicyArchitectureConfig,
    SiteEffortRoundPolicy,
    load_policy_checkpoint,
    read_jsonl_records,
    run_spatial_episode,
    save_policy_checkpoint,
)


def _small_incident(*, budget: int = 18, num_sites: int = 8) -> tuple[IncidentConfig, list[Site], list[Edge]]:
    sites = [
        Site(id=f"site_{i:02d}", x=float(i), y=0.0, habitat_score=0.5, q_model=0.1)
        for i in range(num_sites)
    ]
    edges = [
        Edge(src=f"site_{i:02d}", dst=f"site_{i + 1:02d}", distance=1.0)
        for i in range(num_sites - 1)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_00",
        budget=budget,
        teams=1,
        protocol="r7_test",
        seed=1,
        world_model_id="A_graph_diffusion",
    ), sites, edges


def _hypotheses(sites, edges, *, seed: int = 1):
    world_model = GraphDiffusionWorldModel()
    context = WorldModelContext(tuple(sites), tuple(edges), sites[0].id)
    hypotheses = sample_ecological_hypotheses([world_model], context, draws_per_model=50, seed=seed)
    q_support = [QHypothesis(0.05), QHypothesis(0.10), QHypothesis(0.20)]
    return world_model, hypotheses, q_support


def _policy(hidden_dim: int = 16, num_layers: int = 1, effort_levels=(1, 3, 6)) -> SiteEffortRoundPolicy:
    torch.manual_seed(0)
    return SiteEffortRoundPolicy(
        GNNActorCritic(hidden_dim=hidden_dim, num_layers=num_layers),
        hidden_dim=hidden_dim,
        effort_levels=effort_levels,
    )


# --- Action-contract mechanics ------------------------------------------------


def test_deterministic_act_is_reproducible():
    from adaptive_response import BeliefEngine, GraphStateExporter

    config, sites, edges = _small_incident()
    from adaptive_response.environment import Environment

    env = Environment(config)
    public = env.reset(seed=1)
    belief = BeliefEngine.initialize({s.id: 0.35 for s in public.sites}, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    policy = _policy()
    d1 = policy.act(graph_state, public.remaining_budget, deterministic=True)
    d2 = policy.act(graph_state, public.remaining_budget, deterministic=True)
    assert d1.site_id == d2.site_id
    assert d1.effort_units == d2.effort_units
    assert d1.node_logits == d2.node_logits


def test_effort_levels_above_remaining_budget_are_never_chosen():
    from adaptive_response import BeliefEngine, GraphStateExporter
    from adaptive_response.environment import Environment

    config, sites, edges = _small_incident(budget=2)
    env = Environment(config)
    public = env.reset(seed=1)
    belief = BeliefEngine.initialize({s.id: 0.35 for s in public.sites}, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    policy = _policy()
    for _ in range(20):
        decision = policy.act(graph_state, remaining_budget=2, deterministic=False)
        assert decision.effort_units <= 2
    # Effort levels 3 and 6 must be masked ineligible at every node.
    for row in decision.eligible_mask:
        assert row[1] is False and row[2] is False


def test_act_rejects_budget_below_smallest_effort_level():
    from adaptive_response import BeliefEngine, GraphStateExporter
    from adaptive_response.environment import Environment

    config, sites, edges = _small_incident()
    env = Environment(config)
    public = env.reset(seed=1)
    belief = BeliefEngine.initialize({s.id: 0.35 for s in public.sites}, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    policy = _policy()
    with pytest.raises(ValueError, match="insufficient budget"):
        policy.act(graph_state, remaining_budget=0, deterministic=True)


# --- Episode-level: horizon vs. budget-exhaustion termination -----------------


def test_episode_stops_at_horizon_when_budget_would_allow_more_rounds():
    config, sites, edges = _small_incident(budget=18)
    world_model, hypotheses, q_support = _hypotheses(sites, edges)
    policy = _policy(effort_levels=(1,))  # forces effort=1 every round -> 18 possible rounds

    rollout = run_spatial_episode(
        policy, config, world_model=world_model, q_true=0.10,
        ecological_hypotheses=hypotheses, q_hypotheses=q_support,
        max_rounds=6, seed=3,
    )
    assert rollout.num_rounds == 6
    assert rollout.effort_spent == 6  # horizon hit long before budget exhausted
    assert rollout.metrics is not None


def test_episode_stops_at_budget_exhaustion_before_horizon():
    config, sites, edges = _small_incident(budget=6)
    world_model, hypotheses, q_support = _hypotheses(sites, edges)
    policy = _policy(effort_levels=(6,))  # every pick spends the whole budget at once

    rollout = run_spatial_episode(
        policy, config, world_model=world_model, q_true=0.10,
        ecological_hypotheses=hypotheses, q_hypotheses=q_support,
        max_rounds=6, seed=4,
    )
    assert rollout.num_rounds == 1
    assert rollout.effort_spent == 6
    assert rollout.metrics is not None


def test_decision_logger_produces_one_record_per_round_with_effort_and_eligibility(tmp_path):
    config, sites, edges = _small_incident(budget=18)
    world_model, hypotheses, q_support = _hypotheses(sites, edges)
    policy = _policy()

    log_path = tmp_path / "spatial_decisions.jsonl"
    with JsonlDecisionLogger(log_path) as logger:
        rollout = run_spatial_episode(
            policy, config, world_model=world_model, q_true=0.10,
            ecological_hypotheses=hypotheses, q_hypotheses=q_support,
            max_rounds=6, seed=5, decision_logger=logger, episode_index=2,
        )

    records = read_jsonl_records(log_path)
    assert len(records) == rollout.num_rounds
    assert all(r["episode_index"] == 2 for r in records)
    assert records[-1]["done"] is True
    assert all(not r["done"] for r in records[:-1])
    for r in records:
        assert r["effort_units"] in (1, 3, 6)
        assert len(r["node_logits"]) == len(r["eligible_mask"]) == len(r["node_ids"])
        assert all(len(row) == 3 for row in r["eligible_mask"])
    assert "missed_occupied_fraction" in records[-1]["reward_components"]
    assert all("missed_occupied_fraction" not in r["reward_components"] for r in records[:-1])


# --- Checkpointing -------------------------------------------------------------


def test_site_effort_checkpoint_round_trip_is_deterministic(tmp_path):
    from adaptive_response import BeliefEngine, GraphStateExporter
    from adaptive_response.environment import Environment

    config, sites, edges = _small_incident()
    env = Environment(config)
    public = env.reset(seed=1)
    belief = BeliefEngine.initialize({s.id: 0.35 for s in public.sites}, confirmed_sites={public.initial_detection})
    graph_state = GraphStateExporter().export(public, belief)

    architecture = SiteEffortPolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_levels=(1, 3, 6))
    policy = architecture.build()
    before = policy.act(graph_state, public.remaining_budget, deterministic=True)

    path = tmp_path / "site_effort.pt"
    save_policy_checkpoint(path, policy, architecture=architecture, update_idx=10)
    loaded = load_policy_checkpoint(path)

    assert isinstance(loaded.architecture, SiteEffortPolicyArchitectureConfig)
    after = loaded.policy.act(graph_state, public.remaining_budget, deterministic=True)
    assert before.site_id == after.site_id
    assert before.effort_units == after.effort_units
    assert before.node_logits == after.node_logits


def test_old_round_policy_checkpoints_still_default_to_round_policy_kind(tmp_path):
    """A checkpoint saved before architecture_kind existed has no such key and
    must still load as the older RoundPolicy - this is what the whole
    save/load path already produced before this handoff's changes."""

    from adaptive_response.rl import PolicyArchitectureConfig, RoundPolicy

    architecture = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = architecture.build()
    path = tmp_path / "legacy.pt"
    save_policy_checkpoint(path, policy, architecture=architecture, update_idx=1)

    payload = torch.load(path, weights_only=False)
    del payload["architecture_kind"]  # simulate a pre-R7 checkpoint on disk
    torch.save(payload, path)

    loaded = load_policy_checkpoint(path)
    assert isinstance(loaded.policy, RoundPolicy)
    assert isinstance(loaded.architecture, PolicyArchitectureConfig)


# --- Structural hidden-truth / q_true firewall, extended to the new modules --


def test_site_effort_policy_module_does_not_import_hidden_world() -> None:
    """spatial_metrics/spatial_training_env are trainer/evaluator-side (like the
    older reward.py/training_env.py) and legitimately touch HiddenWorld only
    after loop.reveal()/force_complete() - this check is scoped to the actual
    policy-facing module, same as the existing round_policy/backbone checks."""

    import importlib

    module = importlib.import_module("adaptive_response.rl.site_effort_policy")
    assert not hasattr(module, "HiddenWorld")
    assert "HiddenWorld" not in dir(module)


def test_site_effort_act_signature_has_no_hidden_or_q_true_parameter() -> None:
    sig = inspect.signature(SiteEffortRoundPolicy.act)
    param_names = set(sig.parameters) - {"self"}
    assert param_names == {"graph_state", "remaining_budget", "deterministic"}
    assert HiddenWorld.__name__ not in [
        getattr(p.annotation, "__name__", str(p.annotation)) for p in sig.parameters.values()
    ]
