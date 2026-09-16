"""Engineering block (2026-09-12): hardening tests for action masking, budget
conservation, STOP behavior, duplicate-action prevention, variable graph size
(incl. the ~30-node stress size), and the hidden-truth boundary.

Deliberately does not touch GraphState, MissionAction, node/edge features,
action semantics, or reward semantics - it only adds coverage around the
existing, unchanged contracts. Requires torch; skipped otherwise.
"""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_response import (  # noqa: E402
    BeliefEngine,
    Environment,
    GraphStateExporter,
)
from adaptive_response.models import HiddenWorld  # noqa: E402
from adaptive_response.rl import (  # noqa: E402
    GNNActorCritic,
    IncidentSamplerConfig,
    RoundPolicy,
    sample_incident,
)


def _graph_state_and_budget(num_sites: int, *, seed: int):
    incident = sample_incident(
        np.random.default_rng(seed), IncidentSamplerConfig(min_sites=num_sites, max_sites=num_sites)
    )
    env = Environment(incident)
    public = env.reset(seed=seed)
    prior = {s.id: 0.5 for s in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    return GraphStateExporter().export(public, belief), public.remaining_budget


def _policy(hidden_dim: int = 16, num_layers: int = 1) -> RoundPolicy:
    torch.manual_seed(0)
    return RoundPolicy(GNNActorCritic(hidden_dim=hidden_dim, num_layers=num_layers), hidden_dim=hidden_dim)


# --- Variable graph size, including the ~30-node stress size -----------------


@pytest.mark.parametrize("num_sites", [1, 2, 12, 20, 24, 30, 40])
def test_policy_handles_graph_sizes_from_singleton_to_stress_size(num_sites: int) -> None:
    policy = _policy()
    graph_state, budget = _graph_state_and_budget(num_sites, seed=num_sites)

    decision = policy.act(graph_state, budget, deterministic=True)

    assert decision.mission.total_cost > 0
    assert decision.mission.total_cost <= budget
    picked = [a.site_id for a in decision.mission.allocations]
    assert len(picked) == len(set(picked)), "duplicate site picked within one round"
    assert set(picked).issubset(set(graph_state.node_ids))


def test_same_instance_across_wildly_different_sizes_back_to_back() -> None:
    """40 down to 1: no fixed-N parameter anywhere should make this fail."""

    policy = _policy()
    for num_sites in (40, 24, 12, 2, 1, 30, 20):
        graph_state, budget = _graph_state_and_budget(num_sites, seed=num_sites * 7)
        decision = policy.act(graph_state, budget, deterministic=True)
        assert decision.mission.total_cost > 0


# --- Budget edge cases --------------------------------------------------------


def test_minimal_budget_of_exactly_one_effort_unit() -> None:
    policy = _policy()
    graph_state, _ = _graph_state_and_budget(12, seed=1)

    decision = policy.act(graph_state, remaining_budget=1, deterministic=True)

    assert decision.mission.total_cost == 1
    assert len(decision.mission.allocations) == 1


def test_act_rejects_budget_smaller_than_effort_per_pick() -> None:
    policy = RoundPolicy(GNNActorCritic(hidden_dim=16, num_layers=1), hidden_dim=16, effort_per_pick=5)
    graph_state, _ = _graph_state_and_budget(12, seed=2)

    with pytest.raises(ValueError, match="insufficient budget"):
        policy.act(graph_state, remaining_budget=4, deterministic=True)


# --- Stress: many episodes, invariants must never break ----------------------


def test_duplicate_and_budget_invariants_hold_over_many_stochastic_episodes() -> None:
    """Not a single example: run many episodes end-to-end (via Environment, so
    the mission actually gets executed and validated by Environment's own
    checks too) with stochastic (non-deterministic) action sampling, which
    exercises far more of the action distribution than the deterministic
    argmax path alone.
    """

    policy = _policy()
    rng = np.random.default_rng(123)

    for i in range(40):
        num_sites = int(rng.integers(12, 31))
        incident = sample_incident(rng, IncidentSamplerConfig(min_sites=num_sites, max_sites=num_sites))
        env = Environment(incident)
        public = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        prior = {s.id: 0.5 for s in public.sites}
        belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
        exporter = GraphStateExporter()

        rounds = 0
        while public.remaining_budget > 0 and rounds < 200:  # safety cap, not a real limit
            graph_state = exporter.export(public, belief)
            decision = policy.act(graph_state, public.remaining_budget, deterministic=False)

            picked = [a.site_id for a in decision.mission.allocations]
            assert len(picked) == len(set(picked)), f"episode {i}: duplicate pick"
            assert decision.mission.total_cost <= public.remaining_budget, (
                f"episode {i}: mission exceeds remaining budget"
            )

            observations, done, metrics = env.step(decision.mission)
            public = env.current_public_state
            q_by_site = exporter.planner_constraints(public)["q_by_site"]
            belief = BeliefEngine.update(belief, observations, q_by_site)
            rounds += 1
            if done:
                break
        assert rounds < 200, f"episode {i}: did not terminate within safety cap"


# --- Structural hidden-truth boundary: broader than the Checkpoint A tests ---


@pytest.mark.parametrize(
    "module_name",
    ["round_policy", "backbone", "tensor_adapter", "layers", "checkpointing"],
)
def test_no_module_in_rl_package_imports_hidden_world(module_name: str) -> None:
    """Extends the Checkpoint A tensor_adapter-only check to every module in
    the policy's forward/inference path. A future edit that starts importing
    HiddenWorld anywhere here should fail this test immediately.
    """

    import importlib

    module = importlib.import_module(f"adaptive_response.rl.{module_name}")
    assert not hasattr(module, "HiddenWorld")
    assert "HiddenWorld" not in dir(module)


def test_round_policy_act_signature_cannot_accept_hidden_world() -> None:
    sig = inspect.signature(RoundPolicy.act)
    param_names = set(sig.parameters) - {"self"}
    assert param_names == {"graph_state", "remaining_budget", "deterministic"}
    # Structural guarantee: HiddenWorld is a real, importable type here only to
    # prove it is never one of the accepted parameter types above - this
    # module's own use of it does not create a path into the policy.
    assert HiddenWorld.__name__ not in [
        getattr(p.annotation, "__name__", str(p.annotation)) for p in sig.parameters.values()
    ]


def test_gnn_actor_critic_forward_only_accepts_graph_tensors() -> None:
    sig = inspect.signature(GNNActorCritic.forward)
    param_names = list(sig.parameters)
    assert param_names == ["self", "tensors"]
