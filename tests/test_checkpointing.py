"""Engineering-block hardening: checkpoint save/load + deterministic inference.

Requires torch; skipped entirely otherwise, same as the other rl/ test modules.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_response.rl import (  # noqa: E402
    GNNActorCritic,
    IncidentSamplerConfig,
    RoundPolicy,
    sample_incident,
)
from adaptive_response.rl.checkpointing import (  # noqa: E402
    PolicyArchitectureConfig,
    load_optimizer_state,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from adaptive_response.rl.tensor_adapter import graph_state_to_tensors  # noqa: E402


def _sample_graph_state():
    rng = np.random.default_rng(0)
    return sample_incident(rng, IncidentSamplerConfig(min_sites=10, max_sites=10))


def _build_graph_state_for_forward_pass():
    from adaptive_response import BeliefEngine, Environment, GraphStateExporter

    incident = _sample_graph_state()
    env = Environment(incident)
    public = env.reset(seed=1)
    prior = {s.id: 0.5 for s in public.sites}
    belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
    return GraphStateExporter().export(public, belief), public.remaining_budget


def test_save_load_roundtrip_reproduces_identical_deterministic_output(tmp_path) -> None:
    torch.manual_seed(0)
    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = arch.build()

    graph_state, budget = _build_graph_state_for_forward_pass()
    with torch.no_grad():
        original = policy.act(graph_state, budget, deterministic=True)

    ckpt_path = tmp_path / "policy.pt"
    save_policy_checkpoint(ckpt_path, policy, architecture=arch, update_idx=42)

    loaded = load_policy_checkpoint(ckpt_path)
    assert loaded.update_idx == 42
    assert loaded.architecture == arch

    with torch.no_grad():
        reloaded_decision = loaded.policy.act(graph_state, budget, deterministic=True)

    assert reloaded_decision.mission == original.mission
    assert torch.equal(reloaded_decision.value, original.value)
    assert torch.equal(reloaded_decision.log_prob, original.log_prob)


def test_fresh_instance_from_same_checkpoint_matches_across_multiple_loads(tmp_path) -> None:
    """Loading the same file twice into two independent policy objects must
    give bit-identical inference, not just 'close enough' - this is exactly
    what demo-mode determinism depends on.
    """

    torch.manual_seed(1)
    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=2, effort_per_pick=1)
    policy = arch.build()
    ckpt_path = tmp_path / "policy.pt"
    save_policy_checkpoint(ckpt_path, policy, architecture=arch, update_idx=0)

    graph_state, budget = _build_graph_state_for_forward_pass()

    loaded_a = load_policy_checkpoint(ckpt_path)
    loaded_b = load_policy_checkpoint(ckpt_path)

    with torch.no_grad():
        decision_a = loaded_a.policy.act(graph_state, budget, deterministic=True)
        decision_b = loaded_b.policy.act(graph_state, budget, deterministic=True)

    assert decision_a.mission == decision_b.mission
    assert torch.equal(decision_a.value, decision_b.value)


def test_checkpoint_without_optimizer_is_inference_only(tmp_path) -> None:
    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = arch.build()
    ckpt_path = tmp_path / "policy.pt"

    save_policy_checkpoint(ckpt_path, policy, architecture=arch, update_idx=5)
    loaded = load_policy_checkpoint(ckpt_path)
    assert loaded.policy is not None

    optimizer = torch.optim.Adam(policy.parameters())
    with pytest.raises(ValueError, match="without optimizer state"):
        load_optimizer_state(ckpt_path, optimizer)


def test_checkpoint_with_optimizer_restores_state(tmp_path) -> None:
    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = arch.build()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    # Take one step so optimizer state (Adam moment buffers) is non-trivial.
    loss = sum(p.sum() for p in policy.parameters())
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "policy.pt"
    save_policy_checkpoint(
        ckpt_path, policy, architecture=arch, update_idx=1, optimizer=optimizer
    )

    fresh_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    load_optimizer_state(ckpt_path, fresh_optimizer)
    assert len(fresh_optimizer.state) == len(optimizer.state)


def test_extra_metadata_round_trips(tmp_path) -> None:
    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = arch.build()
    ckpt_path = tmp_path / "policy.pt"

    save_policy_checkpoint(
        ckpt_path,
        policy,
        architecture=arch,
        update_idx=7,
        extra={"run_name": "engineering_smoke", "seed": 123},
    )
    loaded = load_policy_checkpoint(ckpt_path)
    assert loaded.extra == {"run_name": "engineering_smoke", "seed": 123}


def test_loading_a_non_checkpoint_file_raises_clearly(tmp_path) -> None:
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"some": "other", "unrelated": "dict"}, path)
    with pytest.raises(ValueError, match="format_version"):
        load_policy_checkpoint(path)


def test_variable_graph_size_after_load(tmp_path) -> None:
    """A loaded checkpoint must remain graph-size agnostic, not just work on
    whatever size it happened to be saved/tested with."""

    arch = PolicyArchitectureConfig(hidden_dim=16, num_layers=1, effort_per_pick=1)
    policy = arch.build()
    ckpt_path = tmp_path / "policy.pt"
    save_policy_checkpoint(ckpt_path, policy, architecture=arch, update_idx=0)
    loaded = load_policy_checkpoint(ckpt_path)

    from adaptive_response import BeliefEngine, Environment, GraphStateExporter

    for num_sites in (12, 20, 24, 30):
        rng = np.random.default_rng(num_sites)
        incident = sample_incident(
            rng, IncidentSamplerConfig(min_sites=num_sites, max_sites=num_sites)
        )
        env = Environment(incident)
        public = env.reset(seed=num_sites)
        prior = {s.id: 0.5 for s in public.sites}
        belief = BeliefEngine.initialize(prior, confirmed_sites={public.initial_detection})
        graph_state = GraphStateExporter().export(public, belief)

        with torch.no_grad():
            decision = loaded.policy.act(graph_state, public.remaining_budget, deterministic=True)
        assert decision.mission.total_cost > 0
