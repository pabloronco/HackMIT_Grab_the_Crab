"""Learned decision engine (GNN + RL) for the adaptive first-response planner.

This subpackage is deliberately isolated from the core `adaptive_response`
package: importing `adaptive_response` must never require torch. Only code
that explicitly does `from adaptive_response.rl import ...` pays that cost.

Scope as of Checkpoint A (see docs/DECISION_LOG.md, "GNN/RL backbone v0"):
GraphState -> tensors -> message passing -> node embeddings -> graph/global
representation -> per-node policy logits + a critic value estimate.

Explicitly OUT of scope here: action-space semantics, effort representation,
reward, and anything that would require freezing how a MissionAction gets
built from policy output. Those are Checkpoint B decisions and must be made
with the team, not unilaterally in this module.
"""

from .backbone import ActorCriticOutput, GNNActorCritic, SiteEffortActorHead
from .benchmark import (
    BenchmarkCase,
    BenchmarkRow,
    make_benchmark_cases,
    run_benchmark_suite,
    write_benchmark_csv,
)
from .checkpointing import (
    LoadedCheckpoint,
    PolicyArchitectureConfig,
    SiteEffortPolicyArchitectureConfig,
    load_optimizer_state,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from .decision_logger import (
    JsonlDecisionLogger,
    RoundLogRecord,
    SpatialRoundLogRecord,
    read_jsonl_records,
)
from .eval_utils import EpisodeMetrics, RLPlannerAdapter, run_planner_episode
from .incident_sampler import IncidentSamplerConfig, sample_incident
from .reward import (
    RewardConfig,
    SpatialRewardConfig,
    round_reward,
    round_reward_components,
    spatial_round_reward_components,
    spatial_terminal_reward_components,
    terminal_missed_extent_penalty,
)
from .round_policy import RoundDecision, RoundPolicy
from .site_effort_policy import (
    DEFAULT_EFFORT_LEVELS,
    SiteEffortDecision,
    SiteEffortRoundPolicy,
)
from .real_graph_cases import RealGraphCase, build_real_incident_case, eligible_incident_seed_sites
from .spatial_benchmark import (
    RLSpatialPlannerAdapter,
    SpatialBenchmarkCase,
    SpatialBenchmarkRow,
    run_spatial_benchmark_suite,
    run_spatial_planner_case,
    write_spatial_benchmark_csv,
)
from .spatial_metrics import SpatialPrimaryMetrics, compute_spatial_primary_metrics
from .spatial_training_env import SpatialEpisodeRollout, run_spatial_episode
from .tensor_adapter import GraphTensors, graph_state_to_tensors
from .trainer import ActorCriticTrainer, TrainerConfig, UpdateStats
from .training_env import EpisodeRollout, run_episode

__all__ = [
    "ActorCriticOutput",
    "ActorCriticTrainer",
    "BenchmarkCase",
    "BenchmarkRow",
    "DEFAULT_EFFORT_LEVELS",
    "EpisodeMetrics",
    "EpisodeRollout",
    "GNNActorCritic",
    "GraphTensors",
    "IncidentSamplerConfig",
    "JsonlDecisionLogger",
    "LoadedCheckpoint",
    "PolicyArchitectureConfig",
    "RLPlannerAdapter",
    "RLSpatialPlannerAdapter",
    "RealGraphCase",
    "RewardConfig",
    "RoundDecision",
    "RoundLogRecord",
    "RoundPolicy",
    "SiteEffortActorHead",
    "SiteEffortDecision",
    "SiteEffortPolicyArchitectureConfig",
    "SiteEffortRoundPolicy",
    "SpatialBenchmarkCase",
    "SpatialBenchmarkRow",
    "SpatialEpisodeRollout",
    "SpatialPrimaryMetrics",
    "SpatialRewardConfig",
    "SpatialRoundLogRecord",
    "TrainerConfig",
    "UpdateStats",
    "build_real_incident_case",
    "compute_spatial_primary_metrics",
    "eligible_incident_seed_sites",
    "graph_state_to_tensors",
    "load_optimizer_state",
    "load_policy_checkpoint",
    "make_benchmark_cases",
    "read_jsonl_records",
    "round_reward",
    "round_reward_components",
    "run_benchmark_suite",
    "run_episode",
    "run_planner_episode",
    "run_spatial_benchmark_suite",
    "run_spatial_episode",
    "run_spatial_planner_case",
    "sample_incident",
    "save_policy_checkpoint",
    "spatial_round_reward_components",
    "spatial_terminal_reward_components",
    "terminal_missed_extent_penalty",
    "write_benchmark_csv",
    "write_spatial_benchmark_csv",
]
