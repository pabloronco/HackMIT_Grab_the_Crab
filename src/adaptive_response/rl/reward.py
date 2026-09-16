from __future__ import annotations

from dataclasses import dataclass

from ..models import BeliefState, HiddenWorld, PublicState

# PROPOSED CURRENT DEFAULT, not a frozen weighting. Directly follows the
# reward candidate already sketched in the Technical Specification:
# "information gain / useful frontier coverage / detections - unnecessary
# effort - travel; terminal penalty per missed extent. All coefficients OPEN
# until sensitivity tests." These are exactly those coefficients, made
# concrete so training can start; they are meant to be revisited once
# Pablo/Fede's benchmark harness exists.
#
# Training-time-only privileged access: `HiddenWorld` is read here to compute
# the terminal missed-extent term. This module is evaluator/trainer-side code,
# never part of the policy's observation path (the policy only ever sees
# GraphState via adaptive_response.rl.tensor_adapter). This mirrors the
# project's own invariant: HiddenWorld may inform reward/evaluation, it must
# never reach planner/policy inputs.


@dataclass(frozen=True)
class RewardConfig:
    uncertainty_reduction_weight: float = 2.0  # alpha
    detection_weight: float = 1.0  # beta
    effort_cost_weight: float = 0.02  # lambda, per effort unit spent
    missed_extent_weight: float = 1.0  # eta, per truly-occupied site never detected


def round_reward_components(
    *,
    belief_before: BeliefState,
    belief_after: BeliefState,
    detections_this_round: int,
    effort_spent_this_round: int,
    config: RewardConfig | None = None,
) -> dict[str, float]:
    """Same computation as `round_reward`, broken out by term for logging
    (engineering block, 2026-09-12). `round_reward`'s value is unchanged and
    is defined as the sum of these components - see
    test_reward_components_sum_to_round_reward for the pinned equivalence.
    """

    cfg = config or RewardConfig()

    mean_uncertainty_before = _mean(belief_before.uncertainty_by_site)
    mean_uncertainty_after = _mean(belief_after.uncertainty_by_site)
    uncertainty_reduction = mean_uncertainty_before - mean_uncertainty_after

    return {
        "uncertainty_reduction": cfg.uncertainty_reduction_weight * uncertainty_reduction,
        "detections": cfg.detection_weight * float(detections_this_round),
        "effort_cost": -cfg.effort_cost_weight * float(effort_spent_this_round),
    }


def round_reward(
    *,
    belief_before: BeliefState,
    belief_after: BeliefState,
    detections_this_round: int,
    effort_spent_this_round: int,
    config: RewardConfig | None = None,
) -> float:
    """Shaped, per-round reward. Uses only publicly observable quantities."""

    components = round_reward_components(
        belief_before=belief_before,
        belief_after=belief_after,
        detections_this_round=detections_this_round,
        effort_spent_this_round=effort_spent_this_round,
        config=config,
    )
    return sum(components.values())


def terminal_missed_extent_penalty(
    *,
    public_state: PublicState,
    hidden_world: HiddenWorld,
    config: RewardConfig | None = None,
) -> float:
    """Episode-end-only penalty. Requires `HiddenWorld` (evaluator/trainer-only;
    see module docstring). Must never be called from planner/policy code.
    """

    cfg = config or RewardConfig()
    missed = sum(
        1
        for site in public_state.sites
        if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
    )
    return -cfg.missed_extent_weight * float(missed)


def _mean(values_by_key: dict[str, float]) -> float:
    if not values_by_key:
        return 0.0
    return sum(values_by_key.values()) / len(values_by_key)


# R7 spatial-belief reward (docs/DEMU_HANDOFF_R7.md, configs/benchmark_protocol_r7.json
# "recommended_rl_reward_for_cross_team_ack", ACKed 2026-09-12 - see Decision Log).
# Deliberately separate from RewardConfig/round_reward* above rather than reusing
# them: the R7 contract uses different normalization (mean entropy and a
# *fraction* of missed sites, not a per-round sum and a raw count) so the old
# coefficients are not comparable numbers, and the old site-local reward path
# stays untouched for the legacy toy-environment/AdaptiveMissionLoop code path.


@dataclass(frozen=True)
class SpatialRewardConfig:
    mean_uncertainty_reduction_weight: float = 2.0
    new_detection_weight: float = 0.5
    missed_occupied_fraction_weight: float = -2.0  # sign included, matches r7 config


def spatial_round_reward_components(
    *,
    belief_before: BeliefState,
    belief_after: BeliefState,
    new_detections: int,
    config: SpatialRewardConfig | None = None,
) -> dict[str, float]:
    """Per-round R7 reward terms. No effort-cost term: budget + horizon already
    constrain resource use (explicit guardrail in benchmark_protocol_r7.json)."""

    cfg = config or SpatialRewardConfig()
    mean_entropy_before = _mean(belief_before.uncertainty_by_site)
    mean_entropy_after = _mean(belief_after.uncertainty_by_site)
    reduction = mean_entropy_before - mean_entropy_after

    return {
        "mean_uncertainty_reduction": cfg.mean_uncertainty_reduction_weight * reduction,
        "new_detections": cfg.new_detection_weight * float(new_detections),
    }


def spatial_terminal_reward_components(
    *,
    missed_occupied_fraction: float,
    config: SpatialRewardConfig | None = None,
) -> dict[str, float]:
    """Episode-end-only term. `missed_occupied_fraction` must be computed by
    trainer/evaluator code from HiddenWorld (see spatial_benchmark/spatial_training_env);
    this function itself takes only the already-reduced scalar, never HiddenWorld."""

    cfg = config or SpatialRewardConfig()
    if not 0.0 <= missed_occupied_fraction <= 1.0:
        raise ValueError("missed_occupied_fraction must be in [0, 1].")
    return {
        "missed_occupied_fraction": cfg.missed_occupied_fraction_weight * missed_occupied_fraction,
    }
