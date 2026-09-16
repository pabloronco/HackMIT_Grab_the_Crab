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


def round_reward(
    *,
    belief_before: BeliefState,
    belief_after: BeliefState,
    detections_this_round: int,
    effort_spent_this_round: int,
    config: RewardConfig | None = None,
) -> float:
    """Shaped, per-round reward. Uses only publicly observable quantities."""

    cfg = config or RewardConfig()

    mean_uncertainty_before = _mean(belief_before.uncertainty_by_site)
    mean_uncertainty_after = _mean(belief_after.uncertainty_by_site)
    uncertainty_reduction = mean_uncertainty_before - mean_uncertainty_after

    return (
        cfg.uncertainty_reduction_weight * uncertainty_reduction
        + cfg.detection_weight * float(detections_this_round)
        - cfg.effort_cost_weight * float(effort_spent_this_round)
    )


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
