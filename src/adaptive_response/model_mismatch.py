from __future__ import annotations

from dataclasses import dataclass
from math import inf, log2

from .models import Observation
from .spatial_belief import SpatialBeliefEngine, SpatialBeliefState


@dataclass(frozen=True, slots=True)
class PredictiveSurprise:
    """Pre-update posterior-predictive diagnostic for one field observation.

    This is a model-criticism signal, not a new ecological state variable and not a
    planner feature. It quantifies how probable the realized field result was under
    the *current* joint ecological-world/q posterior before that result is used to
    update the belief.
    """

    site_id: str
    effort: int
    detection: bool
    predictive_detection_probability: float
    observation_probability: float
    surprise_bits: float
    impossible_under_current_ensemble: bool
    joint_entropy_before_bits: float
    q_mean_before: float
    effective_sample_size_before: float


def posterior_predictive_surprise(
    belief: SpatialBeliefState,
    observation: Observation,
) -> PredictiveSurprise:
    """Score one observation against the current joint posterior before updating.

    ``surprise_bits = -log2 P(observation | current posterior)``.

    A large value means the observation was poorly predicted by the current
    ecological-world/q ensemble. No universal threshold is imposed here: selecting a
    threshold would be an additional modeling decision requiring calibration. A zero-
    probability observation is reported as infinite surprise and explicitly marked
    impossible under the current ensemble.

    The function receives no HiddenWorld and does not alter the posterior.
    """

    if observation.site_id not in belief.site_ids:
        raise ValueError(f"Unknown site_id {observation.site_id!r}.")
    if isinstance(observation.effort, bool) or not isinstance(observation.effort, int) or observation.effort <= 0:
        raise ValueError("observation effort must be a positive integer.")

    p_detection = SpatialBeliefEngine.predictive_detection_probability(
        belief,
        site_id=observation.site_id,
        effort=observation.effort,
    )
    p_observation = p_detection if observation.detection else 1.0 - p_detection

    # Numerical guard mirrors the probability semantics of SpatialBeliefEngine.
    if p_observation < -1e-12 or p_observation > 1.0 + 1e-12:
        raise RuntimeError("posterior-predictive observation probability left [0, 1].")
    p_observation = min(1.0, max(0.0, p_observation))
    impossible = p_observation == 0.0
    surprise = inf if impossible else -log2(p_observation)

    return PredictiveSurprise(
        site_id=observation.site_id,
        effort=observation.effort,
        detection=observation.detection,
        predictive_detection_probability=p_detection,
        observation_probability=p_observation,
        surprise_bits=surprise,
        impossible_under_current_ensemble=impossible,
        joint_entropy_before_bits=belief.joint_entropy_bits(),
        q_mean_before=belief.q_mean(),
        effective_sample_size_before=belief.effective_sample_size(),
    )
