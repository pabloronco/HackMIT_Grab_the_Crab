"""Model-mismatch alerts: the system flagging when its own ecological ensemble
predicted the field evidence poorly.

Built on `model_mismatch.posterior_predictive_surprise` (R9 diagnostic):
`surprise_bits = -log2 P(observation | current posterior)`, scored BEFORE the
observation is used to update the belief. This module turns that continuous
per-observation diagnostic into a per-round alert object a UI or briefing can
show, plus a deterministic plain-language explanation.

Threshold discipline: `docs/MODEL_MISMATCH_DIAGNOSTIC_R9.md` states there is
no source-grounded calibration for a "surprise > X bits => model failure"
rule, and this module does not pretend otherwise. Two alert levels:

  - IMPOSSIBLE: the observation had zero probability under every hypothesis
    in the ensemble. Unambiguous by construction (the doc names it as the
    strongest signal), no threshold involved.
  - HIGH: `surprise_bits >= alert_bits`, where `alert_bits` is an explicit,
    caller-visible demo setting. The default (3.0) has a plain reading -
    "the realized outcome had probability <= 1/8 under the ensemble" - and
    an empirical basis, not a calibration: on the frozen R8 manifest (180
    real-graph cases, 540 Frontier rounds; `scripts/run_real_graph_demo.py
    --find-alerts`) it fires exactly once, on an out-of-distribution
    world-model case (ood_model_test E_fragmented_patchy #028, round 3,
    3.68 bits). For reference, firing counts out of 540 rounds: 1.5 bits ->
    114, 2.0 -> 40, 2.5 -> 4, 3.0 -> 1, 4.0 -> 0. This is still not a
    calibrated failure criterion; callers should say so when they surface it.

Nothing here reads HiddenWorld, alters the posterior, or feeds a planner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model_mismatch import PredictiveSurprise, posterior_predictive_surprise
from .models import ObservationBatch
from .spatial_belief import SpatialBeliefState

DEFAULT_ALERT_BITS = 3.0


@dataclass(frozen=True)
class SurpriseAlert:
    """One round's mismatch assessment. `level` is 'none', 'high' or 'impossible'."""

    round: int
    level: str
    alert_bits: float
    per_observation: tuple[PredictiveSurprise, ...]
    max_surprise_bits: float  # inf if any observation was impossible
    total_surprise_bits: float  # inf if any observation was impossible
    joint_entropy_before_bits: float
    effective_sample_size_before: float

    @property
    def triggered(self) -> bool:
        return self.level != "none"

    @property
    def most_surprising(self) -> PredictiveSurprise | None:
        if not self.per_observation:
            return None
        return max(self.per_observation, key=lambda s: s.surprise_bits)


def evaluate_round_surprise(
    spatial_belief_before: SpatialBeliefState,
    observations: ObservationBatch,
    *,
    alert_bits: float = DEFAULT_ALERT_BITS,
) -> SurpriseAlert:
    """Score every observation in the round against the pre-update posterior.

    `spatial_belief_before` must be the belief as it was before these
    observations were applied - capture `loop.current_spatial_belief` before
    calling `run_round()`. Scoring against the post-update belief would be
    circular (the belief has already absorbed the evidence).
    """
    if alert_bits <= 0:
        raise ValueError("alert_bits must be positive.")

    scored = tuple(
        posterior_predictive_surprise(spatial_belief_before, obs)
        for obs in observations.observations
    )
    if any(s.impossible_under_current_ensemble for s in scored):
        level = "impossible"
        max_bits = float("inf")
        total_bits = float("inf")
    else:
        max_bits = max((s.surprise_bits for s in scored), default=0.0)
        total_bits = sum(s.surprise_bits for s in scored)
        level = "high" if max_bits >= alert_bits else "none"

    return SurpriseAlert(
        round=observations.round,
        level=level,
        alert_bits=alert_bits,
        per_observation=scored,
        max_surprise_bits=max_bits,
        total_surprise_bits=total_bits,
        joint_entropy_before_bits=spatial_belief_before.joint_entropy_bits(),
        effective_sample_size_before=spatial_belief_before.effective_sample_size(),
    )


def format_alert_facts(alert: SurpriseAlert) -> str:
    """Structured fact block for an alert - the only thing an LLM explainer is
    ever given, mirroring `narrate._format_transition` (numbers are computed
    here, never by the model)."""
    lines = [f"MODEL-MISMATCH CHECK, round {alert.round}: level={alert.level.upper()}"]
    lines.append(
        f"  alert threshold: {alert.alert_bits:.1f} bits (demo setting: fires when the observed "
        f"outcome had probability <= {100 * 2 ** -alert.alert_bits:.1f}% under the models; "
        f"not a calibrated failure criterion)"
    )
    for s in alert.per_observation:
        outcome = "DETECTION" if s.detection else "no detection"
        if s.impossible_under_current_ensemble:
            lines.append(
                f"  {s.site_id}: {outcome} after {s.effort} checks -> IMPOSSIBLE under every hypothesis "
                f"(the models predicted a detection with probability {s.predictive_detection_probability:.4f})"
            )
        else:
            lines.append(
                f"  {s.site_id}: {outcome} after {s.effort} checks -> the models gave this outcome "
                f"probability {s.observation_probability:.4f} (they predicted a detection with "
                f"probability {s.predictive_detection_probability:.4f}) -> surprise {s.surprise_bits:.2f} bits"
            )
    lines.append(f"  ensemble joint entropy before this round: {alert.joint_entropy_before_bits:.2f} bits")
    lines.append(f"  effective number of hypotheses still plausible: {alert.effective_sample_size_before:.1f}")
    return "\n".join(lines)


def template_alert_explanation(alert: SurpriseAlert) -> str:
    """Deterministic, zero-cost plain-language explanation of an alert."""
    if not alert.triggered:
        return (
            f"Round {alert.round}: field results were consistent with the current "
            f"ecological models (max surprise {alert.max_surprise_bits:.2f} bits, "
            f"below the {alert.alert_bits:.1f}-bit alert setting)."
        )

    worst = alert.most_surprising
    assert worst is not None
    outcome = "a detection" if worst.detection else "no detection"
    p_pct = worst.predictive_detection_probability * 100

    if alert.level == "impossible":
        head = (
            f"Round {alert.round}: MODEL MISMATCH. At {worst.site_id}, the field team reported "
            f"{outcome} after {worst.effort} checks, but every ecological hypothesis the system "
            f"is currently considering gave that outcome zero probability "
            f"(predicted detection probability {p_pct:.1f}%)."
        )
    else:
        head = (
            f"Round {alert.round}: model-mismatch warning. At {worst.site_id}, the field team "
            f"reported {outcome} after {worst.effort} checks; the system's ecological models "
            f"had predicted a {p_pct:.1f}% detection probability, making this outcome "
            f"{worst.surprise_bits:.1f} bits more surprising than expected "
            f"(alert setting: {alert.alert_bits:.1f} bits)."
        )

    tail = (
        f" Before this round, {alert.effective_sample_size_before:.1f} hypotheses were still "
        f"effectively plausible. This means the models the plan is based on may not describe "
        f"this site well; treat the next recommendation with extra caution and consider "
        f"re-checking. This is a diagnostic, not proof the models are wrong."
    )
    return head + tail
