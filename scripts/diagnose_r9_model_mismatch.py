from __future__ import annotations

from math import isinf

from adaptive_response.model_mismatch import posterior_predictive_surprise
from adaptive_response.models import Observation, ObservationBatch
from adaptive_response.spatial_belief import (
    EcologicalHypothesis,
    QHypothesis,
    SpatialBeliefEngine,
)


def fmt_bits(value: float) -> str:
    return "inf" if isinf(value) else f"{value:.3f}"


def main() -> None:
    print("=== R9 POSTERIOR-PREDICTIVE MODEL-MISMATCH DIAGNOSTIC ===")
    print("STATUS: model criticism / ensemble stress signal, NOT a new ecological state variable.")
    print()

    belief = SpatialBeliefEngine.initialize(
        (
            EcologicalHypothesis(
                presence_by_site={"A": True, "B": False},
                prior_weight=1.0,
                label="A_occupied",
            ),
            EcologicalHypothesis(
                presence_by_site={"A": False, "B": False},
                prior_weight=1.0,
                label="A_absent",
            ),
        ),
        (
            QHypothesis(q=0.10, prior_weight=1.0, label="q_low"),
            QHypothesis(q=0.30, prior_weight=1.0, label="q_high"),
        ),
    )

    print("CASE 1 — effort changes how surprising a non-detection is")
    for effort in (1, 6):
        result = posterior_predictive_surprise(
            belief,
            Observation(site_id="A", effort=effort, detection=False, round=1),
        )
        print(
            f"  effort={effort}: P(obs)={result.observation_probability:.4f} "
            f"surprise={fmt_bits(result.surprise_bits)} bits"
        )
    print()

    print("CASE 2 — an observation outside the current world support is explicit")
    impossible = posterior_predictive_surprise(
        belief,
        Observation(site_id="B", effort=6, detection=True, round=1),
    )
    print(
        f"  site=B detection=True: P(obs)={impossible.observation_probability:.4f} "
        f"surprise={fmt_bits(impossible.surprise_bits)} bits "
        f"impossible_under_current_ensemble={impossible.impossible_under_current_ensemble}"
    )
    print("  Interpretation: this can indicate world-ensemble inadequacy and/or a violated observation-model assumption.")
    print()

    print("CASE 3 — q can absorb surprising non-detections, so inspect both surprise and q posterior")
    confounded = SpatialBeliefEngine.initialize(
        (
            EcologicalHypothesis(
                presence_by_site={"A": True},
                prior_weight=1.0,
                label="A_always_occupied",
            ),
        ),
        (
            QHypothesis(q=0.05, prior_weight=1.0, label="q_low"),
            QHypothesis(q=0.30, prior_weight=1.0, label="q_high"),
        ),
    )
    for round_index in (1, 2):
        observation = Observation(site_id="A", effort=6, detection=False, round=round_index)
        diagnostic = posterior_predictive_surprise(confounded, observation)
        before_q = confounded.q_posterior()
        confounded = SpatialBeliefEngine.update(
            confounded,
            ObservationBatch(
                observations=(observation,),
                round=round_index,
                total_effort=6,
            ),
        )
        print(
            f"  round={round_index}: P(obs)={diagnostic.observation_probability:.4f} "
            f"surprise={fmt_bits(diagnostic.surprise_bits)} bits "
            f"q_before={before_q} q_after={confounded.q_posterior()}"
        )
    print()

    print("INTERPRETATION GATE")
    print("  - surprise is computed BEFORE conditioning on the observation;")
    print("  - no universal threshold is frozen; the value is a continuous diagnostic;")
    print("  - high/repeated surprise is evidence that the current ensemble predicts the field result poorly;")
    print("  - low surprise does NOT prove the ecological model is correct;")
    print("  - q shifts must not be interpreted as sufficient explanation when model misspecification remains plausible;")
    print("  - HiddenWorld is never supplied to this diagnostic.")


if __name__ == "__main__":
    main()
