from __future__ import annotations

from adaptive_response import (
    EcologicalHypothesis,
    Observation,
    ObservationBatch,
    QHypothesis,
    SpatialBeliefEngine,
)


def one_observation(*, effort: int, detection: bool) -> ObservationBatch:
    observation = Observation(
        site_id="A",
        effort=effort,
        detection=detection,
        round=1,
    )
    return ObservationBatch(
        observations=(observation,),
        round=1,
        total_effort=effort,
    )


def main() -> None:
    worlds = [
        EcologicalHypothesis(
            {"A": True, "B": True, "C": False},
            label="connected_extent",
        ),
        EcologicalHypothesis(
            {"A": False, "B": False, "C": False},
            label="no_local_extent",
        ),
    ]
    q_support = [QHypothesis(0.05, label="low_q"), QHypothesis(0.20, label="high_q")]
    prior = SpatialBeliefEngine.initialize(worlds, q_support)

    after_one = SpatialBeliefEngine.update(
        prior,
        one_observation(effort=1, detection=False),
    )
    after_ten = SpatialBeliefEngine.update(
        prior,
        one_observation(effort=10, detection=False),
    )

    print("=== R4 SPATIAL BELIEF DIAGNOSTIC ===")
    print("STATUS: controlled inference sanity check, not ecological calibration.")
    print("A and B are correlated by the declared hypothesis ensemble; C is absent in both worlds.")
    print("q support {0.05, 0.20} is illustrative only; the numeric q range remains OPEN.")
    print()
    print("Prior occupancy marginals:", prior.p_by_site())
    print("Prior q posterior:", prior.q_posterior())
    print()
    print("After 0 detections / 1 effort at A:")
    print("  occupancy:", {k: round(v, 4) for k, v in after_one.p_by_site().items()})
    print("  q posterior:", {k: round(v, 4) for k, v in after_one.q_posterior().items()})
    print()
    print("After 0 detections / 10 effort at A:")
    print("  occupancy:", {k: round(v, 4) for k, v in after_ten.p_by_site().items()})
    print("  q posterior:", {k: round(v, 4) for k, v in after_ten.q_posterior().items()})
    print()
    print("INTERPRETATION:")
    print("  More effort makes the same non-detection stronger evidence.")
    print("  B changes because the ecological hypotheses couple A and B.")
    print("  C does not move because no declared hypothesis links the A evidence to C occupancy.")
    print("  q uncertainty is updated jointly rather than revealing a simulator q_true.")
    print("  Next gate: connect this kernel to real-data-constrained world-model families.")


if __name__ == "__main__":
    main()
