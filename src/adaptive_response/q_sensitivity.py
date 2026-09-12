from __future__ import annotations

from typing import Any, Iterable


DEFAULT_Q_PROBES = (0.02, 0.05, 0.10, 0.20, 0.35)
DEFAULT_EFFORT_PROBES = (1, 3, 6, 10, 12)


def detection_probability(q: float, effort: float) -> float:
    """Probability of at least one detection if occupied.

    q is the effective per-effort-unit detection probability conditional on occupancy.
    This function encodes the frozen evidence semantics only; it does not claim that
    any particular numeric q is empirically identified for European green crab.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if effort < 0.0:
        raise ValueError("effort must be non-negative")
    return 1.0 - (1.0 - q) ** effort


def posterior_after_nondetection(prior: float, q: float, effort: float) -> float:
    """Bayesian occupancy posterior after a non-detection."""
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must lie in [0, 1]")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if effort < 0.0:
        raise ValueError("effort must be non-negative")

    miss_if_occupied = (1.0 - q) ** effort
    denominator = (1.0 - prior) + prior * miss_if_occupied
    if denominator == 0.0:
        return 0.0
    return prior * miss_if_occupied / denominator


def build_q_sensitivity_table(
    *,
    q_values: Iterable[float] = DEFAULT_Q_PROBES,
    effort_values: Iterable[int] = DEFAULT_EFFORT_PROBES,
    prior: float = 0.5,
) -> list[dict[str, Any]]:
    """Expose how assumed q changes event sensitivity and Bayes evidence strength.

    The default q values are deliberately broad engineering *probes*, not a sourced
    biological range. Their purpose is to make the model consequence of q explicit
    before the team freezes any q distribution for simulation/training.
    """
    q_values = tuple(float(q) for q in q_values)
    effort_values = tuple(int(effort) for effort in effort_values)
    if not q_values:
        raise ValueError("at least one q value is required")
    if not effort_values:
        raise ValueError("at least one effort value is required")
    if any(effort < 0 for effort in effort_values):
        raise ValueError("effort values must be non-negative")

    rows: list[dict[str, Any]] = []
    for q in q_values:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q values must lie in [0, 1]")
        for effort in effort_values:
            rows.append(
                {
                    "q_probe": q,
                    "effort": effort,
                    "detection_if_occupied": detection_probability(q, effort),
                    "miss_if_occupied": (1.0 - q) ** effort,
                    "posterior_after_nondetection": posterior_after_nondetection(
                        prior, q, effort
                    ),
                    "prior": prior,
                    "q_status": "DESIGN_SENSITIVITY_PROBE_NOT_EMPIRICAL_ESTIMATE",
                }
            )
    return rows
