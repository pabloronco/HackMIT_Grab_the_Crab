"""Prospective (pre-survey) evaluation of candidate (site, effort) actions.

Everything here is computed from the *observable* joint posterior over
(ecological occupancy extent, q) held by ``SpatialBeliefState``. No function in
this module receives ``HiddenWorld`` or simulator ``q_true``.

Two implementations of the same mathematics coexist on purpose:

* ``SpatialBeliefEngine`` (spatial_belief.py) exposes slow, exact, scalar
  reference functions that reuse the frozen likelihood and ``update()`` kernel
  literally (hypothetical posterior = ``update`` on a copy, never on live state).
* ``ProspectiveEvaluator`` (this module) vectorizes the identical quantities
  over every (site, effort) pair with numpy so a planner can score the whole
  action space per decision round in well under a millisecond.

Tests pin the two paths to agree to floating-point precision.

Ecological extent entropy
-------------------------
The joint ensemble crosses every ecological hypothesis with every q hypothesis.
An identical occupancy bit-vector under q = 0.05 and under q = 0.20 is ONE
ecological extent, so extent entropy is computed after summing posterior weight
over q for each unique occupancy vector.

Expected information gain (bits)
--------------------------------
For a prospective observation y in {detection, no detection} at (site, effort):

    EIG = H_extent(before) - sum_y P(y) * H_extent(after_y)

where after_y is the exact finite-ensemble posterior under the frozen
likelihood P(no detection | occupied, q, e) = (1 - q)^e.

Conditional miss probability if occupied
----------------------------------------
    P(no detection | site occupied, posterior, e)
      = sum_{h: occupied_h} w_h (1-q_h)^e / sum_{h: occupied_h} w_h

computed on the JOINT posterior, so any occupancy-q dependence the evidence
has induced is preserved. Undefined (None) when posterior occupancy is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np

from .spatial_belief import (
    ProspectiveActionEvaluation,
    SpatialBeliefEngine,
    SpatialBeliefState,
    _probability_with_roundoff_guard,
)

_MASS_EPSILON = 1e-15


def _entropy_bits_columns(masses: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each column of a [E, N] mass matrix."""

    safe = np.where(masses > _MASS_EPSILON, masses, 1.0)
    return -np.sum(np.where(masses > _MASS_EPSILON, masses * np.log2(safe), 0.0), axis=0)


def _bernoulli_entropy_bits_rows(p: np.ndarray) -> np.ndarray:
    """Sum over columns of Bernoulli entropies (bits) for each row of a [n, N] matrix."""

    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > _MASS_EPSILON, p * np.log2(np.where(p > _MASS_EPSILON, p, 1.0)), 0.0)
        q = 1.0 - p
        term += np.where(q > _MASS_EPSILON, q * np.log2(np.where(q > _MASS_EPSILON, q, 1.0)), 0.0)
    return -np.sum(term, axis=1)


@dataclass(frozen=True, slots=True)
class DetectabilitySupportPressure:
    """How much posterior q mass sits on the edges of the modeled q support.

    This is a *support* diagnostic, deliberately separate from observation
    surprise / model stress: mass concentrating at q_min can make repeated
    non-detections MORE expected and therefore *reduce* surprise. It says the
    evidence is pushing toward the edge of the modeled detectability range,
    nothing more. ``concentrated_at_boundary`` is only populated when the caller
    supplies an explicit, labelled diagnostic threshold.
    """

    q_support: tuple[float, ...]
    q_min: float
    q_max: float
    mass_at_low_boundary: float
    mass_at_high_boundary: float
    posterior_mean: float
    boundary_mass_threshold: float | None
    concentrated_at_boundary: bool | None
    boundary_side: str | None


def detectability_support_pressure(
    belief: SpatialBeliefState,
    *,
    boundary_mass_threshold: float | None = None,
) -> DetectabilitySupportPressure:
    posterior = belief.q_posterior()
    support = tuple(sorted(posterior))
    if not support:
        raise ValueError("q posterior is empty.")
    q_min, q_max = support[0], support[-1]
    low = float(posterior[q_min])
    high = float(posterior[q_max]) if q_max != q_min else low
    if boundary_mass_threshold is not None:
        if not 0.0 < boundary_mass_threshold <= 1.0:
            raise ValueError("boundary_mass_threshold must be in (0, 1].")
        side = None
        if low >= boundary_mass_threshold:
            side = "low"
        elif high >= boundary_mass_threshold:
            side = "high"
        concentrated: bool | None = side is not None
    else:
        side = None
        concentrated = None
    return DetectabilitySupportPressure(
        q_support=support,
        q_min=q_min,
        q_max=q_max,
        mass_at_low_boundary=low,
        mass_at_high_boundary=high,
        posterior_mean=belief.q_mean(),
        boundary_mass_threshold=boundary_mass_threshold,
        concentrated_at_boundary=concentrated,
        boundary_side=side,
    )


class ProspectiveEvaluator:
    """Vectorized prospective scorer bound to one immutable posterior."""

    def __init__(self, belief: SpatialBeliefState) -> None:
        SpatialBeliefEngine._validate_state(belief)
        self.belief = belief
        self.site_ids: tuple[str, ...] = belief.site_ids
        self._site_index = {site_id: i for i, site_id in enumerate(self.site_ids)}

        self._w = np.asarray(belief.weights, dtype=np.float64)  # [H]
        self._q = np.asarray([h.q for h in belief.hypotheses], dtype=np.float64)  # [H]
        self._presence = np.asarray(
            [h.presence for h in belief.hypotheses], dtype=np.float64
        )  # [H, N]

        # Group joint hypotheses by occupancy vector (marginalize q).
        _, group = np.unique(self._presence, axis=0, return_inverse=True)
        group = np.asarray(group).reshape(-1)
        num_extents = int(group.max()) + 1 if group.size else 0
        self._group_matrix = np.zeros((num_extents, self._w.shape[0]), dtype=np.float64)
        self._group_matrix[group, np.arange(self._w.shape[0])] = 1.0  # [E, H]

        extent_mass = self._group_matrix @ self._w
        self.extent_entropy_bits: float = float(
            _entropy_bits_columns(extent_mass.reshape(-1, 1))[0]
        )
        self.unique_extent_count: int = num_extents
        self.occupancy_belief: np.ndarray = self._w @ self._presence  # [N]
        self.marginal_entropy_sum_bits: float = float(
            _bernoulli_entropy_bits_rows(self.occupancy_belief.reshape(1, -1))[0]
        )

    # ------------------------------------------------------------------ public
    def predictive_detection_probability(self, site_id: str, effort: int) -> float:
        rows = self.evaluate_all((effort,), site_ids=(site_id,))
        return rows[0].predictive_detection_probability

    def evaluate(self, site_id: str, effort: int) -> ProspectiveActionEvaluation:
        return self.evaluate_all((effort,), site_ids=(site_id,))[0]

    def evaluate_all(
        self,
        efforts: Sequence[int],
        *,
        site_ids: Sequence[str] | None = None,
    ) -> list[ProspectiveActionEvaluation]:
        """Evaluate every (site, effort) pair; order is efforts-major, sites-minor."""

        for effort in efforts:
            if isinstance(effort, bool) or not isinstance(effort, int) or effort <= 0:
                raise ValueError("effort must be a positive integer.")
        if site_ids is None:
            columns = np.arange(len(self.site_ids))
        else:
            unknown = [s for s in site_ids if s not in self._site_index]
            if unknown:
                raise ValueError(f"Unknown site_id(s) {unknown!r}.")
            columns = np.asarray([self._site_index[s] for s in site_ids], dtype=int)

        presence = self._presence[:, columns]  # [H, n]
        w = self._w
        p_occ = w @ presence  # [n]
        h_before = self.extent_entropy_bits

        results: list[ProspectiveActionEvaluation] = []
        for effort in efforts:
            miss = (1.0 - self._q) ** effort  # [H]
            det_given_h = presence * (1.0 - miss)[:, None]  # [H, n]
            p_det = w @ det_given_h  # [n]
            p_nodet = 1.0 - p_det

            raw_det = w[:, None] * det_given_h  # [H, n], unnormalized posterior if detection
            raw_nodet = w[:, None] * (1.0 - det_given_h)  # if no detection
            mass_det = self._group_matrix @ raw_det  # [E, n]
            mass_nodet = self._group_matrix @ raw_nodet

            with np.errstate(divide="ignore", invalid="ignore"):
                norm_det = np.where(p_det > _MASS_EPSILON, mass_det / np.where(p_det > _MASS_EPSILON, p_det, 1.0), 0.0)
                norm_nodet = np.where(p_nodet > _MASS_EPSILON, mass_nodet / np.where(p_nodet > _MASS_EPSILON, p_nodet, 1.0), 0.0)
            h_det = _entropy_bits_columns(norm_det)
            h_nodet = _entropy_bits_columns(norm_nodet)
            expected_after = np.where(p_det > _MASS_EPSILON, p_det * h_det, 0.0) + np.where(
                p_nodet > _MASS_EPSILON, p_nodet * h_nodet, 0.0
            )
            eig = np.maximum(0.0, h_before - expected_after)

            miss_if_occ_num = w @ (presence * miss[:, None])  # [n]

            # Secondary information objective: expected reduction in the SUM of
            # marginal occupancy entropies (the R7/R8 Information Gain objective),
            # for configs that prefer it. Column marginals of each hypothetical
            # posterior: p'_i = sum_h w'_h presence[h, i].
            full = self._presence  # [H, N]
            with np.errstate(divide="ignore", invalid="ignore"):
                marg_det = (raw_det / np.where(p_det > _MASS_EPSILON, p_det, 1.0)).T @ full  # [n, N]
                marg_nodet = (raw_nodet / np.where(p_nodet > _MASS_EPSILON, p_nodet, 1.0)).T @ full
            hsum_det = _bernoulli_entropy_bits_rows(marg_det)  # [n]
            hsum_nodet = _bernoulli_entropy_bits_rows(marg_nodet)
            expected_marginal_after = np.where(p_det > _MASS_EPSILON, p_det * hsum_det, 0.0) + np.where(
                p_nodet > _MASS_EPSILON, p_nodet * hsum_nodet, 0.0
            )
            marginal_eig = np.maximum(0.0, self.marginal_entropy_sum_bits - expected_marginal_after)

            for k, column in enumerate(columns):
                site_id = self.site_ids[int(column)]
                p_occ_k = float(p_occ[k])
                results.append(
                    ProspectiveActionEvaluation(
                        site_id=site_id,
                        effort=int(effort),
                        occupancy_belief=_probability_with_roundoff_guard(p_occ_k, name="occupancy belief"),
                        predictive_detection_probability=_probability_with_roundoff_guard(
                            float(p_det[k]), name="predictive detection probability"
                        ),
                        ecological_entropy_before_bits=h_before,
                        ecological_entropy_if_detection_bits=(
                            float(h_det[k]) if p_det[k] > _MASS_EPSILON else None
                        ),
                        ecological_entropy_if_no_detection_bits=(
                            float(h_nodet[k]) if p_nodet[k] > _MASS_EPSILON else None
                        ),
                        expected_ecological_entropy_after_bits=float(expected_after[k]),
                        expected_information_gain_bits=float(eig[k]),
                        expected_marginal_information_gain_bits=float(marginal_eig[k]),
                        conditional_miss_probability_if_occupied=(
                            _probability_with_roundoff_guard(
                                float(miss_if_occ_num[k] / p_occ_k),
                                name="conditional miss probability",
                            )
                            if p_occ_k > _MASS_EPSILON
                            else None
                        ),
                    )
                )
        for row in results:
            if not isfinite(row.expected_information_gain_bits):
                raise RuntimeError("Prospective evaluation produced a non-finite EIG.")
        return results
