from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log, log2
from typing import Mapping, Sequence

from .models import BeliefState, Observation, ObservationBatch


_PROBABILITY_ROUNDOFF_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class EcologicalHypothesis:
    """One plausible spatial occupancy pattern supplied by an ecological model.

    This is a *hypothesis used for inference*, not simulator hidden truth. World-model
    families will later generate many such hypotheses. `prior_weight` represents the
    hypothesis prior before conditioning on field evidence.
    """

    presence_by_site: Mapping[str, bool]
    prior_weight: float = 1.0
    label: str | None = None


@dataclass(frozen=True, slots=True)
class QHypothesis:
    """One effective protocol-level detectability hypothesis.

    R4 deliberately keeps q uncertainty explicit. Numeric q support is not frozen by
    this class and must come from sourced constraints / sensitivity design later.
    """

    q: float
    prior_weight: float = 1.0
    label: str | None = None


@dataclass(frozen=True, slots=True)
class SpatialHypothesis:
    """Internal joint ecological + q hypothesis aligned to `SpatialBeliefState.site_ids`."""

    presence: tuple[bool, ...]
    q: float
    ecological_label: str | None = None
    q_label: str | None = None


@dataclass(frozen=True, slots=True)
class SpatialBeliefState:
    """Posterior distribution over plausible spatial worlds and q hypotheses.

    The state contains hypothesis weights only. It never contains the simulator's
    realized hidden world. Site-level occupancy beliefs exposed to GraphState are
    marginals of this posterior.
    """

    site_ids: tuple[str, ...]
    hypotheses: tuple[SpatialHypothesis, ...]
    weights: tuple[float, ...]
    observed_history: tuple[Observation, ...] = field(default_factory=tuple)

    def p_by_site(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, site_id in enumerate(self.site_ids):
            marginal = sum(
                weight
                for hypothesis, weight in zip(self.hypotheses, self.weights)
                if hypothesis.presence[index]
            )
            result[site_id] = _probability_with_roundoff_guard(
                marginal,
                name=f"occupancy marginal for {site_id!r}",
            )
        return result

    def uncertainty_by_site(self) -> dict[str, float]:
        return {
            site_id: _bernoulli_entropy(p)
            for site_id, p in self.p_by_site().items()
        }

    def q_posterior(self) -> dict[float, float]:
        posterior: dict[float, float] = {}
        for hypothesis, weight in zip(self.hypotheses, self.weights):
            posterior[hypothesis.q] = posterior.get(hypothesis.q, 0.0) + weight
        return dict(sorted(posterior.items()))

    def q_mean(self) -> float:
        return sum(q * weight for q, weight in self.q_posterior().items())

    def joint_entropy_bits(self) -> float:
        return -sum(weight * log2(weight) for weight in self.weights if weight > 0.0)

    def effective_sample_size(self) -> float:
        denominator = sum(weight * weight for weight in self.weights)
        return 0.0 if denominator <= 0.0 else 1.0 / denominator

    def as_belief_state(self) -> BeliefState:
        """Project the spatial posterior into the existing planner-facing schema."""
        return BeliefState(
            p_by_site=self.p_by_site(),
            uncertainty_by_site=self.uncertainty_by_site(),
            observed_history=self.observed_history,
        )


@dataclass(frozen=True, slots=True)
class ProspectiveActionEvaluation:
    """Observable, pre-survey evaluation of one candidate (site, effort) action.

    Every field is derived from the current joint posterior only. ``None`` marks a
    quantity that is undefined for this action (e.g. entropy after a detection
    when detection has zero posterior-predictive probability).
    """

    site_id: str
    effort: int
    occupancy_belief: float
    predictive_detection_probability: float
    ecological_entropy_before_bits: float
    ecological_entropy_if_detection_bits: float | None
    ecological_entropy_if_no_detection_bits: float | None
    expected_ecological_entropy_after_bits: float
    expected_information_gain_bits: float
    conditional_miss_probability_if_occupied: float | None
    # Secondary objective kept for continuity with the R7/R8 Information Gain
    # baseline: expected reduction of the sum of marginal occupancy entropies.
    expected_marginal_information_gain_bits: float | None = None


class SpatialBeliefEngine:
    """Exact finite-ensemble Bayes update for spatial occupancy under imperfect detection.

    R4 CURRENT DEFAULT:
    - ecological structure lives in explicit occupancy hypotheses supplied externally;
    - q is an uncertain effective protocol-level parameter represented by finite support;
    - ecological and q hypotheses are crossed into a joint ensemble;
    - observations reweight that ensemble with the frozen effort-aware likelihood;
    - no false positives in the MVP;
    - no resampling in the kernel, so updates are deterministic and exactly inspectable;
    - GraphState receives only posterior marginals through `BeliefState`.

    The neural policy therefore never learns what a non-detection means. It only sees
    the posterior produced by this explicit evidence model.
    """

    @classmethod
    def initialize(
        cls,
        ecological_hypotheses: Sequence[EcologicalHypothesis],
        q_hypotheses: Sequence[QHypothesis],
        *,
        confirmed_sites: set[str] | None = None,
    ) -> SpatialBeliefState:
        if not ecological_hypotheses:
            raise ValueError("At least one ecological hypothesis is required.")
        if not q_hypotheses:
            raise ValueError("At least one q hypothesis is required.")

        ecological = tuple(ecological_hypotheses)
        q_support = tuple(q_hypotheses)

        first_sites = set(ecological[0].presence_by_site)
        if not first_sites:
            raise ValueError("Ecological hypotheses cannot have an empty site set.")
        site_ids = tuple(sorted(first_sites))

        for index, hypothesis in enumerate(ecological):
            cls._validate_positive_weight(
                hypothesis.prior_weight,
                name=f"ecological_hypotheses[{index}].prior_weight",
            )
            if set(hypothesis.presence_by_site) != first_sites:
                raise ValueError("All ecological hypotheses must use the same site ids.")
            for site_id in site_ids:
                value = hypothesis.presence_by_site[site_id]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"presence_by_site[{site_id!r}] must be bool in every hypothesis."
                    )

        for index, q_hypothesis in enumerate(q_support):
            cls._validate_probability(q_hypothesis.q, name=f"q_hypotheses[{index}].q")
            cls._validate_positive_weight(
                q_hypothesis.prior_weight,
                name=f"q_hypotheses[{index}].prior_weight",
            )

        confirmed = confirmed_sites or set()
        unknown_confirmed = confirmed.difference(first_sites)
        if unknown_confirmed:
            raise ValueError(
                f"confirmed_sites contains unknown sites: {sorted(unknown_confirmed)!r}"
            )

        hypotheses: list[SpatialHypothesis] = []
        raw_weights: list[float] = []
        for ecological_hypothesis in ecological:
            presence = tuple(
                bool(ecological_hypothesis.presence_by_site[site_id])
                for site_id in site_ids
            )
            consistent_with_confirmed = all(
                ecological_hypothesis.presence_by_site[site_id]
                for site_id in confirmed
            )
            for q_hypothesis in q_support:
                hypotheses.append(
                    SpatialHypothesis(
                        presence=presence,
                        q=float(q_hypothesis.q),
                        ecological_label=ecological_hypothesis.label,
                        q_label=q_hypothesis.label,
                    )
                )
                raw_weights.append(
                    float(ecological_hypothesis.prior_weight)
                    * float(q_hypothesis.prior_weight)
                    * float(consistent_with_confirmed)
                )

        weights = cls._normalize_weights(
            raw_weights,
            zero_message=(
                "No ecological hypothesis is compatible with the confirmed-site evidence."
            ),
        )
        return SpatialBeliefState(
            site_ids=site_ids,
            hypotheses=tuple(hypotheses),
            weights=weights,
            observed_history=(),
        )

    @classmethod
    def update(
        cls,
        belief: SpatialBeliefState,
        observation_batch: ObservationBatch,
    ) -> SpatialBeliefState:
        cls._validate_state(belief)
        site_index = {site_id: index for index, site_id in enumerate(belief.site_ids)}

        for observation in observation_batch.observations:
            if observation.site_id not in site_index:
                raise ValueError(
                    f"Observation references unknown belief site {observation.site_id!r}."
                )
            cls._validate_observation(observation)

        log_weights: list[float] = []
        for hypothesis, prior_weight in zip(belief.hypotheses, belief.weights):
            if prior_weight <= 0.0:
                log_weights.append(float("-inf"))
                continue

            value = log(prior_weight)
            for observation in observation_batch.observations:
                index = site_index[observation.site_id]
                likelihood = cls.observation_likelihood(
                    occupied=hypothesis.presence[index],
                    q=hypothesis.q,
                    observation=observation,
                )
                if likelihood <= 0.0:
                    value = float("-inf")
                    break
                value += log(likelihood)
            log_weights.append(value)

        finite = [value for value in log_weights if isfinite(value)]
        if not finite:
            raise ValueError(
                "Observation batch has zero probability under every spatial/q hypothesis."
            )
        offset = max(finite)
        raw_weights = [
            0.0 if not isfinite(value) else exp(value - offset)
            for value in log_weights
        ]
        weights = cls._normalize_weights(
            raw_weights,
            zero_message="Posterior normalization failed: all hypothesis weights are zero.",
        )

        return SpatialBeliefState(
            site_ids=belief.site_ids,
            hypotheses=belief.hypotheses,
            weights=weights,
            observed_history=(
                belief.observed_history + tuple(observation_batch.observations)
            ),
        )

    @classmethod
    def predictive_detection_probability(
        cls,
        belief: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> float:
        """Posterior predictive P(at least one detection) for prospective effort."""
        cls._validate_state(belief)
        if site_id not in belief.site_ids:
            raise ValueError(f"Unknown site_id {site_id!r}.")
        if isinstance(effort, bool) or not isinstance(effort, int) or effort <= 0:
            raise ValueError("effort must be a positive integer.")

        index = belief.site_ids.index(site_id)
        probability = 0.0
        for hypothesis, weight in zip(belief.hypotheses, belief.weights):
            if hypothesis.presence[index]:
                probability += weight * (1.0 - (1.0 - hypothesis.q) ** effort)
        return _probability_with_roundoff_guard(
            probability,
            name=f"predictive detection probability for {site_id!r}",
        )

    @classmethod
    def ecological_extent_entropy_bits(cls, belief: SpatialBeliefState) -> float:
        """Entropy (bits) over unique ecological occupancy extents, q marginalized.

        Joint hypotheses that share an occupancy bit-vector but differ in q are
        ONE ecological extent: their posterior weights are summed before the
        entropy is taken. This is deliberately not ``joint_entropy_bits``.
        """

        cls._validate_state(belief)
        mass_by_extent: dict[tuple[bool, ...], float] = {}
        for hypothesis, weight in zip(belief.hypotheses, belief.weights):
            mass_by_extent[hypothesis.presence] = (
                mass_by_extent.get(hypothesis.presence, 0.0) + float(weight)
            )
        return -sum(m * log2(m) for m in mass_by_extent.values() if m > 0.0)

    @classmethod
    def conditional_miss_probability_if_occupied(
        cls,
        belief: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> float | None:
        """P(no detection | site occupied, current joint posterior, effort).

        Computed on the JOINT posterior so any occupancy/q dependence induced by
        evidence is preserved. Returns ``None`` when posterior occupancy is zero
        (the conditional is undefined).
        """

        cls._validate_state(belief)
        if site_id not in belief.site_ids:
            raise ValueError(f"Unknown site_id {site_id!r}.")
        if isinstance(effort, bool) or not isinstance(effort, int) or effort <= 0:
            raise ValueError("effort must be a positive integer.")
        index = belief.site_ids.index(site_id)
        occupied_mass = 0.0
        miss_mass = 0.0
        for hypothesis, weight in zip(belief.hypotheses, belief.weights):
            if hypothesis.presence[index]:
                occupied_mass += weight
                miss_mass += weight * (1.0 - hypothesis.q) ** effort
        if occupied_mass <= 0.0:
            return None
        return _probability_with_roundoff_guard(
            miss_mass / occupied_mass, name=f"conditional miss probability for {site_id!r}"
        )

    @classmethod
    def expected_information_gain_bits(
        cls,
        belief: SpatialBeliefState,
        *,
        site_id: str,
        effort: int,
    ) -> ProspectiveActionEvaluation:
        """Exact reference EIG over ecological extents for a prospective action.

        Hypothetical posteriors are built with the frozen ``update`` kernel on a
        prospective ``Observation`` (round=0); ``update`` is pure and returns a new
        state, so the live belief is never mutated.
        """

        cls._validate_state(belief)
        if site_id not in belief.site_ids:
            raise ValueError(f"Unknown site_id {site_id!r}.")
        if isinstance(effort, bool) or not isinstance(effort, int) or effort <= 0:
            raise ValueError("effort must be a positive integer.")

        p_detection = cls.predictive_detection_probability(belief, site_id=site_id, effort=effort)
        p_no_detection = 1.0 - p_detection
        h_before = cls.ecological_extent_entropy_bits(belief)

        def _hypothetical(detection: bool) -> float:
            observation = Observation(site_id=site_id, effort=effort, detection=detection, round=0)
            posterior = cls.update(
                belief, ObservationBatch(observations=(observation,), round=0, total_effort=effort)
            )
            return cls.ecological_extent_entropy_bits(posterior)

        def _hypothetical_marginal(detection: bool) -> float:
            observation = Observation(site_id=site_id, effort=effort, detection=detection, round=0)
            posterior = cls.update(
                belief, ObservationBatch(observations=(observation,), round=0, total_effort=effort)
            )
            return sum(posterior.uncertainty_by_site().values())

        h_det = _hypothetical(True) if p_detection > 0.0 else None
        h_nodet = _hypothetical(False) if p_no_detection > 0.0 else None
        expected_after = 0.0
        expected_marginal_after = 0.0
        if h_det is not None:
            expected_after += p_detection * h_det
            expected_marginal_after += p_detection * _hypothetical_marginal(True)
        if h_nodet is not None:
            expected_after += p_no_detection * h_nodet
            expected_marginal_after += p_no_detection * _hypothetical_marginal(False)
        marginal_before = sum(belief.uncertainty_by_site().values())

        return ProspectiveActionEvaluation(
            site_id=site_id,
            effort=int(effort),
            occupancy_belief=belief.p_by_site()[site_id],
            predictive_detection_probability=p_detection,
            ecological_entropy_before_bits=h_before,
            ecological_entropy_if_detection_bits=h_det,
            ecological_entropy_if_no_detection_bits=h_nodet,
            expected_ecological_entropy_after_bits=expected_after,
            expected_information_gain_bits=max(0.0, h_before - expected_after),
            conditional_miss_probability_if_occupied=cls.conditional_miss_probability_if_occupied(
                belief, site_id=site_id, effort=effort
            ),
            expected_marginal_information_gain_bits=max(0.0, marginal_before - expected_marginal_after),
        )

    @staticmethod
    def observation_likelihood(
        *,
        occupied: bool,
        q: float,
        observation: Observation,
    ) -> float:
        SpatialBeliefEngine._validate_probability(q, name="q")
        SpatialBeliefEngine._validate_observation(observation)
        if not occupied:
            return 0.0 if observation.detection else 1.0

        miss_probability = (1.0 - q) ** observation.effort
        return 1.0 - miss_probability if observation.detection else miss_probability

    @staticmethod
    def _validate_state(belief: SpatialBeliefState) -> None:
        if not belief.site_ids:
            raise ValueError("SpatialBeliefState site_ids cannot be empty.")
        if not belief.hypotheses:
            raise ValueError("SpatialBeliefState hypotheses cannot be empty.")
        if len(belief.hypotheses) != len(belief.weights):
            raise ValueError("SpatialBeliefState hypotheses and weights must align.")
        if any(len(hypothesis.presence) != len(belief.site_ids) for hypothesis in belief.hypotheses):
            raise ValueError("Spatial hypothesis presence vectors must align with site_ids.")
        if any(weight < 0.0 for weight in belief.weights):
            raise ValueError("SpatialBeliefState weights cannot be negative.")
        if abs(sum(belief.weights) - 1.0) > 1e-9:
            raise ValueError("SpatialBeliefState weights must sum to 1.")
        for hypothesis in belief.hypotheses:
            SpatialBeliefEngine._validate_probability(hypothesis.q, name="hypothesis.q")

    @staticmethod
    def _validate_observation(observation: Observation) -> None:
        if isinstance(observation.effort, bool) or not isinstance(observation.effort, int):
            raise ValueError("Observation effort must be a positive integer.")
        if observation.effort <= 0:
            raise ValueError("Observation effort must be a positive integer.")
        if not isinstance(observation.detection, bool):
            raise ValueError("Observation detection must be bool.")

    @staticmethod
    def _normalize_weights(
        raw_weights: Sequence[float],
        *,
        zero_message: str,
    ) -> tuple[float, ...]:
        total = sum(float(weight) for weight in raw_weights)
        if total <= 0.0:
            raise ValueError(zero_message)
        return tuple(float(weight) / total for weight in raw_weights)

    @staticmethod
    def _validate_probability(value: float, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric in [0, 1].")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    @staticmethod
    def _validate_positive_weight(value: float, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a positive number.")
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be > 0.")


def _probability_with_roundoff_guard(value: float, *, name: str) -> float:
    """Clamp machine-epsilon boundary noise without hiding real probability bugs.

    Finite-ensemble marginalization can produce values such as
    1.0000000000000002 after summing normalized floating-point weights. That is a
    numerical artifact, not a probabilistic state above one. Values materially
    outside [0, 1] still raise so genuine inference errors remain visible.
    """

    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value < -_PROBABILITY_ROUNDOFF_TOLERANCE or value > 1.0 + _PROBABILITY_ROUNDOFF_TOLERANCE:
        raise ValueError(f"{name} fell outside [0, 1]: {value!r}.")
    return min(1.0, max(0.0, value))


def _bernoulli_entropy(p: float) -> float:
    p = _probability_with_roundoff_guard(p, name="Bernoulli probability")
    if p in (0.0, 1.0):
        return 0.0
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p))
