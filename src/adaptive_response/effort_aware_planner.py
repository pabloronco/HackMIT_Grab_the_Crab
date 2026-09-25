"""Effort-aware transparent planners over the (site, effort) action space.

RAMP product priority (docs/DECISION_LOG.md, 2026-09-19 "Effort-aware planner"):
Marine must answer WHERE to survey and HOW MUCH effort to spend there, and must
be able to return unused field resources (STOP) when the remaining expected
value per unit cost is below a configured, labelled threshold.

All quantities are computed from the observable joint posterior through
``ProspectiveEvaluator``; no planner here receives hidden truth. Every score
component is exported in ``MissionAction.diagnostics`` so the UI can show *why*
a site and an effort level were selected.

Every coefficient in ``EffortAwarePlannerConfig`` / ``OperationalCostConfig`` is a
DESIGN CHOICE in normalized cost units, tuned on the frozen validation split
only (scripts/benchmark_effort_aware.py --tune) and frozen before the formal
test split is read. Nothing here is an agency cost, a biological standard, or a
professional preference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .graph_state import NODE_FEATURE_NAMES
from .models import GraphState, MissionAction, MissionAllocation
from .planners import FrontierPlanner
from .prospective import ProspectiveEvaluator
from .spatial_belief import ProspectiveActionEvaluation, SpatialBeliefState

DEFAULT_EFFORT_LEVELS: tuple[int, ...] = (1, 3, 6)
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "effort_aware_planner.json"

STOP_DIAGNOSTIC_KEY = "stop"


@dataclass(frozen=True)
class OperationalCostConfig:
    """Normalized operational cost of one field mission.

    cost(site, e) = dispatch_overhead_cost + effort_unit_cost * e
                    + travel_cost_weight * travel_proxy(site)

    ``travel_cost_weight`` defaults to 0.0: the frozen incident artifacts carry a
    per-EDGE navigable-route proxy (Edge.travel_cost) but no team base location
    or mission-level route model, so no defensible per-mission travel cost
    exists. The term is kept in the interface and documented as unavailable.

    Regime-adaptive dispatch cost (R11, "regime-adaptive dispatch cost" decision
    log entry): a single fixed ``dispatch_overhead_cost`` was found empirically to
    win clearly in some observable regimes and lose in others -- specifically,
    high posterior mean detectability (q_mean) and high posterior-expected
    prevalence both favor cheap, broad effort (low dispatch cost -> more effort-3
    missions), while low q_mean or low expected prevalence favor committing real
    effort to few candidates (high dispatch cost -> more effort-6 missions,
    closer to the Fixed High Effort baseline). Both q_mean and posterior-expected
    prevalence (mean occupancy belief across the incident) are observable —
    computed only from ``SpatialBeliefState``, never from simulator truth — and
    were verified to track their hidden counterparts on the frozen R8 formal
    cases before this mechanism was built (see the decision log entry: q_mean
    tracks q_true on average; posterior-expected occupied count correlates
    r=0.376 with the true occupied count, well-calibrated by bucket).

    When ``regime_adaptive`` is enabled, ``dispatch_overhead_cost`` above is
    ignored and ``resolved_dispatch_overhead_cost`` is used instead: a dispatch
    cost that departs from ``dispatch_base`` by ``q_slope`` per unit q_mean is
    BELOW ``q_reference`` and by ``prevalence_slope`` per unit expected
    prevalence below ``prevalence_reference``, clamped to
    [``dispatch_min``, ``dispatch_max``]. ``q_mean`` starts every episode at the
    prior mean (no field evidence differentiates q yet) and only departs from
    ``q_reference`` as detectability evidence accumulates within the campaign —
    this is a within-episode-adaptive rule, not a day-one prediction, and that is
    intentional: q genuinely is unknown at t0.
    """

    dispatch_overhead_cost: float = 1.0
    effort_unit_cost: float = 1.0
    travel_cost_weight: float = 0.0
    units: str = "normalized_cost_units"
    provenance: str = "DESIGN_CHOICE_not_agency_truth"

    regime_adaptive: bool = False
    dispatch_base: float = 1.0
    q_reference: float = 0.1167  # prior mean of the {0.05, 0.10, 0.20} belief support
    q_slope: float = 0.0  # added cost per unit q_mean BELOW q_reference (>=0: low q -> more effort)
    prevalence_reference: float = 0.35
    prevalence_slope: float = 0.0  # added cost per unit expected prevalence BELOW prevalence_reference
    dispatch_min: float = 0.0
    dispatch_max: float = 3.0

    def __post_init__(self) -> None:
        for name in ("dispatch_overhead_cost", "effort_unit_cost", "travel_cost_weight", "dispatch_base", "dispatch_min", "dispatch_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number.")
        if self.effort_unit_cost <= 0 and self.dispatch_overhead_cost <= 0 and not self.regime_adaptive:
            raise ValueError("Cost model must be strictly positive for some effort.")
        if self.dispatch_min > self.dispatch_max:
            raise ValueError("dispatch_min cannot exceed dispatch_max.")
        for name in ("q_reference", "prevalence_reference"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

    def resolved_dispatch_overhead_cost(
        self, *, q_mean: float | None = None, expected_prevalence: float | None = None
    ) -> float:
        """The dispatch cost actually used for one scoring call.

        Falls back to the fixed ``dispatch_overhead_cost`` when regime-adaptive
        mode is off, or when a required observable was not supplied.
        """

        if not self.regime_adaptive:
            return float(self.dispatch_overhead_cost)
        value = float(self.dispatch_base)
        if q_mean is not None:
            value += self.q_slope * (self.q_reference - float(q_mean))
        if expected_prevalence is not None:
            value += self.prevalence_slope * (self.prevalence_reference - float(expected_prevalence))
        return max(self.dispatch_min, min(self.dispatch_max, value))

    def action_cost(
        self,
        effort: int,
        *,
        travel_proxy: float = 0.0,
        q_mean: float | None = None,
        expected_prevalence: float | None = None,
    ) -> float:
        dispatch = self.resolved_dispatch_overhead_cost(q_mean=q_mean, expected_prevalence=expected_prevalence)
        return (
            dispatch
            + float(self.effort_unit_cost) * float(effort)
            + float(self.travel_cost_weight) * float(travel_proxy)
        )


@dataclass(frozen=True)
class StopRuleConfig:
    """Optional early-completion rule. Threshold values are DESIGN CHOICES."""

    enabled: bool = False
    # STOP when the best available expected information gain per unit cost falls
    # below this value (bits per normalized cost unit).
    min_information_efficiency: float | None = None
    # STOP when the current ecological-extent entropy is already below target.
    entropy_target_bits: float | None = None

    def __post_init__(self) -> None:
        if self.min_information_efficiency is not None and self.min_information_efficiency < 0:
            raise ValueError("min_information_efficiency must be >= 0.")
        if self.entropy_target_bits is not None and self.entropy_target_bits < 0:
            raise ValueError("entropy_target_bits must be >= 0.")


@dataclass(frozen=True)
class EffortAwarePlannerConfig:
    effort_levels: tuple[int, ...] = DEFAULT_EFFORT_LEVELS
    # site_ranking:
    #   "belief_frontier": rank SITES exactly like FrontierPlanner (frontier flag, then
    #     occupancy belief, then uncertainty, then site id) -- this is a find-the-target
    #     (exploitation) rule and is what the frozen default uses (see DECISION_LOG,
    #     "R11 site-ranking correction"): a pure entropy/EIG objective is an
    #     uncertainty-reduction (exploration) rule that provably prefers sites near
    #     belief 0.5 over sites with high occupancy belief, which directly hurt
    #     detection coverage in both formal-benchmark and live-demo testing.
    #   "information_efficiency": rank sites by EIG / cost directly (the R7/R8
    #     InformationGainPlanner-style rule); kept for experimentation/comparison,
    #     NOT the frozen default because of the above.
    site_ranking: str = "belief_frontier"
    # effort_ranking_objective: how EFFORT is chosen once a site is fixed.
    #   "information_efficiency": maximize EIG / cost (primary transparent RAMP objective)
    #   "utility": maximize the scalarized utility below
    ranking_objective: str = "information_efficiency"
    # information_objective:
    #   "extent_entropy": EIG over unique ecological extents (q marginalized) - primary
    #   "marginal_entropy_sum": EIG on the sum of marginal occupancy entropies (R7/R8 IG objective)
    information_objective: str = "extent_entropy"
    information_gain_weight: float = 1.0  # alpha
    detection_weight: float = 0.0  # beta
    frontier_weight: float = 0.0  # gamma
    cost_weight: float = 0.0  # lambda_effort (utility mode only)
    travel_weight: float = 0.0  # lambda_travel (utility mode only, no data -> 0)
    cost: OperationalCostConfig = field(default_factory=OperationalCostConfig)
    stop: StopRuleConfig = field(default_factory=StopRuleConfig)
    # revisit_requires_escalation: when a site already has observed_effort > 0,
    # only consider effort levels STRICTLY GREATER than what was already spent
    # there. Motivation (R11 "revisit masking ablation"): under the paired
    # per-site detection draw used for fair cross-track comparison, a revisit
    # at the SAME or LOWER effort as a prior failed attempt is deterministically
    # guaranteed to fail again (same draw, same or lower detection threshold) --
    # empirically, allowing unconstrained revisits measured WORSE than blocking
    # them outright (mean detected fraction 0.390 vs 0.419 on the 100-case demo
    # library, even below fixed_high_effort's 0.397). This flag is the
    # alternative tested to recover revisit value without that failure mode:
    # only ever escalate. Default False preserves every prior behavior exactly.
    revisit_requires_escalation: bool = False
    status: str = "PROVISIONAL_UNTUNED"
    provenance: str = "DESIGN_CHOICE_tune_on_validation_only"

    def __post_init__(self) -> None:
        if not self.effort_levels or any(
            isinstance(e, bool) or not isinstance(e, int) or e <= 0 for e in self.effort_levels
        ):
            raise ValueError("effort_levels must be positive ints.")
        if len(set(self.effort_levels)) != len(self.effort_levels):
            raise ValueError("effort_levels must be unique.")
        if self.ranking_objective not in ("information_efficiency", "utility"):
            raise ValueError("ranking_objective must be 'information_efficiency' or 'utility'.")
        if self.information_objective not in ("extent_entropy", "marginal_entropy_sum"):
            raise ValueError("information_objective must be 'extent_entropy' or 'marginal_entropy_sum'.")
        if self.site_ranking not in ("belief_frontier", "information_efficiency"):
            raise ValueError("site_ranking must be 'belief_frontier' or 'information_efficiency'.")
        object.__setattr__(self, "effort_levels", tuple(sorted(int(e) for e in self.effort_levels)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effort_levels"] = list(self.effort_levels)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffortAwarePlannerConfig":
        data = dict(payload)
        cost = data.pop("cost", {}) or {}
        stop = data.pop("stop", {}) or {}
        data["effort_levels"] = tuple(int(e) for e in data.get("effort_levels", DEFAULT_EFFORT_LEVELS))
        return cls(cost=OperationalCostConfig(**cost), stop=StopRuleConfig(**stop), **data)

    @classmethod
    def load_frozen(cls, path: Path = FROZEN_CONFIG_PATH) -> "EffortAwarePlannerConfig":
        """Load the committed planner config; falls back to defaults if absent."""

        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload["planner_config"])


def make_stop_action(*, reason: str, diagnostics: Mapping[str, Any] | None = None) -> MissionAction:
    """A zero-cost MissionAction signalling the planner wants to end the campaign."""

    return MissionAction(
        allocations=(),
        total_cost=0,
        diagnostics={
            STOP_DIAGNOSTIC_KEY: True,
            "stop_reason": reason,
            "hidden_truth_used": False,
            **(dict(diagnostics) if diagnostics else {}),
        },
    )


def is_stop_action(mission: MissionAction | None) -> bool:
    return bool(
        mission is not None
        and not mission.allocations
        and mission.total_cost == 0
        and bool((mission.diagnostics or {}).get(STOP_DIAGNOSTIC_KEY, False))
    )


def _row_from_evaluation(
    evaluation: ProspectiveActionEvaluation,
    *,
    frontier: bool,
    uncertainty: float,
    config: EffortAwarePlannerConfig,
    travel_proxy: float = 0.0,
    q_mean: float | None = None,
    expected_prevalence: float | None = None,
) -> dict[str, Any]:
    cost = config.cost.action_cost(
        evaluation.effort, travel_proxy=travel_proxy, q_mean=q_mean, expected_prevalence=expected_prevalence
    )
    extent_eig = float(evaluation.expected_information_gain_bits)
    marginal_eig = (
        float(evaluation.expected_marginal_information_gain_bits)
        if evaluation.expected_marginal_information_gain_bits is not None
        else extent_eig
    )
    eig = marginal_eig if config.information_objective == "marginal_entropy_sum" else extent_eig
    efficiency = eig / cost if cost > 0 else float("inf")
    utility = (
        config.information_gain_weight * eig
        + config.detection_weight * float(evaluation.predictive_detection_probability)
        + config.frontier_weight * float(frontier)
        - config.cost_weight * cost
        - config.travel_weight * travel_proxy
    )
    return {
        "site_id": evaluation.site_id,
        "effort_units": int(evaluation.effort),
        "belief": float(evaluation.occupancy_belief),
        "uncertainty": float(uncertainty),
        "frontier": bool(frontier),
        "predictive_detection_probability": float(evaluation.predictive_detection_probability),
        "expected_information_gain_bits": eig,
        "expected_extent_information_gain_bits": extent_eig,
        "expected_marginal_information_gain_bits": marginal_eig,
        "information_objective": config.information_objective,
        "ecological_entropy_before_bits": float(evaluation.ecological_entropy_before_bits),
        "expected_ecological_entropy_after_bits": float(evaluation.expected_ecological_entropy_after_bits),
        "conditional_miss_probability_if_occupied": evaluation.conditional_miss_probability_if_occupied,
        "cost": float(cost),
        "travel_proxy": float(travel_proxy),
        "information_efficiency": float(efficiency),
        "utility": float(utility),
        "dispatch_overhead_cost_used": float(cost) - float(config.cost.effort_unit_cost) * float(evaluation.effort) - float(config.cost.travel_cost_weight) * float(travel_proxy),
    }


def _effort_ranking_key(row: Mapping[str, Any], objective: str) -> tuple:
    """Orders (site, effort) rows for the EFFORT-selection sub-problem at a fixed site,
    and as the fallback full ranking when site_ranking == 'information_efficiency'."""
    primary = row["utility"] if objective == "utility" else row["information_efficiency"]
    return (
        -float(primary),
        -float(row["expected_information_gain_bits"]),
        -float(row["information_efficiency"]),
        -float(row["uncertainty"]),
        -float(row["belief"]),
        str(row["site_id"]),
        int(row["effort_units"]),
    )


def _site_ranking_key(row: Mapping[str, Any]) -> tuple:
    """Frontier-equivalent SITE ordering: frontier first, then belief, then uncertainty,
    then deterministic site id -- see EffortAwarePlannerConfig.site_ranking docstring
    for why this, not EIG, decides WHICH site (find-the-target, not entropy-reduction)."""
    return (
        -int(bool(row["frontier"])),
        -float(row["belief"]),
        -float(row["uncertainty"]),
        str(row["site_id"]),
    )


class EffortAwareInformationGainPlanner:
    """Adaptive (site, effort) planner with optional STOP.

    Requires ``constraints['spatial_belief_state']`` (the observable posterior).
    Re-scores every feasible (site, effort) pair from scratch on every call, so
    wrapping it in an adaptive loop makes it replan after each field return.
    """

    planner_name = "adaptive_effort_aware"

    def __init__(self, config: EffortAwarePlannerConfig | None = None) -> None:
        self.config = config or EffortAwarePlannerConfig()

    # ------------------------------------------------------------- scoring
    def score_actions(
        self,
        graph_state: GraphState,
        spatial_belief: SpatialBeliefState,
        remaining_budget: int,
    ) -> dict[str, Any]:
        """Score all feasible (site, effort) pairs. Pure; no state change."""

        if len(graph_state.node_ids) != len(graph_state.node_features):
            raise ValueError("GraphState node_ids and node_features are misaligned.")
        if len(graph_state.node_ids) != len(graph_state.feasibility_mask):
            raise ValueError("GraphState feasibility_mask is misaligned with nodes.")
        if set(spatial_belief.site_ids) != set(graph_state.node_ids):
            raise ValueError("Spatial belief site ids must match GraphState node ids.")
        if isinstance(remaining_budget, bool) or not isinstance(remaining_budget, int) or remaining_budget < 0:
            raise ValueError("remaining_budget must be a non-negative integer.")

        frontier_idx = NODE_FEATURE_NAMES.index("frontier")
        uncertainty_idx = NODE_FEATURE_NAMES.index("uncertainty")
        observed_effort_idx = NODE_FEATURE_NAMES.index("observed_effort")
        feasible_sites = [
            site_id
            for site_id, feasible in zip(graph_state.node_ids, graph_state.feasibility_mask)
            if feasible
        ]
        efforts = tuple(e for e in self.config.effort_levels if e <= remaining_budget)
        evaluator = ProspectiveEvaluator(spatial_belief)
        node_meta = {
            site_id: (
                bool(graph_state.node_features[i][frontier_idx] > 0.5),
                float(graph_state.node_features[i][uncertainty_idx]),
            )
            for i, site_id in enumerate(graph_state.node_ids)
        }
        observed_effort_by_site = {
            site_id: float(graph_state.node_features[i][observed_effort_idx])
            for i, site_id in enumerate(graph_state.node_ids)
        }

        # Observable regime signals (R11 regime-adaptive dispatch cost): both are
        # pure functions of SpatialBeliefState, recomputed fresh on every call, so
        # this needs no per-episode state and is safe to call any number of times
        # (including read-only UI display calls between rounds). q_mean starts
        # every episode at the prior mean and only departs as field evidence
        # accumulates; expected_prevalence is the posterior-expected occupied
        # fraction (sum of occupancy belief over the incident / site count), an
        # unbiased, empirically-calibrated estimator of the hidden true
        # prevalence (see docs/DECISION_LOG.md).
        q_mean = spatial_belief.q_mean()
        expected_prevalence = (
            sum(evaluator.occupancy_belief) / len(spatial_belief.site_ids)
            if spatial_belief.site_ids
            else None
        )

        rows: list[dict[str, Any]] = []
        if feasible_sites and efforts:
            for evaluation in evaluator.evaluate_all(efforts, site_ids=feasible_sites):
                frontier, uncertainty = node_meta[evaluation.site_id]
                rows.append(
                    _row_from_evaluation(
                        evaluation, frontier=frontier, uncertainty=uncertainty, config=self.config,
                        q_mean=q_mean, expected_prevalence=expected_prevalence,
                    )
                )
        if self.config.revisit_requires_escalation:
            rows = [
                row
                for row in rows
                if row["effort_units"] > observed_effort_by_site.get(row["site_id"], 0.0)
            ]
        # Global action ranking (every feasible (site, effort) pair): drives STOP's
        # "best available marginal value per cost anywhere" and the flat
        # diagnostics list. This is always effort-ranking-key ordered regardless
        # of site_ranking -- STOP asks "is anything left worth doing", not "is the
        # site Marine would actually visit worth it".
        rows.sort(key=lambda row: _effort_ranking_key(row, self.config.ranking_objective))

        # Per-site view: best EFFORT per site (by effort-ranking key -- rows is
        # already sorted that way, so the first row seen per site_id is its best
        # effort) plus the full effort curve, so the UI can explain "why effort 3
        # and not 6" with real numbers.
        per_site: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = per_site.setdefault(row["site_id"], {"best": row, "alternatives": []})
            entry["alternatives"].append(row)
        ranked_sites: list[dict[str, Any]] = []
        for site_id, entry in per_site.items():
            alternatives = sorted(entry["alternatives"], key=lambda r: r["effort_units"])
            max_effort_eig = max(a["expected_information_gain_bits"] for a in alternatives)
            for alternative in alternatives:
                alternative["eig_fraction_of_max_effort"] = (
                    alternative["expected_information_gain_bits"] / max_effort_eig
                    if max_effort_eig > 0
                    else None
                )
            ranked_sites.append({**entry["best"], "effort_alternatives": alternatives})

        # WHICH SITE Marine visits: belief_frontier (the frozen default) uses the
        # same find-the-target ordering as FrontierPlanner/FixedHighEffortPlanner
        # -- see EffortAwarePlannerConfig.site_ranking docstring for why a pure
        # EIG/cost site ordering measurably picks lower-belief sites and costs
        # real detection coverage. HOW MUCH effort at that site remains
        # effort-ranking-key driven either way.
        if self.config.site_ranking == "belief_frontier":
            ranked_sites.sort(key=_site_ranking_key)
        else:
            ranked_sites.sort(key=lambda row: _effort_ranking_key(row, self.config.ranking_objective))
        for rank, row in enumerate(ranked_sites, start=1):
            row["rank"] = rank

        return {
            "ranked_actions": rows,
            "ranked_sites": ranked_sites,
            "ecological_entropy_bits": evaluator.extent_entropy_bits,
            "unique_extent_count": evaluator.unique_extent_count,
            "feasible_efforts": list(efforts),
        }

    def evaluate_stop(self, scored: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the configured STOP rule to a ``score_actions`` result."""

        stop_cfg = self.config.stop
        best = scored["ranked_actions"][0] if scored["ranked_actions"] else None
        entropy = float(scored["ecological_entropy_bits"])
        best_efficiency = float(best["information_efficiency"]) if best else None
        triggers: list[str] = []
        if stop_cfg.enabled and best is not None:
            if (
                stop_cfg.min_information_efficiency is not None
                and best_efficiency is not None
                and best_efficiency < stop_cfg.min_information_efficiency
            ):
                triggers.append("best_information_efficiency_below_threshold")
            if stop_cfg.entropy_target_bits is not None and entropy <= stop_cfg.entropy_target_bits:
                triggers.append("ecological_entropy_at_or_below_target")
        return {
            "enabled": bool(stop_cfg.enabled),
            "recommended": bool(triggers),
            "triggers": triggers,
            "best_information_efficiency": best_efficiency,
            "min_information_efficiency": stop_cfg.min_information_efficiency,
            "ecological_entropy_bits": entropy,
            "entropy_target_bits": stop_cfg.entropy_target_bits,
            "threshold_semantics": "configured diagnostic/design threshold, not a biological standard",
        }

    # ---------------------------------------------------------------- plan
    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        spatial = constraints.get("spatial_belief_state")
        if not isinstance(spatial, SpatialBeliefState):
            raise ValueError(
                "EffortAwareInformationGainPlanner requires constraints['spatial_belief_state']."
            )
        if remaining_budget == 0:
            return MissionAction(
                allocations=(),
                total_cost=0,
                diagnostics={"planner": self.planner_name, "reason": "no_remaining_budget"},
            )

        scored = self.score_actions(graph_state, spatial, remaining_budget)
        stop_eval = self.evaluate_stop(scored)
        base_diagnostics = {
            "planner": self.planner_name,
            "objective": self.config.ranking_objective,
            "ranked_candidates": tuple(scored["ranked_actions"]),
            "ranked_sites": tuple(scored["ranked_sites"]),
            "ecological_entropy_bits": scored["ecological_entropy_bits"],
            "unique_extent_count": scored["unique_extent_count"],
            "stop_evaluation": stop_eval,
            "config": self.config.to_dict(),
            "hidden_truth_used": False,
        }
        if not scored["ranked_actions"]:
            return make_stop_action(reason="no_feasible_action", diagnostics=base_diagnostics)
        if stop_eval["recommended"]:
            return make_stop_action(reason=";".join(stop_eval["triggers"]), diagnostics=base_diagnostics)

        site_row = scored["ranked_sites"][0]
        best = site_row
        return MissionAction(
            allocations=(
                MissionAllocation(site_id=str(best["site_id"]), effort_units=int(best["effort_units"])),
            ),
            total_cost=int(best["effort_units"]),
            diagnostics={
                **base_diagnostics,
                "selected": best,
                "ranked_selected_sites": (site_row,),
                "effort_alternatives": tuple(site_row["effort_alternatives"]),
            },
        )


class StaticEffortAwarePlanner:
    """Same scoring as the adaptive planner, but precommitted at t0.

    On the first call it simulates a no-new-evidence campaign: pick the best
    (site, effort), remove that site from the feasible set (no revisits), reduce
    the budget, repeat until the budget is exhausted or the STOP rule fires.
    Later field evidence is recorded by the loop but cannot change the
    remaining sequence. Comparing this against the adaptive planner isolates
    the value of replanning on evidence, with an identical action space.
    """

    planner_name = "static_effort_aware"

    def __init__(self, config: EffortAwarePlannerConfig | None = None) -> None:
        self._base = EffortAwareInformationGainPlanner(config)
        self._sequence: tuple[MissionAction, ...] | None = None
        self._cursor = 0
        self.committed_actions: tuple[tuple[str, int], ...] = ()

    def precommit(
        self,
        graph_state: GraphState,
        spatial_belief: SpatialBeliefState,
        remaining_budget: int,
    ) -> tuple[MissionAction, ...]:
        mask = list(graph_state.feasibility_mask)
        budget = int(remaining_budget)
        actions: list[MissionAction] = []
        while budget > 0 and any(mask):
            masked = replace(graph_state, feasibility_mask=tuple(mask))
            mission = self._base.plan(masked, budget, {"spatial_belief_state": spatial_belief})
            if is_stop_action(mission) or not mission.allocations:
                actions.append(
                    make_stop_action(
                        reason=str(mission.diagnostics.get("stop_reason", "precommitted_stop")),
                        diagnostics={"planner": self.planner_name, "precommitted_at_t0": True},
                    )
                )
                break
            allocation = mission.allocations[0]
            actions.append(
                MissionAction(
                    allocations=mission.allocations,
                    total_cost=mission.total_cost,
                    diagnostics={
                        **mission.diagnostics,
                        "planner": self.planner_name,
                        "precommitted_at_t0": True,
                    },
                )
            )
            mask[graph_state.node_ids.index(allocation.site_id)] = False
            budget -= allocation.effort_units
        self._sequence = tuple(actions)
        self._cursor = 0
        self.committed_actions = tuple(
            (a.allocations[0].site_id, a.allocations[0].effort_units)
            for a in actions
            if a.allocations
        )
        return self._sequence

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        if self._sequence is None:
            spatial = constraints.get("spatial_belief_state")
            if not isinstance(spatial, SpatialBeliefState):
                raise ValueError("StaticEffortAwarePlanner requires constraints['spatial_belief_state'].")
            self.precommit(graph_state, spatial, remaining_budget)
        assert self._sequence is not None
        if self._cursor >= len(self._sequence):
            return make_stop_action(
                reason="precommitted_sequence_exhausted",
                diagnostics={"planner": self.planner_name, "precommitted_at_t0": True},
            )
        action = self._sequence[self._cursor]
        if action.total_cost > remaining_budget:
            raise RuntimeError("Static effort-aware action exceeds remaining budget.")
        self._cursor += 1
        return action


class FixedHighEffortPlanner:
    """RAMP resource baseline: Frontier site ranking, always the largest feasible effort.

    Not an expert, professional, agency or standard-of-care model. It is the
    "spend the whole budget at high effort" reference the effort-aware planners
    are measured against.
    """

    planner_name = "fixed_high_effort"

    def __init__(self, effort_levels: Sequence[int] = DEFAULT_EFFORT_LEVELS) -> None:
        self._base = FrontierPlanner(max_sites=1, effort_levels=tuple(effort_levels))
        self.effort_levels = tuple(sorted(int(e) for e in effort_levels))

    def rank_candidates(self, graph_state: GraphState) -> tuple[dict[str, Any], ...]:
        return self._base.rank_candidates(graph_state)

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints: Mapping[str, Any],
    ) -> MissionAction:
        mission = self._base.plan(graph_state, remaining_budget=remaining_budget, constraints=constraints)
        return MissionAction(
            allocations=mission.allocations,
            total_cost=mission.total_cost,
            diagnostics={**mission.diagnostics, "planner": self.planner_name, "effort_rule": "largest_feasible"},
        )
