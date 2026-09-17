from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Protocol, Sequence

from .environment import Environment
from .models import (
    BeliefState,
    GraphState,
    HiddenWorld,
    IncidentConfig,
    MissionAction,
    PublicState,
)
from .planners import FrontierPlanner
from .spatial_belief import EcologicalHypothesis, QHypothesis
from .spatial_mission_loop import SpatialAdaptiveMissionLoop
from .world_models import WorldModel


R10_BUDGET = 18
R10_MISSIONS = 3
R10_EFFORT = 6
R10_EFFORT_LEVELS = (1, 3, 6)


class R10CaseLike(Protocol):
    label: str
    incident: IncidentConfig
    world_model: WorldModel
    q_true: float
    ecological_hypotheses: Sequence[EcologicalHypothesis]
    q_hypotheses: Sequence[QHypothesis]
    seed: int
    group: str


@dataclass(frozen=True)
class R10ArmResult:
    arm: str
    mission_sites: tuple[str, ...]
    mission_detections: tuple[bool, ...]
    detections_found: int
    effort_spent: int
    occupied_sites_total: int
    occupied_sites_missed: int
    missed_occupied_fraction: float
    occupied_site_coverage: float
    final_global_uncertainty: float


@dataclass(frozen=True)
class R10CaseComparison:
    case_label: str
    group: str
    static: R10ArmResult
    adaptive: R10ArmResult
    mission_2_divergence: bool
    mission_3_divergence: bool

    @property
    def delta_missed_occupied_fraction(self) -> float:
        return (
            self.static.missed_occupied_fraction
            - self.adaptive.missed_occupied_fraction
        )

    @property
    def overall_mission_divergence_rate(self) -> float:
        return (
            int(self.mission_2_divergence)
            + int(self.mission_3_divergence)
        ) / 2.0

    @property
    def outcome_category(self) -> str:
        delta = self.delta_missed_occupied_fraction
        if delta > 0:
            return "adaptive_better"
        if delta < 0:
            return "adaptive_worse"
        return "equal"


def paired_uniform(case_id: str, site_id: str) -> float:
    key = f"r10-v1|{case_id}|{site_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64)


def paired_uniforms_for_case(
    case_id: str,
    site_ids: Sequence[str],
) -> dict[str, float]:
    return {site_id: paired_uniform(case_id, site_id) for site_id in site_ids}


def _masked_graph_state(
    graph_state: GraphState,
    blocked_sites: set[str],
) -> GraphState:
    mask = tuple(
        feasible and site_id not in blocked_sites
        for site_id, feasible in zip(
            graph_state.node_ids,
            graph_state.feasibility_mask,
        )
    )
    return replace(graph_state, feasibility_mask=mask)


def _validate_r10_action(action: MissionAction) -> str:
    if len(action.allocations) != 1:
        raise RuntimeError("R10 requires exactly one survey site per mission.")
    allocation = action.allocations[0]
    if action.total_cost != R10_EFFORT:
        raise RuntimeError("R10 requires mission total_cost == 6.")
    if allocation.effort_units != R10_EFFORT:
        raise RuntimeError("R10 requires effort_units == 6.")
    return allocation.site_id


def _frontier_planner() -> FrontierPlanner:
    return FrontierPlanner(
        max_sites=1,
        effort_levels=R10_EFFORT_LEVELS,
    )


class StaticFrontierPlanner:
    """Frontier plan fully precommitted from the initial t0 state."""

    planner_name = "static_frontier"

    def __init__(
        self,
        *,
        initially_blocked_sites: Sequence[str] = (),
    ) -> None:
        self._planner = _frontier_planner()
        self._initially_blocked = frozenset(initially_blocked_sites)
        self._precommitted: tuple[MissionAction, ...] | None = None
        self._cursor = 0

    @property
    def precommitted_sites(self) -> tuple[str, ...]:
        if self._precommitted is None:
            return ()
        return tuple(
            action.allocations[0].site_id
            for action in self._precommitted
        )

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints,
    ) -> MissionAction:
        if self._precommitted is None:
            actions: list[MissionAction] = []
            blocked: set[str] = set(self._initially_blocked)
            budget = remaining_budget

            for _ in range(R10_MISSIONS):
                masked = _masked_graph_state(graph_state, blocked)
                action = self._planner.plan(
                    masked,
                    remaining_budget=budget,
                    constraints=constraints,
                )
                site_id = _validate_r10_action(action)
                if site_id in blocked:
                    raise RuntimeError("Static R10 plan attempted a site revisit.")
                actions.append(action)
                blocked.add(site_id)
                budget -= action.total_cost

            self._precommitted = tuple(actions)

        if self._cursor >= len(self._precommitted):
            raise RuntimeError("Static R10 planner exhausted its precommitted missions.")

        action = self._precommitted[self._cursor]
        if action.total_cost > remaining_budget:
            raise RuntimeError("Static precommitted action exceeds remaining budget.")

        self._cursor += 1
        return action


class AdaptiveFrontierPlanner:
    """Same Frontier rule, but rerun after every evidence update."""

    planner_name = "adaptive_frontier"

    def __init__(
        self,
        *,
        initially_blocked_sites: Sequence[str] = (),
    ) -> None:
        self._planner = _frontier_planner()
        self._initially_blocked = frozenset(initially_blocked_sites)
        self._selected: list[str] = []

    @property
    def selected_sites(self) -> tuple[str, ...]:
        return tuple(self._selected)

    def plan(
        self,
        graph_state: GraphState,
        remaining_budget: int,
        constraints,
    ) -> MissionAction:
        blocked = set(self._initially_blocked) | set(self._selected)
        masked = _masked_graph_state(graph_state, blocked)
        action = self._planner.plan(
            masked,
            remaining_budget=remaining_budget,
            constraints=constraints,
        )
        site_id = _validate_r10_action(action)

        if site_id in self._selected:
            raise RuntimeError("Adaptive R10 plan attempted a site revisit.")

        self._selected.append(site_id)
        return action


def _compute_metrics(
    public_state: PublicState,
    hidden_world: HiddenWorld,
    final_belief: BeliefState,
) -> tuple[int, int, float, float, float]:
    # Intentionally matches the frozen R8 metric semantics.
    occupied_total = sum(
        1
        for site in public_state.sites
        if hidden_world.occupied_by_site.get(site.id, False)
    )
    occupied_missed = sum(
        1
        for site in public_state.sites
        if hidden_world.occupied_by_site.get(site.id, False)
        and site.detections == 0
    )

    missed_fraction = (
        occupied_missed / occupied_total
        if occupied_total > 0
        else 0.0
    )
    coverage = 1.0 - missed_fraction

    uncertainty = (
        fmean(final_belief.uncertainty_by_site.values())
        if final_belief.uncertainty_by_site
        else 0.0
    )

    return (
        occupied_total,
        occupied_missed,
        missed_fraction,
        coverage,
        uncertainty,
    )


def _run_arm(
    case: R10CaseLike,
    planner,
    *,
    arm_name: str,
) -> R10ArmResult:
    site_ids = tuple(site.id for site in case.incident.sites)
    uniforms = paired_uniforms_for_case(case.label, site_ids)

    environment = Environment(
        case.incident,
        world_model=case.world_model,
        q_true=case.q_true,
        observation_uniform_by_site=uniforms,
    )
    loop = SpatialAdaptiveMissionLoop(
        environment,
        planner,
        case.ecological_hypotheses,
        case.q_hypotheses,
    )
    loop.reset(seed=case.seed)

    mission_sites: list[str] = []
    mission_detections: list[bool] = []
    detections_found = 0
    effort_spent = 0
    final_transition = None

    for _ in range(R10_MISSIONS):
        transition = loop.run_round()
        final_transition = transition

        site_id = _validate_r10_action(transition.mission)
        mission_sites.append(site_id)

        if len(transition.observations.observations) != 1:
            raise RuntimeError("R10 expects exactly one field observation per mission.")

        detection = bool(transition.observations.observations[0].detection)
        mission_detections.append(detection)
        detections_found += int(detection)
        effort_spent += int(transition.simulator_metrics["effort_spent"])

    assert final_transition is not None

    if not final_transition.done:
        raise RuntimeError(
            "R10 must exhaust budget 18 after exactly three effort-6 missions."
        )

    if len(set(mission_sites)) != R10_MISSIONS:
        raise RuntimeError("R10 requires three distinct survey sites.")

    hidden_world = loop.reveal()

    (
        occupied_total,
        occupied_missed,
        missed_fraction,
        coverage,
        final_uncertainty,
    ) = _compute_metrics(
        final_transition.public_state_after,
        hidden_world,
        final_transition.belief_after,
    )

    return R10ArmResult(
        arm=arm_name,
        mission_sites=tuple(mission_sites),
        mission_detections=tuple(mission_detections),
        detections_found=detections_found,
        effort_spent=effort_spent,
        occupied_sites_total=occupied_total,
        occupied_sites_missed=occupied_missed,
        missed_occupied_fraction=missed_fraction,
        occupied_site_coverage=coverage,
        final_global_uncertainty=final_uncertainty,
    )


def run_r10_case(case: R10CaseLike) -> R10CaseComparison:
    if case.incident.budget != R10_BUDGET:
        raise ValueError("R10 requires incident budget == 18.")

    if len(case.incident.sites) < R10_MISSIONS:
        raise ValueError("R10 requires at least three survey sites.")

    static = _run_arm(
        case,
        StaticFrontierPlanner(
            initially_blocked_sites=(case.incident.initial_detection,),
        ),
        arm_name="static_frontier",
    )
    adaptive = _run_arm(
        case,
        AdaptiveFrontierPlanner(
            initially_blocked_sites=(case.incident.initial_detection,),
        ),
        arm_name="adaptive_frontier",
    )

    if static.mission_sites[0] != adaptive.mission_sites[0]:
        raise RuntimeError(
            "R10 fairness violation: first mission must be identical."
        )

    if static.mission_detections[0] != adaptive.mission_detections[0]:
        raise RuntimeError(
            "R10 paired-randomness violation on the shared first mission."
        )

    return R10CaseComparison(
        case_label=case.label,
        group=case.group,
        static=static,
        adaptive=adaptive,
        mission_2_divergence=(
            static.mission_sites[1] != adaptive.mission_sites[1]
        ),
        mission_3_divergence=(
            static.mission_sites[2] != adaptive.mission_sites[2]
        ),
    )
