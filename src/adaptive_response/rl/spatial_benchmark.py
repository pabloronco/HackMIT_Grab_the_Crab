from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..environment import Environment
from ..models import IncidentConfig, MissionAction
from ..planners import Planner
from ..spatial_belief import EcologicalHypothesis, QHypothesis
from ..spatial_mission_loop import SpatialAdaptiveMissionLoop
from ..world_models import WorldModel
from .site_effort_policy import SiteEffortRoundPolicy
from .spatial_metrics import compute_spatial_primary_metrics

# R7 planner-agnostic benchmark (docs/DEMU_HANDOFF_R7.md "Required
# learned-policy benchmark"): Frontier / spatial-joint Information Gain / RL
# on exactly the same cases, seeds, budget, horizon and observable
# information. Deliberately no score/rank/winner field - same discipline as
# the older engineering-block benchmark.py: reporting is the frozen
# planner-independent primary metrics, not a reward and not a single number.


@dataclass(frozen=True)
class SpatialBenchmarkCase:
    label: str
    incident: IncidentConfig
    world_model: WorldModel
    q_true: float
    ecological_hypotheses: tuple[EcologicalHypothesis, ...]
    q_hypotheses: tuple[QHypothesis, ...]
    max_rounds: int | None
    seed: int
    group: str  # e.g. "id_test", "ood_model_E", "ood_q_low", "ood_q_high"


@dataclass(frozen=True)
class SpatialBenchmarkRow:
    planner_name: str
    case_label: str
    group: str
    seed: int
    num_sites: int
    budget: int
    max_rounds: int
    num_rounds: int
    detections_found: int
    effort_spent: int
    occupied_sites_total: int
    occupied_sites_missed: int
    missed_occupied_fraction: float
    occupied_site_coverage: float
    final_global_uncertainty: float
    wall_clock_seconds: float


class RLSpatialPlannerAdapter:
    """Expose a trained `SiteEffortRoundPolicy` through the shared `Planner`
    protocol for benchmark evaluation. Always deterministic (argmax): the
    benchmark measures committed behavior, not an exploration sample."""

    def __init__(self, policy: SiteEffortRoundPolicy) -> None:
        self._policy = policy

    def plan(self, graph_state, remaining_budget, constraints) -> MissionAction:
        del constraints
        return self._policy.act(graph_state, remaining_budget, deterministic=True).mission


def run_spatial_planner_case(planner: Planner, case: SpatialBenchmarkCase) -> SpatialBenchmarkRow:
    start = time.time()
    environment = Environment(case.incident, world_model=case.world_model, q_true=case.q_true)
    loop = SpatialAdaptiveMissionLoop(
        environment, planner, case.ecological_hypotheses, case.q_hypotheses
    )
    loop.reset(seed=case.seed)

    num_rounds = 0
    detections_found = 0
    effort_spent = 0

    last_transition = None
    while True:
        transition = loop.run_round()
        if transition is None:
            # Planner STOP before this round: campaign complete with budget left.
            assert last_transition is not None, "planner stopped before any round"
            transition = last_transition
            hidden_world = loop.reveal()
            metrics = compute_spatial_primary_metrics(
                sites=transition.public_state_after.sites,
                hidden_world=hidden_world,
                final_belief=transition.belief_after,
            )
            return SpatialBenchmarkRow(
                planner_name=getattr(planner, "planner_name", type(planner).__name__),
                case_label=case.label, group=case.group, seed=case.seed,
                num_sites=len(case.incident.sites), budget=case.incident.budget,
                max_rounds=case.max_rounds if case.max_rounds is not None else 0,
                num_rounds=num_rounds, detections_found=detections_found, effort_spent=effort_spent,
                occupied_sites_total=metrics.occupied_sites_total,
                occupied_sites_missed=metrics.occupied_sites_missed,
                missed_occupied_fraction=metrics.missed_occupied_fraction,
                occupied_site_coverage=metrics.occupied_site_coverage,
                final_global_uncertainty=metrics.final_global_uncertainty,
                wall_clock_seconds=time.time() - start,
            )
        last_transition = transition
        num_rounds += 1
        detections_found += int(transition.simulator_metrics["detections"])
        effort_spent += int(transition.simulator_metrics["effort_spent"])

        horizon_reached = case.max_rounds is not None and num_rounds >= case.max_rounds
        planner_stopped = bool(getattr(transition, "planner_stopped", False))
        episode_over = transition.done or horizon_reached or planner_stopped
        if episode_over and not transition.done and not planner_stopped:
            loop.force_complete()

        if episode_over:
            hidden_world = loop.reveal()
            metrics = compute_spatial_primary_metrics(
                sites=transition.public_state_after.sites,
                hidden_world=hidden_world,
                final_belief=transition.belief_after,
            )
            return SpatialBenchmarkRow(
                planner_name=getattr(planner, "planner_name", type(planner).__name__),
                case_label=case.label,
                group=case.group,
                seed=case.seed,
                num_sites=len(case.incident.sites),
                budget=case.incident.budget,
                max_rounds=case.max_rounds if case.max_rounds is not None else 0,
                num_rounds=num_rounds,
                detections_found=detections_found,
                effort_spent=effort_spent,
                occupied_sites_total=metrics.occupied_sites_total,
                occupied_sites_missed=metrics.occupied_sites_missed,
                missed_occupied_fraction=metrics.missed_occupied_fraction,
                occupied_site_coverage=metrics.occupied_site_coverage,
                final_global_uncertainty=metrics.final_global_uncertainty,
                wall_clock_seconds=time.time() - start,
            )


def run_spatial_benchmark_suite(
    planners: dict[str, Planner],
    cases: Sequence[SpatialBenchmarkCase],
) -> list[SpatialBenchmarkRow]:
    rows: list[SpatialBenchmarkRow] = []
    for case in cases:
        for planner_name, planner in planners.items():
            row = run_spatial_planner_case(planner, case)
            rows.append(
                SpatialBenchmarkRow(**{**row.__dict__, "planner_name": planner_name})
            )
    return rows


def write_spatial_benchmark_csv(rows: Sequence[SpatialBenchmarkRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SpatialBenchmarkRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
