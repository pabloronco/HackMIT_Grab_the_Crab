from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..models import IncidentConfig
from ..planners import Planner
from .eval_utils import run_planner_episode
from .incident_sampler import IncidentSamplerConfig, sample_incident

# Planner-agnostic benchmark runner (engineering block, 2026-09-12).
#
# Deliberately produces raw, un-ranked facts only: no "score", no "winner",
# no ecological ranking metric. Per the team's Data-First Validation Freeze
# (docs/DATA_FIRST_VALIDATION_FREEZE.md, Section 8), the primary competitive
# baseline is Information Gain, evaluation metrics are not yet frozen, and the
# RL go/no-go decision is a cross-team call - none of that belongs in this
# module. This only guarantees that whatever planner is handed to it (today:
# FrontierPlanner, RLPlannerAdapter; tomorrow: an InformationGainPlanner drop-
# in once Pablo/Fede's q-model/spatial-belief work lands) runs against
# identical incidents/seeds/budgets/observable information, and reports what
# happened in a structured, comparison-ready way.


@dataclass(frozen=True)
class BenchmarkCase:
    """One held-out incident + the seed it should be run with. Frozen once
    built (`make_benchmark_cases`), so every planner in a suite sees exactly
    the same case set - the freeze's "same incidents, same seeds, same
    budget, same information" requirement.
    """

    label: str
    incident: IncidentConfig
    seed: int


@dataclass(frozen=True)
class BenchmarkRow:
    """One (planner, case) result. Deliberately just the raw `EpisodeMetrics`
    fields plus identifying columns - no derived ranking metric. Compute
    whatever comparison metric is eventually frozen from these rows, don't
    bake one in here.
    """

    planner_name: str
    case_label: str
    seed: int
    num_sites: int
    budget: int
    num_rounds: int
    detections_found: int
    effort_spent: int
    occupied_sites_total: int
    occupied_sites_missed: int
    wall_clock_seconds: float


def make_benchmark_cases(
    *,
    num_cases: int,
    seed: int,
    sampler_config: IncidentSamplerConfig | None = None,
) -> list[BenchmarkCase]:
    """Build a frozen, reproducible set of held-out incidents.

    `seed` should be independent of whatever seed(s) are used to train any
    planner being benchmarked, the same discipline already used for
    `train_gnn_policy.py`'s eval_rng.
    """

    cfg = sampler_config or IncidentSamplerConfig()
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(num_cases):
        incident = sample_incident(rng, cfg)
        case_seed = int(rng.integers(0, 2**31 - 1))
        cases.append(BenchmarkCase(label=f"case_{i:03d}", incident=incident, seed=case_seed))
    return cases


def run_benchmark_suite(
    planners: dict[str, Planner], cases: list[BenchmarkCase]
) -> list[BenchmarkRow]:
    """Run every planner against every case. O(len(planners) * len(cases))
    full episodes. Each (planner, case) pair is fully independent - a
    crash/exception in one does not need to be tolerated silently here;
    callers wanting partial-results-on-failure should wrap this themselves.
    """

    rows: list[BenchmarkRow] = []
    for planner_name, planner in planners.items():
        for case in cases:
            start = time.perf_counter()
            metrics = run_planner_episode(planner, case.incident, seed=case.seed)
            elapsed = time.perf_counter() - start

            rows.append(
                BenchmarkRow(
                    planner_name=planner_name,
                    case_label=case.label,
                    seed=case.seed,
                    num_sites=len(case.incident.sites),
                    budget=case.incident.budget,
                    num_rounds=metrics.num_rounds,
                    detections_found=metrics.detections_found,
                    effort_spent=metrics.effort_spent,
                    occupied_sites_total=metrics.occupied_sites_total,
                    occupied_sites_missed=metrics.occupied_sites_missed,
                    wall_clock_seconds=elapsed,
                )
            )
    return rows


def write_benchmark_csv(rows: list[BenchmarkRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
