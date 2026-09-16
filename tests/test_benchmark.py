"""Engineering block: planner-agnostic benchmark runner. Requires torch (the
RL planner side does), skipped otherwise like the rest of rl/'s tests."""

import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_response import FrontierPlanner  # noqa: E402
from adaptive_response.rl import (  # noqa: E402
    GNNActorCritic,
    IncidentSamplerConfig,
    RLPlannerAdapter,
    RoundPolicy,
)
from adaptive_response.rl.benchmark import (  # noqa: E402
    BenchmarkCase,
    make_benchmark_cases,
    run_benchmark_suite,
    write_benchmark_csv,
)


def test_make_benchmark_cases_is_reproducible_for_a_fixed_seed() -> None:
    cases_a = make_benchmark_cases(num_cases=5, seed=999)
    cases_b = make_benchmark_cases(num_cases=5, seed=999)

    assert [c.label for c in cases_a] == [c.label for c in cases_b]
    assert [c.seed for c in cases_a] == [c.seed for c in cases_b]
    for a, b in zip(cases_a, cases_b):
        assert [s.id for s in a.incident.sites] == [s.id for s in b.incident.sites]
        assert a.incident.budget == b.incident.budget


def test_different_seeds_give_different_case_sets() -> None:
    cases_a = make_benchmark_cases(num_cases=5, seed=1)
    cases_b = make_benchmark_cases(num_cases=5, seed=2)
    assert [c.incident.budget for c in cases_a] != [c.incident.budget for c in cases_b]


def test_run_benchmark_suite_runs_every_planner_on_every_case() -> None:
    cases = make_benchmark_cases(
        num_cases=3, seed=7, sampler_config=IncidentSamplerConfig(min_sites=10, max_sites=10)
    )
    torch.manual_seed(0)
    rl_policy = RoundPolicy(GNNActorCritic(hidden_dim=16, num_layers=1), hidden_dim=16)
    planners = {"frontier": FrontierPlanner(), "rl": RLPlannerAdapter(rl_policy)}

    rows = run_benchmark_suite(planners, cases)

    assert len(rows) == len(planners) * len(cases)
    seen = {(r.planner_name, r.case_label) for r in rows}
    assert seen == {(p, c.label) for p in planners for c in cases}
    for row in rows:
        assert row.effort_spent == row.budget  # episodes run to exact budget exhaustion
        assert row.wall_clock_seconds >= 0
        assert 0 <= row.occupied_sites_missed <= row.occupied_sites_total


def test_same_planner_and_case_are_identical_across_two_runs() -> None:
    """Determinism check: FrontierPlanner has no randomness at all, so two
    runs on the same case must produce byte-identical metrics."""

    case = make_benchmark_cases(num_cases=1, seed=42)[0]
    planners = {"frontier": FrontierPlanner()}

    rows_a = run_benchmark_suite(planners, [case])
    rows_b = run_benchmark_suite(planners, [case])

    assert rows_a[0].detections_found == rows_b[0].detections_found
    assert rows_a[0].num_rounds == rows_b[0].num_rounds
    assert rows_a[0].occupied_sites_missed == rows_b[0].occupied_sites_missed


def test_write_benchmark_csv_round_trips(tmp_path) -> None:
    cases = make_benchmark_cases(num_cases=2, seed=3)
    rows = run_benchmark_suite({"frontier": FrontierPlanner()}, cases)

    csv_path = tmp_path / "benchmark.csv"
    write_benchmark_csv(rows, csv_path)

    with csv_path.open() as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == len(rows)
    assert read_rows[0]["planner_name"] == "frontier"
    assert int(read_rows[0]["effort_spent"]) == int(read_rows[0]["budget"])


def test_benchmark_row_has_no_ranking_metric_baked_in() -> None:
    """Structural guard for the freeze's 'do not freeze the final ecological
    ranking metrics yet' requirement: BenchmarkRow must stay a flat set of
    raw facts, not grow a 'score'/'rank'/'winner' field."""

    from dataclasses import fields

    from adaptive_response.rl.benchmark import BenchmarkRow

    field_names = {f.name for f in fields(BenchmarkRow)}
    forbidden = {"score", "rank", "winner", "ranking", "is_best"}
    assert field_names.isdisjoint(forbidden)


def test_benchmark_case_is_a_planner_agnostic_placeholder_for_future_ig() -> None:
    """No test can prove an Information Gain planner will work before it
    exists, but this documents/pins the actual requirement: anything
    satisfying the shared `Planner` protocol (plan(graph_state,
    remaining_budget, constraints) -> MissionAction) drops into
    run_benchmark_suite with zero changes, RL or not."""

    class DummyFutureInformationGainPlanner:
        def plan(self, graph_state, remaining_budget, constraints):
            return FrontierPlanner().plan(graph_state, remaining_budget, constraints)

    case = make_benchmark_cases(num_cases=1, seed=5)[0]
    rows = run_benchmark_suite({"future_ig_stub": DummyFutureInformationGainPlanner()}, [case])
    assert len(rows) == 1
