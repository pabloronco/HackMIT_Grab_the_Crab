import json

import pytest

from adaptive_response import Edge, FrontierPlanner, IncidentConfig, Site
from adaptive_response.event_stream import RealGraphEpisode, to_json_line
from adaptive_response.spatial_belief import EcologicalHypothesis, QHypothesis

SITES = [f"s{i}" for i in range(6)]


def _incident(budget: int = 12) -> IncidentConfig:
    sites = [Site(id=s, x=float(i), y=0.0, habitat_score=0.5, q_model=0.25) for i, s in enumerate(SITES)]
    edges = [Edge(src=SITES[i], dst=SITES[i + 1], distance=1.0, connectivity_weight=1.0) for i in range(5)]
    return IncidentConfig(
        sites=sites, edges=edges, initial_detection="s2", budget=budget, teams=1,
        protocol="binary_detection", seed=7, world_model_id="toy_graph_cluster_m1",
    )


def _hypotheses():
    def hyp(present, weight, label):
        return EcologicalHypothesis(
            presence_by_site={s: (s in present) for s in SITES}, prior_weight=weight, label=label
        )
    # An "everything occupied" member keeps every observation possible, so the
    # belief engine's finite-ensemble support failure cannot fire here.
    eco = (
        hyp({"s2"}, 3.0, "only_source"),
        hyp({"s1", "s2", "s3"}, 2.0, "neighbours"),
        hyp({"s0", "s1", "s2", "s3", "s4"}, 1.0, "spread"),
        hyp(set(SITES), 0.5, "everywhere"),
    )
    q = (QHypothesis(q=0.15, prior_weight=1.0, label="low"), QHypothesis(q=0.35, prior_weight=1.0, label="high"))
    return eco, q


def _episode(budget: int = 12, max_rounds: int = 4) -> RealGraphEpisode:
    eco, q = _hypotheses()
    return RealGraphEpisode(
        _incident(budget), planner=FrontierPlanner(effort_levels=(1, 3, 6), max_sites=1),
        ecological_hypotheses=eco, q_hypotheses=q, seed=11, max_rounds=max_rounds,
        label="toy_case", group="unit",
    )


def test_stream_shape_and_ordering() -> None:
    events = list(_episode().run())

    assert events[0]["type"] == "graph"
    assert events[-1]["type"] == "reveal"
    rounds = [e for e in events if e["type"] == "round"]
    assert [r["round"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert rounds[-1]["done"] is True and rounds[-1]["ended_by"] in {"budget", "horizon"}
    assert all(r["done"] is False and r["ended_by"] is None for r in rounds[:-1])
    for line in map(to_json_line, events):
        json.loads(line)  # every event is strict JSON (no NaN/inf)


def test_hidden_truth_only_appears_in_reveal() -> None:
    events = list(_episode().run())
    for event in events[:-1]:
        assert "occupied_by_site" not in json.dumps(event)
    assert set(events[-1]["occupied_by_site"]) == set(SITES)


def test_round_event_is_consistent_with_the_graph() -> None:
    events = list(_episode().run())
    graph, rounds = events[0], [e for e in events if e["type"] == "round"]
    site_ids = {s["id"] for s in graph["sites"]}

    assert set(graph["initial_belief"]) == site_ids
    assert graph["initial_detection"] == "s2" and graph["initial_belief"]["s2"] == 1.0
    budgets = [r["remaining_budget"] for r in rounds]
    assert budgets == sorted(budgets, reverse=True)
    for r in rounds:
        assert set(r["belief_after"]) == site_ids
        assert set(r["belief_delta"]) <= site_ids
        assert all(0.0 <= p <= 1.0 for p in r["belief_after"].values())
        assert r["surprise"]["level"] in {"none", "high", "impossible"}
        assert (r["alert_text"] is None) == (r["surprise"]["level"] == "none")
        assert r["briefing"]  # free deterministic text is always present
        for a in r["mission"]:
            assert a["site_id"] in site_ids and a["effort_units"] in {1, 3, 6}


def test_horizon_ends_the_episode_and_drops_the_unexecuted_next_mission() -> None:
    episode = _episode(budget=60, max_rounds=2)
    events = list(episode.run())
    rounds = [e for e in events if e["type"] == "round"]

    assert len(rounds) == 2
    assert rounds[-1]["ended_by"] == "horizon"
    assert rounds[-1]["next_mission"] is None
    assert rounds[-1]["remaining_budget"] > 0


def test_reveal_metrics_match_the_benchmark_definition() -> None:
    from adaptive_response.rl.spatial_metrics import compute_spatial_primary_metrics

    episode = _episode()
    reveal = list(episode.run())[-1]
    last = episode.rounds[-1].transition
    hidden = episode._hidden_world  # test-only peek, after the reveal gate
    expected = compute_spatial_primary_metrics(
        sites=last.public_state_after.sites, hidden_world=hidden, final_belief=last.belief_after
    )

    assert reveal["occupied_sites_total"] == expected.occupied_sites_total
    assert reveal["occupied_sites_missed"] == expected.occupied_sites_missed
    assert reveal["missed_occupied_fraction"] == pytest.approx(expected.missed_occupied_fraction)
    assert reveal["occupied_site_coverage"] == pytest.approx(expected.occupied_site_coverage)
    assert reveal["final_global_uncertainty"] == pytest.approx(expected.final_global_uncertainty)


def test_lifecycle_guards() -> None:
    episode = _episode()
    with pytest.raises(RuntimeError):
        episode.step()  # before reset
    episode.reset()
    with pytest.raises(RuntimeError):
        episode.reset()  # twice
    with pytest.raises(RuntimeError):
        episode.reveal()  # before completion
    while not episode.done:
        episode.step()
    with pytest.raises(RuntimeError):
        episode.step()  # after completion
    first, second = episode.reveal(), episode.reveal()
    assert first == second  # idempotent


def test_alert_bits_is_threaded_through() -> None:
    episode = _episode()
    episode._alert_bits = 0.01  # anything non-trivial is now an alert
    rounds = [e for e in episode.run() if e["type"] == "round"]
    assert all(r["surprise"]["alert_bits"] == 0.01 for r in rounds)
    assert any(r["surprise"]["level"] != "none" for r in rounds)
