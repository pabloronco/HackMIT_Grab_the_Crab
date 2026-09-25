from adaptive_response import Edge, FrontierPlanner, IncidentConfig, Site
from adaptive_response.copilot import (
    EpisodeTools,
    PlannerCopilot,
    TOOL_SPECS,
    deciding_criterion,
    frontier_ranking,
    site_feature_table,
)
from adaptive_response.event_stream import RealGraphEpisode
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
    eco = (
        hyp({"s2"}, 3.0, "only_source"),
        hyp({"s1", "s2", "s3"}, 2.0, "neighbours"),
        hyp({"s0", "s1", "s2", "s3", "s4"}, 1.0, "spread"),
        hyp(set(SITES), 0.5, "everywhere"),
    )
    q = (QHypothesis(q=0.15, prior_weight=1.0, label="low"), QHypothesis(q=0.35, prior_weight=1.0, label="high"))
    return eco, q


def _finished_episode() -> RealGraphEpisode:
    eco, q = _hypotheses()
    episode = RealGraphEpisode(
        _incident(), planner=FrontierPlanner(effort_levels=(1, 3, 6), max_sites=1),
        ecological_hypotheses=eco, q_hypotheses=q, seed=11, max_rounds=4, label="toy_case",
    )
    for _ in episode.run():
        pass
    return episode


def _frontier_key(row):
    return (-int(row["frontier"]), -row["belief"], -row["uncertainty"], row["site_id"])


def test_ranking_is_the_planners_own_ordering_over_all_feasible_sites() -> None:
    episode = _finished_episode()
    graph = episode.rounds[0].transition.graph_before
    ranking = frontier_ranking(graph)
    table = site_feature_table(graph)

    feasible = [sid for sid, row in table.items() if row["feasible"]]
    assert [r["site_id"] for r in ranking] == sorted(feasible, key=lambda sid: _frontier_key({**table[sid], "site_id": sid}))
    assert [r["rank"] for r in ranking] == list(range(1, len(feasible) + 1))
    # The round-1 mission went to the top-ranked site.
    assert episode.rounds[0].transition.mission.allocations[0].site_id == ranking[0]["site_id"]


def test_compare_names_the_criterion_that_actually_separates_the_sites() -> None:
    tools = EpisodeTools(_finished_episode())
    ranking = tools.rank_sites(round=1, top=100)["ranking"]
    top, second = ranking[0], ranking[1]

    result = tools.compare_sites(top["site_id"], second["site_id"], round=1)
    assert result["ranks_higher"] == top["site_id"]
    assert result["deciding_criterion"] == deciding_criterion(top, second)
    # Symmetric.
    flipped = tools.compare_sites(second["site_id"], top["site_id"], round=1)
    assert flipped["ranks_higher"] == top["site_id"]


def test_tools_return_errors_instead_of_raising() -> None:
    tools = EpisodeTools(_finished_episode())
    assert "error" in tools.get_site("nope")
    assert "error" in tools.get_round(0)
    assert "error" in tools.get_round(99)
    assert "error" in tools.rank_sites(round=99)
    assert "error" in tools.compare_sites("s0", "ghost")
    assert "error" in tools.dispatch("_graph_for", {"round_": None})  # private helpers are not tools
    assert "error" in tools.dispatch("get_round", {"bogus": 1})


def test_round_and_status_tools_reflect_the_recorded_episode() -> None:
    episode = _finished_episode()
    tools = EpisodeTools(episode)
    first = episode.rounds[0].transition

    round1 = tools.get_round(1)
    assert round1["mission"] == [
        {"site_id": a.site_id, "effort_units": a.effort_units} for a in first.mission.allocations
    ]
    assert round1["remaining_budget_after"] == first.public_state_after.remaining_budget
    assert round1["mismatch_alert"]["level"] == episode.rounds[0].alert.level

    status = tools.get_mission_status()
    assert status["rounds_completed"] == len(episode.rounds)
    assert status["episode_complete"] is True
    assert status["initial_detection_site"] == "s2"
    assert status["alerts_triggered_in_rounds"] == [r.index for r in episode.rounds if r.alert.triggered]

    site = tools.get_site("s2")
    assert site["belief_trajectory"][0] == {"round": 0, "belief": 1.0}
    assert len(site["belief_trajectory"]) == len(episode.rounds) + 1
    assert set(tools.get_alerts()) == {"alerts", "note"}


def test_tool_specs_match_the_implementation() -> None:
    tools = EpisodeTools(_finished_episode())
    for spec in TOOL_SPECS:
        name = spec["function"]["name"]
        assert callable(getattr(tools, name))
        assert spec["function"]["parameters"]["additionalProperties"] is False


def test_copilot_needs_no_network_until_asked() -> None:
    copilot = PlannerCopilot(_finished_episode(), api_key="not-a-real-key")
    assert copilot._client is None
    assert copilot.tools.get_mission_status()["rounds_completed"] >= 1
