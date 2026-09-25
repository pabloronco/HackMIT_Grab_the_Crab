"""End-to-end sanity run (see `run_end_to_end_sanity.py`) plus an OpenAI-
generated plain-language field briefing after each round.

Requires the `llm` extra (`pip install -e ".[llm]"`) and `OPENAI_API_KEY` in
the environment. If either is missing, or the API call fails, this prints
the raw round transition instead of crashing the demo.

Usage: OPENAI_API_KEY=... python scripts/run_incident_with_briefing.py
"""

from adaptive_response import (
    AdaptiveMissionLoop,
    Edge,
    Environment,
    FrontierPlanner,
    IncidentConfig,
    Site,
)
from adaptive_response.narrate import narrate_round, template_briefing


def build_incident() -> IncidentConfig:
    sites = [
        Site(id=f"site_{i:02d}", x=float(i), y=0.0, habitat_score=0.5, q_model=0.25)
        for i in range(7)
    ]
    edges = [
        Edge(src=f"site_{i:02d}", dst=f"site_{i + 1:02d}", distance=1.0, connectivity_weight=1.0)
        for i in range(6)
    ]
    return IncidentConfig(
        sites=sites,
        edges=edges,
        initial_detection="site_03",
        budget=6,
        teams=2,
        protocol="binary_detection",
        seed=20260910,
        world_model_id="toy_graph_cluster_m1",
    )


def build_prior() -> dict[str, float]:
    prior = {f"site_{i:02d}": 0.20 for i in range(7)}
    prior["site_02"] = 0.60
    prior["site_04"] = 0.70
    return prior


def print_briefing(round_) -> None:
    print("--- FIELD BRIEFING ---")
    try:
        result = narrate_round(round_)
        print(result.text)
        tier = "escalated to mini" if result.escalated else "nano (default tier)"
        print(f"[{tier}, ${result.total_cost_dollars:.6f}]")
    except Exception as exc:  # network/API failure must not kill a live demo
        print(f"[LLM briefing unavailable ({exc}), falling back to free template]")
        print(template_briefing(round_))
    print()


def main() -> None:
    loop = AdaptiveMissionLoop(
        Environment(build_incident()),
        FrontierPlanner(effort_per_site=3, max_sites=1),
        build_prior(),
    )
    loop.reset()

    print("=== ADAPTIVE FIRST-RESPONSE / BRIEFING DEMO ===\n")

    round_1 = loop.run_round()
    print_briefing(round_1)

    if not round_1.done:
        round_2 = loop.run_round()
        print_briefing(round_2)


if __name__ == "__main__":
    main()
