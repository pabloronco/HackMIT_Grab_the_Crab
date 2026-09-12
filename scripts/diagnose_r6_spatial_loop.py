from __future__ import annotations

from adaptive_response import (
    EcologicalHypothesis,
    Edge,
    Environment,
    GeneratedWorld,
    IncidentConfig,
    InformationGainPlanner,
    QHypothesis,
    Site,
    SpatialAdaptiveMissionLoop,
)


class FixedDiagnosticWorld:
    family_id = "diagnostic_fixed_hidden_world"

    def sample(self, context, *, seed: int):
        # Hidden simulator truth: A/B present, C absent. This is never passed to planner.
        occupied = {
            "seed": True,
            "A": True,
            "B": True,
            "C": False,
        }
        return GeneratedWorld(
            family_id=self.family_id,
            occupied_by_site=occupied,
            seed=seed,
            generator_parameters={"diagnostic_only": 1},
        )


def main() -> None:
    sites = [
        Site("seed", 0.0, 0.0, 0.5, {"status": "belief_q_managed_by_spatial_engine"}),
        Site("A", 1.0, 0.0, 0.5, {"status": "belief_q_managed_by_spatial_engine"}),
        Site("B", 2.0, 0.0, 0.5, {"status": "belief_q_managed_by_spatial_engine"}),
        Site("C", 0.0, 1.0, 0.5, {"status": "belief_q_managed_by_spatial_engine"}),
    ]
    config = IncidentConfig(
        sites=sites,
        edges=[
            Edge("seed", "A", 1.0),
            Edge("A", "B", 1.0),
            Edge("seed", "C", 1.0),
        ],
        initial_detection="seed",
        budget=6,
        teams=1,
        protocol="diagnostic_protocol",
        seed=4,
        world_model_id="diagnostic_fixed_hidden_world",
    )

    # Belief worlds: seed confirmed; A/B correlated; C independent.
    worlds = [
        EcologicalHypothesis({"seed": True, "A": False, "B": False, "C": False}),
        EcologicalHypothesis({"seed": True, "A": False, "B": False, "C": True}),
        EcologicalHypothesis({"seed": True, "A": True, "B": True, "C": False}),
        EcologicalHypothesis({"seed": True, "A": True, "B": True, "C": True}),
    ]
    q_support = [QHypothesis(0.10), QHypothesis(0.30)]

    # Simulator q_true is deliberately separate and hidden from inference/planner.
    environment = Environment(
        config,
        world_model=FixedDiagnosticWorld(),
        q_true=0.25,
    )
    planner = InformationGainPlanner(
        effort_per_site=3,
        max_sites=1,
        require_spatial_belief=True,
    )
    loop = SpatialAdaptiveMissionLoop(
        environment,
        planner,
        worlds,
        q_support,
    )
    loop.reset(seed=4)

    print("=== R6 SPATIAL ADAPTIVE LOOP DIAGNOSTIC ===")
    print("STATUS: controlled integration check, not ecological validation.")
    print("Simulator q_true=0.25 is hidden; belief q support={0.10, 0.30}.")
    print("Hidden truth is unavailable until budget exhaustion.")
    print()

    print(f"Initial belief: { {k: round(v, 4) for k, v in loop.current_belief.p_by_site.items()} }")
    first = loop.plan_next()
    print(
        f"Mission 1: site={first.allocations[0].site_id} "
        f"effort={first.allocations[0].effort_units} mode={first.diagnostics['mode']}"
    )

    transition = loop.execute_pending()
    observation = transition.observations.observations[0]
    print(
        f"Field result: site={observation.site_id} effort={observation.effort} "
        f"detection={observation.detection} metadata={observation.metadata}"
    )
    print(
        "Posterior belief: "
        f"{ {k: round(v, 4) for k, v in transition.belief_after.p_by_site.items()} }"
    )
    print(
        "Posterior q: "
        f"{ {k: round(v, 4) for k, v in loop.current_spatial_belief.q_posterior().items()} }"
    )
    assert transition.next_mission is not None
    print(
        f"MISSION UPDATED: next site={transition.next_mission.allocations[0].site_id} "
        f"effort={transition.next_mission.allocations[0].effort_units}"
    )
    print()
    print("INTERPRETATION")
    print("  Observation changes the explicit joint posterior.")
    print("  B can change after surveying A because the declared ecological worlds correlate them.")
    print("  Information Gain replans from the updated posterior, not from hidden truth.")
    print("  Observation metadata does not expose q_true.")
    print("  This is the intended FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED kernel.")


if __name__ == "__main__":
    main()
