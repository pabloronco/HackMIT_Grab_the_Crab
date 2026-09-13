from __future__ import annotations

from adaptive_response import (
    EcologicalHypothesis,
    GraphState,
    InformationGainPlanner,
    QHypothesis,
    SpatialBeliefEngine,
)
from adaptive_response.graph_state import NODE_FEATURE_NAMES


def graph_from_spatial(spatial) -> GraphState:
    p = spatial.p_by_site()
    u = spatial.uncertainty_by_site()
    node_ids = tuple(sorted(p))
    rows = []
    for site_id in node_ids:
        values = {
            "belief": p[site_id],
            "uncertainty": u[site_id],
            "observed_effort": 0.0,
            "detections": 0.0,
            "habitat_score": 0.5,
            "access_cost": 1.0,
            "frontier": 0.0,
        }
        rows.append(tuple(float(values[name]) for name in NODE_FEATURE_NAMES))
    return GraphState(
        node_ids=node_ids,
        node_features=tuple(rows),
        edge_index=((), ()),
        edge_features=(),
        global_features=(6.0, 0.0, 1.0, sum(u.values()) / len(u)),
        feasibility_mask=tuple(True for _ in node_ids),
    )


def main() -> None:
    # A and B are correlated; C is independent. Every site starts with p=0.5.
    worlds = [
        EcologicalHypothesis({"A": False, "B": False, "C": False}),
        EcologicalHypothesis({"A": False, "B": False, "C": True}),
        EcologicalHypothesis({"A": True, "B": True, "C": False}),
        EcologicalHypothesis({"A": True, "B": True, "C": True}),
    ]
    spatial = SpatialBeliefEngine.initialize(
        worlds,
        [QHypothesis(0.15), QHypothesis(0.30)],
    )
    graph = graph_from_spatial(spatial)

    print("=== R6 INFORMATION GAIN DIAGNOSTIC ===")
    print("STATUS: controlled decision sanity check, not ecological calibration.")
    print("A and B are correlated in the explicit spatial posterior; C is independent.")
    print(f"occupancy marginals={spatial.p_by_site()}")
    print(f"q posterior={spatial.q_posterior()}")
    print()

    for effort in (1, 3, 6):
        planner = InformationGainPlanner(
            effort_per_site=effort,
            max_sites=1,
            require_spatial_belief=True,
        )
        mission = planner.plan(
            graph,
            remaining_budget=6,
            constraints={"spatial_belief_state": spatial},
        )
        rows = mission.diagnostics["ranked_candidates"]
        print(f"effort={effort} | chosen={mission.allocations[0].site_id}")
        for row in rows:
            print(
                "  "
                f"{row['site_id']}: IG={row['expected_information_gain_bits']:.4f} bits | "
                f"P(det)={row['predictive_detection_probability']:.4f}"
            )
        print()

    print("INTERPRETATION")
    print("  A/B have identical local marginals to C, but surveying A or B informs two sites.")
    print("  Therefore spatial-joint VOI must rank A/B above C.")
    print("  Larger effort can increase expected information because non-detection becomes stronger evidence.")
    print("  No HiddenWorld is supplied to the planner.")
    print("  Formal benchmark mode must use mode='spatial_joint', not the site-local fallback.")


if __name__ == "__main__":
    main()
