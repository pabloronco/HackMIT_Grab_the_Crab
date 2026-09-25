"""How much MORE budget would Fixed High Effort need to match Marine's mean
detected fraction (41.9%) at budget=18, on the same 100-case demo library?
This is a real, self-computed result from our own simulator, not an external
assumption."""
import statistics as st
from adaptive_response.effort_aware_planner import FixedHighEffortPlanner
from adaptive_response.mission_control import MissionControlSession
from adaptive_response.mission_loop import LoopPhase
from adaptive_response.environment import Environment
from adaptive_response.spatial_mission_loop import SpatialAdaptiveMissionLoop

MARINE_TARGET = 0.4185  # confirmed mean detected fraction at budget=18 (reports/ramp_v2/demo_population/summary.json)

session = MissionControlSession()

def run_fixed_high_at_budget(budget):
    fractions = []
    for i in range(1, 101):
        case_id = f"incident_{i:03d}"
        # Reuse MissionControlSession's own incident construction for perfect parity,
        # but override the budget by rebuilding the loop with FixedHighEffortPlanner.
        session.reset(case_id=case_id)
        loop = session._loop
        incident = loop._environment._config
        incident.budget = budget
        env = Environment(incident, world_model=loop._environment._world_model, q_true=list(loop._environment._q_true_by_site.values())[0], observation_uniform_by_site=loop._environment._observation_uniform_by_site)
        fresh = SpatialAdaptiveMissionLoop(env, FixedHighEffortPlanner(), loop._ecological_hypotheses, loop._q_hypotheses)
        fresh.reset(seed=incident.seed)
        while fresh.phase not in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
            fresh.run_round()
        hidden = fresh.reveal()
        total = sum(hidden.occupied_by_site.values())
        detected = sum(1 for s in fresh.current_public_state.sites if s.detections > 0 and hidden.occupied_by_site.get(s.id, False))
        fractions.append(detected/total if total else 0.0)
    return st.fmean(fractions)

for budget in (18, 20, 22, 24, 26, 28, 30, 33, 36):
    frac = run_fixed_high_at_budget(budget)
    print(f"budget={budget:3d}  fixed_high mean detected fraction={frac:.4f}  (target {MARINE_TARGET:.4f}) {'>= TARGET REACHED' if frac>=MARINE_TARGET else ''}")
