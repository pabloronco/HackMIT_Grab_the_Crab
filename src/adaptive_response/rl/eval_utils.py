from __future__ import annotations

from dataclasses import dataclass

from ..environment import Environment
from ..mission_loop import AdaptiveMissionLoop
from ..models import IncidentConfig, MissionAction
from ..planners import Planner
from .round_policy import RoundPolicy


@dataclass(frozen=True)
class EpisodeMetrics:
    """Planner-agnostic task metrics, for comparing RL against FrontierPlanner
    (and later Information Gain) on identical held-out incidents/budgets, per
    the team's benchmark discipline (docs/DECISION_LOG.md, M3 entry, and the
    "RL vs Information Gain on held-out incidents" gate in the handoff)."""

    num_rounds: int
    detections_found: int
    effort_spent: int
    occupied_sites_total: int
    occupied_sites_missed: int


class RLPlannerAdapter:
    """Expose a trained `RoundPolicy` through the shared `Planner` protocol so
    it can be evaluated with the exact same harness as `FrontierPlanner`.

    Always acts deterministically (argmax): evaluation should measure the
    policy's committed behavior, not an exploration sample.
    """

    def __init__(self, policy: RoundPolicy) -> None:
        self._policy = policy

    def plan(self, graph_state, remaining_budget, constraints) -> MissionAction:
        del constraints
        return self._policy.act(graph_state, remaining_budget, deterministic=True).mission


def run_planner_episode(
    planner: Planner, incident_config: IncidentConfig, *, seed: int | None = None
) -> EpisodeMetrics:
    """Run one incident under any `Planner` and report task metrics only.

    Uses the shared `AdaptiveMissionLoop` (M4) so evaluation goes through the
    same real orchestration as the product/UI side, including the proper
    `reveal()` gate, instead of re-deriving the plan/execute/Bayes cycle here.
    Safe to use `run_round()` directly (unlike training_env.py's run_episode):
    this function never needs to read anything back from `planner` itself, so
    `execute_pending()`'s internal next-round lookahead planning call is
    harmless here.
    """

    env = Environment(incident_config)
    prior = {site.id: 0.5 for site in incident_config.sites}
    loop = AdaptiveMissionLoop(env, planner, prior_by_site=prior)
    loop.reset(seed=seed)

    num_rounds = 0
    detections_found = 0
    effort_spent = 0

    while True:
        transition = loop.run_round()
        metrics = transition.simulator_metrics
        num_rounds += 1
        detections_found += int(metrics["detections"])
        effort_spent += int(metrics["effort_spent"])

        if transition.done:
            # Returning immediately here (rather than looping once more to let
            # a `while phase is not COMPLETE` condition catch it) matters:
            # reveal() moves the loop's phase from COMPLETE to REVEALED, so
            # that condition would misfire into one extra invalid round.
            hidden_world = loop.reveal()
            public = transition.public_state_after
            occupied_total = sum(hidden_world.occupied_by_site.values())
            missed = sum(
                1
                for site in public.sites
                if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
            )
            return EpisodeMetrics(
                num_rounds=num_rounds,
                detections_found=detections_found,
                effort_spent=effort_spent,
                occupied_sites_total=occupied_total,
                occupied_sites_missed=missed,
            )
