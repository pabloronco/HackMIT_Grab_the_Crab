from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from .environment import Environment
from .graph_state import GraphStateExporter
from .mission_loop import LoopPhase, RoundTransition
from .models import BeliefState, GraphState, HiddenWorld, MissionAction, PublicState
from .planners import Planner
from .spatial_belief import (
    EcologicalHypothesis,
    QHypothesis,
    SpatialBeliefEngine,
    SpatialBeliefState,
)


class SpatialAdaptiveMissionLoop:
    """Planner-agnostic mission loop backed by the explicit spatial Bayes ensemble.

    This is the R6 integration path for formal baseline/learned-policy work. The
    existing ``AdaptiveMissionLoop`` remains as the site-local rehearsal kernel so
    earlier tests and demos stay stable.

    Pipeline:
      Environment observations -> SpatialBeliefEngine -> BeliefState marginals ->
      GraphState -> Planner -> MissionAction -> Environment -> repeat.

    Planner constraints may include the *inference posterior* ``spatial_belief_state``
    and its q posterior. These are derived only from declared hypotheses plus field
    evidence; they are not simulator HiddenWorld or q_true. This lets the strong
    Information Gain baseline compute global VOI without changing GraphState schema.
    """

    def __init__(
        self,
        environment: Environment,
        planner: Planner,
        ecological_hypotheses: Sequence[EcologicalHypothesis],
        q_hypotheses: Sequence[QHypothesis],
        *,
        exporter: GraphStateExporter | None = None,
    ) -> None:
        if not ecological_hypotheses:
            raise ValueError("SpatialAdaptiveMissionLoop requires ecological hypotheses.")
        if not q_hypotheses:
            raise ValueError("SpatialAdaptiveMissionLoop requires q hypotheses.")
        self._environment = environment
        self._planner = planner
        self._ecological_hypotheses = tuple(ecological_hypotheses)
        self._q_hypotheses = tuple(q_hypotheses)
        self._exporter = exporter or GraphStateExporter()

        self._phase = LoopPhase.UNINITIALIZED
        self._public_state: PublicState | None = None
        self._spatial_belief: SpatialBeliefState | None = None
        self._belief: BeliefState | None = None
        self._graph_state: GraphState | None = None
        self._pending_mission: MissionAction | None = None

    @property
    def phase(self) -> LoopPhase:
        return self._phase

    @property
    def current_public_state(self) -> PublicState:
        self._require_initialized()
        assert self._public_state is not None
        return deepcopy(self._public_state)

    @property
    def current_spatial_belief(self) -> SpatialBeliefState:
        self._require_initialized()
        assert self._spatial_belief is not None
        return deepcopy(self._spatial_belief)

    @property
    def current_belief(self) -> BeliefState:
        self._require_initialized()
        assert self._belief is not None
        return deepcopy(self._belief)

    @property
    def current_graph_state(self) -> GraphState:
        self._require_initialized()
        assert self._graph_state is not None
        return deepcopy(self._graph_state)

    def reset(self, seed: int | None = None) -> GraphState:
        public_state = self._environment.reset(seed=seed)
        spatial = SpatialBeliefEngine.initialize(
            self._ecological_hypotheses,
            self._q_hypotheses,
            confirmed_sites={public_state.initial_detection},
        )
        belief = spatial.as_belief_state()
        graph_state = self._exporter.export(public_state, belief)

        self._public_state = public_state
        self._spatial_belief = spatial
        self._belief = belief
        self._graph_state = graph_state
        self._pending_mission = None

        if public_state.remaining_budget == 0:
            self._phase = LoopPhase.COMPLETE
            self._environment._allow_reveal()
        else:
            self._phase = LoopPhase.READY_TO_PLAN
        return deepcopy(graph_state)

    def plan_next(self) -> MissionAction:
        self._require_initialized()
        if self._phase in (LoopPhase.COMPLETE, LoopPhase.REVEALED):
            raise RuntimeError("Cannot plan after the mission loop is complete.")
        if self._pending_mission is not None:
            return self._pending_mission

        assert self._public_state is not None
        assert self._graph_state is not None
        constraints = self._planner_constraints()
        mission = self._planner.plan(
            self._graph_state,
            remaining_budget=self._public_state.remaining_budget,
            constraints=constraints,
        )
        if mission.total_cost <= 0 or not mission.allocations:
            raise RuntimeError("Planner returned an empty mission while positive budget remains.")
        self._pending_mission = mission
        self._phase = LoopPhase.MISSION_PLANNED
        return mission

    def set_pending_mission(self, mission: MissionAction) -> MissionAction:
        """Install or replace the pending mission with an operator-selected action.

        This is a product/orchestration hook for human-in-the-loop mission control.
        It does not bypass Environment validation, spend budget, update belief, or
        expose hidden truth. It only replaces the action that execute_pending will
        send to the Environment. Planner-facing state remains unchanged.
        """

        self._require_initialized()
        if self._phase not in (LoopPhase.READY_TO_PLAN, LoopPhase.MISSION_PLANNED):
            raise RuntimeError(
                "Operator mission selection requires READY_TO_PLAN or MISSION_PLANNED."
            )
        assert self._public_state is not None
        assert self._graph_state is not None

        if mission.total_cost <= 0 or not mission.allocations:
            raise ValueError("Operator mission must allocate positive effort.")
        if mission.total_cost > self._public_state.remaining_budget:
            raise ValueError("Operator mission exceeds remaining budget.")

        known_sites = set(self._graph_state.node_ids)
        allocated_effort = 0
        for allocation in mission.allocations:
            if allocation.site_id not in known_sites:
                raise ValueError(
                    f"Operator mission references unknown site {allocation.site_id!r}."
                )
            if (
                isinstance(allocation.effort_units, bool)
                or not isinstance(allocation.effort_units, int)
                or allocation.effort_units <= 0
            ):
                raise ValueError("Operator mission effort must be a positive integer.")
            allocated_effort += allocation.effort_units

        if allocated_effort != mission.total_cost:
            raise ValueError("Operator mission total_cost must equal allocated effort.")

        self._pending_mission = mission
        self._phase = LoopPhase.MISSION_PLANNED
        return deepcopy(mission)

    def execute_pending(self) -> RoundTransition:
        self._require_initialized()
        if self._phase is not LoopPhase.MISSION_PLANNED or self._pending_mission is None:
            raise RuntimeError("No pending mission is available to execute.")

        assert self._public_state is not None
        assert self._spatial_belief is not None
        assert self._belief is not None
        assert self._graph_state is not None

        mission = self._pending_mission
        public_before = deepcopy(self._public_state)
        belief_before = deepcopy(self._belief)
        graph_before = deepcopy(self._graph_state)

        observations, done, metrics = self._environment.step(mission)
        public_after = self._environment.current_public_state
        spatial_after = SpatialBeliefEngine.update(self._spatial_belief, observations)
        belief_after = spatial_after.as_belief_state()
        graph_after = self._exporter.export(public_after, belief_after)

        self._public_state = public_after
        self._spatial_belief = spatial_after
        self._belief = belief_after
        self._graph_state = graph_after
        self._pending_mission = None

        if done:
            self._phase = LoopPhase.COMPLETE
            self._environment._allow_reveal()
            next_mission = None
        else:
            self._phase = LoopPhase.READY_TO_PLAN
            next_mission = self.plan_next()

        return RoundTransition(
            mission=mission,
            observations=observations,
            public_state_before=public_before,
            public_state_after=deepcopy(public_after),
            belief_before=belief_before,
            belief_after=deepcopy(belief_after),
            graph_before=graph_before,
            graph_after=deepcopy(graph_after),
            next_mission=next_mission,
            done=done,
            simulator_metrics=dict(metrics),
        )

    def run_round(self) -> RoundTransition:
        if self._phase is LoopPhase.READY_TO_PLAN:
            self.plan_next()
        return self.execute_pending()

    def force_complete(self) -> None:
        """Unlock reveal for a round-count/horizon cap the loop itself doesn't know.

        R7 action contract (docs/DEMU_HANDOFF_R7.md): a formal episode also ends
        at a fixed max-round horizon, independent of remaining budget - a
        constraint that lives in the caller (training/eval driver), not in
        IncidentConfig/Environment. Mirrors exactly what execute_pending()
        already does when the environment reports budget-exhaustion (done),
        so a horizon-terminated episode is revealed/evaluated the same way a
        budget-terminated one is.

        Valid right after execute_pending(): that method itself already plans
        one round ahead whenever the just-executed round wasn't done (see its
        own `next_mission = self.plan_next()` call), so phase may already be
        MISSION_PLANNED with a pending mission that will simply never be
        executed - discarding it here is safe, since discarding an unexecuted
        MissionAction spends no budget and touches the environment nowhere.
        """
        self._require_initialized()
        if self._phase not in (LoopPhase.READY_TO_PLAN, LoopPhase.MISSION_PLANNED):
            raise RuntimeError(
                "force_complete() requires phase READY_TO_PLAN or MISSION_PLANNED "
                f"(got {self._phase.value!r}); the loop is already COMPLETE/REVEALED "
                "or was never reset."
            )
        self._pending_mission = None
        self._phase = LoopPhase.COMPLETE
        self._environment._allow_reveal()

    def reveal(self) -> HiddenWorld:
        self._require_initialized()
        if self._phase is not LoopPhase.COMPLETE:
            raise RuntimeError("Reveal is only available after the loop is complete.")
        hidden_world = self._environment.reveal()
        self._phase = LoopPhase.REVEALED
        return hidden_world

    def _planner_constraints(self) -> dict[str, object]:
        assert self._spatial_belief is not None
        q_mean = self._spatial_belief.q_mean()
        return {
            "q_by_site": {site_id: q_mean for site_id in self._spatial_belief.site_ids},
            "q_posterior": self._spatial_belief.q_posterior(),
            "q_semantics": "belief_posterior_not_simulator_truth",
            "spatial_belief_state": deepcopy(self._spatial_belief),
        }

    def _require_initialized(self) -> None:
        if self._phase is LoopPhase.UNINITIALIZED:
            raise RuntimeError("SpatialAdaptiveMissionLoop has not been reset yet.")
