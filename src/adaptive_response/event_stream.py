"""Real-graph episode recorder and JSON event stream for the mission-control UI.

Runs one incident on the real monitoring graph through
`SpatialAdaptiveMissionLoop` exactly the way the frozen R8 benchmark does
(`rl/spatial_benchmark.run_spatial_planner_case`: same Environment
construction, same seed, same max-round horizon, same reveal gate) and, after
every round, emits one JSON-serializable event carrying everything a UI
needs to animate the core thesis - FIELD EVIDENCE -> BELIEF CHANGED ->
MISSION CHANGED - plus the model-mismatch alert for that round.

Event schema (all keys always present; numbers are finite floats or null):

  {"type": "graph"}   emitted once by `reset()`
      case_label, group, seed, budget, max_rounds, initial_detection,
      sites: [{id, x, y, habitat_score}], edges: [{src, dst, distance,
      travel_cost, connectivity_weight}], initial_belief: {site_id: p},
      initial_uncertainty: {site_id: bits}, coordinates_note

  {"type": "round"}   emitted once per `step()`
      round (1-based), mission: [{site_id, effort_units, team_id}],
      observations: [{site_id, effort, detection}],
      belief_before / belief_after / uncertainty_after: {site_id: value},
      belief_delta: {site_id: after - before} (changed sites only),
      remaining_budget, next_mission (list or null), done, ended_by
      ("budget" | "horizon" | null), global_uncertainty, q_mean,
      effective_sample_size, briefing (free deterministic text),
      surprise: {level, alert_bits, max_bits, total_bits,
                 per_observation: [{site_id, effort, detection,
                 predictive_detection_probability, observation_probability,
                 surprise_bits, impossible}]},
      alert_text (deterministic explanation, or null when not triggered)

  {"type": "reveal"}  emitted once by `reveal()`, only after the episode ends
      occupied_by_site: {site_id: bool}, occupied_sites_total,
      occupied_sites_missed, missed_occupied_fraction,
      occupied_site_coverage, final_global_uncertainty

Hidden occupancy appears in the `reveal` event and nowhere else, mirroring
the loop's own reveal gate: no planner, belief, or earlier event ever sees
it. Every event is checked with `json.dumps(..., allow_nan=False)` before it
leaves `to_json_line()`, so an infinite surprise (an "impossible"
observation) is serialized as null with `impossible: true`, never as NaN.

This module deliberately does not import `adaptive_response.rl` (that
subpackage imports torch at package level); benchmark cases are accepted by
duck typing via `RealGraphEpisode.from_benchmark_case()`, and the final
metrics replicate `rl/spatial_metrics.compute_spatial_primary_metrics`
field-for-field (covered by a test).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from .alerts import (
    DEFAULT_ALERT_BITS,
    SurpriseAlert,
    evaluate_round_surprise,
    template_alert_explanation,
)
from .environment import Environment
from .graph_state import GLOBAL_FEATURE_NAMES
from .mission_loop import RoundTransition
from .models import HiddenWorld, IncidentConfig, MissionAction
from .narrate import template_briefing
from .planners import Planner
from .spatial_belief import EcologicalHypothesis, QHypothesis, SpatialBeliefState
from .spatial_mission_loop import SpatialAdaptiveMissionLoop
from .world_models import WorldModel

COORDINATES_NOTE = (
    "x/y are the deterministic graph-layout placeholders used by "
    "adaptive_response.rl.real_graph_cases (raw Dryad site coordinates were not "
    "fetchable from the development environment; see docs/DECISION_LOG.md, "
    "2026-09-15 entry). Graph topology is the real frozen R2 monitoring network."
)

_GLOBAL_UNCERTAINTY_IDX = GLOBAL_FEATURE_NAMES.index("global_uncertainty")


@dataclass(frozen=True)
class RecordedRound:
    """Everything captured for one executed round, kept for the copilot."""

    index: int  # 1-based
    transition: RoundTransition
    alert: SurpriseAlert
    spatial_before: SpatialBeliefState
    spatial_after: SpatialBeliefState
    ended_by: str | None  # "budget", "horizon" or None


@dataclass(frozen=True)
class RevealMetrics:
    occupied_sites_total: int
    occupied_sites_missed: int
    missed_occupied_fraction: float
    occupied_site_coverage: float
    final_global_uncertainty: float


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _mission_payload(mission: MissionAction | None) -> list[dict[str, Any]] | None:
    if mission is None:
        return None
    return [
        {"site_id": a.site_id, "effort_units": int(a.effort_units), "team_id": a.team_id}
        for a in mission.allocations
    ]


class RealGraphEpisode:
    """One recorded episode on a real-graph incident. Drive it with
    `reset()` -> `step()`* -> `reveal()`, or just iterate `run()`."""

    def __init__(
        self,
        incident: IncidentConfig,
        *,
        planner: Planner,
        ecological_hypotheses: Sequence[EcologicalHypothesis],
        q_hypotheses: Sequence[QHypothesis],
        seed: int,
        max_rounds: int,
        world_model: WorldModel | None = None,
        q_true: float | Mapping[str, float] | None = None,
        label: str = "",
        group: str = "",
        alert_bits: float = DEFAULT_ALERT_BITS,
    ) -> None:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive.")
        self._incident = incident
        self._planner = planner
        self._seed = seed
        self._max_rounds = max_rounds
        self._label = label
        self._group = group
        self._alert_bits = alert_bits
        self._loop = SpatialAdaptiveMissionLoop(
            Environment(incident, world_model=world_model, q_true=q_true),
            planner,
            ecological_hypotheses,
            q_hypotheses,
        )
        self._rounds: list[RecordedRound] = []
        self._started = False
        self._done = False
        self._hidden_world: HiddenWorld | None = None
        self._reveal_metrics: RevealMetrics | None = None

    @classmethod
    def from_benchmark_case(
        cls, case: Any, planner: Planner, *, alert_bits: float = DEFAULT_ALERT_BITS
    ) -> "RealGraphEpisode":
        """Accepts an `rl.spatial_benchmark.SpatialBenchmarkCase` (or anything
        with the same fields) without importing the torch-dependent rl package."""
        return cls(
            case.incident,
            planner=planner,
            ecological_hypotheses=case.ecological_hypotheses,
            q_hypotheses=case.q_hypotheses,
            seed=case.seed,
            max_rounds=case.max_rounds,
            world_model=case.world_model,
            q_true=case.q_true,
            label=case.label,
            group=case.group,
            alert_bits=alert_bits,
        )

    # ----- state ---------------------------------------------------------

    @property
    def rounds(self) -> tuple[RecordedRound, ...]:
        return tuple(self._rounds)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def done(self) -> bool:
        return self._done

    @property
    def revealed(self) -> bool:
        return self._hidden_world is not None

    @property
    def incident(self) -> IncidentConfig:
        return self._incident

    @property
    def label(self) -> str:
        return self._label

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    @property
    def current_graph_state(self):
        return self._loop.current_graph_state

    @property
    def current_public_state(self):
        return self._loop.current_public_state

    @property
    def reveal_metrics(self) -> RevealMetrics | None:
        return self._reveal_metrics

    # ----- driving -------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("Episode already started; create a new RealGraphEpisode to rerun.")
        self._loop.reset(seed=self._seed)
        self._started = True
        self._done = self._loop.current_public_state.remaining_budget == 0
        belief = self._loop.current_belief
        return {
            "type": "graph",
            "case_label": self._label,
            "group": self._group,
            "seed": self._seed,
            "budget": int(self._incident.budget),
            "max_rounds": self._max_rounds,
            "initial_detection": self._incident.initial_detection,
            "sites": [
                {
                    "id": s.id,
                    "x": float(s.x),
                    "y": float(s.y),
                    "habitat_score": float(s.habitat_score),
                }
                for s in self._incident.sites
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "distance": float(e.distance),
                    "travel_cost": None if e.travel_cost is None else float(e.travel_cost),
                    "connectivity_weight": (
                        None if e.connectivity_weight is None else float(e.connectivity_weight)
                    ),
                }
                for e in self._incident.edges
            ],
            "initial_belief": {k: float(v) for k, v in belief.p_by_site.items()},
            "initial_uncertainty": {k: float(v) for k, v in belief.uncertainty_by_site.items()},
            "coordinates_note": COORDINATES_NOTE,
        }

    def step(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is complete; call reveal() or start a new episode.")

        spatial_before = self._loop.current_spatial_belief
        transition = self._loop.run_round()
        index = len(self._rounds) + 1
        alert = evaluate_round_surprise(
            spatial_before, transition.observations, alert_bits=self._alert_bits
        )
        spatial_after = self._loop.current_spatial_belief

        horizon_reached = index >= self._max_rounds
        if transition.done:
            ended_by: str | None = "budget"
        elif horizon_reached:
            ended_by = "horizon"
            self._loop.force_complete()
        else:
            ended_by = None
        self._done = ended_by is not None

        record = RecordedRound(
            index=index,
            transition=transition,
            alert=alert,
            spatial_before=spatial_before,
            spatial_after=spatial_after,
            ended_by=ended_by,
        )
        self._rounds.append(record)
        return self._round_event(record)

    def reveal(self) -> dict[str, Any]:
        if not self._done:
            raise RuntimeError("reveal() is only available after the episode is complete.")
        if self._hidden_world is None:
            self._hidden_world = self._loop.reveal()
            last = self._rounds[-1].transition
            self._reveal_metrics = _reveal_metrics(
                sites=last.public_state_after.sites,
                hidden_world=self._hidden_world,
                final_uncertainty_by_site=last.belief_after.uncertainty_by_site,
            )
        m = self._reveal_metrics
        assert m is not None
        return {
            "type": "reveal",
            "occupied_by_site": {k: bool(v) for k, v in self._hidden_world.occupied_by_site.items()},
            "occupied_sites_total": m.occupied_sites_total,
            "occupied_sites_missed": m.occupied_sites_missed,
            "missed_occupied_fraction": m.missed_occupied_fraction,
            "occupied_site_coverage": m.occupied_site_coverage,
            "final_global_uncertainty": m.final_global_uncertainty,
        }

    def run(self) -> Iterator[dict[str, Any]]:
        """Yield graph, every round, then reveal. Safe to iterate lazily."""
        yield self.reset()
        while not self._done:
            yield self.step()
        yield self.reveal()

    # ----- event assembly ------------------------------------------------

    def _round_event(self, record: RecordedRound) -> dict[str, Any]:
        t = record.transition
        before = t.belief_before.p_by_site
        after = t.belief_after.p_by_site
        delta = {
            sid: float(after[sid] - before.get(sid, after[sid]))
            for sid in after
            if abs(after[sid] - before.get(sid, after[sid])) > 1e-9
        }
        alert = record.alert
        return {
            "type": "round",
            "round": record.index,
            "mission": _mission_payload(t.mission),
            "observations": [
                {"site_id": o.site_id, "effort": int(o.effort), "detection": bool(o.detection)}
                for o in t.observations.observations
            ],
            "belief_before": {k: float(v) for k, v in before.items()},
            "belief_after": {k: float(v) for k, v in after.items()},
            "uncertainty_after": {k: float(v) for k, v in t.belief_after.uncertainty_by_site.items()},
            "belief_delta": delta,
            "remaining_budget": int(t.public_state_after.remaining_budget),
            "next_mission": None if record.ended_by else _mission_payload(t.next_mission),
            "done": bool(record.ended_by is not None),
            "ended_by": record.ended_by,
            "global_uncertainty": float(t.graph_after.global_features[_GLOBAL_UNCERTAINTY_IDX]),
            "q_mean": float(record.spatial_after.q_mean()),
            "effective_sample_size": float(record.spatial_after.effective_sample_size()),
            "briefing": template_briefing(t),
            "surprise": {
                "level": alert.level,
                "alert_bits": float(alert.alert_bits),
                "max_bits": _finite_or_none(alert.max_surprise_bits),
                "total_bits": _finite_or_none(alert.total_surprise_bits),
                "per_observation": [
                    {
                        "site_id": s.site_id,
                        "effort": int(s.effort),
                        "detection": bool(s.detection),
                        "predictive_detection_probability": float(s.predictive_detection_probability),
                        "observation_probability": float(s.observation_probability),
                        "surprise_bits": _finite_or_none(s.surprise_bits),
                        "impossible": bool(s.impossible_under_current_ensemble),
                    }
                    for s in alert.per_observation
                ],
            },
            "alert_text": template_alert_explanation(alert) if alert.triggered else None,
        }


def _reveal_metrics(
    *, sites, hidden_world: HiddenWorld, final_uncertainty_by_site: Mapping[str, float]
) -> RevealMetrics:
    """Same definitions as rl/spatial_metrics.compute_spatial_primary_metrics."""
    occupied_total = sum(1 for s in sites if hidden_world.occupied_by_site.get(s.id, False))
    occupied_missed = sum(
        1
        for s in sites
        if hidden_world.occupied_by_site.get(s.id, False) and s.detections == 0
    )
    missed_fraction = (occupied_missed / occupied_total) if occupied_total > 0 else 0.0
    values = list(final_uncertainty_by_site.values())
    return RevealMetrics(
        occupied_sites_total=occupied_total,
        occupied_sites_missed=occupied_missed,
        missed_occupied_fraction=missed_fraction,
        occupied_site_coverage=1.0 - missed_fraction,
        final_global_uncertainty=(sum(values) / len(values)) if values else 0.0,
    )


def to_json_line(event: Mapping[str, Any]) -> str:
    """One event -> one JSONL line. `allow_nan=False` makes a stray NaN/inf a
    loud error here rather than a silently broken line in the UI."""
    return json.dumps(event, allow_nan=False, separators=(",", ":"))
