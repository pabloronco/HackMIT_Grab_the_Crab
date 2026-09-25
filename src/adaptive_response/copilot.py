"""'Ask the planner': an explainability copilot grounded in the planner's own
internals through OpenAI function calling.

A question like "why did you send the team to 533 instead of 349?" is
answered by letting the model call tools that read the recorded episode
(`event_stream.RealGraphEpisode`) - the real belief, uncertainty, frontier
flags and the Frontier planner's own ranking - and then phrase what the
tools returned. The model is never handed the whole state up front; it
requests only the slices it needs (retrieval instead of context stuffing),
and it is instructed to state nothing that a tool did not return.

Grounding, not trust: every tool is pure Python over recorded state, the
ranking is produced by `FrontierPlanner.plan()` itself (called with enough
budget to rank every feasible site; `plan()` has no side effects), and each
answer carries `numbers_traceable` - whether every number in the answer
appears in that turn's tool results. Tool traffic is pruned between turns
so a multi-question session does not grow its prompt with stale JSON.

Reasoning models are used with `reasoning_effort="low"` here (not
"minimal" as in `narrate.py`): choosing which tool to call and reading its
result is a small reasoning task, and "minimal" was validated only for pure
restatement.

Requires the `llm` extra and an `OPENAI_API_KEY` only for `ask()`; the tool
functions themselves need neither and are unit-tested offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .event_stream import RealGraphEpisode, RecordedRound
from .graph_state import NODE_FEATURE_NAMES
from .models import GraphState
from .narrate import (
    RELIABLE_MODEL,
    _PRICES_PER_TOKEN,
    _allowed_numbers,
    _numeric_fidelity_ok,
    resolve_api_key,
)
from .planners import FrontierPlanner

_FEATURE_INDEX = {name: i for i, name in enumerate(NODE_FEATURE_NAMES)}


# ----- pure tools over recorded state ------------------------------------


def site_feature_table(graph: GraphState) -> dict[str, dict[str, Any]]:
    """Per-site observable features exactly as the planner sees them."""
    table: dict[str, dict[str, Any]] = {}
    for i, site_id in enumerate(graph.node_ids):
        features = graph.node_features[i]
        row: dict[str, Any] = {name: float(features[_FEATURE_INDEX[name]]) for name in NODE_FEATURE_NAMES}
        row["frontier"] = bool(row["frontier"] > 0.5)
        row["feasible"] = bool(graph.feasibility_mask[i])
        table[site_id] = row
    return table


def frontier_ranking(graph: GraphState) -> list[dict[str, Any]]:
    """Every feasible site in the order `FrontierPlanner` would pick them:
    frontier flag, then belief, then uncertainty, then site id tie-break -
    produced by the planner's own `rank_candidates()`, not a re-implementation."""
    if not graph.node_ids:
        return []
    ranked = FrontierPlanner().rank_candidates(graph)
    return [
        {
            "rank": position + 1,
            "site_id": str(row["site_id"]),
            "belief": float(row["belief"]),
            "uncertainty": float(row["uncertainty"]),
            "frontier": bool(row["frontier"]),
        }
        for position, row in enumerate(ranked)
    ]


def deciding_criterion(higher: dict[str, Any], lower: dict[str, Any]) -> str:
    """Which Frontier ordering key separated two ranked sites."""
    if higher["frontier"] != lower["frontier"]:
        return "frontier flag (unvisited site adjacent to evidence ranks first)"
    if higher["belief"] != lower["belief"]:
        return "higher belief of occupancy"
    if higher["uncertainty"] != lower["uncertainty"]:
        return "higher uncertainty"
    return "site id tie-break (all ranking features equal)"


class EpisodeTools:
    """Tool implementations. Each returns a JSON-serializable dict; errors
    are returned as {"error": ...} so the model can recover, never raised."""

    def __init__(self, episode: RealGraphEpisode, planner_name: str = "frontier") -> None:
        self._episode = episode
        self._planner_name = planner_name

    # -- helpers --

    def _graph_for(self, round_: int | None) -> GraphState | dict[str, Any]:
        rounds = self._episode.rounds
        if round_ is None:
            if rounds:
                return rounds[-1].transition.graph_after
            return self._episode.current_graph_state
        if not 1 <= round_ <= len(rounds):
            return {"error": f"round must be between 1 and {len(rounds)} (rounds completed so far)."}
        return rounds[round_ - 1].transition.graph_before

    def _current_basis(self) -> str:
        rounds = self._episode.rounds
        if self._episode.done and rounds:
            if rounds[-1].ended_by == "budget":
                return (
                    "current state after the final round: the episode is complete and the "
                    "budget is exhausted, so no site is feasible any more; pass round=<n> to "
                    "see what the planner chose from at round n"
                )
            return (
                "current state after the final round: the episode ended at its round "
                "horizon; pass round=<n> to see what the planner chose from at round n"
            )
        return "current graph state (what the next mission is planned on)"

    def _round(self, round_: int) -> RecordedRound | dict[str, Any]:
        rounds = self._episode.rounds
        if not 1 <= round_ <= len(rounds):
            return {"error": f"round must be between 1 and {len(rounds)} (rounds completed so far)."}
        return rounds[round_ - 1]

    # -- tools --

    def get_mission_status(self) -> dict[str, Any]:
        ep = self._episode
        public = ep.current_public_state
        return {
            "planner": self._planner_name,
            "rounds_completed": len(ep.rounds),
            "max_rounds": ep.max_rounds,
            "episode_complete": ep.done,
            "remaining_budget": int(public.remaining_budget),
            "total_budget": int(ep.incident.budget),
            "initial_detection_site": ep.incident.initial_detection,
            "num_sites": len(ep.incident.sites),
            "alerts_triggered_in_rounds": [r.index for r in ep.rounds if r.alert.triggered],
        }

    def get_site(self, site_id: str) -> dict[str, Any]:
        site_id = str(site_id)
        graph = self._graph_for(None)
        assert isinstance(graph, GraphState)
        table = site_feature_table(graph)
        if site_id not in table:
            return {"error": f"unknown site_id {site_id!r}; known sites: {sorted(table)}"}
        rounds = self._episode.rounds
        history = [
            {"round": r.index, "effort": int(o.effort), "detection": bool(o.detection)}
            for r in rounds
            for o in r.transition.observations.observations
            if o.site_id == site_id
        ]
        trajectory = []
        if rounds:
            trajectory.append({"round": 0, "belief": float(rounds[0].transition.belief_before.p_by_site[site_id])})
            trajectory.extend(
                {"round": r.index, "belief": float(r.transition.belief_after.p_by_site[site_id])} for r in rounds
            )
        ranking = frontier_ranking(graph)
        rank = next((row["rank"] for row in ranking if row["site_id"] == site_id), None)
        return {
            "site_id": site_id,
            "current": table[site_id],
            "current_frontier_rank": rank,
            "observations_at_site": history,
            "belief_trajectory": trajectory,
        }

    def rank_sites(self, round: int | None = None, top: int = 10) -> dict[str, Any]:
        graph = self._graph_for(round)
        if isinstance(graph, dict):
            return graph
        ranking = frontier_ranking(graph)
        top = max(1, int(top))
        infeasible = [sid for sid, row in site_feature_table(graph).items() if not row["feasible"]]
        return {
            "basis": (
                f"graph state the round-{round} mission was planned on"
                if round is not None
                else self._current_basis()
            ),
            "ordering": "frontier flag, then belief, then uncertainty, then site id",
            "ranking": ranking[:top],
            "total_feasible": len(ranking),
            "infeasible_sites": infeasible,
        }

    def compare_sites(self, site_a: str, site_b: str, round: int | None = None) -> dict[str, Any]:
        site_a, site_b = str(site_a), str(site_b)
        graph = self._graph_for(round)
        if isinstance(graph, dict):
            return graph
        ranking = {row["site_id"]: row for row in frontier_ranking(graph)}
        table = site_feature_table(graph)
        missing = [s for s in (site_a, site_b) if s not in table]
        if missing:
            return {"error": f"unknown site(s) {missing}; known sites: {sorted(table)}"}
        entry_a, entry_b = ranking.get(site_a), ranking.get(site_b)
        result: dict[str, Any] = {
            "basis": (
                f"graph state the round-{round} mission was planned on"
                if round is not None
                else self._current_basis()
            ),
            "site_a": {"site_id": site_a, **table[site_a], "rank": entry_a["rank"] if entry_a else None},
            "site_b": {"site_id": site_b, **table[site_b], "rank": entry_b["rank"] if entry_b else None},
        }
        if entry_a is None or entry_b is None:
            result["ranks_higher"] = site_a if entry_a else (site_b if entry_b else None)
            result["deciding_criterion"] = "feasibility (an infeasible site is never ranked)"
            return result
        higher, lower = (entry_a, entry_b) if entry_a["rank"] < entry_b["rank"] else (entry_b, entry_a)
        result["ranks_higher"] = higher["site_id"]
        result["deciding_criterion"] = deciding_criterion(higher, lower)
        return result

    def get_round(self, round: int) -> dict[str, Any]:
        record = self._round(int(round))
        if isinstance(record, dict):
            return record
        t = record.transition
        before, after = t.belief_before.p_by_site, t.belief_after.p_by_site
        changes = {
            sid: {"before": float(before.get(sid, after[sid])), "after": float(after[sid])}
            for sid in after
            if abs(after[sid] - before.get(sid, after[sid])) > 1e-9
        }
        worst = record.alert.most_surprising
        return {
            "round": record.index,
            "mission": [
                {"site_id": a.site_id, "effort_units": int(a.effort_units)} for a in t.mission.allocations
            ],
            "observations": [
                {"site_id": o.site_id, "effort": int(o.effort), "detection": bool(o.detection)}
                for o in t.observations.observations
            ],
            "belief_changes": changes,
            "remaining_budget_after": int(t.public_state_after.remaining_budget),
            "next_mission": (
                None
                if record.ended_by or t.next_mission is None
                else [{"site_id": a.site_id, "effort_units": int(a.effort_units)} for a in t.next_mission.allocations]
            ),
            "ended_by": record.ended_by,
            "mismatch_alert": {
                "level": record.alert.level,
                "max_surprise_bits": (
                    None if record.alert.max_surprise_bits == float("inf") else float(record.alert.max_surprise_bits)
                ),
                "alert_bits_setting": float(record.alert.alert_bits),
                "most_surprising_site": worst.site_id if worst else None,
            },
        }

    def get_alerts(self) -> dict[str, Any]:
        alerts = []
        for r in self._episode.rounds:
            if not r.alert.triggered:
                continue
            worst = r.alert.most_surprising
            alerts.append(
                {
                    "round": r.index,
                    "level": r.alert.level,
                    "max_surprise_bits": (
                        None if r.alert.max_surprise_bits == float("inf") else float(r.alert.max_surprise_bits)
                    ),
                    "site_id": worst.site_id if worst else None,
                    "predicted_detection_probability": (
                        float(worst.predictive_detection_probability) if worst else None
                    ),
                    "observed_detection": bool(worst.detection) if worst else None,
                }
            )
        return {
            "alerts": alerts,
            "note": "threshold is a demo setting, not a calibrated failure criterion (docs/MODEL_MISMATCH_DIAGNOSTIC_R9.md)",
        }

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Entry point for model-issued calls. Floats are rounded to 4 decimals
        here (and only here) so the model quotes readable numbers; the
        tool methods themselves stay full-precision for ranking logic."""
        handler = getattr(self, name, None)
        if name.startswith("_") or handler is None or not callable(handler):
            return {"error": f"unknown tool {name!r}"}
        try:
            return _round_floats(handler(**arguments))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}


def _round_floats(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, ndigits) for v in value]
    return value


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_mission_status",
            "description": "Overall episode status: rounds completed, budget left, whether it is complete, which rounds raised mismatch alerts.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_site",
            "description": "Current belief, uncertainty, frontier flag, effort and detections at one site, its current Frontier rank, every observation made there, and its belief trajectory across rounds.",
            "parameters": {
                "type": "object",
                "properties": {"site_id": {"type": "string"}},
                "required": ["site_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_sites",
            "description": "The Frontier planner's full ranking of feasible sites. Omit round for the current state (what the next mission is planned on); give a round number to see the ranking that round's mission was chosen from.",
            "parameters": {
                "type": "object",
                "properties": {
                    "round": {"type": ["integer", "null"], "description": "1-based round, or null for current"},
                    "top": {"type": "integer", "description": "how many top entries to return", "default": 10},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sites",
            "description": "Compare two sites on the ranking features and report which ranks higher and the single deciding criterion. Use this for 'why X instead of Y' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_a": {"type": "string"},
                    "site_b": {"type": "string"},
                    "round": {"type": ["integer", "null"], "description": "1-based round, or null for current"},
                },
                "required": ["site_a", "site_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_round",
            "description": "What happened in one round: mission, field observations, belief changes, budget after, next mission, and the model-mismatch alert for that round.",
            "parameters": {
                "type": "object",
                "properties": {"round": {"type": "integer", "description": "1-based"}},
                "required": ["round"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": "All rounds where the field evidence was poorly predicted by the ecological models (model-mismatch alerts).",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

_SYSTEM_PROMPT = (
    "You are the explainability assistant for a marine invasive-species mission "
    "planner. Answer questions ONLY from tool results: call tools first, then "
    "explain. If a number or fact is not in a tool result, do not state it. The "
    "planner in use is the Frontier heuristic, which ranks feasible sites by (1) "
    "frontier flag, (2) belief of occupancy, (3) uncertainty, (4) site id "
    "tie-break, and sends effort to the top-ranked site each round. When "
    "explaining a choice between sites, name the single deciding criterion the "
    "compare tool reports. Never speculate about true or hidden occupancy - only "
    "belief is known. If a tool reports the episode is complete, say so plainly "
    "and, when asked what to do next, answer from the last round's ranking "
    "instead (rank_sites or compare_sites with that round number). Be concise: "
    "a short paragraph, numbers as returned."
)


@dataclass
class CopilotAnswer:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_dollars: float = 0.0
    numbers_traceable: bool = True
    model: str = RELIABLE_MODEL


class PlannerCopilot:
    def __init__(
        self,
        episode: RealGraphEpisode,
        *,
        api_key: str | None = None,
        model: str = RELIABLE_MODEL,
        max_tool_rounds: int = 8,
        planner_name: str = "frontier",
    ) -> None:
        self.tools = EpisodeTools(episode, planner_name=planner_name)
        self._api_key = api_key
        self._model = model
        self._max_tool_rounds = max_tool_rounds
        self._history: list[dict[str, str]] = []  # final user/assistant turns only
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import: llm extra, not a core dependency

            self._client = OpenAI(api_key=resolve_api_key(self._api_key))
        return self._client

    def ask(self, question: str) -> CopilotAnswer:
        client = self._get_client()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": question},
        ]
        price_in, price_out = _PRICES_PER_TOKEN[self._model]
        answer = CopilotAnswer(text="", model=self._model)
        tool_result_text: list[str] = []

        for _ in range(self._max_tool_rounds + 1):
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SPECS,
                max_completion_tokens=900,
                reasoning_effort="low",
            )
            usage = response.usage
            answer.tokens_in += usage.prompt_tokens
            answer.tokens_out += usage.completion_tokens
            answer.cost_dollars += usage.prompt_tokens * price_in + usage.completion_tokens * price_out
            message = response.choices[0].message

            if not message.tool_calls:
                answer.text = message.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    arguments, result = {}, {"error": f"unparseable arguments: {exc}"}
                else:
                    result = self.tools.dispatch(tc.function.name, arguments)
                result_json = json.dumps(result, allow_nan=False)
                tool_result_text.append(result_json)
                answer.tool_calls.append({"name": tc.function.name, "arguments": arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})
        else:
            answer.text = message.content or "(stopped: tool-call limit reached without a final answer)"

        allowed = _allowed_numbers("\n".join(tool_result_text)) if tool_result_text else set()
        answer.numbers_traceable = _numeric_fidelity_ok(answer.text, allowed) if tool_result_text else False

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer.text})
        return answer
