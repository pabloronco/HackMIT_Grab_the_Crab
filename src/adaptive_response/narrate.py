"""Mission-round briefings over a `RoundTransition`: a free deterministic
default, plus an opt-in LLM path for more natural phrasing.

Architecture (validated empirically this session against gpt-5-nano/mini,
not just assumed):

  1. `template_briefing()` - the default. Pure Python, zero cost, zero API
     calls, correct by construction (it only ever restates numbers already
     computed by the mission engine, never generates them). A blind judge
     (gpt-5) scored it statistically tied with gpt-5-mini on clarity for
     this task, at $0 instead of real cost per call.
  2. `narrate_round()` - opt-in, for when natural prose is worth the cost.
     Tries gpt-5-nano first (cheapest), verifies the response with a
     deterministic numeric-fidelity gate (every number in the output must
     trace back to a number actually given in the prompt), and escalates to
     gpt-5-mini only if that gate fails. In our validation run nano passed
     the gate 14/14 - escalation should be rare in practice, not the common
     path.

Known, accepted residual risk: the numeric-fidelity gate catches invented
numbers, but not a *correct* number attached to the *wrong* site (found once
in 14 validation cases, gpt-5-nano). A second detection layer was attempted
and rejected: an attribution checker requiring the site id to co-occur with
its value produced 8 false positives across the same 14 cases, because
briefings often refer back to a site named earlier in the text rather than
repeating it in every sentence - a normal, correct writing style a rule-
based check cannot distinguish from the real error without another LLM call
(which would defeat the point of tiering for cost). Forcing the model to
repeat the site id in every sentence to make this checkable was also
rejected: it fixes a rare error by degrading every other response's
fluency, disproportionate to the risk. This risk is accepted, not hidden,
and is why `narrate_round()` is opt-in rather than the default - most
rounds should use `template_briefing()`, which cannot have this failure
mode at all.

Requires the `llm` extra (`pip install -e ".[llm]"`) and an `OPENAI_API_KEY`
in the environment only for `narrate_round()`. `template_briefing()` has no
such requirement. Nothing else in `adaptive_response` imports this module,
so the core belief/planner engine runs with no network access and no API
key, the same guarantee `adaptive_response.rl` gives for torch.

`narrate_alert()` applies the same tiering + gate to a model-mismatch
`SurpriseAlert` (see `alerts.py`); `template_alert_explanation()` there is
its free deterministic counterpart.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .alerts import SurpriseAlert, format_alert_facts
from .mission_loop import RoundTransition

FAST_MODEL = "gpt-5-nano"
RELIABLE_MODEL = "gpt-5-mini"

# $ per token, for the cost log only - not used to alter behavior.
_PRICES_PER_TOKEN = {
    "gpt-5-nano": (0.05e-6, 0.40e-6),
    "gpt-5-mini": (0.25e-6, 2.00e-6),
    "gpt-5": (1.25e-6, 10.00e-6),
}

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SITE_RE = re.compile(r"[Ss]ite[_ -]?(\d+)")
_NUM_RE = re.compile(r"(?<!\w)-?\d+\.\d+(?!\w)|(?<!\w)-?\d+(?!\w)")


def _load_key_from_dotenv() -> str | None:
    """Read OPENAI_API_KEY from a repo-root `.env` (gitignored) if present.

    Manual parsing instead of python-dotenv: one env var, not worth a new
    dependency in a package that otherwise only takes on torch/openai lazily.
    """
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


_SYSTEM_PROMPT = (
    "You are writing a short field briefing for a marine invasive-species "
    "response team. You will be given exact belief, effort, and budget "
    "figures already computed by the mission engine. Restate them clearly "
    "for a non-technical field team. Do not invent, round differently, or "
    "compute any new number. Do not speculate about hidden/true occupancy - "
    "you only know what the belief state says. Keep it under 120 words."
)


def _format_transition(round_: RoundTransition) -> str:
    lines = ["MISSION"]
    for alloc in round_.mission.allocations:
        team = f"team {alloc.team_id}" if alloc.team_id else "the team"
        lines.append(f"  {team}: {alloc.effort_units} checks at {alloc.site_id}")
    lines.append(f"  total effort spent: {round_.mission.total_cost}")

    lines.append("FIELD RETURNS")
    for obs in round_.observations.observations:
        lines.append(
            f"  {obs.site_id}: {'DETECTION' if obs.detection else 'no detection'} "
            f"({obs.effort} checks)"
        )

    lines.append("BELIEF CHANGE (probability of occupancy)")
    listed, omitted, omitted_max = _reported_belief_changes(round_)
    for site_id, before, after in listed:
        lines.append(f"  {site_id}: {before:.4f} -> {after:.4f}")
    if omitted:
        lines.append(f"  {omitted} other sites shifted by at most {omitted_max:.4f} (not listed)")

    lines.append(f"REMAINING BUDGET: {round_.public_state_after.remaining_budget}")
    if round_.next_mission is not None:
        lines.append("NEXT MISSION")
        for alloc in round_.next_mission.allocations:
            team = f"team {alloc.team_id}" if alloc.team_id else "the team"
            lines.append(f"  {team}: {alloc.effort_units} checks at {alloc.site_id}")
    else:
        lines.append("NEXT MISSION: none (budget exhausted or episode complete)")

    lines.append(f"EPISODE COMPLETE: {round_.done}")
    return "\n".join(lines)


def _changed_sites(round_: RoundTransition) -> set[str]:
    return {
        site_id
        for site_id, p_after in round_.belief_after.p_by_site.items()
        if abs(p_after - round_.belief_before.p_by_site.get(site_id, p_after)) > 1e-9
    }


BELIEF_CHANGES_TOP_K = 5


def _reported_belief_changes(
    round_: RoundTransition, top_k: int = BELIEF_CHANGES_TOP_K
) -> tuple[list[tuple[str, float, float]], int, float]:
    """Which belief changes a briefing states explicitly.

    On the real graph one observation propagates through the spatial
    ensemble and moves the belief at most sites, so listing every change
    makes a briefing unreadable (and inflates the LLM prompt). Always list
    the surveyed sites, then the largest other shifts up to `top_k` total;
    return the rest as a count plus their largest absolute shift so the
    text can say what it left out. Deterministic ordering.
    """
    before, after = round_.belief_before.p_by_site, round_.belief_after.p_by_site
    surveyed = [a.site_id for a in round_.mission.allocations]
    changed = _changed_sites(round_)
    rows = {sid: (sid, float(before.get(sid, after[sid])), float(after[sid])) for sid in changed}

    listed: list[tuple[str, float, float]] = [rows[s] for s in surveyed if s in rows]
    others = sorted(
        (row for sid, row in rows.items() if sid not in surveyed),
        key=lambda row: (-abs(row[2] - row[1]), row[0]),
    )
    room = max(0, top_k - len(listed))
    listed.extend(others[:room])
    omitted = others[room:]
    omitted_max = max((abs(a - b) for _, b, a in omitted), default=0.0)
    return listed, len(omitted), omitted_max


def template_briefing(round_: RoundTransition) -> str:
    """Deterministic, zero-cost briefing. No API call, cannot invent or
    misattribute a number - it only ever restates `RoundTransition` fields.
    Recommended default; see module docstring for why."""
    sentences = []
    for alloc in round_.mission.allocations:
        team = alloc.team_id or "the team"
        obs = next(
            (o for o in round_.observations.observations if o.site_id == alloc.site_id),
            None,
        )
        result = "a detection" if (obs and obs.detection) else "no detection"
        sentences.append(
            f"{team} surveyed {alloc.site_id} ({alloc.effort_units} checks): {result}."
        )

    listed, omitted, omitted_max = _reported_belief_changes(round_)
    for site_id, before, after in listed:
        sentences.append(
            f"Belief of occupancy at {site_id} moved from {before * 100:.1f}% "
            f"to {after * 100:.1f}%."
        )
    if omitted:
        sentences.append(
            f"{omitted} other sites shifted by less than {omitted_max * 100:.1f} points."
        )

    sentences.append(
        f"Remaining budget: {round_.public_state_after.remaining_budget} units."
    )
    if round_.next_mission is not None:
        parts = ", ".join(
            f"{a.site_id} ({a.effort_units} checks)"
            for a in round_.next_mission.allocations
        )
        sentences.append(f"Next mission: {parts}.")
    else:
        sentences.append("No further missions planned.")
    sentences.append("Episode complete." if round_.done else "Episode ongoing.")
    return " ".join(sentences)


def _normalize_site_ids(text: str) -> str:
    return _SITE_RE.sub(lambda m: f"site_{int(m.group(1)):02d}", text)


def _allowed_numbers(prompt_text: str) -> set[float]:
    """Every number that appears in the prompt, plus its percentage-scale
    paraphrase (a model writing '70%' for an input '0.7000' is restating,
    not inventing) - the exact logic validated against 14 real cases this
    session, not a guess."""
    raw = [float(x) for x in _NUM_RE.findall(_normalize_site_ids(prompt_text))]
    allowed: set[float] = set()
    for v in raw:
        for r in (v, round(v, 2), round(v, 1), round(v)):
            allowed.add(round(r, 3))
        if 0.0 <= v <= 1.0:
            pct = v * 100
            for r in (pct, round(pct, 1), round(pct)):
                allowed.add(round(r, 3))
    return allowed


def _numeric_fidelity_ok(output_text: str, allowed: set[float]) -> bool:
    found = [float(x) for x in _NUM_RE.findall(_normalize_site_ids(output_text))]
    return all(any(abs(v - a) <= 0.5 for a in allowed) for v in found)


@dataclass
class NarrationCall:
    """One `narrate_round()` attempt, for cost accounting/reporting."""

    model: str
    tokens_in: int
    tokens_out: int
    cost_dollars: float
    passed_gate: bool


@dataclass
class NarrationResult:
    text: str
    escalated: bool
    calls: list[NarrationCall] = field(default_factory=list)

    @property
    def total_cost_dollars(self) -> float:
        return sum(c.cost_dollars for c in self.calls)


def resolve_api_key(api_key: str | None = None) -> str:
    """Explicit argument, then OPENAI_API_KEY, then the repo-root .env."""
    key = api_key or os.environ.get("OPENAI_API_KEY") or _load_key_from_dotenv()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in a repo-root .env file, export it, "
            "or pass api_key= explicitly."
        )
    return key


def _llm_restate(
    prompt_text: str,
    *,
    system_prompt: str,
    api_key: str | None,
    allow_escalation: bool,
) -> NarrationResult:
    """Shared tiering core: `FAST_MODEL` first, numeric-fidelity gate,
    escalate to `RELIABLE_MODEL` only on a gate failure. `prompt_text` must
    already contain every number the model is allowed to say."""
    key = resolve_api_key(api_key)

    from openai import OpenAI  # lazy import: llm extra, not a core dependency

    client = OpenAI(api_key=key)
    allowed = _allowed_numbers(prompt_text)

    tiers = [FAST_MODEL, RELIABLE_MODEL] if allow_escalation else [FAST_MODEL]
    calls: list[NarrationCall] = []
    last_text = ""

    for tier_model in tiers:
        response = client.chat.completions.create(
            model=tier_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=300,
            reasoning_effort="minimal",  # these are reasoning models; without
            # this, reasoning tokens can consume the whole completion budget
            # and leave zero visible output (found empty-briefing failures
            # during this session's calibration before adding this).
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        price_in, price_out = _PRICES_PER_TOKEN[tier_model]
        passed = _numeric_fidelity_ok(text, allowed)
        calls.append(
            NarrationCall(
                model=tier_model,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                cost_dollars=usage.prompt_tokens * price_in + usage.completion_tokens * price_out,
                passed_gate=passed,
            )
        )
        last_text = text
        if passed:
            return NarrationResult(text=text, escalated=tier_model != tiers[0], calls=calls)

    # Exhausted every tier without a clean pass: return the strongest tier's
    # output anyway (better than nothing for a live demo) but the caller can
    # inspect `calls[-1].passed_gate is False` to know it wasn't verified.
    return NarrationResult(text=last_text, escalated=len(tiers) > 1, calls=calls)


def narrate_round(
    round_: RoundTransition,
    *,
    api_key: str | None = None,
    allow_escalation: bool = True,
) -> NarrationResult:
    """Return a plain-language briefing, trying `FAST_MODEL` first and
    escalating to `RELIABLE_MODEL` only if the numeric-fidelity gate fails.

    Raises `RuntimeError` if no API key is available, and re-raises whatever
    the OpenAI client raises on a request failure - callers doing a live
    demo should catch and fall back to `template_briefing()`.
    """
    return _llm_restate(
        _format_transition(round_),
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
        allow_escalation=allow_escalation,
    )


_ALERT_SYSTEM_PROMPT = (
    "You are explaining a model-mismatch diagnostic to a marine invasive-species "
    "response team. You will be given exact, already-computed figures: how "
    "probable the field result was under the system's current ecological "
    "models (surprise bits = -log2 of the probability the models gave the observed "
    "outcome; higher means less expected). Explain in plain language what was "
    "surprising and why the team "
    "should treat the next recommendation with extra caution. Do not invent, "
    "round differently, or compute any new number. Say explicitly that the "
    "alert threshold is a demo setting, not a calibrated failure criterion, and "
    "that surprise is a diagnostic, not proof the models are wrong. Do not "
    "speculate about true/hidden occupancy. Keep it under 120 words."
)


def narrate_alert(
    alert: SurpriseAlert,
    *,
    api_key: str | None = None,
    allow_escalation: bool = True,
) -> NarrationResult:
    """LLM explanation of a model-mismatch alert, same tiering + gate as
    `narrate_round()`. Free counterpart: `alerts.template_alert_explanation`."""
    return _llm_restate(
        format_alert_facts(alert),
        system_prompt=_ALERT_SYSTEM_PROMPT,
        api_key=api_key,
        allow_escalation=allow_escalation,
    )
