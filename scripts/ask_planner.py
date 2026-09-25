"""Interrogate the planner about a real-graph episode in natural language.

    PYTHONPATH=src python scripts/ask_planner.py --case-index 0 \
        --question "Why did you pick the first site you surveyed instead of its neighbour?"
    PYTHONPATH=src python scripts/ask_planner.py --case-index 0          # interactive

Runs the episode first (same frozen R8 case loader and planner config as
`run_real_graph_demo.py`), prints a compact recap so you know what
happened, then answers questions through `adaptive_response.copilot`
(OpenAI function calling over the recorded state). Each answer is printed
with the tools it called, its cost, and whether every number in it traces
back to a tool result. Needs the `llm` extra and OPENAI_API_KEY (repo-root
`.env` works); `--rounds N` stops the episode after N rounds so you can ask
about an in-progress mission - use it for "what should we do next?"
questions, since a completed episode has no next move.
"""

from __future__ import annotations

import argparse
import sys

from adaptive_response import FrontierPlanner
from adaptive_response.copilot import PlannerCopilot
from adaptive_response.event_stream import RealGraphEpisode


def load_cases(budget: int):
    try:
        from adaptive_response.rl.r8_manifest_cases import load_formal_benchmark_cases
    except ImportError as exc:
        sys.exit(f"Loading the frozen real-graph cases needs the `rl` extra: {exc}")
    return load_formal_benchmark_cases(budget=budget)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--rounds", type=int, default=None, help="stop after N rounds (default: run to completion)")
    parser.add_argument("--question", action="append", default=[], help="ask this and exit (repeatable)")
    return parser.parse_args()


def recap(episode: RealGraphEpisode) -> None:
    print(f"=== episode {episode.label}: {len(episode.rounds)} round(s), "
          f"{'complete' if episode.done else 'in progress'} ===", file=sys.stderr)
    for r in episode.rounds:
        t = r.transition
        mission = ", ".join(f"{a.site_id}x{a.effort_units}" for a in t.mission.allocations)
        obs = ", ".join(
            f"{o.site_id}:{'DET' if o.detection else 'no'}" for o in t.observations.observations
        )
        flag = f"  [ALERT {r.alert.level}]" if r.alert.triggered else ""
        print(f"  round {r.index}: mission {mission} -> {obs}; budget left "
              f"{t.public_state_after.remaining_budget}{flag}", file=sys.stderr)
    print(file=sys.stderr)


def print_answer(answer) -> None:
    print(answer.text, flush=True)
    tools = ", ".join(
        f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})" for c in answer.tool_calls
    ) or "none"
    print(
        f"[tools: {tools} | {answer.model} | ${answer.cost_dollars:.6f} | "
        f"numbers traceable: {'yes' if answer.numbers_traceable else 'NO'}]\n",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.budget)
    if not 0 <= args.case_index < len(cases):
        sys.exit(f"--case-index must be in [0, {len(cases) - 1}]")
    case = cases[args.case_index]

    episode = RealGraphEpisode.from_benchmark_case(
        case, FrontierPlanner(effort_levels=(1, 3, 6), max_sites=1)
    )
    episode.reset()
    while not episode.done and (args.rounds is None or len(episode.rounds) < args.rounds):
        episode.step()
    recap(episode)

    copilot = PlannerCopilot(episode)
    if args.question:
        for q in args.question:
            print(f"> {q}", file=sys.stderr, flush=True)
            print_answer(copilot.ask(q))
        return

    print("Ask the planner (empty line or 'exit' to quit):", file=sys.stderr)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print_answer(copilot.ask(q))


if __name__ == "__main__":
    main()
