"""Run one incident on the REAL monitoring graph and emit the round-by-round
event stream the mission-control UI consumes (schema: `adaptive_response/
event_stream.py` module docstring). Every event is one JSON line on stdout
(and in `--out` if given); everything else goes to stderr, so stdout can be
piped straight into a consumer.

    PYTHONPATH=src python scripts/run_real_graph_demo.py --list-cases
    PYTHONPATH=src python scripts/run_real_graph_demo.py --case-index 0 --out reports/demo/real_graph_events.jsonl
    PYTHONPATH=src python scripts/run_real_graph_demo.py --case-index 0 --narrate   # adds an LLM briefing per round
    PYTHONPATH=src python scripts/run_real_graph_demo.py --find-alerts               # which cases/rounds raise a mismatch alert
    PYTHONPATH=src python scripts/run_real_graph_demo.py --case-index 118            # the one manifest case that alerts at the default 3.0 bits

Cases come from the frozen R8 formal manifest (180 real-graph incidents,
budget 18, horizon 6) via `adaptive_response.rl.r8_manifest_cases`; that
subpackage imports torch, so this script needs the `rl` extra even though
the event stream itself does not.

`--narrate` calls `narrate.narrate_round()` (gpt-5-nano first, gpt-5-mini on
gate failure) and attaches the result as `llm_briefing`; if the key/network
is missing the event still carries the free deterministic `briefing` and
`llm_briefing` records the error instead of aborting the demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adaptive_response import FrontierPlanner
from adaptive_response.alerts import DEFAULT_ALERT_BITS
from adaptive_response.event_stream import RealGraphEpisode, to_json_line


def load_cases(budget: int):
    try:
        from adaptive_response.rl.r8_manifest_cases import load_formal_benchmark_cases
    except ImportError as exc:  # torch missing
        sys.exit(
            f"Loading the frozen real-graph cases needs the `rl` extra "
            f"(pip install -e '.[rl]'): {exc}"
        )
    return load_formal_benchmark_cases(budget=budget)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-index", type=int, default=0, help="index into the 180 formal cases")
    parser.add_argument("--budget", type=int, default=18, help="field-check budget (frozen protocol: 18)")
    parser.add_argument("--alert-bits", type=float, default=DEFAULT_ALERT_BITS,
                        help="model-mismatch alert setting in bits (demo setting, not calibrated)")
    parser.add_argument("--out", type=str, default=None, help="also write the JSONL events here")
    parser.add_argument("--narrate", action="store_true", help="attach an LLM briefing per round (needs OPENAI_API_KEY)")
    parser.add_argument("--list-cases", action="store_true", help="print case index/label/group and exit")
    parser.add_argument("--find-alerts", action="store_true",
                        help="run every case and list the rounds that trigger a mismatch alert at --alert-bits, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.budget)

    if args.list_cases:
        for i, case in enumerate(cases):
            print(f"{i:3d}  {case.group:16s}  {case.label}")
        return

    planner = FrontierPlanner(effort_levels=(1, 3, 6), max_sites=1)  # R8's validated fair config

    if args.find_alerts:
        hits = 0
        total_rounds = 0
        for i, case in enumerate(cases):
            ep = RealGraphEpisode.from_benchmark_case(case, planner, alert_bits=args.alert_bits)
            try:
                for _ in ep.run():
                    pass
            except ValueError as exc:  # finite-ensemble support failure, a known edge case (docs/DECISION_LOG.md)
                print(f"{i:3d}  {case.label}  SKIPPED: {exc}", file=sys.stderr)
                continue
            total_rounds += len(ep.rounds)
            for r in ep.rounds:
                if r.alert.triggered:
                    hits += 1
                    worst = r.alert.most_surprising
                    print(
                        f"{i:3d}  {case.label:42s}  round {r.index}  {r.alert.level:10s}  "
                        f"max={r.alert.max_surprise_bits:.2f} bits  site={worst.site_id if worst else '-'}"
                    )
        print(
            f"[find-alerts] {hits} alert round(s) out of {total_rounds} rounds "
            f"across {len(cases)} cases at {args.alert_bits} bits",
            file=sys.stderr,
        )
        return

    if not 0 <= args.case_index < len(cases):
        sys.exit(f"--case-index must be in [0, {len(cases) - 1}]")
    case = cases[args.case_index]

    episode = RealGraphEpisode.from_benchmark_case(case, planner, alert_bits=args.alert_bits)

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w", encoding="utf-8") if out_path else None

    narrate_round = None
    if args.narrate:
        from adaptive_response.narrate import narrate_round  # llm extra

    total_llm_cost = 0.0
    alerts_triggered = 0
    try:
        for event in episode.run():
            if event["type"] == "round":
                if event["surprise"]["level"] != "none":
                    alerts_triggered += 1
                if narrate_round is not None:
                    record = episode.rounds[-1]
                    try:
                        result = narrate_round(record.transition)
                        total_llm_cost += result.total_cost_dollars
                        event["llm_briefing"] = {
                            "text": result.text,
                            "model": result.calls[-1].model,
                            "escalated": result.escalated,
                            "verified": result.calls[-1].passed_gate,
                            "cost_dollars": result.total_cost_dollars,
                        }
                    except Exception as exc:  # never let the LLM path kill the demo
                        event["llm_briefing"] = {"error": str(exc)}
            line = to_json_line(event)
            print(line)
            if out_file is not None:
                out_file.write(line + "\n")
    finally:
        if out_file is not None:
            out_file.close()

    metrics = episode.reveal_metrics
    assert metrics is not None
    print(
        f"[done] case={case.label} rounds={len(episode.rounds)} "
        f"ended_by={episode.rounds[-1].ended_by} alerts={alerts_triggered} "
        f"occupied={metrics.occupied_sites_total} missed={metrics.occupied_sites_missed} "
        f"coverage={metrics.occupied_site_coverage:.3f}"
        + (f" llm_cost=${total_llm_cost:.6f}" if args.narrate else "")
        + (f" -> {out_path}" if out_path else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
