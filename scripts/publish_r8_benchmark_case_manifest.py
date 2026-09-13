from __future__ import annotations

import json
from pathlib import Path

from adaptive_response.benchmark_cases import build_benchmark_case_manifest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/milestones/r2_real_graph_v0/incident_subgraph_audit.json"
R5 = ROOT / "configs/benchmark_protocol_r5.json"
R7 = ROOT / "configs/benchmark_protocol_r7.json"
Q7 = ROOT / "configs/q_protocol_r7.json"
OUT_DIR = ROOT / "reports/milestones/r8_benchmark_case_manifest"
OUT = OUT_DIR / "benchmark_cases.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = build_benchmark_case_manifest(
        load(AUDIT),
        load(R5),
        load(R7),
        load(Q7),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print("=== R8 FORMAL BENCHMARK CASE MANIFEST ===")
    print(f"status={manifest['status']}")
    print(f"case count status={manifest['case_count_status']}")
    print(f"cases={manifest['case_count']}")
    print(f"split counts={manifest['split_counts']}")
    print(f"incident topology sizes={manifest['incident_topology_size_counts']}")
    print(f"eligible topology seeds={manifest['eligible_topology_seed_count']}")
    print(f"belief q support={manifest['belief_q_support']}")
    print("hidden occupancy truth embedded=False")
    print("planner results embedded=False")
    print("action/reward contract embedded=False (frozen externally in configs/benchmark_protocol_r8.json)")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
