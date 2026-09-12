from __future__ import annotations

from pathlib import Path

from adaptive_response.milestone_receipts import publish_r2_real_graph_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = publish_r2_real_graph_receipt(repo_root=REPO_ROOT)
    print("=== R2 REAL GRAPH V0 RECEIPT PUBLISHED ===")
    print(f"out={summary['out_dir']}")
    print(
        f"sites={summary['sites']} | edges={summary['primary_edges']} | "
        f"components={summary['components']} | isolated={summary['isolated_sites']}"
    )
    print("Generated compact versioned artifacts:")
    print("  REAL_GRAPH_V0_RECEIPT.md")
    print("  real_graph_v0_audit.json")
    print("  real_graph_v0_edges.csv")
    print("  incident_subgraph_audit.json")
    print("  real_graph_v0.svg")
    print()
    print("Raw data and caches were NOT copied into reports/.")
    print("Review the generated folder, then git add/commit it if the receipt matches the accepted milestone.")


if __name__ == "__main__":
    main()
