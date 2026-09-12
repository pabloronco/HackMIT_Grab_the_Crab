from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .real_graph import graph_summary, load_real_sites


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_adaptive_water_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "src",
            "dst",
            "distance_km",
            "candidate_local_radius",
            "candidate_sparse_review",
            "straight_marine_fraction",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Adaptive-water table missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def _filter_local_edges(
    rows: list[dict[str, Any]],
    *,
    min_water_fraction: float | None = None,
    exclude_zero_only: bool = False,
) -> list[dict[str, Any]]:
    if min_water_fraction is not None and not 0.0 <= min_water_fraction <= 1.0:
        raise ValueError("min_water_fraction must be between 0 and 1")
    result: list[dict[str, Any]] = []
    for row in rows:
        if not _as_bool(row.get("candidate_local_radius")):
            continue
        water = float(row["straight_marine_fraction"])
        if exclude_zero_only and water <= 0.0:
            continue
        if min_water_fraction is not None and water < min_water_fraction:
            continue
        result.append(dict(row))
    return result


def compare_water_filter_topologies(
    sites: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare transparent water-filter interpretations of the 20 km local pool.

    This is a sensitivity diagnostic only. It deliberately excludes sparse-review-only
    edges from the candidate primary graph so monitoring sparsity cannot force an edge.
    No variant is automatically selected as the final ecological graph.
    """
    variants = {
        "all_local": _filter_local_edges(rows),
        "exclude_zero_water": _filter_local_edges(rows, exclude_zero_only=True),
        "water_ge_0_4": _filter_local_edges(rows, min_water_fraction=0.4),
        "water_ge_0_6": _filter_local_edges(rows, min_water_fraction=0.6),
        "all_sampled_water": _filter_local_edges(rows, min_water_fraction=1.0),
    }

    topology = {name: graph_summary(sites, edges) for name, edges in variants.items()}

    review_only = [
        row
        for row in rows
        if _as_bool(row.get("candidate_sparse_review"))
        and not _as_bool(row.get("candidate_local_radius"))
    ]
    review_summary = {
        "edges": len(review_only),
        "water_gt_0": sum(float(row["straight_marine_fraction"]) > 0.0 for row in review_only),
        "water_ge_0_4": sum(float(row["straight_marine_fraction"]) >= 0.4 for row in review_only),
        "water_ge_0_6": sum(float(row["straight_marine_fraction"]) >= 0.6 for row in review_only),
        "all_sampled_water": sum(float(row["straight_marine_fraction"]) >= 1.0 for row in review_only),
        "over_35km": sum(float(row["distance_km"]) > 35.0 for row in review_only),
    }

    return {
        "topology_variants": topology,
        "sampling_isolation_review_only": review_summary,
        "notes": [
            "All variants use only the <=20 km local candidate tier; sparse-review-only edges never force primary connectivity.",
            "Water thresholds are sensitivity probes, not ecological constants.",
            "The open-saltwater fraction is a coarse straight-line land-barrier diagnostic and can miss narrow estuarine water.",
            "Choose a primary v0 rule for transparency and test alternative variants as topology sensitivity/OOD, rather than claiming one threshold is physically true.",
        ],
    }
