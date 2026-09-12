from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

from .water_plausibility import annotate_open_saltwater_support


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_adaptive_candidate_edges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "src",
            "dst",
            "distance_km",
            "candidate_local_radius",
            "candidate_sparse_review",
            "sparse_review_requested_by",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Adaptive candidate table missing fields: {sorted(missing)}")
        return [dict(row) for row in reader]


def annotate_adaptive_candidates_with_water(
    edges: list[dict[str, Any]],
    site_coordinates: dict[str, tuple[float, float]],
    *,
    samples_per_edge: int = 5,
    checker: Callable[[float, float], bool] | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Attach straight-line open-saltwater support to an adaptive candidate pool.

    This preserves candidate provenance (local-radius vs sparse-monitoring review)
    and adds only a coarse land-barrier diagnostic. It does not accept or reject
    ecological edges.
    """
    return annotate_open_saltwater_support(
        edges,
        site_coordinates,
        samples_per_edge=samples_per_edge,
        checker=checker,
        max_workers=max_workers,
    )


def _tier_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "edges": 0,
            "all_water": 0,
            "mixed_water": 0,
            "no_water": 0,
            "water_fraction_median": None,
            "distance_km_median": None,
        }
    fractions = [float(row["straight_marine_fraction"]) for row in rows]
    distances = [float(row["distance_km"]) for row in rows]
    return {
        "edges": len(rows),
        "all_water": sum(_as_bool(row.get("straight_marine_all")) for row in rows),
        "mixed_water": sum(
            not _as_bool(row.get("straight_marine_all"))
            and not _as_bool(row.get("straight_marine_none"))
            for row in rows
        ),
        "no_water": sum(_as_bool(row.get("straight_marine_none")) for row in rows),
        "water_fraction_median": median(fractions),
        "distance_km_median": median(distances),
    }


def sparse_site_review(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group sparse-monitoring review candidates by the site that requested them."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not _as_bool(row.get("candidate_sparse_review")):
            continue
        requested = str(row.get("sparse_review_requested_by") or "").strip()
        if not requested:
            continue
        for site_id in requested.split("|"):
            site_id = site_id.strip()
            if not site_id:
                continue
            other = str(row["dst"]) if str(row["src"]) == site_id else str(row["src"])
            grouped[site_id].append(
                {
                    "neighbor": other,
                    "distance_km": float(row["distance_km"]),
                    "straight_marine_fraction": float(row["straight_marine_fraction"]),
                    "straight_marine_pattern": str(row["straight_marine_pattern"]),
                    "candidate_local_radius": _as_bool(row.get("candidate_local_radius")),
                }
            )

    result: dict[str, list[dict[str, Any]]] = {}
    for site_id, candidates in grouped.items():
        result[site_id] = sorted(
            candidates,
            key=lambda item: (-float(item["straight_marine_fraction"]), float(item["distance_km"])),
        )
    return dict(sorted(result.items()))


def summarize_adaptive_water_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    local = [row for row in rows if _as_bool(row.get("candidate_local_radius"))]
    review_only = [
        row
        for row in rows
        if _as_bool(row.get("candidate_sparse_review"))
        and not _as_bool(row.get("candidate_local_radius"))
    ]
    sparse = sparse_site_review(rows)
    return {
        "candidate_edges_total": len(rows),
        "local_radius": _tier_summary(local),
        "sampling_isolation_review_only": _tier_summary(review_only),
        "sparse_site_review": sparse,
        "notes": [
            "20 km is currently a candidate-generation design default, not an ecological threshold.",
            "Sparse-monitoring review edges are never auto-accepted; they exist to prevent monitoring sparsity from masquerading as ecological isolation.",
            "Straight-line open-saltwater support is a coarse land-barrier diagnostic, not a water-route or dispersal model.",
            "A zero water fraction may also reflect narrow estuarine features or map resolution and therefore is not an automatic rejection rule.",
        ],
    }
