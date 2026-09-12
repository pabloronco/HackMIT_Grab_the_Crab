from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_real_graph_svg(
    sites_csv: Path,
    edges_csv: Path,
    out_svg: Path,
    *,
    width: int = 1000,
    height: int = 760,
    margin: int = 48,
) -> None:
    """Write a dependency-free static SVG receipt of the accepted graph geometry.

    This is a visualization receipt only. It does not alter coordinates or graph
    semantics and must never be used as a modelling input.
    """
    sites = _load_csv(sites_csv)
    edges = _load_csv(edges_csv)
    if not sites:
        raise ValueError("sites_csv is empty")

    required_site = {"site_id", "latitude", "longitude"}
    if not required_site.issubset(sites[0]):
        raise ValueError(f"sites_csv must contain {sorted(required_site)}")
    if edges and not {"src", "dst"}.issubset(edges[0]):
        raise ValueError("edges_csv must contain src,dst")

    coords: dict[str, tuple[float, float]] = {}
    for row in sites:
        coords[str(row["site_id"])] = (float(row["latitude"]), float(row["longitude"]))

    lats = [value[0] for value in coords.values()]
    lons = [value[1] for value in coords.values()]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_span = max(lat_max - lat_min, 1e-9)
    lon_span = max(lon_max - lon_min, 1e-9)

    def xy(site_id: str) -> tuple[float, float]:
        lat, lon = coords[site_id]
        x = margin + (lon - lon_min) / lon_span * (width - 2 * margin)
        y = height - margin - (lat - lat_min) / lat_span * (height - 2 * margin)
        return x, y

    degree = {site_id: 0 for site_id in coords}
    edge_lines: list[str] = []
    for row in edges:
        src, dst = str(row["src"]), str(row["dst"])
        if src not in coords or dst not in coords:
            raise ValueError(f"edge references unknown site: {src}--{dst}")
        degree[src] += 1
        degree[dst] += 1
        x1, y1 = xy(src)
        x2, y2 = xy(dst)
        edge_lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#9aa0a6" stroke-width="1" stroke-opacity="0.55" />'
        )

    node_marks: list[str] = []
    for site_id in sorted(coords):
        x, y = xy(site_id)
        isolated = degree[site_id] == 0
        radius = 5.5 if isolated else 4.0
        fill = "#ffffff" if isolated else "#202124"
        stroke = "#202124"
        node_marks.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"><title>site {site_id}, degree {degree[site_id]}</title></circle>'
        )

    isolated_ids = [site_id for site_id, value in degree.items() if value == 0]
    subtitle = (
        f"{len(coords)} monitoring sites · {len(edges)} primary edges · "
        f"isolated: {', '.join(sorted(isolated_ids)) if isolated_ids else 'none'}"
    )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            '<text x="48" y="28" font-family="sans-serif" font-size="18" font-weight="600">Real Graph v0 — milestone receipt</text>',
            f'<text x="48" y="47" font-family="sans-serif" font-size="12" fill="#5f6368">{subtitle}</text>',
            *edge_lines,
            *node_marks,
            '</svg>',
        ]
    )
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg + "\n", encoding="utf-8")


def publish_r2_real_graph_receipt(
    *,
    repo_root: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Publish small, versionable receipts from local gitignored R0-R2 outputs."""
    out_dir = out_dir or repo_root / "reports/milestones/r2_real_graph_v0"
    processed = repo_root / "data/processed"
    raw = repo_root / "data/raw"

    required = {
        "audit": processed / "r2_real_graph_v0_audit.json",
        "edges": processed / "r2_real_graph_v0_edges.csv",
        "sites": processed / "r2_real_sites.csv",
        "incident": processed / "r2_incident_subgraph_audit.json",
    }
    missing = [str(path.relative_to(repo_root)) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required processed outputs: " + ", ".join(missing))

    raw_inputs = [
        raw / "wsg_cama/all_abundance_wide_format.csv",
        raw / "wsg_effort_geo/PAMA.Month.CPUE.csv",
        raw / "wsg_effort_geo/Network_PAMA_CPUE.Map.csv",
    ]
    missing_raw = [str(path.relative_to(repo_root)) for path in raw_inputs if not path.is_file()]
    if missing_raw:
        raise FileNotFoundError("Missing required raw inputs: " + ", ".join(missing_raw))

    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    incident = json.loads(required["incident"].read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(required["audit"], out_dir / "real_graph_v0_audit.json")
    shutil.copyfile(required["edges"], out_dir / "real_graph_v0_edges.csv")
    shutil.copyfile(required["incident"], out_dir / "incident_subgraph_audit.json")
    write_real_graph_svg(required["sites"], required["edges"], out_dir / "real_graph_v0.svg")

    source_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for path in raw_inputs
    }
    artifact_hashes = {
        name: sha256_file(out_dir / name)
        for name in (
            "real_graph_v0_audit.json",
            "real_graph_v0_edges.csv",
            "incident_subgraph_audit.json",
            "real_graph_v0.svg",
        )
    }

    graph = audit["primary_graph"]
    preferred = incident.get("seeds_in_preferred_range")
    receipt = f"""# R2 Real Graph v0 — Milestone Receipt

**Status:** CURRENT DEFAULT PRIMARY ADJACENCY; not ecological ground truth.  
**Purpose:** small, versioned evidence receipt for the accepted R0–R2 graph milestone.

## Reproduce

From a repository checkout with the three source CSV files present under `data/raw/`:

```bash
python scripts/reproduce_r0_r2.py
python scripts/publish_r2_real_graph_receipt.py
```

Raw data and caches remain gitignored. This folder contains only compact receipts.

## Result

- monitoring sites: **{audit['sites']}**
- primary edges: **{audit['primary_edges']}**
- connected components: **{graph['components']}**
- largest component: **{graph['largest_component_sites']}/{audit['sites']}**
- isolated sites: **{', '.join(graph['isolated_sites']) if graph['isolated_sites'] else 'none'}**
- candidate seeds naturally producing preferred 12–20 node incident graphs: **{preferred if preferred is not None else 'see incident_subgraph_audit.json'}**
- straight-zero-water local edges retained after curved-route audit: **{audit['straight_zero_water_edges_retained']}**

## Decision semantics

- The **20 km direct radius is a candidate-generation DESIGN CHOICE**, not a species dispersal threshold.
- Straight-line water sampling is diagnostic only; the old `straight_water == 0 => reject` rule is **REJECTED**.
- Curved SalishSeaCast route existence is a geometry sanity check; route length/detour are context and sensitivity inputs, not dispersal probabilities.
- Sparse-monitoring review-only edges do not repair connectivity automatically.
- `connectivity_weight` remains **OPEN**.
- Hidden truth and future detections are forbidden inputs to incident-subgraph extraction.

## Source-file SHA-256

"""
    for name, digest in source_hashes.items():
        receipt += f"- `{name}`: `{digest}`\n"
    receipt += "\n## Receipt-artifact SHA-256\n\n"
    for name, digest in artifact_hashes.items():
        receipt += f"- `{name}`: `{digest}`\n"
    receipt += "\n## Provenance notes\n\n"
    receipt += (
        "Washington Sea Grant / Crab Team monitoring inputs provide observed green-crab outcomes, effort metadata, and monitoring coordinates. "
        "ShoreZone and SalishSeaCast are queried by the reproduction scripts as public external context. "
        "The receipt does not vendor raw datasets or remote caches into Git.\n"
    )
    (out_dir / "REAL_GRAPH_V0_RECEIPT.md").write_text(receipt, encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes,
        "sites": audit["sites"],
        "primary_edges": audit["primary_edges"],
        "components": graph["components"],
        "isolated_sites": graph["isolated_sites"],
    }
