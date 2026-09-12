from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from adaptive_response.shoreline_context import fetch_shoreline_context


WIDTH = 1500
HEIGHT = 1050
PADDING_X = 70
PADDING_TOP = 135
PADDING_BOTTOM = 60


def _pick(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return str(row[name])
    return default


def load_sites(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | str]] = []
        for row in reader:
            rows.append(
                {
                    "site_id": _pick(row, "site_id", "SiteID", "site"),
                    "latitude": float(_pick(row, "latitude", "lat")),
                    "longitude": float(_pick(row, "longitude", "lon", "lng")),
                }
            )
    if not rows:
        raise ValueError("Site table is empty.")
    return rows


def load_edges(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        edges: list[dict[str, str]] = []
        for row in reader:
            src = _pick(row, "src", "source", "u")
            dst = _pick(row, "dst", "target", "v")
            if src and dst:
                edges.append({"src": src, "dst": dst})
    return edges


def bbox_for_sites(
    sites: list[dict[str, float | str]],
    *,
    pad_ratio: float = 0.06,
) -> tuple[float, float, float, float]:
    lats = [float(site["latitude"]) for site in sites]
    lons = [float(site["longitude"]) for site in sites]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_pad = max((max_lat - min_lat) * pad_ratio, 0.025)
    lon_pad = max((max_lon - min_lon) * pad_ratio, 0.025)
    return (
        min_lon - lon_pad,
        min_lat - lat_pad,
        max_lon + lon_pad,
        max_lat + lat_pad,
    )


def _projected_xy(lon: float, lat: float, lat0: float) -> tuple[float, float]:
    # Equirectangular display projection; preserves local aspect much better than
    # treating longitude and latitude as equal-width Cartesian coordinates.
    return lon * math.cos(math.radians(lat0)), lat


def make_projector(bbox: tuple[float, float, float, float]):
    min_lon, min_lat, max_lon, max_lat = bbox
    lat0 = (min_lat + max_lat) / 2.0
    min_x, min_y = _projected_xy(min_lon, min_lat, lat0)
    max_x, max_y = _projected_xy(max_lon, max_lat, lat0)
    data_w = max_x - min_x
    data_h = max_y - min_y
    draw_w = WIDTH - 2 * PADDING_X
    draw_h = HEIGHT - PADDING_TOP - PADDING_BOTTOM
    scale = min(draw_w / data_w, draw_h / data_h)
    used_w = data_w * scale
    used_h = data_h * scale
    left = (WIDTH - used_w) / 2.0
    top = PADDING_TOP + (draw_h - used_h) / 2.0

    def project(lon: float, lat: float) -> tuple[float, float]:
        x, y = _projected_xy(lon, lat, lat0)
        sx = left + (x - min_x) * scale
        sy = top + used_h - (y - min_y) * scale
        return sx, sy

    return project


def degrees_from_edges(
    sites: list[dict[str, float | str]],
    edges: list[dict[str, str]],
) -> dict[str, int]:
    degrees = {str(site["site_id"]): 0 for site in sites}
    for edge in edges:
        if edge["src"] in degrees:
            degrees[edge["src"]] += 1
        if edge["dst"] in degrees:
            degrees[edge["dst"]] += 1
    return degrees


def render_svg(
    sites: list[dict[str, float | str]],
    edges: list[dict[str, str]],
    coastline_paths: tuple[tuple[tuple[float, float], ...], ...],
    *,
    object_count: int,
    source_url: str,
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    project = make_projector(bbox)
    by_id = {str(site["site_id"]): site for site in sites}
    degrees = degrees_from_edges(sites, edges)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>",
        """
        .title { font: 700 29px Arial, sans-serif; fill: #102a43; }
        .subtitle { font: 400 15px Arial, sans-serif; fill: #486581; }
        .source { font: 400 12px Arial, sans-serif; fill: #627d98; }
        .coast-halo { fill: none; stroke: #ffffff; stroke-width: 4.6; opacity: 0.96; stroke-linejoin: round; stroke-linecap: round; }
        .coast { fill: none; stroke: #526d82; stroke-width: 2.25; opacity: 0.92; stroke-linejoin: round; stroke-linecap: round; }
        .edge-halo { stroke: #ffffff; stroke-width: 4.5; opacity: 0.84; }
        .edge { stroke: #2f6f73; stroke-width: 2.0; opacity: 0.52; }
        .node { fill: #0b7285; stroke: #ffffff; stroke-width: 1.8; }
        .node-isolated { fill: #c96f12; stroke: #ffffff; stroke-width: 1.8; }
        .label-halo { font: 700 11px Arial, sans-serif; fill: none; stroke: #ffffff; stroke-width: 3.5; stroke-linejoin: round; }
        .label { font: 700 11px Arial, sans-serif; fill: #102a43; }
        .legend { font: 400 13px Arial, sans-serif; fill: #334e68; }
        .legend-title { font: 700 14px Arial, sans-serif; fill: #243b53; }
        .legend-box { fill: #ffffff; fill-opacity: 0.93; stroke: #bcccdc; stroke-width: 1; }
        .water { fill: #eef8fb; }
        """,
        "</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" class="water"/>',
        '<text x="42" y="48" class="title">R2 Real Monitoring Graph v0</text>',
        '<text x="42" y="75" class="subtitle">49 monitoring sites, primary adjacency, and real Washington DNR ShoreZone coastline</text>',
        f'<text x="42" y="99" class="source">Shoreline context: DNR ShoreZone SZLine (layer 46) · {object_count} intersecting shoreline features · visual context only</text>',
    ]

    # Real shoreline first, beneath the graph. White halo helps preserve the
    # coastline shape even where graph edges overlap it.
    for path in coastline_paths:
        points = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (project(lon, lat) for lon, lat in path)
        )
        parts.append(f'<polyline points="{points}" class="coast-halo"/>')
        parts.append(f'<polyline points="{points}" class="coast"/>')

    for edge in edges:
        a = by_id.get(edge["src"])
        b = by_id.get(edge["dst"])
        if a is None or b is None:
            continue
        x1, y1 = project(float(a["longitude"]), float(a["latitude"]))
        x2, y2 = project(float(b["longitude"]), float(b["latitude"]))
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge-halo"/>'
        )
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge"/>'
        )

    for site in sites:
        site_id = str(site["site_id"])
        x, y = project(float(site["longitude"]), float(site["latitude"]))
        cls = "node-isolated" if degrees[site_id] == 0 else "node"
        radius = 6.8 if degrees[site_id] == 0 else 5.6
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" class="{cls}"/>')
        label_x, label_y = x + 8.0, y - 7.0
        parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="label-halo">{site_id}</text>')
        parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" class="label">{site_id}</text>')

    lx, ly = WIDTH - 330, 34
    parts.extend(
        [
            f'<rect x="{lx}" y="{ly}" width="286" height="132" rx="12" class="legend-box"/>',
            f'<text x="{lx+16}" y="{ly+25}" class="legend-title">Map key</text>',
            f'<line x1="{lx+18}" y1="{ly+49}" x2="{lx+58}" y2="{ly+49}" class="coast"/>',
            f'<text x="{lx+70}" y="{ly+54}" class="legend">DNR ShoreZone coastline</text>',
            f'<line x1="{lx+18}" y1="{ly+76}" x2="{lx+58}" y2="{ly+76}" class="edge"/>',
            f'<text x="{lx+70}" y="{ly+81}" class="legend">Primary graph edge</text>',
            f'<circle cx="{lx+38}" cy="{ly+102}" r="5.6" class="node"/>',
            f'<text x="{lx+70}" y="{ly+107}" class="legend">Connected monitoring site</text>',
            f'<circle cx="{lx+38}" cy="{ly+124}" r="6.5" class="node-isolated"/>',
            f'<text x="{lx+70}" y="{ly+129}" class="legend">Isolated monitoring site</text>',
        ]
    )

    parts.append(f'<!-- shoreline_source={source_url} -->')
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the accepted R2 graph over real DNR ShoreZone coastline geometry."
    )
    parser.add_argument("--sites", default="data/processed/r2_real_sites.csv")
    parser.add_argument("--edges", default="data/processed/r2_real_graph_v0_edges.csv")
    parser.add_argument(
        "--cache",
        default="data/cache/shorezone/r2_real_graph_v0_shoreline.json",
        help="Gitignored cache for fetched DNR ShoreZone polylines.",
    )
    parser.add_argument(
        "--out",
        default="reports/milestones/r2_real_graph_v0/real_graph_v0.svg",
    )
    parser.add_argument(
        "--receipt",
        default="reports/milestones/r2_real_graph_v0/shoreline_context_receipt.json",
    )
    args = parser.parse_args()

    sites_path = Path(args.sites)
    edges_path = Path(args.edges)
    if not sites_path.is_file():
        raise SystemExit(f"Missing sites CSV: {sites_path}")
    if not edges_path.is_file():
        raise SystemExit(f"Missing edges CSV: {edges_path}")

    sites = load_sites(sites_path)
    edges = load_edges(edges_path)
    bbox = bbox_for_sites(sites)

    print("Fetching/reusing REAL DNR ShoreZone coastline geometry (SZLine layer 46)...")
    print("No schematic fallback is allowed in this renderer.")
    shoreline = fetch_shoreline_context(
        bbox,
        cache_path=Path(args.cache),
    )

    out_path = Path(args.out)
    render_svg(
        sites,
        edges,
        shoreline.paths,
        object_count=shoreline.object_count,
        source_url=shoreline.source_url,
        bbox=bbox,
        out_path=out_path,
    )

    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "status": "VISUAL_CONTEXT_ONLY",
                "source": "Washington DNR ShoreZone full inventory SZLine",
                "layer_id": 46,
                "source_url": shoreline.source_url,
                "bbox_wgs84": list(bbox),
                "shoreline_feature_count": shoreline.object_count,
                "shoreline_path_count": shoreline.path_count,
                "graph_sites": len(sites),
                "graph_edges": len(edges),
                "note": "Shoreline geometry improves map legibility; it is not dispersal probability or graph connectivity evidence.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"SVG written to: {out_path}")
    print(f"Shoreline receipt: {receipt_path}")
    print(
        f"REAL coastline features={shoreline.object_count} | paths={shoreline.path_count} | "
        f"sites={len(sites)} | edges={len(edges)}"
    )


if __name__ == "__main__":
    main()
