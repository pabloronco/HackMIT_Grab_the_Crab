from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adaptive_response.milestone_receipts import (
    publish_r2_real_graph_receipt,
    sha256_file,
    write_real_graph_svg,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_repo(tmp_path: Path) -> None:
    raw = tmp_path / "data/raw"
    for relative, payload in (
        ("wsg_cama/all_abundance_wide_format.csv", "a,b\n1,2\n"),
        ("wsg_effort_geo/PAMA.Month.CPUE.csv", "a,b\n3,4\n"),
        ("wsg_effort_geo/Network_PAMA_CPUE.Map.csv", "a,b\n5,6\n"),
    ):
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    processed = tmp_path / "data/processed"
    _write_csv(
        processed / "r2_real_sites.csv",
        ["site_id", "latitude", "longitude"],
        [
            {"site_id": "A", "latitude": 48.0, "longitude": -122.0},
            {"site_id": "B", "latitude": 48.1, "longitude": -122.1},
            {"site_id": "C", "latitude": 48.2, "longitude": -122.2},
        ],
    )
    _write_csv(
        processed / "r2_real_graph_v0_edges.csv",
        ["src", "dst", "distance_km"],
        [{"src": "A", "dst": "B", "distance_km": 10.0}],
    )
    (processed / "r2_real_graph_v0_audit.json").write_text(
        json.dumps(
            {
                "sites": 3,
                "primary_edges": 1,
                "straight_zero_water_edges_retained": 1,
                "primary_graph": {
                    "components": 2,
                    "largest_component_sites": 2,
                    "isolated_sites": ["C"],
                },
            }
        ),
        encoding="utf-8",
    )
    (processed / "r2_incident_subgraph_audit.json").write_text(
        json.dumps({"seeds_in_preferred_range": 2}), encoding="utf-8"
    )


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_svg_receipt_marks_isolated_sites(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    out = tmp_path / "graph.svg"
    write_real_graph_svg(
        tmp_path / "data/processed/r2_real_sites.csv",
        tmp_path / "data/processed/r2_real_graph_v0_edges.csv",
        out,
    )
    text = out.read_text(encoding="utf-8")
    assert "Real Graph v0" in text
    assert "isolated: C" in text
    assert "site C, degree 0" in text


def test_publish_receipt_copies_only_small_milestone_outputs(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    summary = publish_r2_real_graph_receipt(repo_root=tmp_path)
    out = tmp_path / "reports/milestones/r2_real_graph_v0"
    assert summary["sites"] == 3
    assert (out / "REAL_GRAPH_V0_RECEIPT.md").is_file()
    assert (out / "real_graph_v0_edges.csv").is_file()
    assert (out / "real_graph_v0.svg").is_file()
    assert not (out / "all_abundance_wide_format.csv").exists()
    assert "candidate-generation DESIGN CHOICE" in (out / "REAL_GRAPH_V0_RECEIPT.md").read_text(encoding="utf-8")


def test_publish_receipt_fails_when_processed_outputs_missing(tmp_path: Path) -> None:
    for relative in (
        "data/raw/wsg_cama/all_abundance_wide_format.csv",
        "data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv",
        "data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="processed outputs"):
        publish_r2_real_graph_receipt(repo_root=tmp_path)
