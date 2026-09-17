from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from adaptive_response.real_incident_source import (
    build_real_incident,
    eligible_incident_seed_sites,
)
from adaptive_response.rl.real_graph_cases import (
    build_real_incident_case,
    eligible_incident_seed_sites as rl_eligible_incident_seed_sites,
)

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def test_eligible_incident_seed_sites_matches_the_rl_version() -> None:
    assert sorted(eligible_incident_seed_sites()) == sorted(rl_eligible_incident_seed_sites())


def test_build_real_incident_matches_rl_build_real_incident_case() -> None:
    """real_incident_source.build_real_incident is a torch-free re-derivation of
    rl/real_graph_cases.py's build_real_incident_case (see this module's
    docstring for why it is duplicated rather than imported). It must produce
    the identical IncidentConfig for the same inputs, or the UI would be
    running a silently-diverged real-data pipeline."""

    for seed_site_id in sorted(eligible_incident_seed_sites())[:3]:
        for rl_seed in (0, 1234):
            ours = build_real_incident(seed_site_id, budget=18, rl_seed=rl_seed)
            reference = build_real_incident_case(seed_site_id, budget=18, rl_seed=rl_seed)

            assert ours.seed_site_id == reference.seed_site_id
            assert ours.below_preferred_min == reference.below_preferred_min
            assert [asdict(s) for s in ours.incident.sites] == [
                asdict(s) for s in reference.incident.sites
            ]
            assert [asdict(e) for e in ours.incident.edges] == [
                asdict(e) for e in reference.incident.edges
            ]
            assert ours.incident.initial_detection == reference.incident.initial_detection
            assert ours.incident.budget == reference.incident.budget
            assert ours.incident.protocol == reference.incident.protocol


def test_real_incident_source_does_not_require_torch() -> None:
    """A product/UI module must be importable without the optional `rl` extra
    (torch) installed - see pyproject.toml's `rl` extra comment and
    rl/__init__.py's module docstring. Checked in a fresh subprocess: once any
    test in this session has imported adaptive_response.rl, torch is already
    in sys.modules for the rest of the process, which would hide a regression
    here if checked in-process."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import adaptive_response.real_incident_source; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)",
        ],
        cwd=REPO_SRC,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
