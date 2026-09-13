from __future__ import annotations

from dataclasses import dataclass

import pytest

from adaptive_response.models import Edge, Site
from adaptive_response.simulator_criticism import (
    RealContextBundle,
    effort_summary,
    habitat_detection_proxy,
    simulator_criticism_report,
    split_years,
    structural_signature,
)
from adaptive_response.world_models import GeneratedWorld, WorldModelContext


def context() -> WorldModelContext:
    return WorldModelContext(
        sites=(
            Site("a", 0.0, 0.0, 0.2, {"status": "OPEN"}),
            Site("b", 1.0, 0.0, 0.8, {"status": "OPEN"}),
            Site("c", 3.0, 0.0, 0.5, {"status": "OPEN"}),
        ),
        edges=(Edge("a", "b", 1.0),),
        initial_detection="a",
    )


def test_split_years_holds_out_latest_years() -> None:
    calibration, reality = split_years([2020, 2021, 2022, 2023, 2024], holdout_years=2)
    assert calibration == (2020, 2021, 2022)
    assert reality == (2023, 2024)


def test_habitat_proxy_uses_site_year_detection_and_smoothing() -> None:
    rows = [
        {"site_id": "a", "year": 2020, "habitat": "Lagoon", "detected": False},
        {"site_id": "a", "year": 2020, "habitat": "Lagoon", "detected": True},
        {"site_id": "b", "year": 2020, "habitat": "Channel", "detected": False},
        {"site_id": "b", "year": 2021, "habitat": "Channel", "detected": False},
        # Held-out year must not enter the calibration proxy.
        {"site_id": "b", "year": 2022, "habitat": "Channel", "detected": True},
    ]
    scores, support, counts = habitat_detection_proxy(rows, [2020, 2021])

    assert counts["Lagoon"] == (1, 1)
    assert counts["Channel"] == (0, 2)
    assert scores["Lagoon"] == pytest.approx(2 / 3)
    assert scores["Channel"] == pytest.approx(1 / 4)
    assert support == {"Channel": 2, "Lagoon": 1}


def test_structural_signature_distinguishes_connected_and_fragmented_extent() -> None:
    connected = structural_signature(context(), {"a": True, "b": True, "c": False})
    fragmented = structural_signature(context(), {"a": True, "b": False, "c": True})

    assert connected.occupied_count == 2
    assert connected.occupied_components == 1
    assert connected.occupied_with_neighbor_fraction == pytest.approx(1.0)
    assert connected.median_nearest_neighbor_km == pytest.approx(1.0)

    assert fragmented.occupied_components == 2
    assert fragmented.occupied_with_neighbor_fraction == pytest.approx(0.0)
    assert fragmented.median_nearest_neighbor_km == pytest.approx(3.0)


def test_effort_summary_preserves_missing_effort() -> None:
    rows = [
        {"site_id": "a", "trap_sets": 6.0, "effort_missing": False},
        {"site_id": "a", "trap_sets": None, "effort_missing": True},
        {"site_id": "b", "trap_sets": 3.0, "effort_missing": False},
    ]
    summary = effort_summary(rows)
    assert summary["known_rows"] == 2
    assert summary["missing_rows"] == 1
    assert summary["median"] == pytest.approx(4.5)


@dataclass(frozen=True)
class TinyVariableModel:
    family_id: str = "tiny_variable"

    def sample(self, ctx: WorldModelContext, *, seed: int) -> GeneratedWorld:
        occupied = {site.id: False for site in ctx.sites}
        occupied[ctx.initial_detection] = True
        if seed % 2 == 0:
            occupied["b"] = True
        return GeneratedWorld(self.family_id, occupied, seed)


def bundle() -> RealContextBundle:
    rows = (
        {"site_id": "a", "year": 2020, "month": 1, "habitat": "Lagoon", "trap_sets": 6.0, "effort_missing": False, "detected": True},
        {"site_id": "b", "year": 2020, "month": 1, "habitat": "Channel", "trap_sets": 6.0, "effort_missing": False, "detected": False},
        {"site_id": "a", "year": 2021, "month": 1, "habitat": "Lagoon", "trap_sets": 6.0, "effort_missing": False, "detected": True},
        {"site_id": "b", "year": 2021, "month": 1, "habitat": "Channel", "trap_sets": 6.0, "effort_missing": False, "detected": True},
        {"site_id": "a", "year": 2022, "month": 1, "habitat": "Lagoon", "trap_sets": 6.0, "effort_missing": False, "detected": True},
    )
    return RealContextBundle(
        context=context(),
        canonical_rows=rows,
        selected_site_ids=("a", "b", "c"),
        seed_site_id="a",
        incident_summary={
            "selection_inputs": ["initial_detection_site", "frozen_primary_adjacency"],
            "forbidden_inputs": ["future_detections", "hidden_synthetic_truth"],
        },
        calibration_years=(2020,),
        reality_check_years=(2021, 2022),
        habitat_score_by_category={"Lagoon": 2 / 3, "Channel": 1 / 3},
        habitat_support_by_category={"Lagoon": 1, "Channel": 1},
    )


def test_report_is_descriptive_and_does_not_call_observed_positives_truth() -> None:
    report = simulator_criticism_report(
        bundle(),
        [TinyVariableModel()],
        draws_per_family=20,
        seed_start=100,
    )

    assert report["status"] == "SIMULATOR_CRITICISM_REALITY_CHECK_NOT_FIELD_VALIDATION"
    assert report["families"][0]["family_id"] == "tiny_variable"
    assert report["families"][0]["unique_occupancy_patterns"] == 2
    interpretation = " ".join(report["interpretation"]).lower()
    assert "not complete occupancy truth" in interpretation
    assert "do not tune simulator ranges" in interpretation
