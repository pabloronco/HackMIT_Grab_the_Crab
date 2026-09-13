from adaptive_response.real_multisite_evidence import (
    connected_components,
    observed_positive_footprint_report,
)


def _rows():
    return [
        {"site_id": "1", "year": 2020, "month": 1, "detected": True},
        {"site_id": "1", "year": 2020, "month": 2, "detected": True},
        {"site_id": "2", "year": 2020, "month": 1, "detected": False},
        {"site_id": "2", "year": 2021, "month": 1, "detected": True},
        {"site_id": "1", "year": 2021, "month": 1, "detected": False},
        {"site_id": "3", "year": 2021, "month": 1, "detected": False},
        {"site_id": "3", "year": 2022, "month": 1, "detected": True},
        {"site_id": "4", "year": 2020, "month": 1, "detected": False},
        {"site_id": "4", "year": 2022, "month": 1, "detected": True},
    ]


def _edges():
    return [
        {"src": "1", "dst": "2"},
        {"src": "2", "dst": "3"},
    ]


def test_connected_components_preserve_real_disconnection() -> None:
    components = connected_components(["1", "2", "3", "4"], _edges())
    assert components == (("1", "2", "3"), ("4",))


def test_monthly_duplicate_detections_collapse_to_one_positive_site_year() -> None:
    report = observed_positive_footprint_report(_rows(), ["1", "2", "3", "4"], _edges())
    first = report["network"]["timeline"][0]
    assert first["year"] == 2020
    assert first["observed_positive_site_count"] == 1
    assert first["observed_positive_sites"] == ["1"]


def test_new_and_cumulative_positive_footprints_are_distinct() -> None:
    report = observed_positive_footprint_report(_rows(), ["1", "2", "3", "4"], _edges())
    timeline = report["network"]["timeline"]
    assert timeline[0]["new_observed_positive_sites"] == ["1"]
    assert timeline[1]["new_observed_positive_sites"] == ["2"]
    assert timeline[1]["cumulative_observed_positive_sites"] == ["1", "2"]
    assert timeline[2]["new_observed_positive_sites"] == ["3", "4"]
    assert timeline[2]["cumulative_observed_positive_site_count"] == 4


def test_nondetection_never_removes_a_site_from_cumulative_footprint() -> None:
    report = observed_positive_footprint_report(_rows(), ["1", "2", "3", "4"], _edges())
    timeline = report["network"]["timeline"]
    # Site 1 is a non-detection in 2021 but remains in observed-positive history.
    assert "1" in timeline[1]["cumulative_observed_positive_sites"]


def test_threshold_crossings_are_descriptive_and_component_scoped() -> None:
    report = observed_positive_footprint_report(_rows(), ["1", "2", "3", "4"], _edges())
    assert report["network"]["threshold_crossings"]["3"] == {
        "reached": True,
        "first_year_reached": 2022,
    }
    main_component = report["components"][0]
    assert main_component["threshold_crossings"]["3"]["reached"] is True
    assert main_component["threshold_crossings"]["5"]["reached"] is False
    assert report["components"][1]["threshold_crossings"]["3"]["reached"] is False


def test_report_never_claims_latent_occupancy_or_biological_episode_identity() -> None:
    report = observed_positive_footprint_report(_rows(), ["1", "2", "3", "4"], _edges())
    assert report["status"] == "DESCRIPTIVE_OBSERVED_POSITIVE_FOOTPRINT_NOT_LATENT_OCCUPANCY"
    assert "never treated as true absence" in report["semantics"]["nondetection"]
    assert "not a claim" in report["semantics"]["episode_proxy"]
