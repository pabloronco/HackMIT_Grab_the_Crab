from adaptive_response.adaptive_water_audit import (
    annotate_adaptive_candidates_with_water,
    sparse_site_review,
    summarize_adaptive_water_audit,
)


def _edges():
    return [
        {
            "src": "A",
            "dst": "B",
            "distance_km": 5.0,
            "candidate_local_radius": True,
            "candidate_sparse_review": False,
            "sparse_review_requested_by": "",
        },
        {
            "src": "B",
            "dst": "C",
            "distance_km": 25.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "sparse_review_requested_by": "C",
        },
    ]


def test_annotation_preserves_candidate_provenance() -> None:
    coordinates = {
        "A": (48.0, -122.0),
        "B": (48.0, -122.1),
        "C": (48.0, -122.2),
    }
    rows = annotate_adaptive_candidates_with_water(
        _edges(),
        coordinates,
        samples_per_edge=3,
        checker=lambda lat, lon: True,
    )
    assert rows[0]["candidate_local_radius"] is True
    assert rows[1]["candidate_sparse_review"] is True
    assert rows[1]["straight_marine_fraction"] == 1.0


def test_summary_separates_local_and_review_only() -> None:
    rows = _edges()
    rows[0].update(
        {
            "straight_marine_fraction": 1.0,
            "straight_marine_all": True,
            "straight_marine_none": False,
            "straight_marine_pattern": "11111",
        }
    )
    rows[1].update(
        {
            "straight_marine_fraction": 0.0,
            "straight_marine_all": False,
            "straight_marine_none": True,
            "straight_marine_pattern": "00000",
        }
    )
    summary = summarize_adaptive_water_audit(rows)
    assert summary["local_radius"]["edges"] == 1
    assert summary["local_radius"]["all_water"] == 1
    assert summary["sampling_isolation_review_only"]["edges"] == 1
    assert summary["sampling_isolation_review_only"]["no_water"] == 1


def test_sparse_site_review_is_grouped_by_requesting_site() -> None:
    rows = _edges()
    rows[1].update(
        {
            "straight_marine_fraction": 0.6,
            "straight_marine_pattern": "11100",
        }
    )
    grouped = sparse_site_review(rows)
    assert list(grouped) == ["C"]
    assert grouped["C"][0]["neighbor"] == "B"
    assert grouped["C"][0]["distance_km"] == 25.0


def test_sparse_site_review_orders_by_water_support_then_distance() -> None:
    rows = [
        {
            "src": "X",
            "dst": "A",
            "distance_km": 30.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "sparse_review_requested_by": "X",
            "straight_marine_fraction": 0.4,
            "straight_marine_pattern": "11000",
        },
        {
            "src": "X",
            "dst": "B",
            "distance_km": 40.0,
            "candidate_local_radius": False,
            "candidate_sparse_review": True,
            "sparse_review_requested_by": "X",
            "straight_marine_fraction": 1.0,
            "straight_marine_pattern": "11111",
        },
    ]
    grouped = sparse_site_review(rows)
    assert [item["neighbor"] for item in grouped["X"]] == ["B", "A"]
