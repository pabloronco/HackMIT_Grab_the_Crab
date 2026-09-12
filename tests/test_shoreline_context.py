from pathlib import Path

import pytest

from adaptive_response.shoreline_context import fetch_shoreline_context


def test_fetches_ids_then_geometry_in_batches(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_fetch(url: str, params: dict, timeout: float) -> dict:
        calls.append(dict(params))
        if params.get("returnIdsOnly") == "true":
            return {"objectIds": [7, 2, 5]}
        assert params["objectIds"] == "2,5" or params["objectIds"] == "7"
        ids = [int(value) for value in params["objectIds"].split(",")]
        return {
            "features": [
                {
                    "attributes": {"OBJECTID": object_id},
                    "geometry": {
                        "paths": [
                            [[-123.0 + object_id / 1000.0, 48.0], [-122.9, 48.1]]
                        ]
                    },
                }
                for object_id in ids
            ]
        }

    cache = tmp_path / "shoreline.json"
    context = fetch_shoreline_context(
        (-123.5, 47.5, -122.0, 49.0),
        cache_path=cache,
        fetch_json=fake_fetch,
        batch_size=2,
    )

    assert context.object_count == 3
    assert context.path_count == 3
    assert cache.is_file()
    assert len(calls) == 3
    assert calls[0]["geometryType"] == "esriGeometryEnvelope"
    assert calls[0]["inSR"] == 4326
    assert calls[1]["outSR"] == 4326


def test_exact_bbox_cache_avoids_network(tmp_path: Path) -> None:
    responses = iter(
        [
            {"objectIds": [1]},
            {
                "features": [
                    {"geometry": {"paths": [[[-123.0, 48.0], [-122.9, 48.1]]]}}
                ]
            },
        ]
    )

    def first_fetch(url: str, params: dict, timeout: float) -> dict:
        return next(responses)

    bbox = (-123.5, 47.5, -122.0, 49.0)
    cache = tmp_path / "shoreline.json"
    original = fetch_shoreline_context(
        bbox,
        cache_path=cache,
        fetch_json=first_fetch,
    )

    def must_not_fetch(url: str, params: dict, timeout: float) -> dict:
        raise AssertionError("network should not be used for exact-bbox cache")

    cached = fetch_shoreline_context(
        bbox,
        cache_path=cache,
        fetch_json=must_not_fetch,
    )
    assert cached == original


def test_no_real_shoreline_is_error_not_fake_fallback() -> None:
    def fake_fetch(url: str, params: dict, timeout: float) -> dict:
        return {"objectIds": []}

    with pytest.raises(RuntimeError, match="no shoreline features"):
        fetch_shoreline_context(
            (-123.5, 47.5, -122.0, 49.0),
            fetch_json=fake_fetch,
        )


def test_invalid_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        fetch_shoreline_context((-122.0, 49.0, -123.0, 47.0))
