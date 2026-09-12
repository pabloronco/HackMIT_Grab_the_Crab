from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from adaptive_response.data_fetch import (
    DryadTarget,
    _dataset_download_url,
    _embedded_items,
    _extract_exact_from_archive,
    _metadata_download_url,
    _resource_id,
    select_file_metadata,
)


def test_embedded_items_accepts_hal_payload() -> None:
    payload = {
        "_embedded": {
            "stash:files": [
                {"_links": {"self": {"href": "/api/v2/files/1"}}, "path": "a.csv"},
                {"_links": {"self": {"href": "/api/v2/files/2"}}, "path": "b.csv"},
            ]
        }
    }
    assert [item["path"] for item in _embedded_items(payload)] == ["a.csv", "b.csv"]


def test_resource_id_accepts_current_dryad_hal_self_links() -> None:
    version = {"_links": {"self": {"href": "/api/v2/versions/355108"}}}
    file_item = {
        "_links": {
            "self": {"href": "/api/v2/files/3985003"},
            "stash:download": {"href": "/api/v2/files/3985003/download"},
        }
    }
    assert _resource_id(version, "version") == 355108
    assert _resource_id(file_item, "file") == 3985003


def test_resource_id_keeps_legacy_top_level_id_compatibility() -> None:
    assert _resource_id({"id": "123"}, "file") == 123


def test_metadata_download_url_resolves_relative_hal_link() -> None:
    metadata = {
        "_links": {"stash:download": {"href": "/api/v2/files/3985003/download"}}
    }
    assert _metadata_download_url(metadata) == "https://datadryad.org/api/v2/files/3985003/download"


def test_dataset_download_url_encodes_doi_for_public_archive_endpoint() -> None:
    url = _dataset_download_url("doi:10.5061/dryad.example/test")
    assert url == "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.example%2Ftest/download"


def test_extract_exact_from_archive_uses_exact_basename(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    destination = tmp_path / "out" / "DailyMaxTemperature.csv"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/DailyMaxTemperature.csv", "a,b\n1,2\n")
        archive.writestr("nested/not_DailyMaxTemperature.csv", "wrong\n")

    _extract_exact_from_archive(archive_path, "DailyMaxTemperature.csv", destination)
    assert destination.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_extract_exact_from_archive_rejects_missing_or_duplicate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("other.csv", "x\n")
    with pytest.raises(RuntimeError, match="exactly one archive member"):
        _extract_exact_from_archive(missing, "wanted.csv", tmp_path / "wanted.csv")

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("a/x.csv", "one\n")
        archive.writestr("b/x.csv", "two\n")
    with pytest.raises(RuntimeError, match="exactly one archive member"):
        _extract_exact_from_archive(duplicate, "x.csv", tmp_path / "x.csv")


def test_select_file_metadata_matches_exact_basename() -> None:
    files = [
        {"_links": {"self": {"href": "/api/v2/files/1"}}, "path": "folder/all_abundance_wide_format.csv"},
        {"_links": {"self": {"href": "/api/v2/files/2"}}, "path": "folder/DailyMaxTemperature.csv"},
    ]
    selected = select_file_metadata(files, "DailyMaxTemperature.csv")
    assert _resource_id(selected, "file") == 2


def test_select_file_metadata_rejects_missing_or_ambiguous() -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        select_file_metadata([], "x.csv")
    with pytest.raises(RuntimeError, match="exactly one"):
        select_file_metadata(
            [
                {"_links": {"self": {"href": "/api/v2/files/1"}}, "path": "x.csv"},
                {"_links": {"self": {"href": "/api/v2/files/2"}}, "path": "nested/x.csv"},
            ],
            "x.csv",
        )


def test_dryad_target_is_explicit_about_destination() -> None:
    target = DryadTarget("doi:example", "data.csv", Path("data/raw/data.csv"))
    assert target.filename == "data.csv"
    assert target.destination.as_posix() == "data/raw/data.csv"
