from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_response.data_fetch import DryadTarget
from adaptive_response.local_data_import import import_downloaded_targets


def test_import_downloaded_targets_rejects_missing_exact_filename(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "downloads"
    source_dir.mkdir()
    target = DryadTarget("doi:test", "exact.csv", tmp_path / "out" / "exact.csv")
    monkeypatch.setattr(
        "adaptive_response.local_data_import.list_dataset_files",
        lambda doi: [{"path": "exact.csv", "size": 3}],
    )
    with pytest.raises(RuntimeError, match="Missing exact file"):
        import_downloaded_targets(source_dir, (target,))


def test_import_downloaded_targets_checks_size_and_copies(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "downloads"
    source_dir.mkdir()
    (source_dir / "exact.csv").write_bytes(b"abc")
    destination = tmp_path / "raw" / "exact.csv"
    target = DryadTarget("doi:test", "exact.csv", destination)
    monkeypatch.setattr(
        "adaptive_response.local_data_import.list_dataset_files",
        lambda doi: [{"path": "exact.csv", "size": 3}],
    )

    records = import_downloaded_targets(source_dir, (target,))

    assert destination.read_bytes() == b"abc"
    assert records[0]["status"] == "verified_local_import"
    assert records[0]["dryad_size_verified"] is True


def test_import_downloaded_targets_rejects_wrong_size(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "downloads"
    source_dir.mkdir()
    (source_dir / "exact.csv").write_bytes(b"abc")
    target = DryadTarget("doi:test", "exact.csv", tmp_path / "out.csv")
    monkeypatch.setattr(
        "adaptive_response.local_data_import.list_dataset_files",
        lambda doi: [{"path": "exact.csv", "size": 4}],
    )
    with pytest.raises(RuntimeError, match="Size mismatch"):
        import_downloaded_targets(source_dir, (target,))
