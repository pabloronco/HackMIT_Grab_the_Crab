from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .data_fetch import R0_TARGETS, DryadTarget, list_dataset_files, select_file_metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_size(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("size")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metadata_digest(metadata: dict[str, Any]) -> tuple[str, str] | None:
    """Return (algorithm, hex digest) when Dryad exposes a recognizable checksum."""
    digest = metadata.get("digest")
    if isinstance(digest, dict):
        algorithm = str(digest.get("algorithm", "")).lower().replace("-", "")
        value = str(digest.get("value", "")).lower()
        if algorithm and value:
            return algorithm, value
    if isinstance(digest, str) and ":" in digest:
        algorithm, value = digest.split(":", 1)
        return algorithm.lower().replace("-", ""), value.lower()
    return None


def _hash_with(path: Path, algorithm: str) -> str | None:
    try:
        digest = hashlib.new(algorithm)
    except ValueError:
        return None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def import_downloaded_targets(
    source_dir: Path,
    targets: Iterable[DryadTarget] = R0_TARGETS,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Verify exact user-downloaded Dryad files against public metadata, then copy them.

    This is the supported fallback when Dryad's public landing page permits browser
    downloads but its API download endpoints require authentication. File selection is
    exact by filename; no fuzzy matching or silent substitution is allowed.
    """
    source_dir = source_dir.expanduser()
    if not source_dir.is_dir():
        raise RuntimeError(f"Download directory does not exist: {source_dir}")

    files_by_doi: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []

    for target in targets:
        source = source_dir / target.filename
        if not source.is_file():
            raise RuntimeError(
                f"Missing exact file {target.filename!r} in {source_dir}. "
                "Download it from the Dryad dataset page without renaming it."
            )

        metadata_files = files_by_doi.setdefault(target.doi, list_dataset_files(target.doi))
        metadata = select_file_metadata(metadata_files, target.filename)

        actual_size = source.stat().st_size
        expected_size = _expected_size(metadata)
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {target.filename}: Dryad metadata says "
                f"{expected_size} bytes, local file has {actual_size} bytes."
            )

        digest_check = _metadata_digest(metadata)
        dryad_digest_verified = None
        if digest_check is not None:
            algorithm, expected_digest = digest_check
            local_digest = _hash_with(source, algorithm)
            if local_digest is not None:
                if local_digest != expected_digest:
                    raise RuntimeError(
                        f"Checksum mismatch for {target.filename} using {algorithm}."
                    )
                dryad_digest_verified = True

        target.destination.parent.mkdir(parents=True, exist_ok=True)
        if target.destination.exists() and not overwrite:
            raise RuntimeError(
                f"Destination already exists: {target.destination}. "
                "Use --overwrite only if you intentionally want to replace it."
            )
        shutil.copy2(source, target.destination)

        manifest.append(
            {
                "doi": target.doi,
                "filename": target.filename,
                "source": str(source),
                "destination": str(target.destination),
                "status": "verified_local_import",
                "size": actual_size,
                "sha256": _sha256(target.destination),
                "dryad_size_verified": expected_size is not None,
                "dryad_digest_verified": dryad_digest_verified,
            }
        )

    return manifest


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
