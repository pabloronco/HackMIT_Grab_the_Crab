from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DRYAD_API_BASE = "https://datadryad.org/api/v2"
DRYAD_ORIGIN = "https://datadryad.org"
DRYAD_FILE_STREAM_BASES = (
    "https://datadryad.org/downloads/file_stream/{file_id}",
    "https://datadryad.org/stash/downloads/file_stream/{file_id}",
)
USER_AGENT = "adaptive-first-response-r0-data-audit/0.1"


@dataclass(frozen=True)
class DryadTarget:
    doi: str
    filename: str
    destination: Path


R0_TARGETS: tuple[DryadTarget, ...] = (
    DryadTarget(
        doi="doi:10.5061/dryad.0rxwdbsdt",
        filename="all_abundance_wide_format.csv",
        destination=Path("data/raw/wsg_cama/all_abundance_wide_format.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.0rxwdbsdt",
        filename="DailyMaxTemperature.csv",
        destination=Path("data/raw/wsg_cama/DailyMaxTemperature.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.sqv9s4ndp",
        filename="PAMA.Month.CPUE.csv",
        destination=Path("data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv"),
    ),
    DryadTarget(
        doi="doi:10.5061/dryad.sqv9s4ndp",
        filename="Network_PAMA_CPUE.Map.csv",
        destination=Path("data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv"),
    ),
)


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _embedded_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return HAL-style embedded object lists without depending on Dryad link names."""
    embedded = payload.get("_embedded", {})
    if isinstance(embedded, dict):
        for value in embedded.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    for key in ("items", "versions", "files"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    if isinstance(payload, list):  # pragma: no cover - defensive, type kept for API convenience
        return payload
    return []


def _resource_id(item: dict[str, Any], resource: str) -> int:
    """Extract a Dryad numeric id from either a legacy `id` or HAL link."""
    direct = item.get("id")
    try:
        if direct is not None:
            return int(direct)
    except (TypeError, ValueError):
        pass

    plural = f"{resource}s"
    links = item.get("_links", {})
    if isinstance(links, dict):
        ordered_keys = ["self", f"stash:{resource}", "stash:download"]
        for key in ordered_keys + [key for key in links if key not in ordered_keys]:
            link = links.get(key)
            if not isinstance(link, dict):
                continue
            href = link.get("href")
            if not isinstance(href, str):
                continue
            match = re.search(rf"/{re.escape(plural)}/(\d+)(?:/|$)", href)
            if match:
                return int(match.group(1))

    raise RuntimeError(f"Dryad {resource} metadata has no usable numeric id")


def _latest_version_id(doi: str) -> int:
    encoded = urllib.parse.quote(doi, safe="")
    payload = _request_json(f"{DRYAD_API_BASE}/datasets/{encoded}/versions")
    versions = _embedded_items(payload)
    if not versions:
        raise RuntimeError(f"Dryad returned no versions for {doi}")

    def version_key(item: dict[str, Any]) -> tuple[int, int]:
        number = item.get("versionNumber")
        try:
            number_i = int(number)
        except (TypeError, ValueError):
            number_i = -1
        try:
            id_i = _resource_id(item, "version")
        except RuntimeError:
            id_i = -1
        return number_i, id_i

    latest = max(versions, key=version_key)
    try:
        return _resource_id(latest, "version")
    except RuntimeError as exc:
        raise RuntimeError(f"Dryad version metadata for {doi} has no usable id/link") from exc


def list_dataset_files(doi: str) -> list[dict[str, Any]]:
    version_id = _latest_version_id(doi)
    payload = _request_json(f"{DRYAD_API_BASE}/versions/{version_id}/files")
    files = _embedded_items(payload)
    if not files:
        raise RuntimeError(f"Dryad returned no files for {doi} version {version_id}")
    return files


def select_file_metadata(files: Iterable[dict[str, Any]], filename: str) -> dict[str, Any]:
    matches = [item for item in files if Path(str(item.get("path", ""))).name == filename]
    if len(matches) != 1:
        available = sorted(Path(str(item.get("path", ""))).name for item in files)
        raise RuntimeError(
            f"Expected exactly one Dryad file named {filename!r}; found {len(matches)}. "
            f"Available files: {available}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_download_url(metadata: dict[str, Any]) -> str | None:
    links = metadata.get("_links", {})
    if not isinstance(links, dict):
        return None
    download = links.get("stash:download")
    if not isinstance(download, dict):
        return None
    href = download.get("href")
    if not isinstance(href, str) or not href:
        return None
    return urllib.parse.urljoin(DRYAD_ORIGIN, href)


def _dataset_download_url(doi: str) -> str:
    encoded = urllib.parse.quote(doi, safe="")
    return f"{DRYAD_API_BASE}/datasets/{encoded}/download"


def _download_url(url: str, destination: Path, *, timeout: int = 300) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/zip,application/octet-stream,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{url} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{url} -> {exc.reason}") from exc


def _download_file(metadata: dict[str, Any], file_id: int, destination: Path) -> None:
    """Fallback: try individual-file endpoints when the dataset archive is unavailable."""
    urls: list[str] = []
    advertised = _metadata_download_url(metadata)
    if advertised:
        urls.append(advertised)
    urls.extend(template.format(file_id=file_id) for template in DRYAD_FILE_STREAM_BASES)

    errors: list[str] = []
    for url in dict.fromkeys(urls):
        try:
            _download_url(url, destination, timeout=120)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
            destination.unlink(missing_ok=True)

    raise RuntimeError("Unable to download Dryad file. " + "; ".join(errors))


def _download_dataset_archive(doi: str, destination: Path) -> None:
    """Download Dryad's documented public latest-version dataset ZIP endpoint."""
    url = _dataset_download_url(doi)
    _download_url(url, destination, timeout=600)
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Dryad dataset download for {doi} did not return a ZIP archive")


def _extract_exact_from_archive(archive_path: Path, filename: str, destination: Path) -> None:
    """Extract exactly one member matching the requested basename, never a fuzzy match."""
    with zipfile.ZipFile(archive_path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == filename]
        if len(matches) != 1:
            available = sorted(Path(name).name for name in archive.namelist() if not name.endswith("/"))
            raise RuntimeError(
                f"Expected exactly one archive member named {filename!r}; found {len(matches)}. "
                f"Available files: {available}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(matches[0]) as source, destination.open("wb") as out:
            shutil.copyfileobj(source, out)


def _expected_digest(metadata: dict[str, Any]) -> str | None:
    digest = metadata.get("digest")
    digest_type = str(metadata.get("digestType", "")).lower().replace("_", "-")
    if isinstance(digest, str) and digest and digest_type in {"sha-256", "sha256"}:
        return digest.lower()
    return None


def fetch_targets(
    targets: Iterable[DryadTarget] = R0_TARGETS,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    target_list = list(targets)
    by_doi: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="r0_dryad_archives_") as archive_dir_raw:
        archive_dir = Path(archive_dir_raw)
        archive_paths: dict[str, Path | None] = {}
        archive_errors: dict[str, str] = {}

        for target in target_list:
            target.destination.parent.mkdir(parents=True, exist_ok=True)
            if target.destination.exists() and not overwrite:
                manifest.append(
                    {
                        "doi": target.doi,
                        "filename": target.filename,
                        "destination": str(target.destination),
                        "status": "already_present",
                        "size": target.destination.stat().st_size,
                        "sha256": _sha256(target.destination),
                    }
                )
                continue

            files = by_doi.setdefault(target.doi, list_dataset_files(target.doi))
            metadata = select_file_metadata(files, target.filename)
            try:
                file_id = _resource_id(metadata, "file")
            except RuntimeError as exc:
                raise RuntimeError(f"Dryad metadata for {target.filename} has no usable file id/link") from exc

            expected_size_raw = metadata.get("size")
            try:
                expected_size = int(expected_size_raw) if expected_size_raw is not None else None
            except (TypeError, ValueError):
                expected_size = None
            expected_digest = _expected_digest(metadata)

            with tempfile.TemporaryDirectory(prefix="r0_dryad_file_") as temp_dir:
                temp_path = Path(temp_dir) / target.filename
                method = "dataset_archive"

                if target.doi not in archive_paths:
                    archive_path = archive_dir / f"dataset_{len(archive_paths):02d}.zip"
                    try:
                        _download_dataset_archive(target.doi, archive_path)
                        archive_paths[target.doi] = archive_path
                    except RuntimeError as exc:
                        archive_paths[target.doi] = None
                        archive_errors[target.doi] = str(exc)

                archive_path = archive_paths[target.doi]
                if archive_path is not None:
                    _extract_exact_from_archive(archive_path, target.filename, temp_path)
                else:
                    method = "individual_file_fallback"
                    try:
                        _download_file(metadata, file_id, temp_path)
                    except RuntimeError as exc:
                        archive_error = archive_errors.get(target.doi, "unknown archive error")
                        raise RuntimeError(
                            f"Dryad download failed for {target.filename}. "
                            f"Dataset archive attempt: {archive_error}. "
                            f"Individual-file attempt: {exc}"
                        ) from exc

                actual_size = temp_path.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    raise RuntimeError(
                        f"Size mismatch for {target.filename}: expected {expected_size}, got {actual_size}"
                    )
                actual_digest = _sha256(temp_path)
                if expected_digest is not None and actual_digest.lower() != expected_digest:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {target.filename}: expected {expected_digest}, got {actual_digest}"
                    )
                shutil.move(str(temp_path), str(target.destination))

            manifest.append(
                {
                    "doi": target.doi,
                    "filename": target.filename,
                    "dryad_file_id": file_id,
                    "destination": str(target.destination),
                    "status": "downloaded",
                    "download_method": method,
                    "size": target.destination.stat().st_size,
                    "sha256": _sha256(target.destination),
                    "dryad_sha256": expected_digest,
                }
            )

    return manifest
