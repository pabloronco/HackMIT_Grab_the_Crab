from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


REAL_SITE_FIELDS = [
    "site_id",
    "latitude",
    "longitude",
    "crabteam_habitat",
    "crabteam_habitat_ambiguous",
    "shorezone_unit_id",
    "shorezone_distance_m",
    "shorezone_accepted",
    "substrate",
    "shoreline_type",
    "exposure",
    "eelgrass",
    "salt_marsh",
    "source_monitoring",
    "source_shorezone",
]


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_real_site_table(
    canonical_path: Path,
    shorezone_matches_path: Path,
    *,
    shorezone_acceptance_m: float = 100.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one static R2 record per monitoring site.

    Design contract:
    - monitoring coordinates remain the authoritative node geometry;
    - Crab Team habitat is retained and audited for within-site consistency;
    - ShoreZone thematic attributes are accepted only when the nearest mapped
      shoreline segment is within an explicit conservative distance threshold;
    - rejected ShoreZone candidates remain visible through distance/unit id but
      their thematic fields are nulled rather than silently trusted.
    """
    if shorezone_acceptance_m <= 0:
        raise ValueError("shorezone_acceptance_m must be positive")

    by_site: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"lat": [], "lon": [], "habitats": set()}
    )
    with canonical_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"site_id", "latitude", "longitude", "habitat"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical table missing fields: {sorted(missing)}")
        for row in reader:
            site_id = str(row["site_id"]).strip()
            if not site_id:
                raise ValueError("Canonical table contains blank site_id")
            by_site[site_id]["lat"].append(float(row["latitude"]))
            by_site[site_id]["lon"].append(float(row["longitude"]))
            habitat = str(row.get("habitat") or "").strip()
            if habitat:
                by_site[site_id]["habitats"].add(habitat)

    shorezone_by_site: dict[str, dict[str, str]] = {}
    with shorezone_matches_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "site_id",
            "shorezone_unit_id",
            "shorezone_distance_m",
            "substrate",
            "shoreline_type",
            "exposure",
            "eelgrass",
            "salt_marsh",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ShoreZone match table missing fields: {sorted(missing)}")
        for row in reader:
            site_id = str(row["site_id"]).strip()
            if site_id in shorezone_by_site:
                raise ValueError(f"Duplicate ShoreZone match for site_id={site_id}")
            shorezone_by_site[site_id] = row

    canonical_sites = set(by_site)
    shorezone_sites = set(shorezone_by_site)
    missing_matches = sorted(canonical_sites - shorezone_sites)
    extra_matches = sorted(shorezone_sites - canonical_sites)
    if missing_matches:
        raise ValueError(f"Missing ShoreZone diagnostic rows for sites={missing_matches[:10]}")
    if extra_matches:
        raise ValueError(f"Unexpected ShoreZone sites not in canonical data={extra_matches[:10]}")

    result: list[dict[str, Any]] = []
    ambiguous_habitat_sites: list[str] = []
    rejected_shorezone_sites: list[str] = []

    for site_id in sorted(canonical_sites, key=lambda value: (len(value), value)):
        values = by_site[site_id]
        habitats = sorted(values["habitats"])
        habitat_ambiguous = len(habitats) > 1
        if habitat_ambiguous:
            ambiguous_habitat_sites.append(site_id)
        habitat_value = "|".join(habitats) if habitats else ""

        sz = shorezone_by_site[site_id]
        distance_text = str(sz.get("shorezone_distance_m") or "").strip()
        distance_m = float(distance_text) if distance_text else None
        accepted = distance_m is not None and distance_m <= shorezone_acceptance_m
        if not accepted:
            rejected_shorezone_sites.append(site_id)

        result.append(
            {
                "site_id": site_id,
                "latitude": median(values["lat"]),
                "longitude": median(values["lon"]),
                "crabteam_habitat": habitat_value,
                "crabteam_habitat_ambiguous": habitat_ambiguous,
                "shorezone_unit_id": str(sz.get("shorezone_unit_id") or "").strip(),
                "shorezone_distance_m": distance_m,
                "shorezone_accepted": accepted,
                "substrate": str(sz.get("substrate") or "").strip() if accepted else "",
                "shoreline_type": str(sz.get("shoreline_type") or "").strip() if accepted else "",
                "exposure": str(sz.get("exposure") or "").strip() if accepted else "",
                "eelgrass": str(sz.get("eelgrass") or "").strip() if accepted else "",
                "salt_marsh": str(sz.get("salt_marsh") or "").strip() if accepted else "",
                "source_monitoring": canonical_path.name,
                "source_shorezone": shorezone_matches_path.name,
            }
        )

    summary = {
        "sites": len(result),
        "shorezone_acceptance_m": shorezone_acceptance_m,
        "shorezone_accepted_sites": sum(bool(row["shorezone_accepted"]) for row in result),
        "shorezone_rejected_sites": sum(not bool(row["shorezone_accepted"]) for row in result),
        "shorezone_rejected_site_ids": rejected_shorezone_sites,
        "crabteam_habitat_ambiguous_sites": len(ambiguous_habitat_sites),
        "crabteam_habitat_ambiguous_site_ids": ambiguous_habitat_sites,
        "notes": [
            "Monitoring coordinates define node geometry; ShoreZone does not move sites.",
            "100 m is a conservative project design threshold, not an ecological law.",
            "Rejected ShoreZone candidates keep their distance/unit provenance but thematic fields are blank.",
        ],
    }
    return result, summary


def write_real_sites_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REAL_SITE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_real_sites_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
