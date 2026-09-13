from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence


def _site_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def connected_components(
    site_ids: Sequence[str],
    edges: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic connected components of the frozen undirected graph."""

    ordered_sites = tuple(str(site_id) for site_id in site_ids)
    if len(ordered_sites) != len(set(ordered_sites)):
        raise ValueError("site_ids must be unique")

    known = set(ordered_sites)
    adjacency = {site_id: set() for site_id in ordered_sites}
    for row in edges:
        src = str(row["src"])
        dst = str(row["dst"])
        if src not in known or dst not in known:
            raise ValueError(f"Edge references unknown site: {src}--{dst}")
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    remaining = set(ordered_sites)
    result: list[tuple[str, ...]] = []
    while remaining:
        start = min(remaining, key=_site_sort_key)
        remaining.remove(start)
        queue: deque[str] = deque([start])
        component = {start}
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node], key=_site_sort_key):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(tuple(sorted(component, key=_site_sort_key)))

    result.sort(key=lambda component: (-len(component), _site_sort_key(component[0])))
    return tuple(result)


def _yearly_observed_positive_sites(
    canonical_rows: Sequence[Mapping[str, Any]],
    allowed_sites: set[str],
) -> dict[int, set[str]]:
    by_year: dict[int, set[str]] = {}
    for row in canonical_rows:
        site_id = str(row["site_id"])
        if site_id not in allowed_sites:
            continue
        year = int(row["year"])
        by_year.setdefault(year, set())
        if bool(row["detected"]):
            by_year[year].add(site_id)
    return by_year


def _footprint_timeline(
    years: Sequence[int],
    positive_by_year: Mapping[int, set[str]],
    component_sites: set[str],
) -> list[dict[str, Any]]:
    first_positive_year: dict[str, int] = {}
    cumulative: set[str] = set()
    timeline: list[dict[str, Any]] = []

    for year in years:
        positive = set(positive_by_year.get(year, set())).intersection(component_sites)
        for site_id in sorted(positive, key=_site_sort_key):
            first_positive_year.setdefault(site_id, int(year))
        new_positive = {site_id for site_id in positive if first_positive_year[site_id] == year}
        cumulative.update(positive)
        timeline.append(
            {
                "year": int(year),
                "observed_positive_sites": sorted(positive, key=_site_sort_key),
                "observed_positive_site_count": len(positive),
                "new_observed_positive_sites": sorted(new_positive, key=_site_sort_key),
                "new_observed_positive_site_count": len(new_positive),
                "cumulative_observed_positive_sites": sorted(cumulative, key=_site_sort_key),
                "cumulative_observed_positive_site_count": len(cumulative),
            }
        )
    return timeline


def _threshold_crossings(
    timeline: Sequence[Mapping[str, Any]],
    thresholds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("thresholds must be positive integers")
        first_year = next(
            (
                int(row["year"])
                for row in timeline
                if int(row["cumulative_observed_positive_site_count"]) >= threshold
            ),
            None,
        )
        result[str(threshold)] = {
            "reached": first_year is not None,
            "first_year_reached": first_year,
        }
    return result


def observed_positive_footprint_report(
    canonical_rows: Sequence[Mapping[str, Any]],
    site_ids: Sequence[str],
    edges: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[int] = (3, 5, 8),
) -> dict[str, Any]:
    """Describe observed-positive footprint through time without inferring absence.

    Monthly detections collapse to site-year presence of at least one observed
    detection. Non-detections are never converted to true absence and never remove a
    site from the cumulative observed-positive footprint.
    """

    components = connected_components(site_ids, edges)
    years = tuple(
        sorted(
            {
                int(row["year"])
                for row in canonical_rows
                if str(row["site_id"]) in set(str(site_id) for site_id in site_ids)
            }
        )
    )
    if not years:
        raise ValueError("No canonical monitoring rows align with the supplied sites.")

    allowed = set(str(site_id) for site_id in site_ids)
    positive_by_year = _yearly_observed_positive_sites(canonical_rows, allowed)

    component_reports: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        component_set = set(component)
        timeline = _footprint_timeline(years, positive_by_year, component_set)
        ever_positive = sorted(
            set().union(*(set(row["observed_positive_sites"]) for row in timeline)),
            key=_site_sort_key,
        )
        component_reports.append(
            {
                "component_id": f"cc_{index:02d}",
                "site_count": len(component),
                "site_ids": list(component),
                "ever_observed_positive_site_count": len(ever_positive),
                "ever_observed_positive_sites": ever_positive,
                "max_observed_positive_sites_in_one_year": max(
                    int(row["observed_positive_site_count"]) for row in timeline
                ),
                "timeline": timeline,
                "threshold_crossings": _threshold_crossings(timeline, thresholds),
            }
        )

    network_timeline = _footprint_timeline(years, positive_by_year, allowed)
    reached_component_counts = {
        str(threshold): sum(
            bool(component["threshold_crossings"][str(threshold)]["reached"])
            for component in component_reports
        )
        for threshold in thresholds
    }

    return {
        "status": "DESCRIPTIVE_OBSERVED_POSITIVE_FOOTPRINT_NOT_LATENT_OCCUPANCY",
        "years": list(years),
        "site_count": len(allowed),
        "component_count": len(components),
        "components": component_reports,
        "network": {
            "timeline": network_timeline,
            "threshold_crossings": _threshold_crossings(network_timeline, thresholds),
            "ever_observed_positive_site_count": int(
                network_timeline[-1]["cumulative_observed_positive_site_count"]
            ),
            "max_observed_positive_sites_in_one_year": max(
                int(row["observed_positive_site_count"]) for row in network_timeline
            ),
        },
        "components_reaching_cumulative_threshold": reached_component_counts,
        "semantics": {
            "positive": "at least one recorded detection at a monitoring site in the year",
            "new_positive": "first year in this table with a recorded detection at the site",
            "cumulative_footprint": "sites ever observed positive up to that year",
            "nondetection": "sampling outcome only; never treated as true absence",
            "episode_proxy": "frozen real-graph connected component; not a claim of a single biological invasion episode",
        },
        "claim_guardrails": [
            "Observed-positive footprint is not complete true occupancy because detection is imperfect.",
            "Growth in cumulative observed-positive footprint can reflect detection, surveillance coverage, spread, persistence, or combinations of these.",
            "This audit is descriptive reality-check evidence and does not calibrate simulator prevalence or world-model ranges by itself.",
        ],
    }
