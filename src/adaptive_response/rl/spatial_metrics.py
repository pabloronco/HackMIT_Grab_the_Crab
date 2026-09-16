from __future__ import annotations

from dataclasses import dataclass

from ..models import BeliefState, HiddenWorld, Site

# R7 primary metrics (docs/DEMU_HANDOFF_R7.md: "Primary evaluation metrics are
# planner-independent: missed occupied sites, occupied-site coverage, final
# global uncertainty"). Deliberately its own module with zero dependency on
# any reward weighting: reward is not an evaluation metric (benchmark_protocol_r7.json),
# so training code and the benchmark runner must compute these the same way,
# independent of whatever reward coefficients a planner (if any) was trained under.


@dataclass(frozen=True)
class SpatialPrimaryMetrics:
    occupied_sites_total: int
    occupied_sites_missed: int
    missed_occupied_fraction: float
    occupied_site_coverage: float
    final_global_uncertainty: float


def compute_spatial_primary_metrics(
    *,
    sites: list[Site],
    hidden_world: HiddenWorld,
    final_belief: BeliefState,
) -> SpatialPrimaryMetrics:
    """`sites` must be the episode's final PublicState.sites (carries observed
    detections). `hidden_world` is trainer/evaluator-only (post-reveal)."""

    occupied_total = sum(
        1 for site in sites if hidden_world.occupied_by_site.get(site.id, False)
    )
    occupied_missed = sum(
        1
        for site in sites
        if hidden_world.occupied_by_site.get(site.id, False) and site.detections == 0
    )
    missed_fraction = (occupied_missed / occupied_total) if occupied_total > 0 else 0.0
    coverage = 1.0 - missed_fraction

    uncertainty_values = list(final_belief.uncertainty_by_site.values())
    final_uncertainty = (
        sum(uncertainty_values) / len(uncertainty_values) if uncertainty_values else 0.0
    )

    return SpatialPrimaryMetrics(
        occupied_sites_total=occupied_total,
        occupied_sites_missed=occupied_missed,
        missed_occupied_fraction=missed_fraction,
        occupied_site_coverage=coverage,
        final_global_uncertainty=final_uncertainty,
    )
