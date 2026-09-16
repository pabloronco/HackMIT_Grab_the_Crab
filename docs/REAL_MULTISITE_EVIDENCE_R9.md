# R9 Real Multi-Site Observed-Positive Evidence Audit

**Status:** descriptive reality-check lane; not latent-occupancy reconstruction and not simulator calibration.

## Question

The synthetic world families can contain multi-site extents. Before making stronger ecological-plausibility claims, we need to know how much multi-site signal is actually visible in the historical monitoring table.

This audit therefore asks only what the observed record supports:

- how many monitoring sites were observed positive in each year;
- which sites were newly observed positive in that table year;
- how the cumulative observed-positive footprint changed through time;
- whether frozen real-graph connected components ever reached cumulative footprints of at least 3, 5 or 8 observed-positive sites.

## Semantics

A site-year is observed positive when at least one monthly monitoring row for that site/year records a detection. Multiple positive months in the same site-year count once.

A non-detection is never treated as true absence. Once a site has been observed positive, a later non-detection does not erase it from the cumulative observed-positive footprint.

The connected components of the frozen R2 graph are used only as reproducible spatial audit units. They are **not** asserted to be separate biological invasion episodes.

Therefore changes in cumulative observed-positive footprint may reflect spread, persistence, changing surveillance coverage, imperfect detection, newly detectable abundance, or combinations of these. The audit does not identify those mechanisms.

## Reproduce

The processed monitoring table is intentionally not committed. On a machine with the existing R1/R2 processed data:

```bash
python scripts/diagnose_r9_real_multisite_evidence.py
```

The script reads:

- `data/processed/canonical_site_visit.csv`;
- the frozen R2 real-site receipt;
- the frozen R2 graph edges.

It writes:

`reports/milestones/r9_real_multisite_evidence/observed_positive_footprint.json`

Review the generated receipt before committing it.

## Claim discipline

This audit can support statements about **observed multi-site monitoring evidence**. It cannot by itself establish true invasion extent, spread rate, occupancy prevalence, or historical counterfactual planner performance. It must not be used to retune world-model ranges after seeing planner results.

## Related frozen MVP scope

`configs/mvp_scope_r9.json` freezes two separate MVP design choices:

1. hidden ecological extent is static within one short rapid-response episode; evidence changes belief, not the latent world between modeled rounds;
2. graph v0 is an unweighted, undirected adjacency model. Neutral `connectivity_weight=1` encoding is an implementation convention, not a sourced ecological dispersal probability.

Dynamic world evolution and source-grounded directed/weighted ecological connectivity remain post-MVP extensions.
