# R5 Simulator Criticism — Decision Receipt

**Status:** ACCEPTED REALITY-CHECK MILESTONE; **not** field validation and **not** ecological calibration.

## Observed run

The accepted diagnostic run used the real R2 monitoring graph and the canonical monitoring table.

- incident seed: `3`, selected by topology/static graph inputs only;
- incident graph: 15 sites, 29 edges;
- calibration years: 2017–2021;
- held-out reality-check years: 2022–2023;
- known effort rows in incident context: 591; missing effort rows: 3;
- effort min / median / p90 / max: 4 / 6 / 6 / 12 trap sets;
- observed positive sites in this incident context: 1/15 both across all years and in the held-out years.

The observed positive-site pattern is **not latent occupancy truth**. With only one observed positive site in this incident subgraph it has very little power to identify spatial cohesion or extent.

## Family review

250 seeded draws were inspected per family.

- **A graph diffusion:** occupied-count p10/median/p90 = 1/2/4; no criticism warning.
- **B spatial cluster:** occupied-count p10/median/p90 = 3/6/9; warning that observed-positive cohesion lies outside the synthetic p10–p90 envelope.
- **C habitat driven:** occupied-count p10/median/p90 = 2/4/7; no criticism warning.
- **E fragmented/patchy:** occupied-count p10/median/p90 = 3/6/9; same observed-positive cohesion warning as B.

## Decision

**KEEP the current A/B/C/E parameter ranges as benchmark DESIGN RANGES for the next implementation stage. Do not fit them to the single-positive observed pattern.**

Reasoning:

1. No family is grossly degenerate (for example, almost always one site or almost always the full graph).
2. B and E produce substantial seeded variation rather than collapsing to one pattern.
3. The B/E cohesion warning is driven by an observed pattern containing only one positive site, for which neighbor fraction is mechanically zero; treating that as complete occupancy would violate the imperfect-detection premise.
4. Adjusting B/E downward to match this sparse observed pattern would be pseudo-calibration to detection outcomes, not defensible inference about hidden extent.
5. The goal of the multi-family benchmark is robustness to simulator assumptions, not selecting the family that visually resembles one incomplete historical footprint.

Therefore the current ranges are now frozen **for the first formal benchmark protocol** unless a later source-grounded ecological constraint reveals a clear incompatibility. Any later change must be versioned and cannot be made after inspecting planner winners.

## Frozen family split

- TRAIN / ID families: `A_graph_diffusion`, `B_spatial_cluster`, `C_habitat_driven`
- OOD MODEL holdout: `E_fragmented_patchy`

## Important open items

This receipt does **not** freeze:

- ecological numeric interpretation of the family parameters;
- q support / detectability range;
- action effort granularity;
- formal budget grid;
- RL reward weights;
- exact OOD parameter shifts;
- exact topology stress cases.

Those remain explicit cross-team or evidence-dependent decisions.

## Claim guardrail

The accepted result supports only that the simulator suite is diverse enough to proceed to controlled planner benchmarking without an obvious gross mismatch revealed by the available reality checks. It does not show that any family is a true model of the invasion or that benchmark performance transfers to field operations.
