# R8 Review of Demu R7 Multi-Seed Benchmark

**Status:** engineering integration ACCEPTED; provisional performance evidence NOT YET ACCEPTED as the final planner comparison.

## What is accepted

Demu's R7 handoff successfully integrated the learned policy with the spatial Bayesian mission loop under the cross-team action/reward contract. The full combined suite passed on his branch, the spatial smoke training ran end-to-end, and three independent 300-update training seeds reproduced the same qualitative direction on his provisional case sampler.

This is meaningful engineering evidence that GNN+RL is now viable enough to test seriously. It is not discarded.

## Provisional multi-seed result

Across three independent RL training seeds on Demu's provisional 120-case benchmark, mean missed occupied fraction was numerically lower for GNN+RL than both Frontier and Information Gain in ID, held-out family E, and q-high; q-low was indistinguishable. Training-seed variation remained visible, especially under q-high.

These results must stay labeled **provisional** because the final frozen case manifest was not used.

## BLOCKER 1 — final world generation used placeholder site context

Demu correctly disclosed that his environment could not fetch the raw Dryad tables. His provisional `real_graph_cases.py` therefore used graph-layout x/y coordinates and a neutral habitat score.

That is acceptable for an integration smoke test but not for the final real-data-constrained benchmark:

- Family B and held-out Family E use geometry, so graph-layout coordinates change the generated incidents.
- Family C uses habitat score, so a constant neutral value removes the real-data-derived habitat variation.

### Resolution

R8 now versions the required context in:

- `reports/milestones/r2_real_graph_v0/real_sites_v0.csv`
- `configs/real_site_context_r8.json`

Formal training/evaluation must use the monitoring-authoritative coordinates and the frozen R5 habitat proxy. The proxy remains explicitly a MODEL PROXY, not latent suitability.

Because this changes the training world distribution for B/C and the OOD world distribution for E, the formal RL seeds must be retrained after the replacement.

## BLOCKER 2 — provisional baselines under-used the available budget

The provisional R7 raw CSV shows representative rows where:

- Frontier spent 6 / 18 effort units;
- Information Gain spent 6 / 18 effort units;
- GNN+RL spent 18 / 18 effort units.

The source is transparent: Frontier was pinned to effort 1 for each of six rounds, while Information Gain optimized information gain **per effort**, which strongly preferred effort 1. RL could select `{1,3,6}` and commonly used the full budget.

This means the provisional missed-extent advantage is confounded by resource utilization. It may still contain a genuine policy effect, but it cannot yet be interpreted as one.

### Resolution for the final benchmark

The final baseline rules are frozen in `configs/benchmark_protocol_r8.json`:

- **Frontier:** existing site heuristic; effort 6 (the standard-event level) when budget permits, otherwise the largest allowed level not exceeding remaining budget.
- **Spatial Information Gain:** evaluate every feasible `(site, effort)` pair and maximize **absolute** expected reduction in the sum of marginal occupancy entropy. Information gain per effort is only a tie-break.

This deliberately makes the competitive baseline stronger. If RL still adds value, that result is much more defensible. If it does not, the project uses the best planner.

## Frozen case/runtime gate

The R8 planner-independent manifest is now frozen at 240 total cases:

- validation: 60;
- ID test: 90;
- OOD family E: 30;
- OOD q-low: 30;
- OOD q-high: 30.

Only the 180 ID/OOD cases are formal reporting cases. Demu measured 148.3 seconds for 120 cases x 3 planners on the provisional benchmark, so the frozen case count is practical at hackathon scale.

## Final rerun required

Do not spend time on new architecture or broad hyperparameter search.

Required sequence:

1. sync the R8 real-site context + frozen manifest/protocol;
2. replace placeholder coordinates/habitat;
3. implement the frozen Frontier and absolute-IG baseline rules;
4. rerun full tests and one smoke benchmark;
5. retrain the same three RL seeds under the corrected real-data-constrained world context;
6. evaluate all three checkpoints on the exact frozen R8 formal cases;
7. report mean + dispersion across cases, variation across RL training seeds, `effort_spent`, failure cases, and runtime.

A fourth/fifth training seed is optional only if runtime is comfortably available after the corrected three-seed benchmark.

## Claim discipline

Before the corrected R8 rerun, the right statement is:

> "Three provisional training seeds show a repeatable learned-policy signal on an integration benchmark, but the final comparison is pending corrected real-site covariates, matched benchmark cases, and stronger budget-aware baselines."

After the rerun, wording depends entirely on the frozen results. No real-world-effectiveness, operator-comparison, or optimality claim is allowed.
