# R8 Corrected Benchmark Report — Frontier vs. spatial-joint Information Gain vs. GNN+RL

**Date:** 2026-09-13
**Status:** Final formal-pipeline result under the R8 corrections
(`docs/R8_DEMU_BENCHMARK_REVIEW.md`, `configs/benchmark_protocol_r8.json`):
real monitoring-site coordinates/habitat, budget-fair baselines, exact frozen
180-case manifest. Supersedes the R7 provisional result
(`docs/R7_BENCHMARK_REPORT.md`), which is retained as a record of what the
R8 review corrected and why.

## 1. What changed since R7 and why it matters

- **Real-site context**: world models B/E now use monitoring-authoritative
  lat/lon (locally projected to km); world model C uses the frozen R5
  habitat proxy by `crabteam_habitat` class. Graph-layout coordinates and a
  neutral habitat constant (both used in R7) are gone.
- **Budget-fair baselines**: Frontier now uses the standard-event effort (6)
  when budget permits, falling back to the largest feasible level - not a
  fixed `effort_per_site=1`. Information Gain now searches `(site, effort)`
  and ranks by **absolute** expected entropy reduction, with information gain
  per effort only a tie-break - reversed from R7's per-effort-primary
  ranking. Result: **every planner now spends the full 18/18 budget in every
  case** (previously Frontier/IG spent 6/18 while RL spent 18/18 - a real
  resource-utilization confound the R7 numbers could not be interpreted
  through).
- **Exact frozen manifest**: 180 formal cases (`id_test` 90, `ood_model_test`
  30, `ood_q_low` 30, `ood_q_high` 30) loaded verbatim from
  `reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json`, not
  resampled. The 60 `validation` cases are excluded from these numbers per
  the manifest's own instruction.
- **Retraining**: all three RL seeds (0, 1, 2) were retrained from scratch
  under the corrected world context - the R7 checkpoints were not reused.
  Same hyperparameters as R7 (hidden_dim=64, num_layers=2, 300 updates x 16
  episodes/update); no new architecture or hyperparameter search.

## 2. Results (mean across 3 RL training seeds; Frontier/IG are deterministic
given a fixed case, so their numbers do not vary by seed)

| Group | Planner | missed_occupied_fraction | occupied_site_coverage | final_global_uncertainty | effort_spent | detections | rounds |
|---|---|---|---|---|---|---|---|
| id_test (n=90) | frontier | 0.589 | 0.411 | 0.733 | 18.0 | 0.90 | 3.0 |
| id_test | information_gain | 0.672 | 0.328 | 0.715 | 18.0 | 0.52 | 3.0 |
| id_test | gnn_rl | 0.595 (+/-0.004 across seeds) | 0.405 | 0.739 | 18.0 | 0.89 | 3.0 |
| ood_model_test (n=30) | frontier | 0.760 | 0.240 | 0.750 | 18.0 | 0.50 | 3.0 |
| ood_model_test | information_gain | 0.742 | 0.258 | 0.685 | 18.0 | 0.63 | 3.0 |
| ood_model_test | gnn_rl | 0.760 (+/-0.004) | 0.240 | 0.743 | 18.0 | 0.67 | 3.0 |
| ood_q_low (n=30, q_true=0.02) | frontier | 0.744 | 0.256 | 0.773 | 18.0 | 0.20 | 3.0 |
| ood_q_low | information_gain | 0.775 | 0.225 | 0.741 | 18.0 | 0.13 | 3.0 |
| ood_q_low | gnn_rl | 0.755 (+/-0.008) | 0.245 | 0.773 | 18.0 | 0.20 | 3.0 |
| ood_q_high (n=30, q_true=0.35) | frontier | 0.483 | 0.517 | 0.691 | 18.0 | 1.53 | 3.0 |
| ood_q_high | information_gain | 0.613 | 0.387 | 0.693 | 18.0 | 0.83 | 3.0 |
| ood_q_high | gnn_rl | 0.497 (+/-0.004) | 0.503 | 0.691 | 18.0 | 1.55 | 3.0 |

Raw per-episode rows: `reports/r8_benchmark_seed{0,1,2}.csv`.

## 3. Honest reading

**The R7 apparent RL advantage does not survive the fairness correction.**
Once Frontier and Information Gain are allowed to spend the full budget
(exactly what R8 Blocker 2 predicted), GNN+RL is no longer clearly ahead of
either baseline in any of the four groups:

- **id_test**: Frontier (0.589) is numerically best; RL (0.595) is close
  behind; Information Gain (0.672) is clearly worst here.
- **ood_model_test**: Information Gain (0.742) is numerically best; Frontier
  and RL are statistically tied (0.760 both).
- **ood_q_low**: Frontier (0.744) is numerically best; RL (0.755) close
  behind; Information Gain (0.775) worst.
- **ood_q_high**: Frontier (0.483) and RL (0.497) are close and both clearly
  better than Information Gain (0.613) - this was the group with the
  largest apparent RL edge in the R7 provisional report (RL 0.42-0.55 vs.
  Frontier/IG ~0.65); under the corrected fair baseline, Frontier alone
  closes almost the entire gap.

RL's variance across the three retrained seeds is now small (stdev
0.004-0.008 missed-fraction) - much tighter than the R7 provisional runs
(0.006-0.066) - because Frontier/Information Gain are now the ones spending
comparable effort, so there is less headroom for one seed's stochastic
training trajectory to produce a dramatically different-looking outcome.

**What this does support**: the RL policy is not broken or degenerate - it
consistently spends its budget, detects at a comparable rate to Frontier,
and is competitive with (though not clearly better than) the simplest
baseline once that baseline is no longer artificially budget-starved.
**What it does not support**: any claim that GNN+RL adds measurable value
over Frontier or Information Gain on this benchmark, at this training scale
(300 updates, 3 seeds), under this reward.

## 4. Visible failure cases

Several cases are hard for every planner alike regardless of strategy, e.g.
`ood_q_low__A_graph_diffusion__002` (missed_fraction 0.938 for all three
planners, 16 truly-occupied sites) and `id_test__B_spatial_cluster__010`
(0.923 for all three). This indicates some incident realizations are simply
information-poor at this budget/horizon, not a planner-specific weakness.

## 5. Runtime and a real engineering blocker found and mitigated

Each seed's 180-case x 3-planner benchmark ran in ~117s (540 episodes,
~0.22s/episode). Training: three seeds x 300 updates x 16 episodes, ~650s
each, run in parallel.

**Blocker found during retraining, not before it**: one training run
(seed 2's first attempt) crashed at update 219/300 with
`SpatialBeliefEngine.update()` raising "Observation batch has zero
probability under every spatial/q hypothesis" - a real, occasionally-hit
numerical edge case where the hidden simulator's realized detection is
inconsistent with every hypothesis in the (necessarily finite) sampled
ecological-hypothesis ensemble. Mitigated on the training side only (a
bounded resample-and-retry, occurrence count logged; never applied to the
frozen benchmark, where a case must never be silently reseeded) - see
`scripts/train_spatial_gnn_policy.py` and the Decision Log entry dated
2026-09-13. The corrected seed-2 retrain completed all 300 updates with
zero retries needed; this is reported as a real, disclosed finding for the
team, not swept under the rug.

## 6. Claim discipline

Per `configs/benchmark_protocol_r8.json`'s guardrail: these results support
performance on the frozen synthetic sequential-decision benchmark and
robustness to the tested model/q assumptions only. `id_test` is a
matched-model-class benchmark, not ecological validation. `ood_model_test`
and the q-shift groups are specific, narrow OOD tests (hidden family E;
hidden q outside belief support) - **not** a generic "OOD robustness" claim,
since no formal OOD-topology or ecological-parameter-OOD lane exists yet.
No real-world-effectiveness, operator-comparison, or optimality claim is
made or supported by this report.

**The honest one-line summary**: under the corrected, budget-fair
comparison, GNN+RL is competitive with but does not clearly outperform the
simple Frontier heuristic, and both typically outperform the current
spatial Information Gain baseline in three of four formal groups - the
project's earlier apparent RL advantage was substantially a resource-
utilization artifact.
