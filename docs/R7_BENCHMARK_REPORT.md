# R7 Benchmark Report — Frontier vs. spatial-joint Information Gain vs. GNN+RL

**Date:** 2026-09-12
**Status:** First formal-pipeline result under the R7 ACKed action/reward contract
(`docs/DEMU_HANDOFF_R7.md`). One training seed, one checkpoint — see Section 5
for exactly what this does and does not license claiming.

## 1. What ran

- **Training**: `scripts/train_spatial_gnn_policy.py`, run `r7_serious_v0`,
  seed 0, hidden_dim=64/num_layers=2, 300 updates x 16 episodes/update (4,800
  training episodes), 675.8s wall-clock (~0.19 core-hours). Real graph
  topology (49 sites, 118 edges), world models sampled from the train split
  (A/B/C) per episode, `q_true` sampled from `{0.05, 0.10, 0.20}`, budget 18,
  horizon 6, effort levels `{1, 3, 6}` — exactly the ACKed contract.
- **Benchmark**: `scripts/run_spatial_benchmark_r7.py` against that run's
  `final.pt` (update 300), seed 12345, 6 held-out real-graph seed sites x 5
  cases each = 30 cases per group x 4 groups = 120 cases x 3 planners = 360
  (planner, case) episodes, 148.3s wall-clock. Raw rows:
  `reports/r7_benchmark_v0.csv`. Frontier pinned to `effort_per_site=1,
  max_sites=1` (see Decision Log, 2026-09-12) so all three planners share the
  same one-(site,effort)-per-round action feasibility.

## 2. Results (mean +/- population stdev across n=30 cases per group)

| Group | Planner | missed_occupied_fraction | occupied_site_coverage | final_global_uncertainty |
|---|---|---|---|---|
| id_test | frontier | 0.774 +/- 0.180 | 0.226 +/- 0.180 | 0.789 |
| id_test | information_gain | 0.760 +/- 0.194 | 0.240 +/- 0.194 | 0.764 |
| id_test | gnn_rl | 0.668 +/- 0.205 | 0.332 +/- 0.205 | 0.732 |
| ood_model_E | frontier | 0.753 +/- 0.168 | 0.247 +/- 0.168 | 0.779 |
| ood_model_E | information_gain | 0.782 +/- 0.163 | 0.218 +/- 0.163 | 0.773 |
| ood_model_E | gnn_rl | 0.709 +/- 0.191 | 0.291 +/- 0.191 | 0.747 |
| ood_q_low (q_true=0.02) | frontier | 0.751 +/- 0.175 | 0.249 +/- 0.175 | 0.795 |
| ood_q_low | information_gain | 0.756 +/- 0.176 | 0.244 +/- 0.176 | 0.780 |
| ood_q_low | gnn_rl | 0.757 +/- 0.175 | 0.243 +/- 0.175 | 0.781 |
| ood_q_high (q_true=0.35) | frontier | 0.647 +/- 0.163 | 0.353 +/- 0.163 | 0.731 |
| ood_q_high | information_gain | 0.664 +/- 0.206 | 0.336 +/- 0.206 | 0.720 |
| ood_q_high | gnn_rl | 0.416 +/- 0.246 | 0.584 +/- 0.246 | 0.649 |

## 3. Reading this honestly

- **id_test / ood_model_E**: GNN+RL's mean missed-fraction is numerically
  lower than Frontier and Information Gain, but the per-case standard
  deviation (~0.16-0.21) is large relative to the gap between planners
  (~0.02-0.11). With n=30 cases and one training seed, this is **not**
  a validated result - it is consistent with, but does not prove, a real
  effect.
- **ood_q_low**: all three planners are statistically indistinguishable
  (0.751-0.757). Expected: at `q_true=0.02`, detection is rare regardless of
  where effort is spent, so strategy matters little.
- **ood_q_high**: the one group with a visually large gap - GNN+RL's mean
  missed-fraction (0.416) is well below Frontier/IG (0.647/0.664), i.e.
  notably higher coverage (0.584 vs 0.34-0.35). This is the most interesting
  single number in this report, and also the one most in need of replication
  before it means anything (see Section 5).
- **Visible failure cases**: several `id_test` cases hit missed_fraction as
  high as 0.917 for every planner alike (e.g. `id_test/128/0`,
  `id_test/161/3`) - some incident realizations are hard for everyone, not
  a planner-specific failure.

## 4. Runtime / seed-stability note

Training: 675.8s for 300 updates (4,800 episodes). Benchmark: 148.3s for 360
episodes (0.41s/episode, dominated by the RL policy's forward passes -
Frontier/Information Gain have no torch cost). No crashes, no NaN losses,
no budget/horizon violations across either run.

**Seed stability was not tested here** - this report reflects exactly one
training seed (0) and one resulting checkpoint. The prior D/E/F ablation on
this same branch (pre-R7, different action/reward contract) found that
run-to-run variance from a single-seed REINFORCE-style estimator is real and
can produce a misleadingly good- or bad-looking individual run. That finding
carries over structurally to this contract too: nothing here should be read
as "GNN+RL beats X" until at least 2-3 independent training seeds are
compared and averaged.

## 5. What this report does not claim

Per `configs/benchmark_protocol_r7.json`'s own guardrail: these results
demonstrate performance on this formalized synthetic decision task and
robustness to the tested simulator/q assumptions, not proven field
effectiveness. This is explicitly **not** a winner-only score, **not** a
GNN+RL-beats-Information-Gain claim (300 updates on one seed is far short of
what the D/E/F ablation's own thousands-of-updates runs needed even on the
simpler pre-R7 contract), and not a claim that any world-model family,
q value, or the 12-20-site graph range are ecologically representative -
`configs/benchmark_protocol_r7.json`/`configs/q_protocol_r7.json` already say
so explicitly for the underlying scenario design.

## 6. Multi-seed replication (added after Section 1-5 were first written)

Two more independent training seeds (1, 2) were run with identical
hyperparameters, then benchmarked against the exact same 120 cases (same
`--seed 12345` for case sampling) as seed 0. Per-group, per-planner mean
missed_occupied_fraction across the three seed-level means:

| Group | Planner | seed 0 | seed 1 | seed 2 | mean across seeds | stdev across seeds |
|---|---|---|---|---|---|---|
| id_test | frontier | 0.774 | 0.773 | 0.778 | 0.775 | 0.002 |
| id_test | information_gain | 0.760 | 0.760 | 0.760 | 0.760 | 0.000 |
| id_test | gnn_rl | 0.668 | 0.683 | 0.682 | 0.678 | 0.007 |
| ood_model_E | frontier | 0.753 | 0.755 | 0.756 | 0.754 | 0.001 |
| ood_model_E | information_gain | 0.782 | 0.776 | 0.779 | 0.779 | 0.002 |
| ood_model_E | gnn_rl | 0.709 | 0.712 | 0.731 | 0.717 | 0.010 |
| ood_q_low | frontier | 0.751 | 0.751 | 0.751 | 0.751 | 0.000 |
| ood_q_low | information_gain | 0.756 | 0.756 | 0.756 | 0.756 | 0.000 |
| ood_q_low | gnn_rl | 0.757 | 0.745 | 0.758 | 0.753 | 0.006 |
| ood_q_high | frontier | 0.647 | 0.646 | 0.645 | 0.646 | 0.001 |
| ood_q_high | information_gain | 0.664 | 0.664 | 0.655 | 0.661 | 0.005 |
| ood_q_high | gnn_rl | 0.416 | 0.554 | 0.411 | 0.461 | 0.066 |

**Reading this**: Frontier/Information Gain are near-deterministic across
training seeds (stdev <=0.005 - expected, they don't depend on RL training at
all; the tiny residual variation is incidental, not signal). GNN+RL carries
real seed-to-seed variance (stdev 0.006-0.066), confirming the same
single-seed-REINFORCE-variance lesson from the pre-R7 D/E/F ablation still
applies here. **But the direction is now consistent across all three
independent training runs**: in id_test, ood_model_E, and ood_q_high, every
one of the three RL seeds landed clearly below both baselines - not one seed
happened to look good while the others didn't. ood_q_low stays
indistinguishable across all three, as expected.

**Updated read**: this is meaningfully stronger evidence than the single-seed
result in Sections 1-5 - a consistent direction across 3 independent training
runs is not nothing. It is still **not** a fully validated claim: n=3 seeds,
one architecture/hyperparameter choice, 300 updates (not run to convergence),
and the case set is this report's provisional real-graph sampling, not the
team's eventual frozen OOD manifest. Worth taking seriously; not yet
something to present as settled.

## 7. Suggested next steps (not started here)

1. More updates per seed if runtime allows - 300 updates is a first pass,
   not a convergence claim, and it is unclear whether the gap grows, holds,
   or shrinks with more training.
2. Once the team's exact frozen OOD topology case manifest exists, rerun
   this exact benchmark against it instead of the provisional real-graph
   seed sampling used here (`real_graph_cases.py`).
3. If runtime allows, a couple more seeds (5 total) would tighten the
   stdev-across-seeds estimate, especially for ood_q_high where it's largest.
