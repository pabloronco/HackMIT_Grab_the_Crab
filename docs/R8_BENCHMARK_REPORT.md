# R8 Corrected Benchmark Report — Frontier vs. spatial-joint Information Gain vs. GNN+RL

**Date:** 2026-09-15
**Status:** Final formal-pipeline result under the R8 corrections, **second revision**.
Supersedes the 2026-09-13 version of this same report (git history retains it
in full), which had two methodological defects found in a follow-up review
(`docs/DECISION_LOG.md`, 2026-09-15 entry): truth/belief seed leakage on
`family_id == "A_graph_diffusion"` cases in the frozen-manifest materializer,
and `Edge.distance` silently using a navigable-route-distance proxy instead
of direct distance on most edges (both fixed; see the decision log entry for
the mechanistic detail). Neither fix changed the case manifest, protocol,
reward, architecture, or action contract — only case geometry/q-hypothesis
independence. All three RL seeds were retrained from scratch under the
corrected code before this rerun.

## 1. What changed since the 2026-09-13 version

- **Truth/belief independence restored**: the frozen-manifest materializer
  (`r8_manifest_cases.py`) now derives the belief-ensemble sampling seed
  independently of the truth seed (disjoint namespace, hash-derived from
  `case_id`), instead of reusing the same `world_seed` for both. This
  specifically affected `family_id == "A_graph_diffusion"` cases, where
  hypothesis draw 0 had been generated identically to the truth draw.
- **Edge distance corrected**: `real_graph_cases.py` now always sets
  `Edge.distance` from `distance_km` (the direct distance every planner,
  including the GNN, reads via `graph_state.py`); the
  `salishseacast_total_route_proxy_km` navigable-route proxy is preserved
  on `Edge.travel_cost` (not read by any current GraphState feature) as
  audit/context metadata rather than silently substituting for the direct
  distance. This changes a GNN input, so retraining was required.
- **Retraining**: all three RL seeds (0, 1, 2) retrained from scratch under
  the corrected code, identical hyperparameters to every prior R7/R8 run
  (hidden_dim=64, num_layers=2, 300 updates x 16 episodes/update); no new
  architecture or hyperparameter search. `--decision-log` was enabled this
  time to support the three diagnostics in section 5.
- **Unchanged from 2026-09-13**: real-site context (versioned coordinates,
  frozen R5 habitat proxy), the budget-fair Frontier (effort=6 when
  possible) and Information Gain (absolute-IG-primary ranking) baselines,
  the exact frozen 180-case manifest, the reward contract, the action
  contract.

## 2. Results (mean across 3 RL training seeds; Frontier/IG are deterministic
given a fixed case, confirmed identical across all three benchmark CSVs)

| Group | Planner | missed_occupied_fraction | occupied_site_coverage | final_global_uncertainty | effort_spent | detections | rounds |
|---|---|---|---|---|---|---|---|
| id_test (n=90) | frontier | 0.601 | 0.399 | 0.739 | 18.0 | 0.89 | 3.0 |
| id_test | information_gain | 0.684 | 0.316 | 0.723 | 18.0 | 0.48 | 3.0 |
| id_test | gnn_rl | 0.610 (+/-0.014 across seeds) | 0.390 | 0.741 | 18.0 | 0.86 | 3.0 |
| ood_model_test (n=30) | frontier | 0.750 | 0.250 | 0.751 | 18.0 | 0.60 | 3.0 |
| ood_model_test | information_gain | 0.749 | 0.251 | 0.705 | 18.0 | 0.57 | 3.0 |
| ood_model_test | gnn_rl | 0.762 (+/-0.017) | 0.238 | 0.742 | 18.0 | 0.63 | 3.0 |
| ood_q_low (n=30, q_true=0.02) | frontier | 0.739 | 0.261 | 0.773 | 18.0 | 0.23 | 3.0 |
| ood_q_low | information_gain | 0.778 | 0.222 | 0.741 | 18.0 | 0.10 | 3.0 |
| ood_q_low | gnn_rl | 0.752 (+/-0.015) | 0.248 | 0.770 | 18.0 | 0.24 | 3.0 |
| ood_q_high (n=30, q_true=0.35) | frontier | 0.502 | 0.498 | 0.699 | 18.0 | 1.50 | 3.0 |
| ood_q_high | information_gain | 0.571 | 0.429 | 0.657 | 18.0 | 0.97 | 3.0 |
| ood_q_high | gnn_rl | 0.507 (+/-0.008) | 0.493 | 0.690 | 18.0 | 1.50 | 3.0 |

Raw per-episode rows: `reports/r8_benchmark_seed{0,1,2}.csv` (regenerated
under the corrected code; the pre-fix versions are in git history at the
2026-09-13 commit). Every planner spends the full 18/18 effort in every
case, as in the 2026-09-13 version.

**These numbers are close to the 2026-09-13 (buggy) version** — the two
fixes were narrow (one family, one geometry input), not a change to the
algorithm or protocol, so a broadly similar qualitative picture is the
expected outcome, not a sign the fixes didn't matter. Seed-to-seed spread
for RL is somewhat larger than the buggy version reported (0.008-0.017 vs.
0.004-0.008), which is itself informative (section 4).

## 3. Paired case-wise comparison — this is the new, statistically load-bearing part

Group-level means above can look similar between two planners while hiding
whether either one wins or loses consistently case-by-case. `scripts/r8_paired_comparison.py`
computes `delta = baseline_missed_fraction - rl_missed_fraction` per case
(positive = RL better on that case) and bootstraps a 95% CI on the mean
delta, both per training seed and pooled across all three via a
case-clustered bootstrap (resampling cases, not case-seed rows
independently — see the script docstring for why that distinction matters).
Full table: `reports/r8_paired_comparison.md`.

### Frontier <-> GNN+RL

| Group | Pooled mean delta | 95% CI | Reading |
|---|---|---|---|
| ALL (180 cases) | -0.0095 | [-0.0202, +0.0010] | not distinguishable from 0 (leans slightly negative) |
| id_test | -0.0087 | [-0.0250, +0.0077] | not distinguishable from 0 |
| ood_model_test | -0.0125 | [-0.0391, +0.0131] | not distinguishable from 0 |
| ood_q_high | -0.0055 | [-0.0226, +0.0129] | not distinguishable from 0 |
| ood_q_low | -0.0130 | [-0.0407, +0.0074] | not distinguishable from 0 |

**No group shows RL statistically beating Frontier.** One individual seed
(seed 2) shows RL *statistically worse* than Frontier overall (mean
delta -0.0233, CI [-0.0416, -0.0054], excludes 0) and seed 0 shows RL worse
specifically on `ood_model_test` (CI [-0.0589, -0.0114]). No seed, no group,
shows RL statistically better than Frontier. **Per the pre-declared rule for
this review ("if RL doesn't beat Frontier after this correction, use
Frontier"): RL does not beat Frontier here.**

### Information Gain <-> GNN+RL

| Group | Pooled mean delta | 95% CI | Reading |
|---|---|---|---|
| ALL (180 cases) | +0.0499 | [+0.0305, +0.0698] | **RL better** (CI excludes 0) |
| id_test | +0.0741 | [+0.0494, +0.1001] | **RL better** (CI excludes 0), replicated in all 3 individual seeds |
| ood_model_test | -0.0127 | [-0.0509, +0.0218] | not distinguishable from 0 |
| ood_q_high | +0.0638 | [-0.0067, +0.1318] | not distinguishable from 0 (point estimate positive, CI just crosses 0) |
| ood_q_low | +0.0258 | [-0.0008, +0.0563] | not distinguishable from 0 (CI barely crosses 0) |

**RL does robustly, statistically beat Information Gain overall and on
`id_test` specifically** — every one of the 3 independent training seeds
shows a positive, zero-excluding CI on `id_test` individually. This is the
one place in this benchmark where GNN+RL shows a real, repeated, paired
advantage over a baseline. It is an advantage over the *weaker* of the two
baselines, not over Frontier.

## 4. Training diagnostics (observation-only; did not and must not change
this rerun — see `reports/r8_training_diagnostics.md` for full per-seed detail)

Requested to decide whether a **separately preregistered future ablation**
is justified, per the R7/R8 reward contract's own watch-item.

- **Validation-curve plateau check**: all three seeds' `eval.csv` show a
  flat or very slightly *positive* (i.e., not improving) tail slope over
  the last 10 evaluation points (+0.0006 to +0.0017 missed_fraction/update)
  and near-zero first-half-vs-second-half improvement (-0.004 to +0.018).
  **300 updates was not cut short mid-improvement for any seed** — no
  "still strongly improving" flag triggered.
- **Reward term magnitude**: terminal `missed_occupied_fraction` term's
  mean absolute magnitude is **2.5-2.65x larger** than the summed dense
  terms (`mean_uncertainty_reduction` + `new_detections`) per episode,
  across all three seeds. **The terminal term is not drowned out** — no
  flag triggered on the reward contract's preregistered watch-item.
- **Action statistics**: effort levels are used in reasonable proportion
  (effort=1: 38-44%, effort=3: 21-28%, effort=6: 33-37%) across all three
  seeds — not collapsed onto a single level. Same-episode site revisit rate
  13-14%; mean 4.6-4.8 distinct sites visited per ~5.2-5.4-round episode
  (before the terminal horizon check — see section 2's `rounds=3.0`, which
  is the frozen-benchmark deterministic-eval number, not the noisier
  training-time rollout number these action stats are drawn from).

**No diagnostic trigger fired.** These are reported for the record and to
close out the review's request, not acted on further per instruction ("no
reward, architecture, training length, or hyperparameter changes based on
the R8 test results").

## 5. Honest reading

**The corrected fixes did not change the qualitative conclusion from
2026-09-13, and the new paired analysis makes that conclusion firmer, not
weaker.** GNN+RL ties or trails Frontier on every formal group — never
statistically ahead, and statistically behind on one seed's overall result
and one seed's `ood_model_test` result. It does robustly beat Information
Gain, which remains the weakest of the three planners on this benchmark in
three of four groups (matching the 2026-09-13 finding).

RL's seed-to-seed spread grew somewhat under the corrected code
(missed-fraction stdev 0.008-0.017, vs. 0.004-0.008 in the buggy version) —
plausibly because the corrected, more heterogeneous edge distances (real
`distance_km` instead of a systematically-inflated route proxy) and the
now-independent belief ensembles for family-A cases make the effective
training/eval distribution somewhat less uniform. This is not large enough
to change any of the paired-comparison conclusions above, which already
account for it via bootstrap CIs rather than point estimates alone.

**What this does and does not support:** the RL policy is not broken — it
spends its full budget, detects at a comparable rate to Frontier, and (per
section 4) trained to a genuine plateau with a reward signal that isn't
structurally starved. It does not, at this training scale (300 updates, 3
seeds, this reward/architecture, this corrected case geometry),
demonstrate measurable value over the Frontier heuristic — only over the
weaker Information Gain baseline. **Per the pre-declared decision rule:
Frontier is the planner this benchmark currently supports, not GNN+RL.**

## 6. Visible failure cases

Same hard cases as the 2026-09-13 version, unaffected by either fix:
`ood_q_low__A_graph_diffusion__002` (missed_fraction 0.938 for all three
planners, 16 truly-occupied sites) and `id_test__B_spatial_cluster__010`
(0.923 for all three). Some incident realizations are information-poor at
this budget/horizon for every planner alike, not a planner-specific
weakness.

## 7. Claim discipline

Unchanged from 2026-09-13: these results support performance on the frozen
synthetic sequential-decision benchmark and robustness to the tested
model/q assumptions only. `id_test` is a matched-model-class benchmark, not
ecological validation. `ood_model_test` and the q-shift groups are
specific, narrow OOD tests (hidden family E; hidden q outside belief
support) — not a generic "OOD robustness" claim. No real-world-
effectiveness, operator-comparison, or optimality claim is made or
supported by this report.

**The honest one-line summary**: under the corrected geometry and
independent belief ensembles, and now backed by a paired bootstrap
analysis rather than group means alone, GNN+RL ties Frontier and clearly
beats Information Gain — the project's decision rule for this review
therefore points to Frontier as the benchmark-supported planner, not
GNN+RL.
