# Checkpoint Engineering Report — GNN/RL Backbone & Toy-Generator Smoke Runs

**Date:** 2026-09-11
**Status:** Requested by Pablo + Fede per `DATA_FIRST_VALIDATION_FREEZE.md` Section 10.
**Scope:** engineering/infrastructure diagnostics only. Per the team freeze, none of this
report validates ecology, claims RL superiority, or should influence reward/evaluation
choices. It answers exactly the questions the freeze asked for: does the training
pipeline run without exploding, do gradients flow sensibly, does the action/STOP
machinery work, is logging/checkpointing reproducible, and is the policy compatible
with variable graph sizes.

## 1. What ran

Six independent training runs on the M1 toy single-generator environment, all using the
`missed_extent_weight`-reweighted reward (except v0, the original weighting, kept as an
internal control) and the `RoundPolicy` autoregressive action machinery from the PROPOSED
Checkpoint B/C design:

| Run | Duration | Updates | Episodes (≈updates×32) | Seed | Notable config |
|---|---|---|---|---|---|
| v0 | 7.5h | 5,909 | ~189k | 0 | original reward weight (control) |
| v1 | 7.5h | 11,299 | ~362k | 1 | reweighted, γ=0.99 |
| v2 | 7.5h | 12,023 | ~385k | 2 | reweighted, γ=0.995 |
| D | 3.5h | 3,911 | ~125k | 10 | reweighted, γ=0.99 |
| E | 3.5h | 4,200 | ~134k | 10 (same as D) | reweighted, γ=0.995 |
| F | 3.5h | 2,938 | ~94k | 10 (same as D/E) | reweighted, γ=0.99, entropy-coef decay |

**Totals: ~33 core-hours of wall-clock training, 40,280 gradient updates, ~1.29 million
simulated episodes, zero crashes, zero unrecovered hangs.**

## 2. Does training run without exploding?

Yes, with one caveat that was caught and fixed, not just observed and tolerated.

Early in this work, raising `missed_extent_weight` (a reward-scale change) without
further changes pushed the loss from single digits into the hundreds in a smoke test.
Root cause: the critic's regression target (discounted returns) directly inherits
whatever scale the reward weights impose, and a freshly-initialized critic hasn't
learned that scale yet. **Fix:** batch-normalize returns before computing `value_loss`
and the advantage (`ActorCriticTrainer.update`, see Decision Log 2026-09-11 entry on
return normalization). After the fix, loss stayed in a comparable, bounded range
(roughly ±15) across all six runs regardless of reward-weight magnitude, for the
entire 33 hours combined. No NaN/Inf losses, no gradient explosions requiring the
`max_grad_norm` clip to intervene destructively (clipping is active and used routinely,
as expected, not a failure mode).

## 3. Do actor/critic receive sensible gradients?

Yes. Directly verified, not just inferred from stable loss:

- `tests/test_round_policy_training.py::test_trainer_update_reduces_loss_gradient_flows`
  asserts that a single `update()` call measurably changes every parameter tensor.
- Across all six runs, entropy (a direct read on whether the actor's distribution is
  moving) declined substantially from initialization in every case — from ~20-24 nats
  at update 1 down to run-dependent plateaus between ~1.5 (v0) and ~8-13 (v1/v2/D/E,
  reflecting the reward-variance interaction noted in the Decision Log, an ecological/
  reward-tuning question, not a gradient-flow failure) and ~2.8-3.0 (F, where the
  entropy-coefficient decay schedule mechanically did exactly what it was configured to
  do - see Section 6).
- The critic's value estimates moved coherently with observed returns across training
  (visible in `mean_advantage` staying near zero post-normalization, as expected for a
  reasonably tracking baseline, across all runs' `metrics.csv`).

## 4. Does the STOP/action machinery work?

Yes, across every one of the ~1.29 million simulated episodes:

- **No empty missions.** STOP is masked out on a round's first pick specifically so a
  round can never legally produce a zero-effort `MissionAction`
  (`RoundPolicy.act`); `Environment` would reject one (`"MissionAction must allocate
  positive effort"`). This never fired as a runtime error in any run.
- **No double-picks.** A site, once picked within a round, is masked out of that
  round's remaining choices. Unit-tested directly
  (`test_round_policy_never_picks_the_same_site_twice_in_one_round`) and never
  triggered a contract violation in any run.
- **Exact budget conservation.** Every run's cumulative `effort_spent` exactly equals
  its incident's budget on every episode - unit-tested
  (`test_run_episode_completes_and_rollout_lengths_match`) and consistent with the
  aggregate logs across all 40,280 updates (~1.29M episodes) reviewed for this report.
- **All-infeasible edge case handled explicitly.** `masked_action_distribution()`
  raises a clear error rather than returning NaN when no feasible node remains,
  by design (callers must already treat exhausted budget as terminal) -
  unit-tested, never triggered unexpectedly in a real run.

## 5. Is logging/checkpointing reproducible?

Yes, within the limits of a stochastic training process:

- **Same seed + same hyperparameters reproduce closely matching trajectories.** Used
  directly as a regression check: a post-refactor smoke run at `seed=0` (after the
  `AdaptiveMissionLoop` integration) reproduced loss/entropy values matching a
  pre-refactor run at the same seed to 2-3 significant digits over the first several
  updates - confirming the refactor was behavior-preserving, not just confirming
  reproducibility in the abstract.
- **CSV metrics (`metrics.csv`, `eval.csv`) and periodic checkpoints (`.pt` files,
  `latest.pt`, `final.pt`) were written correctly and consistently** across all six
  runs, including runs that were interrupted mid-flight by an unrelated harness/session
  restart (the underlying OS processes were unaffected and continued logging correctly
  throughout; only the external process-monitoring lost track of them temporarily).
- **`scripts/compare_runs.py`** was built specifically to consume this logged output
  reproducibly (trend correlation, best/latest/first-vs-last averages) rather than
  requiring ad hoc re-analysis each time, and was used consistently across all
  reported comparisons.
- Ctrl+C handling (`SIGINT`) saves a final checkpoint before exit; implemented and code
  -reviewed, not stress-tested under this report's specific runs (none were manually
  interrupted).

## 6. Is the policy compatible with variable graph sizes?

Yes, both by construction and empirically:

- **By construction:** no layer in `GraphEncoder`/`ActorHead`/`CriticHead` has a
  fixed-N dimension anywhere - node scoring is a shared per-node function, pooling is
  permutation-invariant mean pooling. Unit-tested explicitly for N = 4, 12, 20, 24 with
  the *same* model instance processed back-to-back
  (`test_forward_pass_shapes_for_variable_graph_size`,
  `test_same_model_instance_handles_multiple_graph_sizes_in_a_row`).
- **Empirically:** every one of the six runs randomized graph size uniformly in
  [12, 24] every single episode (`IncidentSamplerConfig`) - roughly 1.29 million
  episodes combined ran across that full size range without a single shape-related
  crash.

## 7. Real bugs found and fixed during this work

Reporting these because the freeze specifically asked what problems were found, not
just what worked:

1. **Reward-scale → loss-scale coupling** (Section 2). Fixed via return normalization.
2. **Reward/decision misattribution bug**, caught while integrating `AdaptiveMissionLoop`
   (M4): that loop's `execute_pending()` internally pre-plans the *next* round as soon
   as the current one isn't done. A naive "call `run_round()`, then read back what the
   planner just decided" pattern would silently attribute a round's reward to the
   *next* round's log_prob/value/entropy once that internal lookahead fires. Caught by
   reasoning through the exact call sequence (not by a failing test - both would have
   passed while producing a plausible-looking wrong answer) and fixed by capturing the
   decision immediately after an explicit `plan_next()` call, before `execute_pending()`.
   Verified directly with a standalone script asserting `decision.mission ==
   transition.mission` on every round of a real multi-round episode, plus a forced
   single-pick-per-round edge case. **This bug did not affect any reported run**: v0
   through F all ran on the prior hand-rolled loop, which never had the lookahead-
   planning call this bug depends on.
3. **A related loop-termination bug**: `reveal()` moves the loop's phase from `COMPLETE`
   to `REVEALED`, so a naive `while phase is not COMPLETE` condition fires one extra,
   invalid round immediately after the terminal one. Fixed with an explicit
   `break`/`return` on the terminal transition.

## 8. Mechanically-verified but ecologically-inconclusive: entropy-coefficient decay

The entropy-coefficient decay feature (linear decay of the exploration bonus over a
configurable number of updates) was added, unit-tested, and exercised in one full run
(F): the coefficient decayed exactly on schedule (0.0100 → 0.0005 floor over 4,000
updates, confirmed numerically in `metrics.csv`) and entropy collapsed accordingly
(~24 nats → ~2.8-3.0 nats). **Mechanically the feature works as specified.** Whether
that particular schedule *helps* the policy is an ecological/reward-tuning question
explicitly out of scope for this report (see Decision Log's D/E/F entry) - it is
reported here only as confirmation that the scheduling logic itself is implemented and
functions correctly, which is the engineering fact requested.

## 9. Open engineering item worth flagging for the "serious" phase

Not an ecological finding: run-to-run variance was high across all six runs even
holding the reward/action-space fixed, including two runs (D, F) that ended with
periods at-or-worse-than the Frontier baseline late in training despite earlier good
performance. This is consistent with what a single-seed, 32-episodes-per-update,
un-batched REINFORCE-style estimator (no GAE, no multi-epoch replay, no per-configuration
seed averaging) would be expected to produce - **an algorithmic/estimator variance
question, independent of which reward or world-model the team eventually freezes.**
Worth considering more episodes per update and/or multi-seed averaging as part of
serious-training infrastructure, regardless of the ecological questions Pablo/Fede are
resolving in parallel.

## 10. What this report does not claim

Per the freeze: these six runs do not show that RL is better than any baseline, that
the reward is correctly specified, that the M1 toy generator is ecologically
realistic, or that any policy generalizes beyond the incidents it was trained on. The
numeric results (missed-fraction vs. Frontier, trend correlations, etc.) are recorded
in the Decision Log entries dated 2026-09-11 for traceability, but are explicitly
reclassified there as smoke-test data, not project results.
