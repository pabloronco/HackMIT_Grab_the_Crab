# Rehearsal Decision Log

This file mirrors project-relevant decisions made after Project Freeze 3.0 for the rehearsal repository. The canonical project Decision Log remains the governing source of truth.

## 2026-09-10 — M0 interface indexing review

**Status:** APPROVED by team.

**Decision:** Extend `GraphState` with explicit tensor/index mappings required for all planners and GNN message passing:

- add `node_ids`, index-aligned with `node_features` and `feasibility_mask`;
- add `edge_index`, edge-aligned with `edge_features`;
- make `q_by_site` available through the shared planner `constraints` / context so the Information Gain planner can evaluate the observation model without adding `q_i` to learned node features by default.

**Reason:** Demu's interface review identified two structural blockers: planner outputs could not map tensor rows back to real `site_id`s, and a GNN could not perform message passing from edge attributes without graph connectivity indices. The Information Gain baseline also requires explicit detectability for prospective likelihood calculations.

**Impact:** No change to the frozen evidence semantics, feature list, hidden-truth boundary, planner ladder, or product architecture. This makes the existing interface executable rather than redefining it.

**Owner:** Team.

## 2026-09-10 — M2 belief-engine implementation defaults

**Status:** CURRENT DEFAULT for rehearsal implementation; not an ecological fact and not a new frozen project decision.

**Decision:**

- Priors are supplied explicitly to `BeliefEngine`; the engine does not invent occupancy priors from habitat, distance, or the synthetic hidden world.
- A confirmed initial detection may be initialized at occupancy belief 1 under the MVP no-false-positive assumption.
- Site-level evidence updates use the explicit effort-aware binary observation model from the Technical Specification.
- Belief uncertainty is represented for v0 as Bernoulli entropy in bits, with range `[0, 1]`.
- The first belief engine is site-local; it does not propagate a positive detection to connected nodes unless a later explicit spatial/world model is introduced.

**Reason:** Project Freeze 3.0 freezes explicit effort/q Bayesian evidence semantics but does not freeze the numerical prior scheme or exact uncertainty scalar. Keeping priors caller-supplied prevents a design choice from being presented as ecological knowledge. Bernoulli entropy is simple, inspectable, and directly useful to the later Information Gain baseline.

**Impact:** M2 can be tested without coupling inference to the toy simulator. Spatial belief coupling, real-data-informed priors, and q range calibration remain later validation/modeling work.

**Owner:** Pablo + Fede, with cross-team review if these choices change `GraphState` semantics.

## 2026-09-10 — GraphState exporter encoding

**Status:** APPROVED CURRENT DEFAULT after Demu consumer validation.

**Decision:** Implement the already-approved `GraphState` contract with a framework-agnostic numeric encoding:

- node feature order: `belief, uncertainty, observed_effort, detections, habitat_score, access_cost, frontier`;
- edge feature order: `distance, connectivity_weight`;
- global feature order: `remaining_budget, round, team_capacity, global_uncertainty`;
- `edge_index` uses shape `[2, E]` and indexes directly into `node_ids`;
- because the current Environment treats the graph as undirected, each configured edge is exported in both directions for message passing;
- frontier is currently a binary observable indicator: a non-positive node directly adjacent to a publicly detected/confirmed-positive node;
- missing access cost and connectivity weight use configurable neutral defaults of `1.0`;
- global uncertainty is the mean node Bernoulli entropy;
- `q_by_site` remains outside learned node features and is exposed separately through planner constraints;
- GraphState remains Python/serialization friendly; the learned planner owns conversion to PyTorch/PyG tensors.

**Consumer validation:** Demu checked the real branch locally, read the producer contract and implementation, ran the full 30-test suite and GraphState sanity script, and confirmed that the payload can be consumed by a variable-size GNN/PyG adapter without structural changes. He also confirmed the node-id mapping, bidirectional `[2,E]` edge format, separate `q_by_site` planner context, and ML-side normalization ownership.

**Known limitation:** `feasibility_mask` is currently uniform per node while budget remains positive. Per-site closures, effort caps, and richer action-feasibility logic remain OPEN until action-space design and must be reviewed cross-team before training semantics are frozen.

**Reason:** The Technical Specification freezes/candidates the feature families but not their tensor ordering, missing-value encoding, frontier definition, or framework representation. These choices make the producer/consumer contract executable without coupling the environment package to Demu's ML stack.

**Impact:** M2.5 producer/consumer interface is now accepted. Demu may proceed with the GraphState-to-tensor adapter and GNN forward-pass work. Action-space, masking, reward, and evaluation semantics remain cross-team decisions.

**Owner:** Team; implementation by Pablo + Fede, consumer validation by Demu.

## 2026-09-10 — M3 FrontierPlanner baseline

**Status:** APPROVED CURRENT DEFAULT for the interpretable frontier baseline; not an ecological optimality claim and not the competitive benchmark target.

**Decision:** Implement `FrontierPlanner` under the frozen shared planner interface `plan(graph_state, remaining_budget, constraints) -> MissionAction` with deterministic ranking:

- feasible frontier nodes rank before feasible non-frontier nodes;
- within each group, higher occupancy belief ranks first;
- ties are broken by higher uncertainty, then deterministic `site_id` order;
- if budget remains after frontier selections, the planner may allocate to the best remaining feasible non-frontier nodes using the same ranking;
- per-site effort is configurable (`effort_per_site`) so M3 does not freeze the later RL action-space granularity;
- the planner does not use `q_by_site` in v0 and never receives hidden occupancy.

**Validation:** Local suite reported 39/39 tests passing. The M3 sanity script showed a controlled causal replan: before new evidence `site_04` was preferred (`p=0.7000` vs `site_02=0.6000`); after `0 detections / 5 checks` at `site_04`, Bayes reduced its belief to `0.3564` and the next mission switched to `site_02`. Hidden occupancy was not used by the planner.

**Known limitation:** This planner is intentionally simple and myopic. It is an interpretable fallback/integration probe, not the strong competitive baseline. Greedy Information Gain / entropy-VOI remains the main baseline for judging whether GNN+RL adds measurable value.

**Impact:** M3 satisfies the roadmap gate that a heuristic planner returns valid `MissionAction`s and demonstrates `FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED` through the shared interfaces. The next system milestone is the full no-RL adaptive loop (M4).

**Owner:** Pablo + Fede; planner contract shared with team.

## 2026-09-11 — M4 end-to-end no-RL adaptive loop

**Status:** APPROVED CURRENT DEFAULT for the rehearsal kernel.

**Decision:** Add a planner-agnostic `AdaptiveMissionLoop` that orchestrates the real executable cycle `plan -> Environment.step -> ObservationBatch -> Bayes -> GraphState -> replan`, with explicit phases `UNINITIALIZED`, `READY_TO_PLAN`, `MISSION_PLANNED`, `COMPLETE`, and `REVEALED`.

- the mission displayed/planned is the mission actually executed by the simulator;
- field observations are generated by `Environment.step`, not injected manually into the loop;
- Bayes consumes those observations and the observable `q_by_site` context;
- the updated belief is re-exported to a new `GraphState` before replanning;
- the loop is planner-agnostic through the shared `Planner` contract;
- hidden truth remains owned by `Environment` and is unavailable through public transitions, belief, graph state, or planner inputs;
- reveal is locked on reset and only becomes available after the episode reaches `COMPLETE`;
- CURRENT DEFAULT completion criterion is budget exhaustion.

**Validation:** Pablo reported 47/47 tests passing. The end-to-end sanity run used a 6-unit budget: Mission 1 allocated 3 checks to `site_04`; the simulator returned 0 detections / 3 checks; Bayes changed `p(site_04)` from `0.7000` to `0.4961`; the next mission changed to `site_02`; a second 3-check mission exhausted the budget; only then was hidden extent revealed. The planner never received hidden occupancy. Deterministic-seed, reset/relock, transition-payload and reveal-gate tests also pass.

**What this demonstrates:** The central software/decision kernel works end-to-end: real simulated field evidence changes probabilistic belief and the changed observable state changes the next mission under finite budget.

**What this does NOT demonstrate:** The current single toy hidden-world generator and caller-supplied priors do not establish ecological realism, real-world effectiveness, or learned-policy superiority. Multiple real-data-constrained simulator families, strong Information Gain comparison, held-out/OOD testing and historical replay remain later validation work.

**Implementation note:** `AdaptiveMissionLoop` currently calls an internal environment reveal-unlock hook when the state machine reaches completion. This is acceptable for the rehearsal kernel because planners never receive the Environment object, but the reveal boundary should remain covered by tests and must not be exposed through planner-facing interfaces.

**Impact:** M4 satisfies the roadmap gate `plan -> observe -> update -> replan -> reveal`. The next roadmap milestone for Pablo + Fede is the mission-control UI shell (M5), while Demu can continue the learned-planner backbone in parallel.

**Owner:** Pablo + Fede for orchestration/product integration; planner contract shared with team.

## 2026-09-10 — GNN/RL backbone v0 (Checkpoint A)

**Status:** CURRENT DEFAULT implementation detail, isolated to Demu's `adaptive_response.rl` subpackage. Does not touch or freeze action space, reward, or evaluation semantics, which remain OPEN per the M2.5 entry above.

**Decision:** Implement the GraphState -> tensors -> GNN -> per-node logits + critic value forward path as a new `src/adaptive_response/rl/` subpackage, deliberately isolated from the core package so importing `adaptive_response` never requires torch:

- `tensor_adapter.py`: `graph_state_to_tensors(GraphState) -> GraphTensors`. Only accepts `GraphState` (never `PublicState`/`HiddenWorld`), so there is no code path for hidden occupancy to reach the network. Applies a documented, adapter-local `log1p` transform to unbounded count-like features (`observed_effort`, `detections`, `remaining_budget`, `round`); bounded features (`belief`, `uncertainty`, `frontier`, `global_uncertainty`) are left as-is. This is scaling, not a `GraphState` contract change.
- `layers.py` / `backbone.py`: a small custom message-passing GNN in plain PyTorch (no PyTorch Geometric / torch-scatter, to avoid platform-specific wheel issues) — edge-conditioned messages, mean aggregation, residual + LayerNorm update, 2 layers by default. Mean-pooled node embeddings + an encoded `global_features` vector form a graph context vector. An `ActorHead` scores each node with a *shared* per-node MLP (no fixed-N layer anywhere, so the same weights run on any graph size) and applies `feasibility_mask` as `-inf` before returning logits. A `CriticHead` gives one scalar value per graph. Default `hidden_dim=64, num_layers=2` -> ~105k parameters.
- `q_by_site` is not wired into node features here, matching the M0 interface decision; using it would be a separate cross-team benchmarked decision.
- New optional dependency: `torch>=2.2,<3` under a new `rl` extra in `pyproject.toml` (`pip install -e ".[dev,rl]"`), not added to core `dependencies`.

**Validation:** New `tests/test_gnn_backbone.py` (skipped automatically via `pytest.importorskip("torch")` when torch is absent, so it cannot break the non-RL side's `pytest` run). Full local suite: 50/50 passing (39 prior + 11 new) on Python 3.10 CPU-only torch 2.14. Forward pass verified correct and shape-stable for N = 4, 12, 20, 24 with the *same* model instance back-to-back. `scripts/run_gnn_backbone_sanity.py` shows the untrained network's per-node logits shifting after the same `site_04: 0/10 checks` evidence used in the M2.5 sanity script, and reports a live value estimate — end-to-end wiring confirmed, not learning behavior (weights are untrained/random). Confirmed no `q` and no hidden-occupancy path at the tensor boundary (mirrors the M2.5 leakage tests).

**Known limitation:** `masked_action_distribution()` (a convenience helper, not a frozen action interface) raises rather than returning NaN when `feasibility_mask` is all-`False`; callers must already treat an exhausted-budget state as terminal before querying the policy, consistent with `Environment.step`'s existing `done` signal.

**Reason:** Checkpoint A brief: reach a small, robust, graph-size-agnostic forward path (`GraphState -> tensors -> GNN -> logits/value`) without pre-empting action-space/reward decisions reserved for Checkpoint B, and without adding a hard ML dependency to the shared core package.

**Impact:** Non-RL side is unaffected — no changes to `models.py`, `graph_state.py`, `belief.py`, `environment.py`, `planners.py`, or their tests. Demu can proceed toward Checkpoint B (action-space + reward, jointly with the team once `FrontierPlanner`'s full sequential loop exists) and Checkpoint C (training) using this backbone.

**Owner:** Demu; branch `demu/gnn-rl-backbone`.

## 2026-09-10 — Action space + reward v0, first training run (Checkpoint B/C, PROPOSED — NOT yet reviewed by Pablo/Fede)

**Status:** PROPOSED. This is explicitly the kind of decision the handoff reserved for a joint Checkpoint B session ("action space + effort semantics + masking + reward"). It was implemented and a training run was started before that review happened, at Demu's discretion under explicit time pressure, once `m4/end-to-end-loop` showed the full sequential loop existed for real. Flag this entry to Pablo/Fede for review; nothing here should be treated as frozen.

**Decision:**

- **Action space**: per round, the policy autoregressively picks sites one at a time (`effort_per_pick=1`, matching `FrontierPlanner`'s existing default rather than inventing a new granularity), each pick adding one `MissionAllocation` and marking that site ineligible for further picks in the same round. A learned STOP token (extra logit from a small head on the graph context) lets the policy end the round early to preserve budget for future evidence; STOP is masked out on a round's first pick so a round can never legally produce an empty `MissionAction`. `node_embeddings`/`graph_context`/`node_logits` are computed once per round and reused across picks (the underlying `GraphState` does not change mid-round). Implementation: `src/adaptive_response/rl/round_policy.py` (`RoundPolicy`).
- **Credit assignment**: one round's full autoregressive pick sequence (including the final STOP) is treated as one factorized joint action, credited with one round-level reward via a single summed log-probability. No finer-grained (per-pick) credit assignment is implemented.
- **Reward** (`src/adaptive_response/rl/reward.py`), directly instantiating the reward candidate already named in the Technical Specification ("information gain / useful frontier coverage / detections - unnecessary effort - travel; terminal penalty per missed extent"):
  - per-round: `+2.0 * (mean uncertainty before − mean uncertainty after)` + `+1.0 * detections this round` − `0.02 * effort spent this round` (all from publicly observable `BeliefState`/`ObservationBatch`, no hidden truth);
  - terminal-only: `−1.0 * count(occupied sites with zero detections)`, which does need `HiddenWorld` — read via `Environment._hidden_world` (a private attribute) because `main` does not yet have M4's `Environment.reveal()` public API. This is training/evaluator-only code; it is never on the policy's observation path (`RoundPolicy.act` only ever receives `GraphState`). **TODO once `m4/end-to-end-loop` merges to `main`: switch `training_env.py` and `eval_utils.py` from the private attribute to `Environment.reveal()`.**
- **Algorithm**: on-policy actor-critic (REINFORCE + learned baseline, full-episode discounted returns, advantage normalization, entropy bonus), not clipped PPO — both are named as candidates in the Technical Specification, and the simpler formulation was chosen because a single gradient step per freshly-collected rollout batch has no stale-rollout importance-sampling problem for PPO's clip to solve. `src/adaptive_response/rl/trainer.py` (`ActorCriticTrainer`).
- **Domain randomization for training incidents** (`src/adaptive_response/rl/incident_sampler.py`): still only the M1 toy single-generator family (Pablo/Fede's multi-family simulator does not exist yet), but sites (12–24), budget (20–40), teams (1–3), habitat/q per site, and topology (random recursive-attachment tree + ~15% extra edges, not just a chain) are all randomized per episode.
- **Evaluation harness** (`src/adaptive_response/rl/eval_utils.py`): a planner-agnostic episode runner + `RLPlannerAdapter` so the trained policy can be benchmarked through the exact same code path as `FrontierPlanner`, on a fixed held-out incident/seed set (seeded independently of training draws) — this is the harness the "RL vs Information Gain on identical held-out incidents" gate from the handoff will eventually run through, once Information Gain exists.

**Validation:** New `tests/test_round_policy_training.py` (11 tests, same `pytest.importorskip("torch")` isolation). Full local suite: 56/56 passing. Confirmed: sampled incidents respect configured ranges and produce a connected, resettable `Environment`; `RoundPolicy` never double-picks a site within one round and always returns a mission `Environment.step()` accepts; a full episode's rollout lengths match `num_rounds` and always spends exactly the incident's budget; one `ActorCriticTrainer.update()` call produces finite losses and measurably changes parameters; `FrontierPlanner` and `RLPlannerAdapter` produce comparable `EpisodeMetrics` through the same harness on the same incident/seed. `scripts/train_gnn_policy.py` was smoke-tested end to end (tiny run: files/checkpoints/CSVs all written correctly) and timed at the intended default config (~1.8s/update at `episodes_per_update=32`, hidden_dim=64/num_layers=2) before starting a real run.

**Known limitations / explicitly not decided here:**
- Per-pick (rather than per-round) credit assignment was not attempted.
- `effort_per_pick=1` and the reward weights are defaults carried over from existing precedent (Frontier's default) or lifted directly from the Spec's named reward terms with round numbers filled in — none of them have had a sensitivity analysis, as the Spec itself requires before treating them as settled.
- Training happens only against the M1 toy generator; per the Data/Simulator/Validation Spec this must not be read as anything beyond "the loop and gradients work," and specifically not as evidence of real-world or even cross-simulator-family effectiveness.
- The `Environment._hidden_world` private-attribute access is a known, flagged, temporary workaround, not a new public contract.

**Reason:** Explicit user instruction to start a long (~7.5h) training run immediately, while stepping away and unable to review a design proposal first. Given `m4/end-to-end-loop` already demonstrates the real sequential loop the handoff wanted Checkpoint B to look at, filling in the open action-space/reward parameters now (rather than blocking) was judged lower-risk than leaving a "come back to us" checkpoint unresolved for hours with no training happening at all. This entry exists specifically so Pablo/Fede see exactly what was decided unilaterally and can override any of it.

**Impact:** A training run was launched in the background (see terminal transcript / `runs/<run-name>_<timestamp>/` for the specific run). Nothing here changes `GraphState`, `MissionAction`, `Planner`, or any file outside `src/adaptive_response/rl/` and `scripts/`. If Pablo/Fede reject any part of this design at review, the affected run(s) should be treated as disposable — checkpoints are cheap to regenerate once the design is corrected, per the project's own "don't count reward-curve-only progress" discipline.

**Owner:** Demu (unilateral, PROPOSED); requires Pablo + Fede review before being treated as CURRENT DEFAULT or FROZEN.

## 2026-09-11 — First empirical read: reward reweighting, three parallel runs

**Status:** Still PROPOSED/unreviewed; this is a data point for that review, not a new decision.

**What was compared**, all on the same fixed 20-incident held-out eval set (`eval_rng` seeded independently of `--seed`), same code otherwise:
- **v0** `checkpointB_v0_unattended` (seed=0): original weights (`missed_extent_weight=1.0`, `gamma=0.99`). Ran ~7.5h, 5800+ updates.
- **v1** `checkpointB_v1_reward_reweight` (seed=1): `missed_extent_weight=8.0`, `gamma=0.99` (isolates the reward-weight change).
- **v2** `checkpointB_v2_reward_reweight_lessdiscount` (seed=2): `missed_extent_weight=8.0`, `gamma=0.995`.

(Return normalization, see the entry above this one, was required first: raising `missed_extent_weight` alone without it pushed the loss into the hundreds in a smoke test.)

**Result at v0's end (~5800 updates) vs v1/v2 at ~2500-2600 updates each:**
- v0 never beat FrontierPlanner's held-out missed_fraction (0.2543, constant): best 0.390 at update 3600, and its last-20-point trend actually reversed (correlation +0.65 — got noisier/worse late, not better) despite entropy having collapsed to ~1.5-2.
- v1: missed_fraction 0.209 (first 10 evals avg) -> 0.183 (last 10) -> 0.155 (latest, also its best), consistently *below* Frontier's 0.254 essentially throughout, trend correlation -0.59 (still improving, not flat).
- v2: 0.251 -> 0.199 -> 0.184 latest (best 0.178), also consistently below Frontier once past its first few evals, trend correlation -0.53.

**Why this is more than "one lucky number"**: two independent seeds (v1, v2) sharing only the `missed_extent_weight` change both show the same qualitative shift - consistent, still-improving outperformance of Frontier on the fixed eval set - while v0 (same seed family, old weight) never got there in more total updates. If this were purely seed-init luck rather than the reward change, both would not be expected to move the same way. It does NOT yet separate "the weight change" from "the gamma change" (v1 vs v2 differ on both seed and gamma), and it does not yet say whether v1's slight edge over v2 is the smaller discounting or seed luck.

**What this does NOT show** (per the project's own claim discipline): validity beyond this one M1 toy-generator family and this one fixed 20-incident eval set; that RL beats Information Gain (doesn't exist yet); that this generalizes to different graph-size ranges, budgets, or a real simulator family. It is a same-generator, same-eval-set result on early-in-training checkpoints (v1/v2 are ~20% through their planned 7.5h budget), not a benchmark-grade claim.

**Decision:** Let v1 and v2 keep running to use their full compute budget rather than stopping early on a promising-but-partial trend; v0 is allowed to finish and then retired (it has clearly plateaued/regressed, not worth further wall-clock). No further reward/hyperparameter changes made mid-run — comparing more variants now would re-introduce the "changed too many things at once" problem this entry is trying to avoid.

**Owner:** Demu (unilateral, PROPOSED); flagging to Pablo + Fede that the Checkpoint B reward design looks directionally validated on this narrow test, pending their review and, eventually, benchmarking against Information Gain rather than only Frontier.

## 2026-09-11 — Follow-up read at ~5700-6000 updates: plateau, not continued improvement, and v2 instability

**Status:** Still PROPOSED/unreviewed; refines the entry above rather than replacing it.

**What changed since the last read:** v1/v2 ran ~3 more hours (v1: update 5600, v2: update 6000; both roughly half their 7.5h budget).

**v1**: stabilized into a consistent 0.156-0.199 missed_fraction band, every single one of the last 15 eval points below Frontier's 0.254. Trend correlation over the last 20 points is ~0.05 (flat) - it has stopped improving, but it has not degraded either.

**v2**: noticeably less stable. In the update-4900-5600 window it repeatedly matched or exceeded Frontier (0.299, 0.254, 0.294, 0.256), before recovering to 0.18-0.21 in its last few evals. Best single point (0.156 at update 4700) is comparable to v1's, but the path there is far noisier.

**Root cause, from `metrics.csv`**: entropy for both v1 and v2 plateaued around 8-13 nats by roughly update 1500-2000 and never continued collapsing toward a confident policy the way v0's did (v0 reached ~1.5-2 by its end). Working hypothesis: multiplying `missed_extent_weight` by 8x also multiplies the *variance* of the terminal reward term - the underlying detection outcome is still a stochastic Bernoulli draw (`q` in the 0.15-0.45 range), so a bigger weight on a noisy binary-ish outcome makes the policy-gradient signal noisier, which plausibly stalls the natural entropy reduction rather than only reweighting it. Not confirmed, just the most consistent explanation for what's observed.

**Interpretation:** the win over Frontier established in the previous entry holds up over 3 more hours (v1 especially - it did not regress into a v0-style late decline), but neither variant is still improving; both found a plateau early and are oscillating around it. v2's extra instability adds more (still seed-confounded) weight to "the gamma change is not obviously helping, and may hurt" from the previous entry.

**Decision:** No mid-run changes (same reasoning as before - don't add more moving pieces to an already-confounded comparison). Let both finish their 7.5h. Flag for the actual Checkpoint C training run (once Pablo/Fede review this and it becomes real, not disposable): reward-scale changes should probably be accompanied by variance-reduction measures (more episodes per update, and/or an entropy-coefficient or reward-scale schedule) rather than assuming more wall-clock time alone will finish the convergence a raw weight bump stalls.

**Owner:** Demu (unilateral, PROPOSED).

## 2026-09-11 — Final wrap-up: v0 vs v1 vs v2, all three runs complete

**Status:** PROPOSED, ready for Pablo/Fede review. This closes out the three-run comparison; it does not freeze anything.

**All three finished their planned 7.5h**: v0 5909 updates (retired earlier, see above), v1 11299 updates, v2 12023 updates. All evaluated throughout on the same fixed 20-incident held-out set.

**Final state:**
- **v1** (`missed_extent_weight=8.0`, `gamma=0.99`, seed=1): last 20 eval points range 0.155-0.218, *every single one* below Frontier's constant 0.2543. Best 0.1517 (update 8700). No degradation across the whole second half of training - the most stable of the three runs by a clear margin.
- **v2** (`missed_extent_weight=8.0`, `gamma=0.995`, seed=2): mostly matched v1 in its final stretch (19 of its last 20 points also below Frontier, best 0.1488 at update 6500, comparable or slightly better than v1's best), but its *very last* checkpoint spiked to 0.2795 - worse than Frontier - and it separately passed through a multi-checkpoint bad patch mid-training (updates ~4900-5600, documented in the previous entry). v1 never did either.
- **v0** (original `missed_extent_weight=1.0`, `gamma=0.99`, seed=0): never beat Frontier in 5909 updates; regressed rather than improved in its final stretch. Confirms the reward reweighting, not just "more training," is what changed the outcome.

**Verdict:** the `missed_extent_weight` reweighting (1.0 -> 8.0) is a real, load-bearing fix - two independent seeds with that change both decisively and durably beat FrontierPlanner on this held-out set, where the unweighted version never did. Between v1 and v2, **v1's configuration (`gamma=0.99`, i.e. only the reward change, not also the discount change) looks like the more reliable of the two** - fewer moving parts, zero excursions above Frontier in its back half versus v2's two. This is *not* a confirmed causal claim about gamma specifically: v1 and v2 also differ by seed, and this run never included a same-seed ablation to isolate it. Final checkpoints (`final.pt` in each run directory) are saved for either variant if useful, but are PROPOSED-design artifacts, not validated policies - treat as disposable/re-trainable, not as something to demo or benchmark-report without review.

**What this does and does not show, restated once more for anyone reading only this final entry**: shows a real, reproducible-across-seed improvement over a simple frontier heuristic, on one synthetic toy-generator family, on one fixed 20-incident eval set, using a reward/action-space design nobody but Demu has reviewed. Does not show anything about Information Gain (doesn't exist yet), other world-model families (don't exist yet), or real-world validity.

**Recommended next steps, in order:**
1. Pablo/Fede review this whole PROPOSED thread (action space, reward, and now these results) and either approve, adjust, or reject before any of it is treated as CURRENT DEFAULT.
2. If approved-with-changes or approved-as-is: re-run with the entropy-coefficient decay fix now available (`--entropy-coef-final`/`--entropy-coef-decay-updates`, added 2026-09-11, not used in this comparison) to address the entropy-plateau finding, and this time control for seed (same seed across variants, only vary the hyperparameter under test) to properly separate reward-weight effects from initialization luck.
3. Once Information Gain exists, re-benchmark against that, not just Frontier - beating Frontier is necessary but was never the actual bar in the original handoff.
4. Once multiple world-model families exist, re-test there before claiming anything about robustness (Checkpoint D).

**Owner:** Demu (unilateral, PROPOSED); awaiting Pablo + Fede review.

## 2026-09-11 — RL training/eval now route through AdaptiveMissionLoop (M4)

**Status:** Implementation cleanup, not a design decision. No action-space/reward semantics changed.

**Decision:** Merged `main` (which now has M4's `AdaptiveMissionLoop`/`Environment.reveal()`) into `demu/gnn-rl-backbone`. Rewrote `training_env.run_episode()` and `eval_utils.run_planner_episode()` to drive the shared `AdaptiveMissionLoop` instead of re-implementing plan/execute/Bayes/replan by hand against `Environment` directly. This retires the private `Environment._hidden_world` access flagged as a temporary workaround in every prior entry in this thread - `HiddenWorld` now only ever reaches training/eval code through the real `loop.reveal()` gate.

**Bug caught while doing this (not just a style change):** `AdaptiveMissionLoop.execute_pending()` internally calls `plan_next()` again for the *next* round as soon as the current one isn't done, to have a mission ready for the following `run_round()`. A naive "call `run_round()`, then read back what the planner just decided" pattern - the obvious way to recover the log_prob/value/entropy a training loop needs, since `run_round()` only returns a `MissionAction` - would silently misattribute: the planner gets invoked a second time for the next round *inside* that same call, overwriting whatever it stashed for the round that actually just executed. Fixed by calling `plan_next()` explicitly and reading the stashed decision back immediately, before calling `execute_pending()`. A second, related bug: `reveal()` moves the loop's phase from `COMPLETE` to `REVEALED`, so a `while phase is not COMPLETE` loop condition fires one extra (invalid) round after the terminal one; fixed with an explicit `break`/`return` on the terminal transition instead of trusting the loop condition.

**Validation:** Neither bug was caught by the existing test suite passing - both are logic errors that produce a *plausible-looking* wrong answer (misattributed rewards, one bogus extra round) rather than a crash. Verified directly with a standalone script asserting `decision.mission == transition.mission` for every round of a multi-round episode, plus a forced single-pick-per-round edge case (6 rounds for a 6-unit budget) checking rollout length/effort consistency. Full 64/64 suite passes; `scripts/run_gnn_backbone_sanity.py` and a `train_gnn_policy.py` smoke run at the same seed as an earlier pre-refactor run reproduce closely matching loss/entropy trajectories.

**Why this matters for reviewing the earlier entries in this thread:** the v0/v1/v2 comparison runs were all executed *before* this fix, using the old hand-rolled loop (not `AdaptiveMissionLoop`, no lookahead-planning call, so the misattribution bug described above could not have occurred in that code path - it only exists in code that calls `execute_pending()`/`run_round()` and then reads planner state back out, which the old loop never did). Their results stand as reported. This entry exists so nobody has to independently re-derive whether the fix invalidates prior numbers: it does not, because the bug it fixes didn't exist in the code that produced them.

**Owner:** Demu.

## 2026-09-11 — Same-seed ablation (D/E/F): humbler result than v0/v1/v2 suggested

**Status:** PROPOSED, still unreviewed. This tempers the optimism of the earlier v0/v1/v2 entries rather than confirming it - read both together, not this one in isolation.

**Setup**: three runs on the now-refactored (`AdaptiveMissionLoop`-based) code, all `missed_extent_weight=8.0`, all **seed=10** (unlike v0/v1/v2's one-seed-per-variant design), 3.5h each: **D** (`gamma=0.99`, replicates v1's hyperparameters at a new seed), **E** (`gamma=0.995`, same seed as D - isolates gamma cleanly for the first time), **F** (`gamma=0.99` + entropy-coefficient decay 0.01->0.0005 over 4000 updates, testing the fix from the entry above).

**Final numbers** (all on the same fixed 20-incident held-out set as every prior entry):
- **D**: final 0.1546 (its best ever), but the full trajectory is highly volatile - its last 15 evals range from 0.155 to 0.341, with three separate points *worse* than Frontier's 0.2543 scattered through them. The good final number looks like it landed on a lucky checkpoint, not a converged, trustworthy policy.
- **E**: mostly tighter (0.16-0.22 for 14 of its last 15 evals) but with one sharp outlier at update 3800 (0.390, well worse than Frontier). Ends at 0.2006. More consistent than D overall, but not clean.
- **F**: the noisiest of the three - five separate points at or worse than Frontier scattered through its last 15 evals (0.255, 0.255, 0.359, and a final 0.3054), interspersed with good ones (0.183-0.201). Entropy collapsed to ~2.8-3.0 nats (coefficient near its 0.0005 floor) without settling into reliably good behavior.

**What this changes about the previous entry's conclusions:**
- The gamma question (0.99 vs 0.995) is **still not resolved**, and if anything looks less decidable than before: with seed now controlled, neither D nor E clearly dominates the other - D swings harder in both directions, E is tighter except for one bad excursion. No confident causal claim about gamma is supportable from two seeds.
- **v1's remarkable stability in the first round (never once below Frontier across its whole second half) does not replicate here.** Neither D (same gamma as v1) nor E nor F comes close to that level of consistency at similar update counts. The honest reading is that v1's specific run was an unusually good outcome - a favorable seed and/or training trajectory - not a property of `missed_extent_weight=8.0` that reproduces reliably on its own.
- The entropy-decay fix (F) does **not** look like a win here - if anything, F's second half shows *more* frequent excursions to Frontier-or-worse than D or E, right as its entropy bottoms out. Collapsing entropy this aggressively (to a 0.0005 floor within 4000 updates) plausibly removes the policy's ability to recover from a bad patch rather than helping it commit to a good one. Do not carry this specific schedule forward without revisiting it.

**Revised overall picture across both rounds (six seeds total: v0, v1, v2, D, E, F)**: `missed_extent_weight=8.0` reliably produces policies that spend *some* portion of training clearly ahead of Frontier - none of the five reweighted runs (v1, v2, D, E, F) failed to beat Frontier at some point, unlike the unweighted v0, which never did. But **consistency is highly run-to-run variable**, and at least three of the five (v2 briefly, D, F) show real excursions to at-or-worse-than-Frontier even late in training. This is not yet a "GNN/RL reliably beats Frontier" result - it is "the reward shape can produce that outcome, inconsistently, with a single-seed actor-critic setup that has no variance-reduction beyond return normalization."

**Recommended next steps, revised:**
1. Same as before: Pablo/Fede review is the actual next step, not more solo runs by Demu.
2. If the team wants a more decisive answer before committing further compute: the highest-value next experiment is not another gamma/entropy variant, but **variance reduction at the algorithm level** - more episodes per update (currently 32), and/or averaging multiple seeds per configuration before comparing, rather than reading single-seed trajectories as if they were representative.
3. Any demo-facing claim should report a distribution across seeds/checkpoints ("beats Frontier in N of M evaluated checkpoints"), not a single cherry-picked best number - several of this run's "best ever" readings (e.g. D's 0.1546) sit right next to much worse ones a few hundred updates away.

**Owner:** Demu (unilateral, PROPOSED); awaiting Pablo + Fede review of the whole thread, now with a more complete and more honest picture than the first round alone would have given.

## 2026-09-11 — Team acknowledgment: Data-First Training & Validation Freeze; v0-v2/D-F reclassified

**Status:** ACKNOWLEDGED. `docs/DATA_FIRST_VALIDATION_FREEZE.md` (PR #10, merged to main, then merged into this branch) is now the governing methodological constraint for anything ecological in this project. This entry records compliance, it does not add new findings.

**What changed:**
- **Checkpoint A is formally APPROVED** by the team (reviewed against the actual commit `7d7b638`). The `GraphState -> tensors -> GNN actor/critic` backbone stands as accepted infrastructure, independent of the ecological generator, and is not to be rewritten.
- **Every run in this thread (v0, v1, v2, D, E, F) is reclassified as SMOKE / ENGINEERING EXPERIMENTS ONLY**, per the freeze's explicit Section 10. This is a *reclassification of interpretation*, not a retraction of the numbers - the numbers stand exactly as measured and reported in the entries above. What changes is what they are allowed to be used for. Restated explicitly per the freeze's own language: these results are not project results, do not validate ecology, must not determine final reward/evaluation choices, and must not be used to claim RL superiority. They remain valid answers to engineering questions: does training run without exploding, do gradients flow, does the STOP/action machinery work, is logging/checkpointing reproducible, does the policy handle variable graph sizes. See the separate consolidated engineering report requested by the team for that framing.
- **No further reward/hyperparameter tuning aimed at improving those scores.** Action space, effort semantics, masking, and reward remain PROPOSED, to be reviewed jointly before any "serious" training, which is now explicitly gated on Pablo/Fede freezing: real-data pack, real graph, q/effort treatment, world-model families (>=3 train, >=1 held-out), held-out/OOD split, action/masking contract, and reward + independent evaluation metrics.
- **`Environment._hidden_world` private-attribute access was already retired** before this freeze landed (see the "RL training/eval now route through `AdaptiveMissionLoop`" entry above, commit `bbb7686`) - `training_env.py` and `eval_utils.py` already only reach hidden truth through `loop.reveal()`, gated by the loop's own state machine. No further change needed for this requirement; confirmed by grep on this branch at merge time (2026-09-11) that no `_hidden_world` reference remains anywhere under `src/adaptive_response/rl/`.
- **The competitive bar remains Information Gain, not Frontier.** Nothing in this thread has benchmarked against Information Gain (it doesn't exist yet) - beating Frontier was never sufficient and is restated as such here for the record.

**What Demu does next, per the freeze's own instruction:** wait for Pablo/Fede's real-data/world-model/action-space freeze before any further training aimed at a project result. In the meantime, permitted work is explicitly non-ecological: backbone robustness, deterministic inference, logging, checkpointing, training-pipeline reliability, and tests - i.e. engineering hardening of what already exists, not new experiments on the toy generator.

**Owner:** Demu; acknowledging Pablo + Fede's team-approved freeze (PR #10).

## 2026-09-12 — Engineering block complete: checkpointing, hardening, benchmark runner, decision logging

**Status:** DONE, per the team's engineering-block instruction (non-ecological work permitted while the second ecological handoff is pending). No action-space/reward/evaluation semantics changed. Full detail in `docs/ENGINEERING_BLOCK_REPORT.md`; this entry is the Decision Log pointer to it.

**Decision/what changed:**
- `src/adaptive_response/rl/checkpointing.py` (commit `22a257e`): self-describing checkpoints (`PolicyArchitectureConfig` embedded alongside weights), `save_policy_checkpoint`/`load_policy_checkpoint`/`load_optimizer_state`, inference-only loads supported, foreign files rejected with a clear `ValueError`. `train_gnn_policy.py` now builds policies through `PolicyArchitectureConfig.build()` and saves through this module instead of ad hoc `torch.save`.
- `tests/test_hardening.py` (commit `50327bb`, 18 tests): variable graph size N in {1,2,12,20,24,30,40} including a ~40-episode stochastic stress test against the real `Environment` (no duplicate picks, no budget overrun, terminates within a safety cap), plus widened structural/signature checks that no module under `src/adaptive_response/rl/` can import or accept `HiddenWorld`.
- `src/adaptive_response/rl/benchmark.py` + `scripts/run_benchmark.py` (commit `5ad0e87`): planner-agnostic benchmark runner (`Planner` protocol only - Frontier/RL/future InformationGain are interchangeable). `BenchmarkRow` is deliberately a raw fact table with no score/rank/winner field, pinned by `test_benchmark_row_has_no_ranking_metric_baked_in`, so this does not freeze evaluation metrics.
- Per-round decision logging: `RoundDecision` gained `node_ids`/`node_logits` (commit `c607402`); `reward.py` gained `round_reward_components()` (value-preserving refactor, `round_reward()` is now its sum); `decision_logger.py` (new) writes one `RoundLogRecord` per round (logits, entropy, value, log-prob, reward components, chosen sites, budget before/after); `training_env.run_episode()` takes an opt-in `decision_logger` param; `train_gnn_policy.py` exposes it as `--decision-log` (commit `d6ad9fc`), off by default.

**Validation:** Full suite 100/100 (`test_belief`8, `test_benchmark`7, `test_checkpointing`7, `test_decision_logging`4, `test_environment`6, `test_frontier_planner`9, `test_gnn_backbone`11, `test_graph_state`8, `test_hardening`18, `test_mission_loop`8, `test_round_policy_training`6, `test_simulator_step`8). Team-authorized smoke run after all four pieces landed together (`engineering_block_smoke`, 10 updates x 4 episodes/update, `--decision-log` on): zero crashes, checkpoints saved/loadable, `decisions.jsonl` produced 175 well-formed records with `done`/`terminal_missed_extent` set only on each episode's true final round. No hyperparameter campaign, no claim of learning quality drawn from this run.

**Owner:** Demu.

## 2026-09-12 — Team ecology/data update acknowledged; engineering block confirmed still frozen; tiny eligible-mask hardening added

**Status:** ACKNOWLEDGED. Recording the team's substantive real-data progress for traceability, confirming compliance with their explicit instruction to keep the engineering block frozen, and logging the one small permitted addition. No design decision made unilaterally here.

**What the team reported:** real monitoring network now at 49 sites; primary real graph v0 (variable degree, disconnected allowed, 118 local routed edges) - straight-line-water rejection was explicitly discarded after curved SalishSeaCast routing showed all 12 previously-rejected local edges actually have viable water routes; incident extraction is now deterministic from the initial detection + static graph only; 46/49 possible seeds naturally produce 14-17-site incident graphs, 3 genuinely isolated monitoring sites stay size-1 rather than being artificially bridged; monthly monitoring table has 1,975 site-year-month observations at 99.6% effort coverage.

**Important ecological finding, not a design choice:** per-effort detectability `q` is **not identifiable** from this dataset alone - the team will not infer it from raw detection fraction. The functional form `P(no detection | occupied, e, q) = (1-q)^e` is FROZEN; the numeric value of `q` is explicitly NOT frozen yet. Current MVP direction (leaning, not decided): an uncertain effective protocol-level `q` - simulator-side `q_true` sampled from a constrained distribution, belief-side uncertainty over `q`, plus dedicated q-shift/OOD tests. **No `q` node feature or GraphState schema change is authorized yet.**

**Explicit instruction, reconfirmed:** the engineering block stays FROZEN. No serious training, no reward tuning, no variance reduction, no Information Gain implementation, no GraphState/action-schema changes - all still "yet," pending the second ecological handoff (spatial belief + world-model/validation contracts).

**Two notes recorded for that eventual integration:**
1. Final benchmark cases must come from the externally frozen ecological benchmark/world-model pack, not the current toy `sample_incident` generator. The team confirmed `BenchmarkCase` (added this engineering block, see the entry above) "looks suitable for that" - no interface change anticipated, just a different case source plugged in later.
2. When integration begins: rebase/sync once against the then-current ecological branch and rerun the combined suite. Branches have evolved in parallel and diverged in size - this branch is at 100 tests, the team's data/ecology branch is already at 135 - purely a consequence of parallel work, not a discrepancy to resolve now.

**Tiny hardening actually done (explicitly authorized as "optional, only if genuinely quick," no policy-behavior change permitted):** `RoundDecision`/`RoundLogRecord` gained `eligible_mask: tuple[bool, ...]`, index-aligned with the existing `node_ids`/`node_logits`, so a later reader can tell a genuinely low-scoring legal action apart from a high-scoring one that was never legal - a direct read-out of the feasibility mask `act()` already computes, not a new computation (commit `4888f0f`). Full 100/100 suite still green; verified with a real `--decision-log` CLI smoke run.

**What happens next:** nothing else, per instruction. Waiting for the real second ecological handoff (real graph + observation model/q + spatial belief + world-model families + frozen validation splits/metrics) before resuming any training-adjacent work.

**Owner:** Demu; acknowledging Pablo + Fede's ecology/data update.

## 2026-09-12 — Second ecological handoff (R7): ACK on action contract, ACK-with-a-watch-item on reward; GNN/RL integrated with SpatialAdaptiveMissionLoop

**Status:** ACK delivered per docs/DEMU_HANDOFF_R7.md's explicit request ("either ACK the recommended action/reward contract or return one concrete counterproposal"). Both read in full, along with configs/benchmark_protocol_r7.json and configs/q_protocol_r7.json, before responding - not answered from the chat summary alone.

**Action contract: ACK, no counterproposal.** One (site, effort) pick per round, effort in {1,3,6}, revisits allowed across rounds, budget 18, horizon 6. This is a real simplification versus the Checkpoint-B autoregressive multi-site design (no STOP token, no within-round duplicate-exclusion logic - moot with one pick per round), reuses the existing GNN encoder/critic unchanged, and is more REINFORCE-friendly (one discrete choice per round instead of an autoregressive sequence). No RL-engineering reason to prefer the old contract.

**Reward contract: ACK, with one flagged watch-item, not a counterproposal.** The recommended terminal weight (-2.0 x missed_occupied_fraction) is smaller in raw magnitude than the missed_extent_weight=8.0 this branch's own D/E/F ablation found necessary to avoid the terminal term being drowned out by cumulative per-round dense reward - but the underlying quantities changed (mean entropy and a *fraction*, not a per-round sum and a raw *count*), so that old number does not mechanically transfer as evidence for a different one here. The protocol itself already reserves room for this ("a single reward-weight ablation is acceptable later if runtime permits"). Decision: use the recommended weights as given; watch the logged reward_components (dense vs. terminal magnitude) during training/benchmark and flag if the terminal term looks drowned out, rather than pre-emptively changing a number with no new evidence.

**Integration delivered** (branch `demu/gnn-rl-backbone`, synced once from `origin/r4/spatial-belief` - commit `bb0e872`, no conflicts):
- `backbone.py` +`SiteEffortActorHead` (K logits/node instead of 1); `site_effort_policy.py` (new `SiteEffortRoundPolicy`/`SiteEffortDecision`) - reuses the unchanged encoder/global-context/critic, no new message-passing (commit `2d3a6a6`).
- `reward.py` +`SpatialRewardConfig`/`spatial_round_reward_components`/`spatial_terminal_reward_components`, matching the ACKed weights exactly, kept separate from the older `RewardConfig` (different normalization, different code path).
- `spatial_metrics.py`: the three planner-independent primary metrics (missed occupied sites, occupied-site coverage, final global uncertainty), zero reward-weight coupling.
- `spatial_training_env.py`: `run_spatial_episode` drives the real `SpatialAdaptiveMissionLoop`, same stash-before-execute_pending discipline as the original engineering-block fix to avoid the lookahead-misattribution bug, enforces the max_rounds horizon.
- `spatial_mission_loop.py` (Pablo/Fede's file) +`force_complete()`: the one small hook needed to end an episode at the horizon (a concept the loop itself has no notion of, since it lives in the R7 action contract, not IncidentConfig/Environment) the same way budget-exhaustion already does.
- `checkpointing.py` +`SiteEffortPolicyArchitectureConfig`, dispatched via a new `architecture_kind` payload field defaulting to `"round_policy"` for checkpoints written before this field existed - old checkpoints still load correctly.
- `decision_logger.py` +`SpatialRoundLogRecord` (site, effort, [N,K] logits/eligibility, reward components) - sibling to the older `RoundLogRecord`, not an overload of its shape.
- `real_graph_cases.py` (new): builds `IncidentConfig` from the real, committed 49-site/118-edge graph (`reports/milestones/r2_real_graph_v0/real_graph_v0_edges.csv`), not the toy `sample_incident` generator (commit `1d5f9b6`). Recomputing `eligible_incident_seed_sites()` reproduces the team's own reported 46/49-seeds-in-range finding exactly, confirming this is reading the real, correct graph.
- `planners.py`: `InformationGainPlanner` gained an additive `effort_levels` parameter (default `None` = unchanged original behavior) so it searches (site, effort) jointly and ranks by information gain per effort unit, per `benchmark_protocol_r7.json`'s `recommended_information_gain_contract` - needed so IG and RL compete under the identical action contract.
- `spatial_benchmark.py` + `scripts/run_spatial_benchmark_r7.py`: the required Frontier/IG/RL benchmark across id_test / ood_model_E / ood_q_low / ood_q_high groups, reporting mean+dispersion, visible failure cases, and the claim guardrail verbatim - no winner-only score.
- **Real bug found and fixed while building the benchmark**: `FrontierPlanner()`'s own defaults (`max_sites=None`) let it pick many sites in one round with unbounded effort, silently violating "same feasibility constraints for every planner." Fixed by pinning it to `effort_per_site=1, max_sites=1` in both the training script's periodic eval and the benchmark script - a disclosed asymmetry (Frontier has no per-effort search, so it always uses the smallest level), not silently different action feasibility (commit `8e39ec2`).

**Real, disclosed blocker**: raw Dryad monitoring CSVs (site coordinates, Crab Team habitat calibration) are not fetchable from this environment - `scripts/fetch_r0_data.py` returns HTTP 401/403 from datadryad.org, and `data/raw/README.md` itself says not to substitute look-alike files. `real_graph_cases.py` therefore uses a deterministic graph-layout placeholder for site x/y and a neutral constant for `habitat_score`, clearly labeled as such - the graph TOPOLOGY (site ids, adjacency) is the real, frozen R2 artifact, unaffected by this.

**Validation**: full suite 253/253 (the pre-existing 100 + 135 from the r4/spatial-belief merge + `test_spatial_rl_integration.py` (10 new) + 3 new `InformationGainPlanner` effort-level tests). One mechanical smoke run (`r7_smoke_official`, 15 updates x 8 episodes, hidden_dim=32) proved the full pipeline end-to-end - real graph -> spatial belief + uncertain q -> GraphState -> planner -> MissionAction -> hidden simulator -> observation -> posterior -> replan - zero crashes, checkpoints/decision-log correct, periodic eval against Frontier/InformationGain ran without error. No learning-quality claim drawn from it. A longer training run (`r7_serious_v0`, 300 updates x 16 episodes, hidden_dim=64/num_layers=2) and the full id_test/ood_model_E/ood_q_low/ood_q_high benchmark are reported separately once complete.

**Owner:** Demu.

## 2026-09-12 — R7 serious training + required Frontier/IG/RL benchmark: first result, single seed

**Status:** First formal-pipeline benchmark result under the ACKed R7 contract. Explicitly a single-seed, 300-update first pass, not a validated claim - see `docs/R7_BENCHMARK_REPORT.md` for the full write-up and all required fields (ID mean+dispersion, OOD model E, OOD q-low/q-high, visible failure cases, runtime/seed-stability note, claim guardrail).

**What ran:** `scripts/train_spatial_gnn_policy.py` run `r7_serious_v0` (seed 0, hidden_dim=64/num_layers=2, 300 updates x 16 episodes/update = 4,800 episodes, 675.8s). `scripts/run_spatial_benchmark_r7.py` against its `final.pt`: 6 held-out real-graph seed sites x 5 cases x 4 groups (id_test, ood_model_E, ood_q_low, ood_q_high) x 3 planners = 360 episodes, 148.3s. Raw rows in `reports/r7_benchmark_v0.csv`.

**Headline numbers (mean missed_occupied_fraction +/- stdev, n=30/group):** id_test - frontier 0.774+/-0.180, information_gain 0.760+/-0.194, gnn_rl 0.668+/-0.205. ood_model_E - frontier 0.753+/-0.168, information_gain 0.782+/-0.163, gnn_rl 0.709+/-0.191. ood_q_low - all three within 0.751-0.757 (expected: q_true=0.02 makes strategy matter little). ood_q_high - frontier 0.647+/-0.163, information_gain 0.664+/-0.206, gnn_rl 0.416+/-0.246 (the one visually large gap in this report).

**What this does NOT license claiming:** per-case dispersion (~0.16-0.25) is large relative to the gaps between planners (~0.02-0.23); this is one training seed and one checkpoint. The prior D/E/F ablation on this branch already showed single-seed REINFORCE-style variance is real and can make an individual run look better or worse than the underlying setup reliably is - that finding carries over structurally here. Nothing above should be read as "GNN+RL beats Information Gain" until multiple independent training seeds are compared and averaged. Explicitly not a winner-only score, not a claim of ecological representativeness (world-model ranges/q values are declared benchmark scenario anchors in `configs/q_protocol_r7.json`, not calibrated estimates), not proven field effectiveness.

**Owner:** Demu.

## 2026-09-12 — R7 multi-seed replication: the id_test/ood_model_E/ood_q_high gap holds across 3 independent training seeds

**Status:** Strengthens, does not settle, the single-seed R7 benchmark entry above. Full table in `docs/R7_BENCHMARK_REPORT.md` Section 6.

**What ran:** two more independent training seeds (1, 2), same hyperparameters as `r7_serious_v0` (seed 0), benchmarked against the identical 120 cases (same `--seed 12345`). Raw rows: `reports/r7_benchmark_seed1.csv`, `reports/r7_benchmark_seed2.csv`.

**Finding:** Frontier/Information Gain are near-deterministic across training seeds (stdev-across-seeds <=0.005 - they don't depend on RL training). GNN+RL carries real seed-to-seed variance (stdev 0.006-0.066, largest in ood_q_high) - the same single-seed-REINFORCE-variance lesson from the pre-R7 D/E/F ablation still applies. But the DIRECTION is now consistent across all three independent runs: every one of the three RL seeds landed clearly below both baselines in id_test, ood_model_E, and ood_q_high (mean missed_occupied_fraction across seeds: id_test 0.678 vs frontier 0.775/IG 0.760; ood_model_E 0.717 vs 0.754/0.779; ood_q_high 0.461 vs 0.646/0.661). ood_q_low stays indistinguishable across all three, as expected (q_true=0.02 makes strategy matter little).

**What this still does not license:** n=3 seeds, 300 updates each (not run to convergence), one architecture/hyperparameter choice, and the case set is this branch's provisional real-graph sampling, not the team's eventual frozen OOD manifest. This is meaningfully stronger evidence than the single-seed result - a consistent direction across 3 independent runs is not nothing - but still not a claim to present as settled.

**Owner:** Demu.

## 2026-09-13 — R8 methodological review acknowledged: real-site context + corrected baseline fairness; retraining under way

**Status:** ACKNOWLEDGED and acted on. Team's `docs/R8_DEMU_BENCHMARK_REVIEW.md` correctly identified two real defects in the R7 provisional benchmark and froze the corrected rules (`configs/benchmark_protocol_r8.json`, `configs/real_site_context_r8.json`) plus a frozen 240-case manifest (`reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json`). Both are accepted without objection - they are correct.

**BLOCKER 1 resolved (real-site context)**: `real_graph_cases.py` rewritten to use versioned real monitoring coordinates (`reports/milestones/r2_real_graph_v0/real_sites_v0.csv`, lat/lon locally projected to km) and the frozen R5 habitat proxy by `crabteam_habitat` class (`configs/real_site_context_r8.json`), replacing the R7-provisional graph-layout coordinates and neutral habitat placeholder. Topology (site ids, adjacency) is unchanged - it was already real in R7.

**BLOCKER 2 resolved (baseline fairness)**: the team's own evidence was exactly right - Frontier/IG were structurally trapped at 6/18 effort while RL commonly used 18/18, confounding the R7 gap with resource utilization, not just policy quality. Corrected: `FrontierPlanner` gained an additive `effort_levels` mode (standard-event effort 6 when budget permits, else the largest feasible level); `InformationGainPlanner`'s effort search now ranks by ABSOLUTE expected information gain with IG-per-effort only a tie-break (reversed from R7's per-effort-primary ranking, which is exactly what caused the effort=1 bias).

**Frozen manifest consumer built**: `r8_manifest_cases.py` loads the exact 240-case manifest and materializes runnable cases (180 formal: id_test 90, ood_model_test 30, ood_q_low 30, ood_q_high 30; 60 validation cases correctly excluded from formal reporting per the manifest doc). `scripts/run_spatial_benchmark_r8.py` runs Frontier/IG/RL against these exact cases and reports the mandatory secondary fields (effort_spent, detections_found, num_rounds, wall_clock_seconds) alongside the primary metrics.

**Retraining**: because B/C/E world generation changes under real coordinates/habitat, all three formal RL seeds (0/1/2) are being retrained from scratch with identical hyperparameters to the R7 runs (hidden_dim=64/num_layers=2, 300 updates x 16 episodes) - no new architecture or hyperparameter search, per instruction. R7's checkpoints are not reused for the R8 report.

**Full suite green** after the merge + corrections. Mechanical smoke-check (12-case subset spanning all 4 formal groups, using a stale R7 checkpoint only to prove the pipeline runs) confirms Frontier/IG now consistently spend the full budget (18/18 in 3 rounds at effort=6) instead of 6/18.

**Claim discipline going forward, per the review's own wording**: until the corrected rerun completes, the right statement is "three provisional training seeds showed a repeatable learned-policy signal on an integration benchmark, but the final comparison is pending corrected real-site covariates, matched benchmark cases, and stronger budget-aware baselines." Report to follow once retraining + the frozen R8 benchmark are done.

**Owner:** Demu.

## 2026-09-13 — R8 corrected benchmark result: the R7 apparent RL advantage does not survive the fairness correction

**Status:** Final result for the R8 rerun requested in `docs/R8_DEMU_BENCHMARK_REVIEW.md`. Full write-up: `docs/R8_BENCHMARK_REPORT.md`. This supersedes the R7 provisional numbers, which stay on record as exactly what the R8 review corrected and why.

**What ran:** three RL seeds (0/1/2) retrained from scratch under the real-site context (real coordinates for B/E, frozen R5 habitat proxy for C), identical hyperparameters to R7 (hidden_dim=64/num_layers=2, 300 updates x 16 episodes). Benchmarked against the exact frozen 180-case manifest (id_test 90, ood_model_test 30, ood_q_low 30, ood_q_high 30) with the corrected budget-fair Frontier (effort=6 when possible) and Information Gain (absolute-IG-primary ranking) baselines.

**Headline (mean missed_occupied_fraction, lower=better; RL is mean +/- stdev across 3 seeds, Frontier/IG deterministic given a fixed case):** id_test - frontier 0.589, information_gain 0.672, gnn_rl 0.595+/-0.004. ood_model_test - frontier 0.760, information_gain 0.742, gnn_rl 0.760+/-0.004. ood_q_low - frontier 0.744, information_gain 0.775, gnn_rl 0.755+/-0.008. ood_q_high - frontier 0.483, information_gain 0.613, gnn_rl 0.497+/-0.004. Every planner now spends exactly 18/18 effort in every case (the R7 confound - Frontier/IG spending 6/18 while RL spent 18/18 - is gone).

**Honest reading:** GNN+RL is no longer clearly ahead of Frontier in any group once Frontier can use its full budget - it is close to Frontier in id_test/ood_q_low/ood_q_high and tied with it in ood_model_test. Information Gain is now the worst performer in three of four groups. The R7 finding that looked like a real learned-policy signal, especially in ood_q_high (where RL had looked dramatically better, 0.42-0.55 vs. Frontier/IG ~0.65), was substantially a resource-utilization confound exactly as the R8 review's BLOCKER 2 predicted - correcting it closes almost the entire gap. RL's own seed-to-seed variance also shrank sharply (stdev 0.004-0.008 vs. R7's 0.006-0.066), consistent with there being less room for one lucky/unlucky training run to look dramatically different once the baselines are no longer artificially weak.

**What this does and does not support:** the RL policy is not broken - it spends its budget fully and detects at a comparable rate to Frontier. It does not, at this training scale (300 updates, 3 seeds, this reward/architecture), demonstrate measurable value over the simple Frontier heuristic. Per the review's own framing: "If it does not [add value], the project uses the best planner" - on this evidence, that is currently Frontier, not GNN+RL.

**Real blocker found and disclosed, not hidden:** one training seed (2, first attempt) crashed mid-run with a genuine `SpatialBeliefEngine` edge case (a finite sampled hypothesis ensemble occasionally fails to cover the realized observation, causing a zero-likelihood ValueError - observed once in ~14,000+ training rounds). Mitigated with a disclosed, training-side-only resample-and-retry (never applied to the frozen benchmark); the corrected retrain completed cleanly with zero retries needed. See the 2026-09-13 entry above and `scripts/train_spatial_gnn_policy.py`.

**Owner:** Demu.

## 2026-09-15 — R8 review of the corrected benchmark: two methodological defects found and fixed; retraining under way

**Status:** ACKNOWLEDGED and acted on. The team's review of the 2026-09-13 corrected-benchmark commit found two real defects in the case-materialization code itself (not in the frozen manifest/protocol, which stand). Both confirmed by reading the code directly, not assumed from the review text, before fixing.

**Correction 1 - truth/belief seed leakage:** `r8_manifest_cases.py`'s `manifest_case_to_benchmark_case()` passed the same `spec["world_seed"]` as both the truth seed (`build_real_incident_case`'s `rl_seed`, which `Environment.reset()` uses to draw the actual hidden world) and the belief-ensemble seed (`sample_ecological_hypotheses()`). Traced mechanistically: `sample_ecological_hypotheses`'s `model_index=0, draw_index=0` world_seed is the input seed unmodified, and family A (`A_graph_diffusion`) is `train_models[0]` - so for every truth `family_id == "A_graph_diffusion"` case (a meaningful fraction of the 240-case manifest), hypothesis draw 0 was generated with the identical model/context/seed as the truth draw, making the true hidden world exactly recoverable inside what the belief update is supposed to treat as an uncertain prior. This is a real truth-vs-belief independence violation, not a style nit. Fixed by deriving a `belief_seed` via SHA256(case_id, world_seed) offset into a namespace (>=10,000,000,000) disjoint from the manifest's actual world_seed range (10,000-47,009); `tests/test_r8_manifest_cases.py` checks zero collisions against all 240 real frozen cases, not a synthetic example, plus a structural (not just empirical) disjointness check.

**Correction 2 - edge-distance proxy substitution:** `real_graph_cases.py`'s edge-weight logic preferred `salishseacast_total_route_proxy_km` over `distance_km` whenever the proxy was present, so `Edge.distance` (what `graph_state.py` puts in the GraphState edge-distance feature every planner, including the GNN, reads) silently became a navigable-route-distance proxy instead of the direct distance on most edges - real data check: edge 108-128 has `distance_km`=14.2 vs. route proxy=113.2, an 8x difference. Fixed: `Edge.distance` is always `distance_km`; the route proxy is preserved on `Edge.travel_cost` (not read by any current GraphState feature) as audit/context metadata rather than dropped. `tests/test_real_graph_cases_r8_edge_distance.py` checks this against the real committed CSV.

**Also renamed** the training-only retry mechanism from "zero-likelihood"/"numerical edge case" to "finite-ensemble support failure" throughout `scripts/train_spatial_gnn_policy.py`, per review: the cause is a finite Monte Carlo hypothesis sample not covering the realized observation's support, not a numerical-precision issue, and the old name invited exactly that wrong mental model.

**No changes** to architecture, reward weights, q values, world-model ranges, the case manifest, budget, or the action contract. Full test suite green after both fixes (6 new tests). Because correction 2 changes a GNN input (edge distance is a frozen node/edge feature the encoder consumes), all three RL seeds must be retrained before the frozen benchmark is rerun - correction 1 alone would not have required this (training's own case sampler already draws truth and belief seeds independently from a continuing RNG stream; the leak was specific to the frozen-manifest materialization path used for benchmarking).

**Retraining launched:** three seeds (0/1/2), identical hyperparameters to R7/R8 (hidden_dim=64/num_layers=2, 300 updates x 16 episodes/update), this time with `--decision-log` enabled to support three requested observation-only diagnostics (validation-curve plateau check, reward-term magnitude breakdown, action statistics) without any additional training-code changes. Per instruction: no reward/architecture/hyperparameter changes will be made based on the R8 rerun's results; if either diagnostic trigger condition fires (still materially improving at update 300, or the reward contract's terminal-term-drowned-out watch-item), that will be reported before any further training, not acted on unilaterally. A new paired case-wise comparison (`scripts/r8_paired_comparison.py`, bootstrap 95% CI on the missed-fraction delta, Frontier<->RL and Information Gain<->RL) will accompany the rerun report.

**Owner:** Demu.

## 2026-09-15 — R8 corrected-corrected rerun complete: RL still does not beat Frontier; now backed by a paired bootstrap analysis

**Status:** Closes the review from earlier today. Full write-up:
`docs/R8_BENCHMARK_REPORT.md` (rewritten in place; the pre-fix 2026-09-13
version is preserved in git history), `reports/r8_paired_comparison.md`
(new paired case-wise bootstrap analysis), `reports/r8_training_diagnostics.md`
(the three requested diagnostics). Training and benchmark both completed
cleanly: zero finite-ensemble support failure retries across all three
seeds (down from 1 occurrence in the prior retrain).

**Headline (mean missed_occupied_fraction across 3 seeds; Frontier/IG
deterministic and confirmed identical across all three seed benchmark
CSVs):** id_test - frontier 0.601, information_gain 0.684, gnn_rl
0.610+/-0.014. ood_model_test - frontier 0.750, information_gain 0.749,
gnn_rl 0.762+/-0.017. ood_q_low - frontier 0.739, information_gain 0.778,
gnn_rl 0.752+/-0.015. ood_q_high - frontier 0.502, information_gain 0.571,
gnn_rl 0.507+/-0.008. Close to the pre-fix numbers - both corrections were
narrow (one world-model family's hypothesis independence, one geometry
input), not a protocol or algorithm change, so a similar qualitative result
is the expected outcome of a correct fix, not evidence the fixes didn't
matter.

**New this round - paired case-wise bootstrap (the statistically
load-bearing addition):** Frontier<->RL delta (baseline_missed -
rl_missed, positive = RL better), pooled across 3 seeds with a
case-clustered bootstrap: ALL groups -0.0095, 95% CI [-0.0202, +0.0010] -
not distinguishable from 0. No group or seed shows RL statistically ahead
of Frontier; one seed (2) shows RL statistically *behind* Frontier overall
(CI [-0.0416,-0.0054]) and one seed (0) statistically behind on
ood_model_test specifically. Information Gain<->RL delta: ALL groups
+0.0499, 95% CI [+0.0305,+0.0698] - RL statistically beats Information
Gain overall, and on id_test specifically in all 3 individual seeds
(smallest per-seed CI lower bound +0.0292). The other three groups are not
statistically distinguishable for the IG comparison.

**Diagnostics (observation-only, no changes made because of them):** no
trigger fired. All three seeds' validation curves were flat/plateaued by
update 300 (tail slope +0.0006 to +0.0017 missed_fraction/update - not
"still improving"). Terminal reward term's mean magnitude is 2.5-2.65x the
summed dense terms per episode across all three seeds - not drowned out,
closing the R7/R8 reward contract's preregistered watch-item with a clean
answer. Effort-level usage and site-revisit rate look like ordinary
exploration, not degenerate collapse.

**Decision, per the review's own pre-declared rule ("if RL doesn't beat
Frontier after this correction, use Frontier"):** RL does not beat
Frontier here - it ties on point estimates and loses on one seed's paired
comparison. **Frontier is the benchmark-supported planner from this
result, not GNN+RL.** RL's real, replicated advantage is specifically over
Information Gain, not over the strongest baseline - worth keeping on
record (GNN+RL is not broken, trains to a genuine plateau, isn't reward-
starved) but not sufficient to recommend it as the project's planner on
this evidence.

**Owner:** Demu.


## 2026-09-19 — Hackathon judging pivot: interactive human-in-the-loop mission control

**Status:** FROZEN for the judging/demo branch `ui/interactive-judge-mode`. This does not change the ecological kernel, benchmark claims, R8/R10 interpretation, hidden-truth boundary, or evidence semantics.

Hackathon feedback changed the product presentation priority from validation-led to transformation-led. Marine is still introduced as ecological rapid response, but the judge now acts as the response coordinator instead of passively watching the planner.

**Judging loop now frozen:**

`inspect current belief -> inspect Marine recommendation -> choose any site -> choose effort {1,3,6} -> deploy -> receive field result -> explicit Bayes update -> inspect propagated belief changes -> inspect reordered plausible extents -> see next Marine recommendation -> repeat -> reveal hidden extent -> compare outcomes`.

**Human-in-the-loop boundary:** Marine is advisory. The judge may follow Marine's top recommendation or override it. The browser contains no ecological inference or hidden truth. User selections are converted into a normal `MissionAction`; `Environment -> SpatialBeliefEngine -> GraphState -> planner` remains the evidence/decision path.

**Possible-world UI:** show top unique ecological occupancy extents after marginalizing q. A world card is an explainability view only; clicking one does not assert hidden truth and does not invoke a new world-conditioned planner. Family provenance may be multi-family because identical occupancy maps are deduplicated.

**Map semantics:** discrete monitoring sites only. No interpolated continuous probability heatmap is claimed. Primary layer is posterior occupancy belief; habitat and observed survey effort are contextual layers. Temperature remains stretch/out of the MVP unless coverage/provenance are made explicit.

**Effort semantics:** the operator may allocate 1, 3, or 6 effort units. Marine does not claim an optimized LOW/MEDIUM/HIGH effort recommendation under Frontier. The UI may show posterior-predictive detection probability at each allowed effort because that quantity is explicitly computed by `SpatialBeliefEngine`.

**Model-stress semantics:** the R9 posterior-predictive surprise diagnostic may be displayed after a field return. It diagnoses how expected the result was under the current ensemble. It does not automatically expand or repair the scenario ensemble.

**Comparator semantics:** the hero comparison is `Marine adaptive Frontier vs Static Response vs You`. Static Response precommits the same Frontier ranking at t0 and does not use subsequent evidence to change its remaining targets. It is not labelled "expert", "professional", "WDFW", or "standard of care". Historical field trajectories remain a separate evidence/replay feature when data support them.

**Outcome receipt:** after reveal, compare the number/fraction of true occupied sites that were actually confirmed/detected, including the known initial detection. Do not use "occupied sites surveyed" as a success metric because imperfect detection means survey is not confirmation. X-axis may be cumulative field effort; hidden truth stays locked until completion.

**Reproducibility:** a committed 100-case demo manifest freezes public demo seeds. Cases are playable synthetic hidden incidents on the real monitoring graph; they are not represented as 100 historical outbreaks. Field outcomes use deterministic simulator-only potential outcomes so the same case is reproducible.

**Implementation priority:** interactive site+effort deployment -> deterministic cases -> top candidate ranking/why-here -> top unique extents -> propagated belief animation -> comparator receipt -> explainability drawers -> polish. World-conditioned mission ranking and temperature are not MVP blockers.

**Owner:** team. Demu may work on the UI in parallel; cross-cutting interface/claim changes require team review.


## 2026-09-19 — Judge-mode hero-case selection rubric frozen before scanning results

**Status:** FROZEN for illustrative demo-case selection only. This is not a new benchmark and must not replace or modify R8/R10 claims.

The first technical smoke case (`incident_001`) was intentionally not selected for storytelling and happened to be uninformative: Marine, Static Response and the follow-Marine judge path all used the same three sites and confirmed only the already-known initial occupied site. Before inspecting the remaining 99 cases, the team freezes the following hero-case selection semantics.

**Positive illustrative candidate gate:**
- Marine confirms more true occupied sites than Static Response on the same frozen incident;
- Marine and Static Response diverge in at least one mission target;
- following Marine produces at least one visible `MISSION UPDATED` event;
- at least one field detection occurs beyond the already-confirmed initial site.

**Positive-candidate ordering, lexicographically:**
1. larger Marine-minus-Static confirmed/detected occupied-site count;
2. more mission-target divergences;
3. more visible mission-updated rounds;
4. more field detections beyond the initial detection;
5. larger maximum propagated absolute occupancy-belief change;
6. more top-world turnovers;
7. deterministic case-id tie-break.

**Causal-legibility fallback gate:** if no positive candidate exists, prioritize incidents with mission divergence plus visible mission update, then larger propagated belief changes and top-world turnover. Such a case may demonstrate adaptive causality without claiming an outcome advantage.

The selector must also report counts of Marine-better / equal / worse cases across the full 100-case UI library so illustrative selection cannot be mistaken for an aggregate performance result.

**Claim rule:** any selected hero incident must be described as an illustrative blinded scenario chosen for causal legibility. Formal statements about whether replanning reliably improves performance remain governed by the frozen R10 paired analysis.


## 2026-09-19 — UI audit correction: old MISSION UPDATED flag was not causal

**Status:** BLOCKER FIXED before selecting the hero case.

The first 100-case UI audit exposed a semantic bug in the judging adapter: `mission_changed` was computed by comparing the just-executed mission with the next mission. Under a no-revisit sequential campaign those are almost always different, so the flag could fire even when the new field evidence did not alter what Marine would otherwise have recommended. The first audit therefore reported `mission_changed_rounds=2` for every three-mission case and must **not** be used to freeze the hero case.

The aggregate outcome counts from that scan (Marine better/equal/worse than Static) and the actual Marine/Static mission sequences remain descriptive outputs of those cases, but any criterion depending on `mission_changed_rounds` must be rerun after this fix.

**Correct semantics now:** after each survey, construct a counterfactual public state that keeps the action itself (effort spent, remaining budget, round, observed effort) while erasing the just-returned ecological result (new detections/status). Hold occupancy belief at the pre-observation posterior. Run the same current Frontier product planner on that counterfactual state. `MISSION UPDATED` is true only when the actual post-evidence recommendation differs from this no-new-evidence counterfactual recommendation.

This makes the UI statement causal and aligned with the project thesis:

`FIELD EVIDENCE -> BELIEF/OBSERVABLE EVIDENCE CHANGED -> NEXT MISSION CHANGED`.

The frozen hero-case rubric from the preceding entry remains unchanged, but the 100-case selector must be rerun on the corrected semantics before a hero case is chosen.


## 2026-09-19 — Judging UI visual direction and hero default implemented

**Status:** CURRENT DEFAULT on `ui/interactive-judge-mode`.

The judging interface has been rebuilt around the visual/product direction reviewed by the team: a bright professional marine operations dashboard with a dominant coastal map, a left mission/choice panel, a right probable-worlds panel, explicit effort controls, inspectable recommendation reasons, live observable incident telemetry, and reveal-only policy comparison.

**Hero case:** `incident_097` is now the default case loaded by the dashboard. It remains an illustrative frozen blinded incident selected under the previously frozen hero-case rubric, not an aggregate performance claim. All 100 cases remain selectable.

**Real-coast map:** the UI now renders the current incident subgraph at the real monitoring-site latitude/longitude coordinates from `real_sites_v0.csv`. Online, the browser requests OpenStreetMap raster tiles only as a cartographic basemap; the ecological graph, site states, beliefs, edges, mission recommendations and evidence are all produced by the local Marine backend. If tiles are unavailable, the SVG graph and coastal-water fallback remain usable. OpenStreetMap attribution is shown in the map.

**Map probability semantics:** site-centered colored halos visualize node-level posterior occupancy belief / habitat context. They are not presented as an interpolated continuous ecological probability surface. The graph remains the decision state.

**Left mission panel:** shows Marine's current Frontier recommendation, the frozen Static Response t0 route, operator site choice, effort `{1,3,6}`, and deploy control. Static Response is still a benchmark design, not a field-professional or agency simulation.

**Why-this-mission panel:** only exposes quantities that the current system actually supports: Frontier status, current occupancy belief, and posterior-predictive detection probability at the operator-selected effort. Habitat is available as map/site context but is not falsely presented as a Frontier ranking cause.

**Right panel:** shows top unique ecological extents marginalized over q plus the current q posterior. World explanations use only explicit extent size, provenance labels and posterior movement; they do not invent ecological narratives or world-conditioned planning.

**Live chart:** before reveal, the chart uses observable quantities only: cumulative confirmed detections and field budget used. Hidden occupied-site coverage is not exposed. After reveal, the evaluator receipt separately compares Marine / Static Response / You on true occupied sites confirmed/detected.

**Owner:** team.
