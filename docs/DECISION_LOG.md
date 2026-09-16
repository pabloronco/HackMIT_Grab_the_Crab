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
