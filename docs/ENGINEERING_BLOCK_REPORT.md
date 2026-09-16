# Engineering Block Report — Checkpointing, Hardening, Benchmark Runner, Decision Logging

**Date:** 2026-09-12
**Status:** Complete. Requested by Pablo + Fede as the work permitted while the second
ecological handoff (real graph, q/observation model, spatial belief, world-model
families, frozen validation splits/metrics) is prepared.
**Scope:** engineering/infrastructure only, per the team's instruction. Nothing here
changes GraphState, MissionAction, node/edge features, action semantics, reward
semantics, or evaluation metrics, and nothing here is an ecological or RL-superiority
claim. See `docs/DECISION_LOG.md`, 2026-09-12 entry, for the pointer and commit list.

## 1. What changed

- **Deterministic, self-describing checkpointing** (`src/adaptive_response/rl/checkpointing.py`).
  `save_policy_checkpoint`/`load_policy_checkpoint` embed the architecture
  (`PolicyArchitectureConfig`) alongside the weights, so a checkpoint reconstructs its
  own policy rather than depending on a sibling config file or hardcoded shape.
  Inference-only loading is supported (optimizer state is optional); loading a foreign
  or corrupt file raises a clear `ValueError` instead of a confusing tensor-shape error.
  `train_gnn_policy.py` was switched to build and save through this module.
- **Hardening tests** (`tests/test_hardening.py`, 18 tests). Variable graph size
  (N = 1, 2, 12, 20, 24, 30, 40, including the same model instance handling several
  sizes back-to-back), a minimal budget=1 edge case, a rejection test for
  budget smaller than one pick's effort cost, and a ~40-episode stochastic stress test
  against the real `Environment` asserting no duplicate picks, no budget overrun, and
  termination within a safety cap. Also widened the structural checks that no module
  under `src/adaptive_response/rl/` can import `HiddenWorld` or accept it as an
  argument, at both the module and function-signature level.
- **Planner-agnostic benchmark runner** (`src/adaptive_response/rl/benchmark.py`,
  `scripts/run_benchmark.py`). Runs any object satisfying the existing `Planner`
  protocol (Frontier, RL, and later Information Gain, unchanged) on the same
  incidents/seeds/budgets and writes a raw fact table (sites, budget, rounds,
  detections, effort spent, occupied-sites total/missed, wall-clock time) with
  **no score, rank, or winner field** — deliberately, so this doesn't freeze the
  ecological ranking metric the team hasn't decided yet.
- **Per-round decision logging.** `RoundDecision` (in `round_policy.py`) now also
  carries `node_ids`/`node_logits`. `reward.py` gained `round_reward_components()`
  (a value-preserving refactor — `round_reward()` is now defined as its sum, pinned by
  a test that they stay equal). A new `JsonlDecisionLogger` writes one `RoundLogRecord`
  per round — logits, entropy, value estimate, log-prob, reward components, chosen
  sites, budget before/after, done flag — to an append-only JSONL file.
  `training_env.run_episode()` takes this logger as an opt-in parameter (default
  `None`, verified not to change existing behavior at all); `train_gnn_policy.py`
  exposes it as `--decision-log`, off by default since it can grow large over a long run.

## 2. Tests passed

**100/100**, full suite, no regressions:
`test_belief`(8), `test_benchmark`(7, new), `test_checkpointing`(7, new),
`test_decision_logging`(4, new), `test_environment`(6), `test_frontier_planner`(9),
`test_gnn_backbone`(11), `test_graph_state`(8), `test_hardening`(18, new),
`test_mission_loop`(8), `test_round_policy_training`(6), `test_simulator_step`(8).

The team-authorized smoke run (`engineering_block_smoke`, 10 updates x 4
episodes/update, all four pieces exercised together including `--decision-log`) ran
with zero crashes, produced loadable periodic/latest/final checkpoints, and wrote
175 well-formed `decisions.jsonl` records with `done`/`terminal_missed_extent`
correctly set only on each episode's true final round. This is a mechanical check
only — no hyperparameter campaign, no learning-quality claim drawn from it.

## 3. Deterministic inference status

**Deterministic and verified end-to-end**, not just assumed:
- `test_save_load_roundtrip_reproduces_identical_deterministic_output`: a policy's
  `act(..., deterministic=True)` output before saving is bit-identical to the same
  call after a save/load round-trip.
- `test_fresh_instance_from_same_checkpoint_matches_across_multiple_loads`: two
  independent loads of the same checkpoint file agree with each other.
- `test_variable_graph_size_after_load`: a loaded policy still handles N = 12/20/24/30
  correctly (checkpointing doesn't silently bake in a training-time graph size).
- Demo-mode deterministic inference is the same `deterministic=True` code path
  `RoundPolicy.act` already exposed — unchanged, and now additionally exercised through
  the checkpoint round-trip above, so a saved checkpoint reloaded for a demo reproduces
  the same picks it made at save time.

## 4. Remaining technical risks

- **Run-to-run learning variance is still unresolved** (carried over from the D/E/F
  ablation, not new): the current on-policy REINFORCE-style estimator has no
  variance reduction beyond return normalization. This is an algorithmic question
  independent of everything in this block and independent of the ecological handoff —
  worth more episodes/update and multi-seed averaging whenever serious training resumes.
- **Ctrl+C / SIGINT checkpoint-on-exit is implemented but not freshly stress-tested**
  in this block (carried over from the prior report; no run in this block was
  manually interrupted).
- **`--decision-log` file growth is uncapped.** For a long serious-training run this
  could produce a large JSONL file; it's off by default for exactly this reason, but
  no rotation/truncation exists yet if someone wants it on for a long run.
- **Benchmark runner has no Information Gain planner to run yet** — architecturally
  ready (anything satisfying `Planner` works, confirmed by a dedicated placeholder
  test), but there is nothing to plug in until Pablo/Fede's q model and spatial belief
  engine exist.

## 5. What requires the second ecological handoff

Everything this block deliberately left alone, per the team's instruction: reward
weights, feature set, action granularity, world-model assumptions, final evaluation/
ranking metrics, and the RL go/no-go decision. Concretely, the benchmark runner and
decision logging built here are ready to receive, but cannot yet be exercised on:
the real graph, the frozen observation/q model, the spatial belief engine, world-model
families (>=3 train, >=1 held-out), and the frozen validation splits — all still
pending from Pablo/Fede. No schema change to `GraphState`/`MissionAction` was needed
or made in this block; if one becomes necessary once the real data pack lands, it will
be flagged to the team before being made, per standing instruction.

**No new hyperparameter campaign was run and no claims are made from the smoke run
above.** Final/serious training remains gated on the second ecological handoff.
