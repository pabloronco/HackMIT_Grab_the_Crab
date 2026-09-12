# DEMU HANDOFF R7 — Serious RL Integration

**Status:** DEMU HANDOFF NEEDED.

The ecological/probabilistic side is now stable enough for the learned-policy block to restart without chasing moving semantics.

## What is now frozen upstream

1. Real monitoring graph v0 and incident-subgraph extraction.
2. Explicit spatial Bayesian belief engine over ecological-world hypotheses x q hypotheses.
3. Multiple world-model families: A/B/C train, E held out as OOD model family.
4. Real-data simulator criticism; current world-model ranges are frozen for the first formal benchmark as design ranges, not ecological estimates.
5. q evidence semantics and benchmark scenario protocol in `configs/q_protocol_r7.json`.
6. Hidden q_true firewall: q_true is simulator/evaluator-only and is not exposed through GraphState, planner constraints or observation metadata.
7. Strong baseline: `InformationGainPlanner` in `mode='spatial_joint'`.
8. Planner-facing GraphState schema is unchanged. Do not add q or hidden-truth features.
9. Primary evaluation metrics are planner-independent: missed occupied sites, occupied-site coverage, final global uncertainty.

## What changed since your engineering block

The old site-local belief loop is no longer the formal path. The new formal path is:

`world-model ensemble + uncertain q -> SpatialBeliefEngine -> BeliefState projection -> GraphState -> planner -> MissionAction -> hidden simulator -> field observation -> posterior -> replan`

Use `SpatialAdaptiveMissionLoop` / the spatial posterior path for formal training and evaluation. Do not train serious RL on the old toy environment semantics and do not let the policy receive particle/world identities or q_true.

## Cross-team decision needed before serious training

Please review `configs/benchmark_protocol_r7.json` and either ACK the recommended action/reward contract or return one concrete counterproposal. Do not start hyperparameter tuning until this is resolved.

### Recommended action contract

For the 24h implementation, simplify the formal policy action to exactly one `(site, effort)` allocation per decision round:

- effort levels `{1, 3, 6}` trap-equivalent units;
- site revisits allowed across rounds;
- recommended total episode budget `18`;
- recommended max horizon `6` rounds;
- same action feasibility for Information Gain and RL.

Rationale: this preserves the core sequential question “where next and how much effort?” while avoiding combinatorial multi-site actions. Your existing GNN actor/critic remains useful; the action head should score/select site plus discrete effort rather than a fixed effort-per-pick multi-site bundle.

If you believe retaining the existing autoregressive multi-site mission is materially better and still benchmark-fair within the time constraint, respond with the exact alternative action contract and why. Do not silently keep the old fixed-effort action.

### Recommended primary training reward

Use normalized terms so graph-size shifts do not automatically rescale reward:

- per round: `2.0 * reduction_in_mean_occupancy_entropy + 0.5 * new_detections`;
- terminal: `-2.0 * missed_occupied_fraction` using HiddenWorld trainer/evaluator-side only;
- no primary effort-cost term because budget + horizon already impose the resource constraint;
- reward is not an evaluation metric; formal reporting uses the frozen planner-independent metrics.

A single reward-weight ablation is acceptable later if runtime permits. Do not tune simulator ranges to improve the RL result.

## q protocol to consume

Training belief support: `{0.05, 0.10, 0.20}` with equal prior mass.

Training simulator q_true: sampled from the same three scenario values, but hidden from the planner.

OOD q mismatch tests:

- `q_true=0.02`, belief support unchanged;
- `q_true=0.35`, belief support unchanged.

These are benchmark scenario anchors, not biological estimates.

## Required learned-policy benchmark

Minimum planners on exactly the same formal cases:

1. Frontier heuristic.
2. Spatial-joint Information Gain.
3. GNN + RL.

The learned policy earns its narrative only if it adds measurable value against Information Gain on at least some meaningful primary metric / OOD condition without collapsing elsewhere. RL > Frontier alone is not sufficient.

Required report:

- ID test mean + dispersion;
- OOD held-out model family E;
- OOD q-low and q-high mismatch;
- visible failure cases;
- runtime / seed stability note;
- no winner-only score and no claim of real-world optimality.

## Engineering integration requirements

- Sync once from `r4/spatial-belief`; do not manually copy isolated files.
- Preserve deterministic checkpointing and decision logging.
- Log action eligibility/mask, chosen site, chosen effort, logits/probabilities, value, entropy and reward components.
- Keep `GraphState -> MissionAction` as the policy boundary.
- HiddenWorld must remain absent from tensor adapter / actor / critic inputs.
- Add tests that q_true cannot leak into policy-facing data.
- Benchmark runner must consume externally frozen cases/config, not generate its own toy incident distribution.

## Deliverable back to Pablo/Fede

Return:

1. ACK or exact counterproposal for action contract.
2. ACK or exact counterproposal for reward contract.
3. branch/commit after integration sync.
4. one smoke-training result proving the new spatial environment runs end-to-end.
5. only after smoke passes: serious training + Frontier/InfoGain/RL benchmark tables.

Do not spend time on architecture novelty unless the current GNN fails mechanically. Reliability and fair comparison are higher priority than a more sophisticated network.
