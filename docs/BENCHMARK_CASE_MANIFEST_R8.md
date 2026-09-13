# R8 Formal Benchmark Case Manifest

**Status:** FROZEN planner-independent manifest. Runtime gate passed; action/reward cross-team ACK received. Final planner comparison is still blocked on the R8 real-context + baseline-fairness corrections in `configs/benchmark_protocol_r8.json`.

## Why this exists

Formal planner comparison must use the same incidents, seeds, q scenarios and topology context for Frontier, spatial Information Gain and GNN/RL. The case manifest was generated and frozen **before** the final planner comparison so the test set cannot drift in response to who wins.

It does not contain planner outputs, rewards, missions, or latent occupancy vectors.

## Frozen case lanes

The first formal manifest contains 240 cases:

- validation: 20 fresh cases per train family A/B/C = 60;
- ID test: 30 cases per train family A/B/C = 90;
- OOD model: 30 cases from held-out family E only;
- OOD q-low: 30 cases across A/B/C with hidden simulator q_true=0.02;
- OOD q-high: 30 cases across A/B/C with hidden simulator q_true=0.35.

The 180 ID/OOD cases are the formal reporting set. The 60 validation cases are reserved for training/checkpoint selection and must not be folded into the final test averages.

The runtime gate is accepted: Demu's provisional benchmark measured 148.3 seconds for 120 cases x 3 planners, making this case count comfortably feasible at hackathon scale. Counts may no longer be changed in response to planner performance without a new versioned protocol.

The normal validation/ID/OOD-model cases cycle over the frozen benchmark q scenarios `{0.05, 0.10, 0.20}`. Belief-side q support remains `{0.05, 0.10, 0.20}` in all lanes.

These q values are benchmark scenarios, not ecological estimates.

## Real topology anchoring

Incident initial-detection seeds are drawn only from the frozen R2 topology audit. The three isolated one-node seeds are excluded from this first formal benchmark because the Project Freeze preferred implementation range is 12–24 nodes and the accepted real graph supplies 46 topology-only seeds producing 14, 15 or 17 node incident graphs.

Seed choice never inspects future detections or synthetic hidden occupancy.

This is **not** an OOD-topology test: exact OOD topology cases remain OPEN and must be versioned separately if implemented.

## Seed namespaces

R8 consumes the namespaces already frozen in `configs/benchmark_protocol_r5.json`:

- validation: 10000+;
- ID test: 20000+;
- OOD model: 30000+;
- OOD parameter/q: 40000+.

World seeds are unique within each split. The incident seed is a deterministic function of the world seed and the versioned eligible topology-seed list, so repeated manifest generation is bit-for-bit deterministic.

## Hidden-truth firewall

A manifest case contains:

- case id;
- split;
- world-model family;
- world seed;
- initial-detection / incident-subgraph seed;
- incident graph size;
- simulator q_true scenario;
- belief q support.

It does **not** contain the generated occupancy vector. Hidden truth is materialized only by the simulator/evaluator when a case is run and must never be passed to a planner.

## Action/reward and fairness boundary

The action/reward contract has now been ACKed and is frozen in `configs/benchmark_protocol_r8.json`: one `(site, effort)` decision per round, effort `{1,3,6}`, budget 18, horizon 6, with the agreed normalized RL reward.

The final benchmark is **not yet accepted** because Demu's provisional run exposed two integration issues that must be corrected before interpretation:

1. formal world models must use the versioned real monitoring coordinates and R5 habitat proxy, not graph-layout / neutral-habitat placeholders;
2. the strong baselines must not be structurally trapped at 6 effort while RL spends 18. The formal Frontier/Information-Gain rules are versioned in `configs/benchmark_protocol_r8.json`.

These corrections change evaluation validity, not case identity, so the manifest remains frozen.

## Reproduce

```bash
python scripts/publish_r8_benchmark_case_manifest.py
```

The versioned receipt is written to:

`reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json`

## Claim guardrail

This manifest makes the synthetic comparison reproducible and prevents test-set drift. It does not make the synthetic worlds real ecological ground truth and does not prove field effectiveness.
