# R8 Formal Benchmark Case Manifest

**Status:** planner-independent case manifest frozen; action/reward contract still awaits cross-team ACK.

## Why this exists

Formal planner comparison must use the same incidents, seeds, q scenarios and topology context for Frontier, spatial Information Gain and GNN/RL. The case manifest is therefore generated and versioned **before** final planner results.

It does not contain planner outputs, rewards, missions, or latent occupancy vectors.

## Case lanes

The first manifest contains 240 cases:

- validation: 20 fresh cases per train family A/B/C = 60;
- ID test: 30 frozen cases per train family A/B/C = 90;
- OOD model: 30 cases from held-out family E only;
- OOD q-low: 30 cases across A/B/C with hidden simulator q_true=0.02;
- OOD q-high: 30 cases across A/B/C with hidden simulator q_true=0.35.

The normal validation/ID/OOD-model cases cycle over the frozen benchmark q scenarios `{0.05, 0.10, 0.20}`. Belief-side q support remains `{0.05, 0.10, 0.20}` in all lanes.

These q values are benchmark scenarios, not ecological estimates.

## Real topology anchoring

Incident initial-detection seeds are drawn only from the frozen R2 topology audit. The three isolated one-node seeds are excluded from the first formal benchmark because the Project Freeze preferred implementation range is 12–24 nodes and the accepted real graph already supplies 46 topology-only seeds producing 14, 15 or 17 node incident graphs.

Seed choice never inspects future detections or synthetic hidden occupancy.

This is **not** an OOD-topology test: exact OOD topology cases remain OPEN and must be versioned separately.

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

## Action/reward boundary

The manifest intentionally does not freeze budget, horizon, `(site, effort)` action shape or RL reward. Those are the explicit cross-team ACK items in `docs/DEMU_HANDOFF_R7.md`.

Once that ACK is resolved, every planner must consume the same realized cases with the same action feasibility, budget and horizon.

## Reproduce

```bash
python scripts/publish_r8_benchmark_case_manifest.py
```

The versioned receipt is written to:

`reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json`

## Claim guardrail

This manifest makes the synthetic comparison reproducible and prevents test-set drift. It does not make the synthetic worlds real ecological ground truth and does not prove field effectiveness.
