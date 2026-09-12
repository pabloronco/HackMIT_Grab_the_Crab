# R5 Simulator Criticism — Contract

**Status:** CURRENT DEFAULT implementation contract for rehearsal; not ecological validation.

## Why this exists

The project must not claim `synthetic train -> synthetic test -> real-world effectiveness`.
Real monitoring data therefore constrains and criticizes the synthetic world-model ensemble before planner comparison.

This layer asks a narrow question:

> Do the current simulator families generate obviously degenerate or structurally implausible incident patterns relative to the real monitoring network and observed monitoring patterns?

It does **not** ask whether the simulator recovered the unknown true historical extent.

## Real-data lanes used here

- monitoring coordinates and the frozen R2 primary graph define incident geometry;
- observed effort constrains realistic effort scale;
- observed positive-site patterns provide descriptive spatial reality checks;
- a weak calibration-only habitat proxy is derived from observed site-year detections;
- the latest two observed years are held out from that habitat-proxy construction as a simple reality-check split.

## Important evidence discipline

Observed detections are **not occupancy ground truth**. Detection is imperfect, unsampled occupancy is unknown, and all-year observed positives accumulate over time.

Therefore:

- positive-site count is a descriptive/lower-bound-style anchor, not target prevalence;
- positive-site clustering and spacing are reality-check summaries, not hidden extent labels;
- simulator warnings are prompts for review, not automatic parameter fitting;
- no family is tuned to improve Frontier, Information Gain, or RL performance.

## Topology-only incident context

The diagnostic chooses a real incident subgraph using only:

- initial seed candidate;
- frozen primary adjacency;
- static route/distance metadata.

It chooses the deterministic smallest topology-only seed whose incident component lies in the preferred 12–20 node range. Future detections and synthetic hidden truth are forbidden selection inputs.

This seed is a **diagnostic context only**, not the final demo/historical-replay seed.

## Habitat proxy

Family C needs a scalar `habitat_score`. We do not have a source-grounded universal numeric suitability score.

R5 therefore uses a deliberately weak observation-derived proxy **for simulator criticism only**:

1. aggregate calibration data to site-year;
2. mark whether at least one detection occurred that year;
3. group by Crab Team habitat category;
4. compute a Beta(1,1)-smoothed observed-detection probability.

This is explicitly **not** latent habitat suitability and not a causal habitat effect. It is a temporary calibration proxy. The latest two years are excluded from its construction.

## Structural statistics

For real observed positive sites and each synthetic draw, R5 summarizes:

- occupied/positive-site count;
- fraction of sites occupied/positive;
- number of connected components;
- largest-component fraction;
- fraction with at least one occupied/positive graph neighbor;
- median nearest-neighbor distance;
- mean habitat-proxy score.

For each simulator family the diagnostic reports min / p10 / median / p90 / max across deterministic seeded draws plus the number of unique occupancy patterns.

## Automatic warnings

Only conservative warning rules are automated:

- very low seed variation;
- >90% single-site worlds;
- >90% full-graph worlds;
- simulator p90 extent below the all-year observed-positive-site count;
- observed positive-site cohesion outside the synthetic p10–p90 envelope;
- observed positive-site spacing outside the synthetic p10–p90 envelope.

These are **warnings**, not automatic rejection criteria.

## Current world-family status

- A — graph diffusion: train candidate
- B — spatial cluster: train candidate
- C — habitat-driven: train candidate
- E — fragmented/patchy: OOD-model holdout candidate

Numeric parameter ranges remain OPEN until this diagnostic is reviewed. Family E must remain outside the train/inference family set when it is used as the structural holdout.

## Reproduce

```bash
python scripts/diagnose_r5_simulator_criticism.py
```

The compact diagnostic output is written to:

`reports/milestones/r5_simulator_criticism/simulator_criticism.json`

## Next gate

After reviewing the R5 output:

1. keep or minimally adjust absurd parameter regions;
2. record the decision;
3. freeze planner-independent benchmark seeds/splits/metrics;
4. implement the strong Information Gain baseline;
5. hand the frozen ecological benchmark contract to Demu for serious learned-policy benchmarking.

The claim remains **robustness to simulator assumptions**, not proof of real-world effectiveness.
