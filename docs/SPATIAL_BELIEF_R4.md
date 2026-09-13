# R4 Spatial Belief Kernel

**Status:** CURRENT DEFAULT implementation for the rehearsal; not ecological ground truth.  
**Owner:** Pablo + Fede.  
**North star:** `FIELD EVIDENCE -> BELIEF CHANGED -> MISSION CHANGED`.

## Why R4 exists

The original `BeliefEngine` is intentionally site-local. That was sufficient to verify the core effort-aware Bayes semantics, but it cannot express the final project requirement that evidence at one site can change belief at other sites when an explicit ecological model supports that dependence.

R4 introduces a finite ensemble / particle belief layer without changing the planner-facing `GraphState` schema.

## CURRENT DEFAULT design

The inference state is a posterior over joint hypotheses:

- an **ecological hypothesis** = one plausible binary occupancy pattern across all incident sites;
- a **q hypothesis** = one plausible effective protocol-level per-effort detectability value;
- a **joint hypothesis** = ecological hypothesis × q hypothesis;
- a normalized posterior weight for every joint hypothesis.

Ecological hypotheses are supplied externally. The R4 kernel does **not** decide how invasion worlds are generated. Later world-model families will generate these hypotheses from real-data-constrained assumptions.

q support is also supplied externally. R3 established that the current monthly aggregate monitoring table does not identify a unique per-effort q, so R4 does not hard-code or estimate a universal q.

## Observation likelihood

For site `i`, effort `e`, occupancy indicator `z_i`, and q hypothesis `q`:

- if `z_i = 1`: `P(no detection) = (1-q)^e`;
- if `z_i = 1`: `P(detection) = 1-(1-q)^e`;
- if `z_i = 0`: `P(no detection) = 1` under the MVP no-false-positive assumption;
- if `z_i = 0`: `P(detection) = 0`.

Each observation reweights the complete joint ensemble by this explicit likelihood. No neural network interprets evidence semantics.

## Spatial propagation semantics

Evidence changes another site's belief **only through the structure of the ecological hypothesis ensemble**.

Example: if the ensemble says A and B tend to be jointly occupied, a non-detection at A downweights worlds where both are occupied and therefore can reduce `p(B)`. If A and B are independent in the ensemble, observing A leaves `p(B)` unchanged.

This is intentionally different from arbitrary rules such as `detection at A => add 0.2 to neighboring nodes`.

## q uncertainty

R4 treats q as uncertain rather than silently revealing simulator `q_true` to the belief engine.

For the first MVP kernel:

- q is an **effective protocol-level** hypothesis shared across sites within one incident;
- multiple q values may be represented with prior weights;
- observations update the posterior over q jointly with ecological hypotheses;
- the exact numeric q support/range remains **OPEN** pending source constraints and sensitivity design;
- q-shift / misspecification will later be part of OOD evaluation.

This shared-q abstraction is a **MODEL ASSUMPTION**, not a biological fact. Site/protocol-specific q can be added later only if evidence and 24-hour scope justify it.

## Planner boundary

`SpatialBeliefState.as_belief_state()` projects the joint posterior to:

- site-level occupancy marginal `p_i`;
- site-level Bernoulli entropy;
- observed history.

That existing `BeliefState` is then compatible with the existing `GraphStateExporter`.

Therefore R4 introduces **no GraphState feature/schema change** and no learned planner contract change. Demu does not need a handoff for this kernel step.

## Deliberate non-features

R4 does not yet include:

- world-model generation;
- resampling / particle rejuvenation;
- learned spatial propagation;
- hydrodynamic dispersal probabilities;
- fixed numeric q calibration;
- Information Gain planner integration;
- mission-loop replacement of the site-local engine.

Those come only after this inference kernel passes controlled tests.

## Required tests

The kernel must demonstrate:

1. `0 detections / 10 effort` is stronger evidence than `0 / 1` at the same q support;
2. a positive detection eliminates incompatible absent-world hypotheses under the no-false-positive MVP;
3. evidence at A can change B when the ensemble couples A and B;
4. evidence at A does not arbitrarily change an independent B;
5. q uncertainty is retained and updated rather than assuming the simulator's true q is known;
6. posterior weights remain normalized;
7. planner-facing projection contains posterior marginals/history only, not simulator hidden truth.

## Claim discipline

Passing R4 tests demonstrates that the software can perform transparent spatial Bayesian reweighting over declared hypotheses under imperfect detection. It does **not** demonstrate that the ecological hypothesis ensemble is realistic, calibrated, or field-effective. That requires the next world-model + validation work.
