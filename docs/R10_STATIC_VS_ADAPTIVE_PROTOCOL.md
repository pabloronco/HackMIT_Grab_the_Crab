# R10 — Static Frontier vs Adaptive Frontier

**Status: FROZEN BEFORE RESULTS**

R10 tests the project's central adaptive-response claim more directly than
the R8 planner comparison.

R8 asked:

> Given an adaptive decision loop, which planner performs best?

R10 asks:

> Holding the planner itself fixed, does reacting to new field evidence
> improve the mission relative to deciding all future survey locations at t0?

This is intentionally a comparison of **the same Frontier strategy with and
without replanning**, not another comparison of Frontier, Information Gain,
and GNN+RL.

## 1. Analysis status

R10 reuses the 180 formal R8 benchmark cases:

- 90 `id_test`
- 30 `ood_model_test`
- 30 `ood_q_low`
- 30 `ood_q_high`

The 60 R8 validation cases remain excluded.

Because the R8 formal cases have already been inspected during earlier
development, R10 is a **secondary paired analysis**, not a new untouched
external test.

## 2. What is held constant

Both arms receive the same:

- hidden ecological incident;
- initial public state;
- initial spatial posterior;
- real graph/context;
- FrontierPlanner ranking rule;
- total survey budget;
- effort per mission;
- number of missions;
- simulator `q_true`;
- potential field-observation randomness.

The Frontier ranking remains:

1. frontier nodes first;
2. higher occupancy belief;
3. higher uncertainty;
4. deterministic site-id tie-break.

R10 uses:

- budget = 18;
- 3 missions;
- effort = 6 per mission;
- exactly one site per mission;
- three distinct sites;
- no site revisit.

The no-revisit rule is an **R10 experimental design choice**, not a global
Marine rule and not an ecological fact. It is imposed identically on both
arms to isolate replanning rather than revisit behavior.

## 3. Static Frontier

STATIC receives the initial observable state and posterior at t0.

It precommits all three future survey sites before observing any new field
evidence.

The three sites are selected sequentially from the same t0 GraphState and
same t0 belief. After each selection, only that selected site is marked
infeasible so that the next precommitted site is distinct.

Field observations still occur during execution and the Bayesian posterior
is still updated after every mission.

However, those updates cannot alter the already-committed future missions.

Conceptually:

`t0 belief -> A, B, C locked`

then:

`survey A -> evidence -> Bayes update -> B remains B`

`survey B -> evidence -> Bayes update -> C remains C`

## 4. Adaptive Frontier

ADAPTIVE chooses only the next mission.

After each field observation:

`observation -> Bayes -> updated GraphState -> FrontierPlanner -> next mission`

Already-surveyed sites are marked infeasible so that Adaptive also surveys
three distinct sites.

Conceptually:

`t0 belief -> A`

`survey A -> evidence -> Bayes update -> B'`

`survey B' -> evidence -> Bayes update -> C'`

The first mission should be identical between STATIC and ADAPTIVE because
both begin from the same state and use the same deterministic Frontier rule.

The only intended experimental difference is therefore:

**future missions locked at t0 vs future missions evidence-responsive.**

## 5. Paired field randomness

The existing Environment uses a sequential random-number generator during
`step()`. That is appropriate for ordinary simulation, but it is not ideal
for this paired experiment: once two policies choose different sites, the
same RNG sequence can become attached to different sites.

R10 therefore freezes a policy-independent potential-outcome scheme.

For each `(case_id, site_id)`:

`key = "r10-v1|{case_id}|{site_id}"`

Compute SHA-256 over that UTF-8 key. Interpret the first 8 digest bytes as an
unsigned big-endian integer and divide by `2^64` to obtain deterministic
`u in [0,1)`.

With effort fixed at 6:

`P(detection | occupied) = 1 - (1 - q_true)^6`

and:

`detection = occupied AND u < P(detection | occupied)`

Unoccupied sites cannot generate false positives in the MVP.

Because revisits are forbidden in R10, the round number is deliberately not
part of the random key.

If STATIC and ADAPTIVE survey the same site in the same case, they therefore
receive the same potential field outcome regardless of when that site is
visited.

The planner must never receive hidden occupancy, `q_true`, the paired
uniform value, or future observations.

## 6. Metrics

Primary:

- `missed_occupied_fraction` — lower is better;
- `occupied_site_coverage` — higher is better.

Secondary:

- `detections_found`;
- `final_global_uncertainty`;
- mission-2 divergence;
- mission-3 divergence;
- overall mission divergence rate.

The primary paired delta is:

`delta_missed = static_missed - adaptive_missed`

Therefore:

- positive delta: Adaptive missed less occupied extent;
- zero: same missed fraction;
- negative delta: Adaptive missed more occupied extent.

We also report the percentage/count of cases where Adaptive is better,
equal, or worse than Static.

## 7. Statistical reporting

Use a paired case-level bootstrap:

- 10,000 bootstrap replicates;
- resampling unit = case;
- frozen bootstrap seed = `20260917`;
- 95% confidence interval.

Mission divergence and operational outcome are separate questions:

1. Does new evidence actually change later missions?
2. When missions change, does that improve the operational metric?

Both must be reported.

## 8. Interpretation

If the paired interval is clearly positive, R10 supports the claim that
evidence-responsive replanning improves the tested operational metric under
this benchmark regime.

If the interval includes zero, report that no reliable difference was
detected. Do **not** convert failure to detect a difference into a claim of
equivalence.

If the interval is clearly negative, treat that as a serious diagnostic
result: under this tested regime, evidence-responsive replanning is
degrading the operational metric and must be investigated before making a
strong adaptive-value claim.

R10 does not prove:

- real-world effectiveness;
- ecological optimality;
- superiority to field operators;
- generic OOD robustness;
- benefit outside the tested simulator/benchmark regime.

## 9. Freeze rule

After R10 results are observed, do not change:

- Frontier ranking;
- q assumptions;
- world-model families/ranges;
- budget;
- effort;
- observation-randomness protocol;
- primary metrics;
- bootstrap definition

in order to improve the result.

Only genuine implementation bugs that violate this frozen protocol may be
fixed and rerun, with the defect and correction documented explicitly.
