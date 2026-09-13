# R7 Detectability Evidence Freeze

**Status:** FROZEN benchmark treatment; ecological numeric q remains OPEN.

## Why this freeze exists

The project needs a stable q treatment before serious RL training, while preserving the distinction between what the evidence supports and what is only an engineering scenario choice.

The observation model remains:

`P(no detection | occupied, e, q) = (1 - q)^e`

where `e` is observed effort and `q` is an effective probability of at least one detection per effort unit conditional on occupancy.

## SOURCE-GROUNDED FACTS

1. Washington Sea Grant Crab Team standard monitoring uses six baited traps per event: three Fukui traps and three minnow traps, typically fished across one nighttime high tide. The current trap-set effort variable therefore has a defensible operational interpretation.
2. Counihan & Thom (2024), DOI `10.3391/mbi.2024.15.2.02`, show that early detection of rare European green crab can require substantially greater trapping effort and that required effort depends on spatial scale. This supports the project semantics that greater effort makes a non-detection more informative.
3. Ranson (2022) reported low trap-level/per-individual capture-detection estimates for stationary Lummi Sea Pond traps, with strong caveats and limited compatible data. These values are not the project's site-level occupancy-conditional q.
4. Bergshoeff et al. (2018), DOI `10.7717/peerj.4223`, found that only about 16% of observed Fukui-trap entry attempts succeeded. Entry-attempt success is also not site-level occupancy-conditional q.

## SOURCE-GROUNDED LIMITATION

The current monthly site-level monitoring table does **not** directly identify the project's q. It lacks same-occasion trap-level replicate detection histories under a defensible occupancy-closure assumption. Detection fraction / effort is therefore not a valid q estimator here.

## FROZEN DESIGN CHOICE FOR THE BENCHMARK

To unblock reproducible training and comparison, the benchmark uses an explicit finite q scenario support:

- belief support: `{0.05, 0.10, 0.20}` with approximately equal prior mass;
- simulator training q_true support: `{0.05, 0.10, 0.20}` with approximately equal probability;
- q_true remains latent and is never revealed to the planner;
- the Bayesian belief engine updates posterior mass over q after observations.

These values are **engineering scenario anchors**, not an empirical biological prior and not estimated detectability values.

At the standard six-trap-equivalent effort, they imply:

| q | P(at least one detection | occupied, e=6) |
|---:|---:|
| 0.05 | 0.2649 |
| 0.10 | 0.4686 |
| 0.20 | 0.7379 |

The purpose is to expose the policy to meaningfully different evidence strengths without pretending the current real data identify the parameter.

## FROZEN OOD q MISMATCH TESTS

The belief model retains the training support `{0.05, 0.10, 0.20}` while the simulator uses a q_true outside that support:

- low-detectability shift: `q_true = 0.02`;
- high-detectability shift: `q_true = 0.35`.

This deliberately tests robustness to detectability misspecification rather than giving the belief engine the simulator truth.

## INFORMATION FIREWALL

FROZEN:

- `q_true` is simulator/evaluator-only latent state;
- `q_true` is not a GraphState feature;
- `q_true` is not a planner constraint;
- `q_true` is not emitted in field-observation metadata;
- the planner sees only belief quantities produced by explicit Bayesian inference.

## CLAIM DISCIPLINE

Allowed:

- detection is imperfect;
- trapping effort changes evidence strength;
- the benchmark tests policies across multiple q scenarios and deliberate q misspecification.

Forbidden:

- `q=0.10` is the true detection probability of an occupied green-crab site;
- the benchmark q support is a calibrated biological prior;
- trap-entry success or per-individual catchability can be directly substituted for this q.

## Decision

**FROZEN:** evidence semantics, benchmark scenario support, latent-q firewall, and OOD q mismatch cases.

**OPEN:** an ecologically calibrated numeric site-level q distribution. That requires richer repeated-detection data or a defensible dedicated detection study and is not needed to complete the HackMIT benchmark.
