# Detectability q evidence memo

**Status:** OPEN numeric range; evidence semantics frozen.

## Project contract

The project observation model remains:

`P(no detection | occupied, e, q) = (1 - q)^e`

where `e` is observable effort and `q` is detection probability per effort unit conditional on occupancy. The project data/validation freeze requires q to be represented as a range/distribution/sensitivity parameter unless repeated real data support estimation, and explicitly forbids presenting one universal `q = 0.xxx` as ecological truth.

## What the current real table can and cannot identify

The canonical green-crab table is one row per site x year x month, with associated total trap-set effort. It is not trap-level and does not contain same-occasion detection replicates. The R3 identifiability audit therefore rejects direct estimation of occupancy-conditional q from observed detection fractions. Monthly repeated visits do not by themselves guarantee occupancy closure, especially under movement, low abundance and removal.

## Source-grounded protocol facts

Washington Sea Grant Crab Team monitoring uses six baited traps per standard monthly event: three cylindrical minnow traps and three Fukui fish traps, alternated at approximately 10 m spacing and typically fished across an overnight high tide. This means the current `trap_sets=6` mode has a real protocol interpretation, but the two trap types target different crab sizes and do not necessarily share one physical catchability parameter.

Sources:
- Washington Sea Grant Crab Team, 2024 Inland Monitor Handbook.
- Crab Team Phase 2 Final Report: three minnow + three Fukui traps per sampling event.
- Grason et al. / Frontiers 2025 monitoring-method description: six baited traps, overnight high-tide soak.

## External evidence on imperfect trapping

### Counihan & Thom 2024

USGS / Management of Biological Invasions, DOI 10.3391/mbi.2024.15.2.02. The study uses Washington/Salish Sea trapping data and species-accumulation/sample-completeness methods. It concludes that high-probability early detection of rare green crab can require substantially more trapping effort than was used in 2020, and that required effort differs by spatial scale.

**Use here:** strong evidence that effort materially changes detectability and that low-effort non-detection is weak evidence.

**Do not use as:** a direct estimate of the project's scalar occupancy-conditional q; the paper does not supply that parameter in our model semantics.

### Ranson 2022 Lummi Sea Pond MSc analysis

A multinomial-Poisson removal model using repeated stationary traps in the Lummi Sea Pond reported per-individual capture/detection probabilities of roughly 0.6-1.3% for shrimp traps and about 0.3% for Fukui traps. The author explicitly notes substantial data cropping, only 16 stationary Fukui traps, unavailable standard errors for several model outputs, and incompatibility of much of the adaptive raw dataset with the model assumptions.

Most importantly, the report interprets the number as the chance that an **individual EGC** at a trapping location is captured during a 24-hour soak. That is not the same quantity as our `q = P(at least one detection in an occupied site per effort unit)`.

**Use here:** evidence that trap type and capture process matter and that catchability can be low.

**Do not use as:** `q=0.003` or `q=0.013` in the occupancy observation model.

### Bergshoeff et al. 2018/2019 Fukui-trap studies

Underwater video of Fukui traps observed 1,226 green-crab entry attempts with only about 16% successful entries. A subsequent trap-modification study found modifications could raise CPUE substantially.

**Use here:** evidence that trap mechanics and design affect capture efficiency.

**Do not use as:** site-level occupancy detection probability; entry-attempt success is conditional on a crab approaching and attempting the trap.

## Current design decision

**OPEN:** final numeric q range/distribution.

**CURRENT DEFAULT for engineering:** keep one explicit effective per-effort q in the observation model, but treat it as an uncertain model parameter and randomize/stress-test it. The current broad values in `q_sensitivity.py` are **design sensitivity probes only**, not a biological prior or calibrated range.

For the Crab Team-derived monitoring representation, one effort unit can be interpreted as one trap deployment / trap-equivalent within the standardized overnight survey protocol. Because the standard event mixes three minnow and three Fukui traps, a single scalar q is an **effective protocol abstraction**, not a physical catchability constant shared by both trap types.

## Required validation behavior

1. Never infer q by dividing detections by trap sets or by treating all real non-detections as occupied trials.
2. Keep 0/1 effort much weaker than 0/10 in Bayes tests.
3. Run simulator/training sensitivity across q scenarios and include OOD q shifts.
4. Consider deliberate q misspecification tests: simulator `q_true` differs from belief-engine `q_assumed`.
5. Do not expose hidden `q_true` to the learned planner if the simulator treats it as latent uncertainty.
6. Do not select a q merely because it creates a visually dramatic replan.

## Next gate

Use the explicit q sensitivity audit to quantify how candidate q assumptions change event-level detection probability and posterior strength. Only after that should the team freeze an MVP q treatment (e.g. scenario/randomized range and whether the belief engine knows the episode q exactly or uses an assumed calibration value).
