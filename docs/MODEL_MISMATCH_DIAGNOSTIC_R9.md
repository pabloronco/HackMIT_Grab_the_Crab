# R9 Posterior-Predictive Model-Mismatch Diagnostic

**Status:** diagnostic accepted for model criticism; no hard threshold and no unknown-world component in the MVP.

## Why this exists

The spatial posterior is joint over ecological occupancy worlds and effective detectability `q`. That is useful, but it creates a real interpretability risk: when the available ecological worlds are misspecified, repeated field observations can sometimes be explained by shifting posterior mass toward a different `q` value rather than by admitting that the world ensemble itself may be inadequate.

R9 therefore adds a pre-update posterior-predictive diagnostic.

For one prospective observation `y` at site `i` with effort `e`:

`P(y | current posterior)`

is computed by marginalizing over the current joint ecological-world/q posterior. The diagnostic reports:

`surprise_bits = -log2 P(y | current posterior)`

before the observation is used to update the belief.

## Semantics

This value is a **model-criticism signal**, not another ecological latent variable and not a planner feature.

A high value means that the realized field result was poorly predicted by the current ensemble. A zero-probability result has infinite surprise and means that the observation is impossible under the current declared ensemble/observation-model assumptions.

This does **not** uniquely identify the reason. Possible explanations include:

- ecological world-model misspecification;
- detectability support that is too narrow;
- a violated observation-model assumption such as the MVP no-false-positive assumption;
- data/protocol problems.

The diagnostic must therefore not be translated automatically into a new posterior component or into a biological conclusion.

## No frozen threshold

R9 deliberately does not introduce a universal threshold such as `surprise > X bits => model failure`. No source-grounded calibration currently supports such a number. The diagnostic remains continuous and inspectable.

Repeated high surprise is stronger evidence of ensemble stress than one isolated moderately surprising observation, but this statement is qualitative until a threshold/calibration study is added.

## q confounding

The diagnostic is specifically meant to be inspected together with the q posterior. Repeated non-detections may shift posterior mass toward lower q and thereby make later non-detections less surprising even when the ecological world support is structurally incomplete.

Therefore:

> a posterior shift toward low q is not, by itself, proof that low detectability is the correct explanation.

## Information firewall

`posterior_predictive_surprise()` receives only the current `SpatialBeliefState` and the realized `Observation`. It never receives `HiddenWorld` and it is not exported into `GraphState`.

## Diagnostic

Run:

```bash
python scripts/diagnose_r9_model_mismatch.py
```

The script checks three controlled cases:

1. effort changes how surprising a non-detection is;
2. a detection at a site absent in every current hypothesis is explicitly reported as impossible under the ensemble;
3. repeated non-detections can shift q downward, illustrating the exact q-vs-world misspecification confounding that motivated this diagnostic.

## Claim discipline

This addition improves transparency about model inadequacy. It does not prove that the ensemble is ecologically complete, and low surprise does not validate the ecological model.
