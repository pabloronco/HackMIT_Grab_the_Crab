# R4 Ecological World-Model Families

**Status:** CURRENT DEFAULT implementation contract for rehearsal; numeric parameter ranges remain MODEL ASSUMPTIONS until constrained by simulator criticism.  
**Purpose:** provide multiple structurally different synthetic latent incident generators without confusing simulator success with real-world validity.

## Source-of-truth alignment

Project Freeze 3.0 requires real-data-constrained simulation, multiple world-model families, at least three training families, at least one structurally held-out family, OOD testing, and historical replay. Synthetic hidden truth remains necessary because complete real occupancy and counterfactual survey outcomes are not observable.

The implemented family split is:

| ID | Mechanism | Current use |
| --- | --- | --- |
| A | Graph diffusion from the initial detection through graph adjacency | TRAIN |
| B | Distance-decay spatial cluster around the initial detection | TRAIN |
| C | Habitat-driven occupancy with mild local correlation | TRAIN |
| E | Fragmented/patchy geographic alternative with remote patch anchors | OOD MODEL HOLDOUT |

Family D (cluster + satellite) remains optional. It is not required for the minimum `>=3 train + >=1 holdout` contract and should only be added if it materially improves the robustness experiment within the 24-hour constraint.

## Important semantics

### SOURCE-GROUNDED / FROZEN

- hidden truth is synthetic and simulator/evaluator-only;
- the known initial detection is occupied in every generated incident;
- future observations and hidden truth are never inputs to the generator context;
- the policy receives posterior belief, never the generator family or latent occupancy;
- the holdout family must remain absent from serious RL training;
- planner comparisons must use the same frozen incident cases, budgets and observable information.

### MODEL ASSUMPTIONS / CURRENT DEFAULT

- Family A samples edge-transmission strength, habitat modulation and number of diffusion waves;
- Family B defines cluster radius relative to the incident graph's own geometry rather than treating one absolute distance as ecological truth;
- Family C uses habitat score as a probabilistic driver plus a small adjacency boost near the initial detection;
- Family E creates multiple geographic patches and intentionally does not assume connected graph diffusion;
- current numeric parameter ranges are broad engineering defaults, not empirical green-crab parameter estimates.

### OPEN

- calibrated parameter ranges after simulator criticism against real monitoring statistics;
- exact habitat-score mapping for the real incident graph;
- whether Family D adds enough distinct structure to justify inclusion;
- exact train/validation/test case counts;
- exact reward and primary planner-ranking metrics.

### FORBIDDEN

- tuning world-model parameters to make RL or any baseline win;
- calling a generator parameter a measured real spread rate without evidence;
- using the OOD holdout family during policy optimization and then calling it held out;
- feeding `family_id`, generator parameters or hidden occupancy into GraphState/planner inputs;
- interpreting success on these generators as proof of real-world effectiveness.

## Bridge to spatial belief

`sample_ecological_hypotheses(...)` can sample many worlds from a caller-selected family set, collapse duplicate occupancy maps, and convert empirical frequency into prior hypothesis weight. This supplies the explicit `SpatialBeliefEngine` with structured spatial hypotheses while preserving the existing planner-facing `BeliefState -> GraphState` contract.

For the OOD-model experiment, the serious training/inference ensemble uses A/B/C only. Hidden incidents from E are then used as a structurally unseen evaluation condition. This tests robustness to generator misspecification; it does **not** establish real-world validity.

## Next gate

Before serious RL training:

1. compare generated vs real-data-constrained summary statistics and reject absurd parameter regions;
2. freeze TRAIN / VALIDATION / ID TEST / OOD MODEL / OOD PARAMETER / OOD TOPOLOGY case manifests;
3. freeze planner-independent evaluation metrics;
4. implement the strong Information Gain baseline on the same explicit belief model;
5. only then issue the second ecological handoff to Demu.
