# Interactive Judge Mode UI Specification

**Status:** CURRENT DEFAULT for HackMIT judging branch `ui/ramp-final-pass`.

## Product sentence

Marine is ecological first-response mission control: it maintains explicit probabilistic belief over a hidden invasive-species extent, recommends **where to survey and how much effort to spend**, interprets imperfect field evidence, and replans when that evidence changes what remains plausible.

## Frozen judging experience

1. A confirmed first detection appears on the real monitoring-site graph.
2. The hidden ecological extent stays locked.
3. Marine shows the current posterior occupancy belief and a next-site recommendation.
4. Marine also recommends effort from `{1, 3, 6}`.
5. The judge may follow Marine or override site and/or effort.
6. The field result is generated against the same blinded synthetic incident.
7. Explicit spatial Bayes updates occupancy-world weights and q uncertainty.
8. The UI shows local belief change, propagated changes, probable-world movement, q/detectability diagnostics, model stress and any evidence-caused mission update.
9. The judging response window ends after **three field deployments**. Unspent effort remains preserved capacity.
10. Reveal unlocks hidden truth and a receipt comparing Marine, Static Response and the interactive path.

The three-deployment window is a **demo/product horizon**. Formal R8/R10 protocols and conclusions remain unchanged.

## Main information hierarchy

### Hero layer — always visible
- confirmed first detection;
- current incident and deployment count;
- remaining field capacity;
- real monitoring-coast map;
- Marine next site;
- Marine recommended effort;
- judge site/effort controls;
- field return;
- `MISSION UPDATED BY EVIDENCE` when causal replan occurs;
- resource-efficiency capacity-preservation card.

### Explainability layer
- top unique ecological extents marginalized over q;
- q posterior and q mean;
- occupancy belief;
- conditional detection power if occupied;
- posterior-predictive probability of detection;
- effort options and expected information gain;
- decision-quality receipt;
- posterior-predictive model stress.

### Reveal layer
- true occupied sites;
- occupied-and-detected / occupied-but-missed mission receipts;
- Marine / Static / Your Path detection outcomes;
- actual effort spent and capacity preserved;
- effort-vs-detected-occupied trajectory.

## Map semantics

The map uses real monitoring-site coordinates from the frozen real graph.

Available layers are:
- posterior occupancy belief;
- posterior occupancy uncertainty;
- habitat suitability proxy;
- historical direct logger temperature when available;
- exposure;
- eelgrass;
- salt marsh;
- field effort in the current incident.

Environmental values are site context only unless the ecological world model explicitly uses them. Missing data remain missing.

**Temperature:** direct logger summaries only. No interpolation or imputation is allowed. If an incident subgraph has no direct temperature records, the layer is disabled.

Site-centered colored halos are visual emphasis around discrete monitoring sites. They are **not** a continuous interpolated probability surface.

The map supports zoom and pan so closely spaced sites can be separated.

## Nearby low-belief sites

Distance from the confirmed detection is not itself occupancy probability. Site belief is a posterior marginal over the whole ecological hypothesis ensemble conditioned on all current evidence.

Therefore a nearby site can legitimately have lower belief than a farther site. The UI must explain this instead of silently implying a distance-decay rule.

A low-belief frontier site may still be operationally valuable because observing it can distinguish competing plausible invasion extents.

## Site recommendation semantics

Current product site ordering is Frontier:

1. frontier first;
2. higher occupancy belief;
3. higher uncertainty;
4. deterministic site-id tie break.

The browser never reimplements this rule; it renders backend diagnostics.

Static Response uses the same initial Frontier logic but precommits three effort-6 targets at t0. It is a benchmark design, **not** an expert/professional/WDFW simulation.

## Effort recommendation semantics

For a selected candidate site and each feasible `e ∈ {1,3,6}`, Marine calculates:

- `P(detection at site with effort e | current posterior)`;
- `P(detection | occupied, effort e, posterior over q)`;
- expected reduction in marginal occupancy entropy.

Marine then chooses the smallest effort retaining enough of the maximum-effort information value and conditional detection power. Detection-power retention is stricter at higher occupancy belief.

This is a transparent resource-efficiency product rule. Its thresholds are **DESIGN CHOICES**, not ecological constants and not agency preferences.

The judge may override effort.

## q / detectability semantics

`q = P(detection in one check | occupied)`.

q is:
- not occupancy probability;
- not simulator truth exposed to the planner;
- not directly set by the judge;
- inferred jointly with ecological extent from field evidence.

Operationally, the judge controls effort. At effort `e`, the occupied-site detection probability under a fixed q is:

`1 - (1 - q)^e`.

### Boundary pressure vs model stress

If posterior mass piles up at the minimum or maximum q in the tested support, the UI calls this **boundary pressure**.

Boundary pressure may indicate:
- tested q support is narrow;
- occupancy and detectability remain confounded.

It does **not** automatically mean model stress is high.

Model stress is separate: it is the posterior-predictive probability/surprise of the actual observed field result.

## Predicted detection vs detection power

**Detection power if occupied** answers:

> If this site truly contains the invasive species, how likely are we to detect it with this effort?

It marginalizes q uncertainty but conditions on occupancy.

**Predicted detection** answers:

> Before knowing whether the site is occupied, how likely are we to get a detection here with this effort?

It combines:
- occupancy uncertainty;
- detectability uncertainty;
- effort.

This distinction must remain visible in the UI.

## resource-efficiency resource semantics

Maximum available response capacity is 18 effort units.

The judging response window contains up to three field deployments. Marine may leave capacity unspent.

Primary resource-efficiency quantities:
- effort spent;
- capacity preserved;
- effort avoided relative to effort 6 on the same number of deployments;
- detections achieved at that expenditure.

The UI may optionally translate saved effort into local minutes or dollars only from values explicitly entered by the operator. Such conversion values are display-only and never affect inference/planning.

Do not present a generic dollar-savings claim without a user-supplied conversion.

## Good decision vs lucky outcome

Detection is stochastic.

A good ex-ante mission can fail to detect a population that is truly present. Therefore the UI keeps a pre-outcome decision receipt.

After reveal, each surveyed Marine site is classified as:
- `occupied_and_detected`;
- `occupied_but_missed`;
- `surveyed_not_occupied`.

For an occupied-but-missed mission, show the conditional miss probability at the chosen effort. This demonstrates that the engine can make a defensible decision and still receive an unlucky field realization.

Do not retroactively label a mission as poor solely because it produced a non-detection.

## Marine vs Static vs Your Path

**Marine vs Static Response** is the policy comparison in the demo.

**Your Path** is an interactive stochastic realization. It is deliberately secondary. A judge can sometimes obtain more realized detections than Marine by chance; the UI must not rewrite or hide that result.

When that happens, explain that one stochastic realization is not a policy benchmark and show the pre-outcome decision receipts.

Aggregate/fixed-case evaluation, not one live draw, is required for policy-performance claims.

## Probable worlds

The inference engine is joint over ecological extent and q.

World cards aggregate posterior mass over q for each unique occupancy extent. Cards are bounded/scrollable and must never overflow their panel.

Clicking a world enables a labelled **what-if hypothesis view** on the map. It does not reveal truth and does not invoke a world-conditioned planner.

## Model stress

After a field return show:
- `P(observed result | pre-update posterior)`;
- surprise in bits;
- impossible-under-current-ensemble status if probability is zero.

No arbitrary alert threshold is presented as ecological fact.

## UI layout

Desktop hierarchy:

- first row: Mission / dominant Map / Detectability;
- second row: resource-efficiency receipt / Why This Mission / Probable Worlds;
- third row: two simple resource/evidence charts;
- fourth row: reveal/evaluation;
- final full-width row: Decision Log.

The Decision Log must never sit beneath a taller sticky mission panel.

Design direction: quiet white surfaces, strong spacing, minimal border hierarchy, one primary blue action, limited accent colors, progressive disclosure — closer to a polished native productivity tool than a dense dashboard.

## Reproducible case library

`configs/ui_case_manifest_v1.json` contains 100 frozen demo seeds on the real monitoring graph.

These are synthetic hidden incidents for counterfactual scoring; they are not represented as 100 historical outbreaks.

A separate resource-aware case audit chooses illustrative hero cases under a predeclared rubric. Hero selection must not be presented as aggregate performance.

## API contract

- `GET /api/state` — observable UI snapshot.
- `GET /api/cases` — frozen case metadata.
- `POST /api/reset` — reset by case id or ad-hoc seed.
- `POST /api/deploy` — operator site + effort; execute field mission, Bayes update, replan.
- `POST /api/reveal` — hidden truth only after completion.

Legacy rehearsal endpoints may remain but are not the primary UI path.

## Claim discipline

Safe:
- “Non-detection is not absence; effort and detectability matter.”
- “Grab the Crab recommends both where to survey and how much effort to spend.”
- “Unused effort is preserved field capacity in this three-deployment response window.”
- “New evidence can change the next mission.”
- “This selected case is an illustrative blinded synthetic incident on a real monitoring graph.”

Not safe without separate validation:
- proven real-world cost savings;
- universal superiority to human operators;
- ecological optimality;
- WDFW endorsement;
- learned-policy superiority;
- treating the selected hero incident as aggregate evidence.

## Acceptance gate before merge

- targeted mission-control tests green;
- full pytest green;
- same case/action deterministic;
- hidden truth absent before reveal;
- exactly three judging deployments complete the response window;
- unspent effort remains preserved;
- Grab the Crab site+effort recommendation visible;
- map zoom/pan works in Safari;
- environmental layers do not fabricate missing values;
- probable-world panel does not overflow;
- q boundary pressure and model stress remain distinct;
- reveal shows stochastic miss receipts;
- Decision Log remains fully visible;
- 100-case resource-aware audit completed before freezing final hero case;
- one 90-second browser rehearsal completed successfully.

## Local validation

```bash
git fetch origin
git switch ui/catch-the-crab-polish
git pull origin ui/catch-the-crab-polish
uv pip install -e ".[dev,ui]"

pytest tests/test_frontier_planner.py tests/test_mission_control.py tests/test_spatial_mission_loop.py tests/test_real_incident_source.py -q
pytest -q

python scripts/smoke_interactive_judge_mode.py --case-id incident_097 --effort marine
python scripts/audit_ui_hero_cases.py --csv reports/ui_ramp_case_audit.csv
python -m adaptive_response.web_app
```

Open `http://127.0.0.1:8000` and verify the acceptance gate before merging.
