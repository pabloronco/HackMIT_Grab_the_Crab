# Interactive Judge Mode UI Specification

**Status:** CURRENT DEFAULT for HackMIT judging branch `ui/interactive-judge-mode`.

## Product sentence

Marine is mission control for ecological first response: it keeps several plausible invasion extents alive, recommends where to survey next, interprets imperfect field evidence, and changes the next mission when the belief changes.

## Judge experience

1. A confirmed first detection is visible on a real monitoring-site graph.
2. Marine shows the current posterior occupancy belief and its top five recommended survey sites.
3. The judge can follow Marine or override the site.
4. The judge chooses field effort from `{1, 3, 6}`.
5. The field result is simulated against hidden truth that remains unavailable to the browser/planner.
6. The spatial Bayesian engine reweights ecological extents and detectability hypotheses.
7. The UI highlights:
   - surveyed-site belief before/after;
   - largest propagated belief changes at unsurveyed sites;
   - reordered top ecological extents;
   - posterior-predictive model stress;
   - the changed next recommendation when applicable.
8. The loop repeats until the field budget is exhausted.
9. The evaluator reveal unlocks hidden truth.
10. A receipt compares Marine, Static Response, and the judge on the same hidden incident and same budget.

## Main-screen information hierarchy

### Always visible
- confirmed first detection;
- incident number;
- remaining field budget;
- round;
- hidden truth lock;
- monitoring graph;
- posterior occupancy belief at nodes;
- top Marine recommendations;
- selected site;
- selected effort;
- latest field result;
- largest propagated changes.

### Explainability / secondary
- top unique ecological extents marginalized over q;
- q posterior and q mean;
- predictive detection probabilities for effort 1/3/6;
- posterior-predictive observation probability and surprise.

### Reveal-only
- true occupied sites;
- hidden world family provenance;
- comparator outcome scores;
- effort-vs-detected-occupied trajectory chart.

## Map semantics

The map is a graph of discrete monitoring sites. Node intensity represents the chosen observable layer:

- **Probability:** posterior marginal P(occupied).
- **Habitat:** site habitat score used as model/context input.
- **Survey History:** cumulative observed effort.

No continuous interpolated probability surface is claimed.

Clicking a possible-world card temporarily shows that occupancy extent as a **what-if hypothesis view**, explicitly labelled as not hidden truth.

## Recommendation semantics

The current product planner is Frontier. Candidate order is:

1. frontier first;
2. higher occupancy belief;
3. higher uncertainty;
4. deterministic site id tie-break.

The browser does not reproduce this rule. The backend exports the exact ranking.

For each candidate, the backend may also show posterior-predictive detection probability for effort 1, 3, and 6. This is descriptive evidence support, not an optimized effort recommendation under Frontier.

## Human override

The system is advisory.

The judge may select any valid delimitation site and effort allowed by budget. The selection is wrapped in a normal `MissionAction` and executed through the same environment/belief/update path.

The UI may show the chosen site's Marine rank. It must not label the judge's choice as wrong.

## Possible worlds

The inference engine is joint over ecological occupancy extent and q.

UI world cards aggregate posterior mass over q for each unique ecological occupancy map. This avoids showing duplicate `(world, q)` rows.

A card may display:
- posterior mass;
- posterior change since the latest evidence packet;
- occupied-site count;
- family provenance labels when available.

Do not implement "optimal mission if this world were true" in the MVP.

## Model stress

After an observation, show:
- P(observed result | pre-update posterior);
- surprise in bits;
- explicit impossible-under-current-ensemble state when probability is zero.

No universal warning threshold is asserted.
No automatic scenario-set expansion is claimed.

## Comparator receipt

### Marine
Adaptive Frontier: re-runs after every field result.

### Static Response
Same initial Frontier logic, three effort-6 missions committed at t0. Later evidence is recorded but cannot change the precommitted targets.

### You
The judge's actual live choices.

The comparator must not be labelled "expert", "professional", "WDFW", or "standard of care".

Primary visible outcome:
- true occupied sites confirmed/detected;
- true occupied sites remaining undetected;
- cumulative effort trajectory.

Hidden truth is available only after reveal.

## Reproducible case library

`configs/ui_case_manifest_v1.json` contains 100 frozen demo case seeds.

These are:
- real monitoring-site graph geometry/context;
- synthetic hidden ecological truth;
- explicit imperfect detection;
- deterministic potential field outcomes;
- independently sampled belief ensemble.

They are **not** 100 historical outbreaks.

## API contract

- `GET /api/state` — observable UI snapshot.
- `GET /api/cases` — frozen case library metadata.
- `POST /api/reset` — reset by `case_id` or ad-hoc seed.
- `POST /api/deploy` — human-in-loop `site_id + effort`, execute field mission, Bayes update, replan.
- `POST /api/reveal` — reveal only after completion.

Legacy `/api/plan` and `/api/execute` remain for rehearsal/backwards compatibility.

## Claim discipline

Safe demo claims:
- "New evidence changes the posterior belief."
- "Non-detection is not absence; effort and detectability matter."
- "Marine can reallocate the next mission after evidence arrives."
- "The judge can override Marine; the engine remains advisory."
- "The hidden extent is synthetic and blinded so counterfactual policies can be scored."
- "Real monitoring sites and real-data-constrained context are used."

Do not claim:
- real-world operational effectiveness;
- superiority to field professionals;
- ecological optimality;
- WDFW endorsement;
- that every demo case is historical;
- that the learned RL policy is the deployed planner.

## Acceptance gate

The judging UI is ready for merge when:
- one case can run end-to-end through custom site/effort choices;
- truth is absent from all pre-reveal snapshots;
- same case/site/effort is reproducible;
- top recommendations and possible extents update after evidence;
- propagated belief changes are visible;
- reveal produces a same-incident comparator receipt;
- existing core tests remain green;
- a live browser rehearsal can be completed reliably in under ~90 seconds.
