# R0 — Real Data Audit v0

**Date:** 2026-09-11  
**Status:** WORKING AUDIT — source inventory and integration plan, not yet a model freeze.  
**Governing rule:** `docs/DATA_FIRST_VALIDATION_FREEZE.md`.

## Goal

Build the ecological and observation layer from real, reproducible sources wherever possible, and reserve synthetic data only for latent occupancy/counterfactuals that historical monitoring cannot reveal.

The target flow remains:

`real monitoring + real geography + real environment -> ecological/observation model -> spatial belief -> GraphState -> planner`

The learned planner must not absorb hidden ecological assumptions that belong upstream.

## Priority source stack

### A. Washington Sea Grant Crab Team — Dryad 2025 community dataset

**Source:** Rubinoff et al. 2025, Dryad DOI `10.5061/dryad.0rxwdbsdt`.

**Why it matters:** this is the strongest immediately downloadable target-species dataset found so far. It is a published subset of the Crab Team network, designed originally for European green crab early detection, with repeated monthly site sampling.

**Coverage / variables documented by the dataset:**

- Salish Sea monitoring sites;
- 2017–2023 repeated monthly observations;
- 49 sites used in the associated paper's repeated-site analysis;
- `Year`, `SiteID`, `Month`, `habtype`;
- `CAMA` = European green crab count per site-month sample;
- counts of other trapped taxa;
- `DailyMaxTemperature.csv` with in-situ water temperature for a subset of sites/years, logged within ~10 m of trap locations.

**Primary roles:**

1. real detection/non-detection target (`CAMA > 0`);
2. real count outcome (`CAMA`);
3. habitat class (`habtype`);
4. repeated temporal structure by site/month/year;
5. real held-out predictive validation;
6. calibration / criticism of seasonal and zero-heavy behaviour.

**Important limitation:** the abundance file does not itself expose exact trap count per row. The associated study describes six traps per monthly sample, but we should not blindly hard-code six if exact effort can be recovered from the companion Crab Team dataset below.

### B. Washington Sea Grant Crab Team — Dryad 2025 network-effort/geography companion

**Source:** Grason et al. 2025, Dryad DOI `10.5061/dryad.sqv9s4ndp`.

Although the publication studies *Palaemon macrodactylus*, several files describe the same Crab Team sampling events/network and therefore provide reusable survey metadata. We must use only the survey metadata for the green-crab project; PAMA presence is not green-crab evidence.

**Useful files / variables:**

`PAMA.Month.CPUE.csv`

- `SiteID`;
- month;
- year;
- `trap.sets` = total traps set for that site-month effort;
- PAMA count / PAMA CPUE (not a green-crab target).

`Network_PAMA_CPUE.Map.csv`

- `SiteID`;
- year;
- `LatitudeDD`, `LongitudeDD` (WGS84);
- trap sets and annual network totals.

`WDFW.PAMA_2021_-2022.csv`

- per-trap deployment/retrieval/check dates;
- effort type;
- management area / waterbody / site id / site name;
- distance to creek mouth;
- trap effort, trap type, trap number, bait;
- latitude / longitude;
- notes and data source.

**Primary roles:**

1. recover exact `trap.sets` for Crab Team site-month samples where keys overlap with dataset A;
2. recover site coordinates for the real graph where SiteID overlap is valid;
3. understand real field-protocol structure and candidate observation-process covariates.

**Required audit before use:** verify one-to-one / many-to-one join coverage on `(SiteID, Year, Month)` and verify coordinate consistency through time. Never infer missing green-crab outcomes from PAMA columns.

## Proposed canonical real table (first target)

After verified joins, build a `site_visit` table with as many real fields as the sources support:

- `site_id`;
- `year`;
- `month` / season;
- latitude / longitude;
- habitat type;
- number of traps set;
- European green crab count (`CAMA`);
- binary detection (`CAMA > 0`);
- CAMA per 100 traps (derived only when effort is known);
- site-level temperature summaries when directly observed;
- source/provenance flags;
- missingness flags.

No field should be silently imputed in v0.

### C. Washington Sea Grant coastal sentinel monitoring reports (outer coast)

**Sources:** standardized coastal monitoring summaries / reports, including 2021–2025.

Recent reports expose named/numbered sites such as Ocean Shores, Brady's Oyster, Grays Harbor NWR, Stackpole, Nahcotta/Paul's Slough, Tokeland, Cutthroat Creek, Dohman Creek, Newskah, and Wa'atch; annual summaries provide traps set, site capture totals, seasonal patterns and CPUE-style summaries.

**Why it matters:** the Salish Sea network is intentionally an early-detection network and is extremely zero-heavy for CAMA. Coastal sites provide a second real ecological regime with much higher abundance and repeated standardized monitoring.

**Primary roles:**

- external calibration / simulator criticism for a higher-prevalence regime;
- spatial/seasonal abundance sanity checks;
- independent reality check on CPUE distributions;
- candidate geographic domain-shift test.

**Current limitation:** the public annual reports are more aggregated than the Dryad site-month tables. Before promoting them into the canonical training/calibration table, attempt to retrieve raw underlying Crab Team Tableau records or another downloadable source. If raw records are unavailable, keep these reports as independent validation/criticism evidence rather than fabricate rows.

### D. WDFW European Green Crab quarterly / annual response reports

**Source:** Washington Department of Fish & Wildlife European green crab progress reports (2022 onward).

**Observed useful fields in public reports:**

- management area / branch;
- green crab captures/removals;
- trap set days / trap check days;
- trapping event counts;
- new management-area detections;
- effort definitions and operational descriptions.

Example public 2025 reporting includes paired catch and effort totals (e.g. DNR trap-set-days and CAMA captures in Grays Harbor and Willapa Bay).

**Primary roles:**

- independent aggregate simulator criticism;
- check catch-vs-effort scale;
- operational budget realism;
- management-area temporal trends;
- source for historical rapid-response timelines.

**Not suitable by itself for:** site-level occupancy fitting or direct `q_i`, because aggregation hides the within-area sampling process.

### E. 2016 Westcott Bay / Padilla Bay rapid-response data

**Source:** Washington Sea Grant / EPA final report and associated 2016 response documentation.

The final report provides unusually concrete response quantities:

- Westcott Bay: 7 trapping sites, 174 trap sets, ~2-mile radius, 0 additional live CAMA, 1 molt;
- Padilla Bay: 31 trapping sites, 368 trap sets, ~4-mile radius, 3 additional live CAMA.

**Primary role:** historical replay / demo reality check.

This is almost exactly the first-response problem formulation: first detection -> limited rapid survey -> detections/non-detections -> updated delimitation concern.

**Claim discipline:** replay can show operational plausibility; it cannot establish that our counterfactual mission would have beaten the historical response.

### F. Washington DNR ShoreZone Inventory

**Source:** Washington State Department of Natural Resources Nearshore Habitat / ShoreZone GIS services.

Public GIS layers include, among others:

- substrate type;
- exposure class;
- shoreline type / geomorphology;
- eelgrass / seagrass;
- salt marsh;
- kelp and other biological shoreline attributes.

The inventory documents >50 shoreline habitat characteristics statewide.

**Primary roles:**

- spatial join environmental/habitat features to real monitoring sites;
- derive ecological-suitability covariates;
- derive shoreline/pathway structure for a real graph.

**Initial high-value fields:** substrate, wave exposure, eelgrass/seagrass, salt-marsh presence, shoreline type.

Do not ingest all available ShoreZone attributes automatically. Each feature must pass the inclusion rule below.

### G. Washington State Department of Ecology marine water quality

**Source:** WA Ecology Marine Waters long-term monitoring / EIM / yearly netCDF profiles.

Public records from 1999 onward include:

- water temperature;
- salinity;
- conductivity;
- dissolved oxygen;
- turbidity;
- fluorescence;
- pH (coverage varies);
- nutrients (discrete sampling);
- station coordinates and depths.

**Primary roles:**

- candidate temperature/salinity covariates for monitoring sites where spatial/temporal matching is defensible;
- environmental sensitivity / domain-shift analysis.

**Caution:** a distant deep-water station is not automatically representative of a shallow pocket estuary. We must quantify nearest-station distance, temporal alignment and depth mismatch before joining. Temperature from in-situ Crab Team loggers takes priority when available.

### H. Padilla Bay NERR / NERRS CDMO

**Source:** NOAA National Estuarine Research Reserve System Centralized Data Management Office.

Continuous water-quality data are available for NERR stations and can support detailed Padilla Bay historical context (temperature, salinity and other water-quality parameters).

**Primary role:** high-resolution environmental context for a replay/site where direct NERR coverage exists, not automatic network-wide imputation.

### I. USGS Nonindigenous Aquatic Species (NAS) — *Carcinus maenas*

**Source:** USGS NAS collection records, downloadable occurrence database.

The database exposes Washington occurrence records with locality/year and broader North American chronology.

**Primary roles:**

- external occurrence chronology;
- first-detection / spread-history cross-check;
- geographically broader sensitivity context.

**Not suitable as absence data:** occurrence databases are presence-biased and do not encode standardized non-detection effort.

## Environmental/operational parameter acceptance rule

A parameter is included only when all applicable conditions are satisfied:

1. **Ecological or observation role is explicit.** We can explain why it affects suitability, spread, detectability, or operational cost.
2. **Real source exists.** It is reproducibly obtainable for a useful fraction of relevant sites/visits.
3. **Coverage and missingness are measured.** No silent imputation.
4. **Temporal/spatial resolution is compatible enough with the monitoring record.**
5. **It helps.** It improves held-out calibration, simulator realism, robustness, or represents a necessary observation mechanism.

### Priority A — include/audit first

- site coordinates;
- year/month/season;
- trap effort (`trap.sets`; soak/check duration when available);
- protocol / trap type when varying;
- CAMA detection/count;
- habitat morphology/class;
- coastal/hydrological distance and graph connectivity derived from real geography;
- substrate;
- wave exposure;
- eelgrass/saltmarsh;
- water temperature;
- salinity.

### Priority B — evaluate if coverage supports it

- tidal elevation/depth;
- distance to estuary/creek mouth;
- turbidity;
- dissolved oxygen;
- vegetation/shoreline complexity beyond the core categories;
- access/travel constraints.

### Priority C — do not include by default

- pH;
- nutrients;
- fluorescence;
- any additional variable whose coverage is sparse or whose mechanistic role is weak after core covariates are included.

These are not forbidden. They require evidence of added value.

## Detectability (`q`) — special caution

The current `q=0.25` toy value is not a final ecological estimate.

We must distinguish:

- site occupancy / local abundance;
- probability an animal encounters/approaches a trap;
- trap-entry / retention efficiency;
- probability a survey returns >=1 detection given presence and effort.

Published Fukui-trap video work reports low success among observed entry attempts, but this is **not directly equal to our site-level `q`**. It is conditional on crabs already being present/active near the trap and under a different study regime.

Therefore:

- do not set `q` equal to a literature entry-success percentage;
- use repeated standardized Crab Team surveys + known-occupied/high-abundance contexts + protocol literature to constrain plausible detectability;
- if `q` is not identifiable separately from abundance/occupancy, represent it as an explicit distribution/range and perform sensitivity analysis;
- preserve the frozen evidence likelihood `P(no detection | occupied,e) = (1-q)^e` only after defining what one effort unit means for the selected protocol.

## Real-data validation lane

Before final world-model selection, reserve real observations that were not used to tune ecological parameters.

Exact split remains **OPEN** until coverage is audited. Candidate designs:

- temporal holdout;
- geographic/site holdout;
- combined year + region stress holdout.

Target metrics for binary detection prediction:

- Brier score;
- predictive log loss / likelihood;
- calibration diagnostics.

Compare against simple honest baselines (global detection rate, site-history rate where available, habitat-only/non-spatial models as appropriate).

## Synthetic-world construction — allowed only after real-data fit/constraints

Synthetic hidden incidents are allowed because complete occupancy and unobserved action outcomes do not exist in field history.

Final worlds should be generated from multiple data-constrained families, not hand-tuned to produce attractive policy behaviour.

Candidate families to keep under review:

1. habitat-driven spatial occupancy;
2. connectivity / graph-spread process;
3. distance-decay spatial cluster;
4. cluster + satellite population;
5. patchy / fragmented extent.

Each family must document:

- which real data constrain it;
- fitted/allowed parameter ranges;
- what statistics it reproduces;
- where it intentionally differs from other families.

At least one family remains completely held out from RL training.

## Simulator-criticism gates

Reject or revise a world-model family if it badly misses relevant real statistics, including where available:

- zero-detection frequency;
- CAMA detections/counts as a function of effort;
- CPUE distribution;
- seasonality;
- spatial clustering/recurrence;
- habitat/environment association;
- effort distribution.

The simulator is not accepted because an RL policy learns on it.

## Canonical near-term build order

1. Download and checksum the two Dryad datasets.
2. Inspect exact schemas, row counts, missingness and key overlap.
3. Build a reproducible `site_visit` join prototype for CAMA + trap sets + coordinates.
4. Quantify how many site-month rows receive real effort and coordinates.
5. Add direct temperature where available.
6. Spatially join audited ShoreZone attributes.
7. Audit salinity/temperature station matching from WA Ecology/NERR; add only defensible matches.
8. Define train/validation/locked-real-test split after seeing coverage.
9. Fit/compare simple ecological/observation baselines before introducing a complex spatial model.
10. Build >=3 data-constrained world families + >=1 held-out family.
11. Implement spatial Bayesian/particle belief and Information Gain on the same posterior.
12. Only then freeze serious RL action/reward/evaluation and hand the calibrated environment to Demu.

## Current blockers / open questions

- Need actual row-level overlap between Dryad A (`CAMA`) and Dryad B (`trap.sets`, coordinates), not assumed overlap.
- Need a defensible definition of one effort unit for the final observation likelihood.
- Need q identifiability analysis; likely requires uncertainty/range rather than a single fitted constant.
- Need to determine whether raw coastal Crab Team CAMA records can be exported from Tableau or must remain report-level validation.
- Need to measure environmental data coverage before adding salinity/turbidity/etc.
- Need graph-construction rule (coastal distance vs hydrological/pathway connectivity) grounded in real geography.

## Decision boundary with Demu

Checkpoint A backbone remains valid.

Any current training on the toy generator remains engineering smoke testing only. Serious/final RL training remains blocked until this audit has produced the canonical data pack, observation model, real graph, multi-family simulator and frozen benchmark described in `DATA_FIRST_VALIDATION_FREEZE.md`.
