# Economic impact translation — methodology and receipt

**Status:** illustrative projection built from one number we measured ourselves and real, sourced external figures. Not validated or endorsed by WDFW / Washington Sea Grant.

## Step 1 — internally measured: effort-equivalent gap

`effort_equivalent_budget.py` runs `FixedHighEffortPlanner` (the resource baseline) at increasing budgets on the same 100-case demo library used throughout this branch, holding the same hidden worlds fixed, and finds the budget at which its mean detected fraction reaches Grab the Crab's own mean at budget=18 (0.4185, `reports/ramp_v2/demo_population/summary.json`).

| Fixed High Effort budget | Mean detected fraction |
|---|---|
| 18 | 0.3972 |
| 19 | 0.4019 |
| 20 | 0.4041 |
| 21 | 0.4129 |
| 22 | 0.4196 |

Linear interpolation between 21 and 22 for the target 0.4185: **budget ≈ 21.8**. Fixed High Effort needs ~21% more field effort than Grab the Crab's 18-unit budget to reach the same detection outcome.

## Step 2 — real, sourced external figures

- Washington Sea Grant Crab Team monitoring program budget: **$180,000/year** (WSG $25k + WDFW $50k + $105k additional needed). Source: https://wsg.washington.edu/green-crab-and-other-aquatic-invasive-species-monitoring-jeff-adams-washington-sea-grant/
- Washington State green crab management, ongoing operating budget (2023-25 biennium): **$6,082,000/year**. Source: WDFW European Green Crab publications (https://wdfw.wa.gov/species-habitats/invasive/carcinus-maenas) and the 2025-2031 Management Plan report to the legislature (https://app.leg.wa.gov/ReportsToTheLegislature/Home/GetPDF?fileName=European-Green-Crab-2025-2031-Management-Plan-for-Washington_9f014ae3-e42e-46b6-8c16-bd396f9d4854.pdf).
- Crab Team volunteer program structure: **65+ monitoring sites** statewide (current network size), monitored monthly April-September, **2 consecutive days each month** (Day One to set traps, Day Two to retrieve them) — 65 x 6 x 2 = **780 field-days/year** (counting each site-visit-day once, not multiplied by team size). Source: https://wsgcrabteam.uw.edu/programs/monitoring-network/ (primary program page, supersedes an earlier ~54-site figure from a regional volunteer chapter). Team-size/season-months corroborated by https://soundwaterstewards.org/crab-team-spring-volunteering/, https://news.wsu.edu/news/2023/04/21/wsu-extension-helps-train-volunteers-to-find-invasive-european-green-crab/
- Dataset time coverage (`wsg_crabteam_dryad_cama`, `configs/data_sources.yaml`): 2017-2023, 7 years.

## Step 3 — the projection

`0.21 * budget` applied to each sourced figure:

| Base (sourced, real) | Annual | Over the dataset's 7 years |
|---|---|---|
| Crab Team monitoring budget ($180,000/yr) | $37,800 | $264,600 |
| Full state program, current funding scale ($6,082,000/yr) | **$1,277,220** | not applied — that funding level did not exist for all 7 years (escalated from a 2022 emergency proclamation); reported as an at-today's-scale figure only |
| Volunteer field-days (65 sites x 6 visits x 2 days = 780/yr) | 164 field-days | 1,147 field-days |

## What this is and is not

**Is:** an internally-measured resource-efficiency gap (21%, our own simulator, reproducible) applied to real, cited program figures, to illustrate what that gap could represent at real-world scale.

**Is not:** an audited savings estimate, a claim WDFW or Washington Sea Grant has reviewed or endorsed, or a claim that this exact dollar amount was actually saved over 2017-2023 (the 100 demo cases are synthetic incidents on the real monitoring graph, not a re-analysis of actual historical outbreaks).
