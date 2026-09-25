# UI effort-policy audit

**Status:** product behavior / sanity audit, not a scientific performance benchmark. This does not touch or supersede the R8/R11 formal benchmark protocols or their frozen results.

**Purpose:** verify that the live recommendation engine's effort choice {1, 3, 6} is actually reachable and used over time, not just listed as an available action, after the horizon fix and the PROBE -> DELIMIT -> CONFIRM policy change.

## Before (old occupancy-aware-retention rule)

Ran the recommendation engine on all 100 frozen UI incidents (`incident_001`-`incident_100`) to budget exhaustion.

| Effort | Count | % |
|---|---|---|
| e1 | 0 | 0.0% |
| e3 | 0 | 0.0% |
| e6 | 300 | 100.0% |

- 100/100 incidents used exactly 1 effort level (e6).
- Every campaign completed in exactly 3 missions.
- Root cause: the rule required a fixed detection-power retention fraction of max effort (55%/68%/82% by band), but e1's actual retention is only ~19-27% across the belief-q support, so it never cleared the bar and always fell through to the max-effort default.

## After (PROBE -> DELIMIT -> CONFIRM)

Same 100 incidents, same methodology, after replacing the single shared retention rule with a distinct rule per occupancy band (see `MissionControlFrontierPlanner.recommend_effort` in `mission_control.py`).

| Effort | Count | % |
|---|---|---|
| e1 | 117 | 23.8% |
| e3 | 189 | 38.4% |
| e6 | 186 | 37.8% |

Total missions: 492 (up from 300, since campaigns now run until the budget is exhausted rather than a variable number of high-effort rounds).

- Incidents using only 1 effort level: 44/100
- Incidents using >=2 effort levels: 56/100
- Incidents using all 3 effort levels: 8/100
- Guardrail (no single effort > ~90% of actions): max share is 38.4% (e3) — passes comfortably.

### Campaign-length distribution

| Missions | Incidents |
|---|---|
| 3 | 40 |
| 4 | 20 |
| 5 | 17 |
| 6 | 5 |
| 7 | 7 |
| 8 | 2 |
| 10 | 2 |
| 12 | 3 |
| 14 | 4 |

- Mean missions/incident: 4.92
- Mean effort/mission: 3.66
- Mean distinct sites surveyed: 4.92 (equal to mean missions - no revisits, see below)

### Unique effort trajectories (sample)

```
incident_001: e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1 -> e1
incident_004: e6 -> e3 -> e3 -> e1 -> e1 -> e1 -> e3
incident_005: e6 -> e3 -> e3 -> e3 -> e1 -> e1 -> e1
incident_028: e6 -> e3 -> e3 -> e3 -> e3
incident_055: e3 -> e6 -> e3 -> e3 -> e3
incident_079: e6 -> e6 -> e3 -> e1 -> e1 -> e1
```

### Repeated-site check

- Cases with any revisit: **0 / 100**
- Maximum visits to a single site in any campaign: **1**

No revisit-related risk exists anywhere in the 100-case frozen library; nothing needed to be redesigned in the simulator/potential-outcome RNG for this freeze.

### Chosen default demo case

**incident_079**: `e6 -> e6 -> e3 -> e1 -> e1 -> e1` (6 missions, all 3 effort levels, no revisits).

- Mission 3 (site 383, e3, belief 0.514, band "delimitation") produces a non-detection that visibly changes the next recommended site/effort (`mission_changed = True`), i.e. a real evidence-driven replan within the first half of the campaign.
- Selected purely for clarity of sequential adaptation, effort diversity, and presentation length (per the anti-cherry-picking rule) - not for its outcome relative to Static.

## Claim-consistency check against the README's "100-case interactive evaluation" numbers

The README describes a 100-case interactive evaluation result of 41.9% (Adaptive) vs 40.0% (Static) detected, +4.67% relative, 88% match-or-exceed. Re-measured on the same 100 incidents against the **current** live product policy (PROBE -> DELIMIT -> CONFIRM):

| Metric | Old README figure | Newly measured (current live policy) |
|---|---|---|
| Mean detected fraction, Adaptive | 41.9% | 40.5% |
| Mean detected fraction, Static | 40.0% | 40.2% |
| Relative improvement | +4.67% | +0.75% |
| Match-or-exceed rate | 88% | 90% |

The old figures do not reproduce under the current live effort policy. This is expected: the policy that produced 41.9%/40.0% always spent effort 6 (see the "Before" table above), which is a materially different decision rule than the one now live. This audit does not resolve which number the README should carry - that is a product-copy decision, not a planner-tuning one - and no planner or threshold was changed based on this comparison.

## Note

This is a product behavior/sanity audit, not a scientific performance benchmark. It does not modify, and should not be confused with, the frozen R8 (Frontier/Information-Gain/GNN+RL) or R11 (Dynamic Delimitation) formal benchmark results.
