# M5 mission-control shell notes

**Status:** CURRENT DEFAULT for the rehearsal UI shell; not a frozen product claim.

**2026-09-17 update:** the shell was ported from the M5 toy scenario onto the
current real-graph/spatial-belief backend on `ui/current-backend-port` (not
yet merged - see that branch's checkpoint report for the exact diff, tests,
and open items). The "Product boundary" and "Visual guardrails" sections below
still hold; "Demo scenario" and "M5 acceptance target" describe the
now-superseded toy setup and are kept for history rather than rewritten in
place.

## Stack decision

The rehearsal UI uses a small FastAPI server plus framework-free HTML/CSS/JavaScript. This avoids introducing a Node/npm toolchain on the current development machine while preserving full visual control and avoiding a generic dashboard aesthetic.

Runtime UI dependencies are isolated behind the optional `ui` extra. Core simulator/Bayes/planner tests do not depend on the web stack.

## Product boundary

The browser contains no ecological decision logic. It calls a thin `MissionControlSession` adapter which consumes the real `AdaptiveMissionLoop`.

The visible flow is:

`incident -> plan -> field return -> belief update -> mission updated -> repeat -> reveal`

Missions come from the active Planner, observations come from `Environment.step`, and beliefs come from the Bayesian engine. Hidden truth is absent from UI snapshots until the M4 reveal gate is opened after mission completion.

## Demo scenario

The shell currently uses a deterministic 20-site synthetic coastline graph with 30 effort units and 2 field teams.

These values and the simple uniform non-confirmed occupancy prior are **DESIGN CHOICES** for the rehearsal demo, not ecological facts. The single toy hidden-world generator is still not a validation simulator family.

## Visual guardrails

The shell is designed as ecological mission control rather than a heatmap/dashboard/game:

- graph nodes remain discrete survey sites;
- mission targets are highlighted explicitly;
- belief is shown as a node ring and numeric percentage rather than as a continuous map heatmap;
- latest field evidence shows effort, outcome, and belief before/after;
- budget and round remain visible;
- `MISSION UPDATED` is surfaced when the next allocation changes;
- true occupancy appears only after explicit reveal.

## M5 acceptance target

M5 is ready to merge when the real M4 loop is visible and resettable through the shell, the 20-site graph is legible, incident/budget/mission/evidence are visible, reveal remains gated, core tests stay green, and the UI runs locally without Node/npm.
