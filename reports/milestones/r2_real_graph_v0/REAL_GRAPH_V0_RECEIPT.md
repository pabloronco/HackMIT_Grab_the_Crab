# R2 Real Graph v0 — Milestone Receipt

**Status:** CURRENT DEFAULT PRIMARY ADJACENCY; not ecological ground truth.  
**Purpose:** small, versioned evidence receipt for the accepted R0–R2 graph milestone.

## Reproduce

From a repository checkout with the three source CSV files present under `data/raw/`:

```bash
python scripts/reproduce_r0_r2.py
python scripts/publish_r2_real_graph_receipt.py
```

Raw data and caches remain gitignored. This folder contains only compact receipts.

## Result

- monitoring sites: **49**
- primary edges: **118**
- connected components: **6**
- largest component: **17/49**
- isolated sites: **219, 367, 74**
- candidate seeds naturally producing preferred 12–20 node incident graphs: **see incident_subgraph_audit.json**
- straight-zero-water local edges retained after curved-route audit: **12**

## Decision semantics

- The **20 km direct radius is a candidate-generation DESIGN CHOICE**, not a species dispersal threshold.
- Straight-line water sampling is diagnostic only; the old `straight_water == 0 => reject` rule is **REJECTED**.
- Curved SalishSeaCast route existence is a geometry sanity check; route length/detour are context and sensitivity inputs, not dispersal probabilities.
- Sparse-monitoring review-only edges do not repair connectivity automatically.
- `connectivity_weight` remains **OPEN**.
- Hidden truth and future detections are forbidden inputs to incident-subgraph extraction.

## Source-file SHA-256

- `data/raw/wsg_cama/all_abundance_wide_format.csv`: `dad3eb70e73badd16f5496a562a4b9d0e88c95f929fa306aa868aeb4897be2a0`
- `data/raw/wsg_effort_geo/PAMA.Month.CPUE.csv`: `7212b2806bfd561738dfe379cb051ff54e246b2fc38984a8edbe241f4898852a`
- `data/raw/wsg_effort_geo/Network_PAMA_CPUE.Map.csv`: `7c595f7786d9ed016dd9e08bdcda1880f8246658d60e9444fe99e57d2c53a2d2`

## Receipt-artifact SHA-256

- `real_graph_v0_audit.json`: `54802f5e40c1327d7d509bcd9e9b6aa8699526a1095a664cd573a193a5cf25cb`
- `real_graph_v0_edges.csv`: `995841419de6e9b73840f99b100d68d03dcaaa3d2877cb6cb07110dcde4badac`
- `incident_subgraph_audit.json`: `1f43790502271b2bd3c569c27d09446a4685dc08618a3bfe32a46738a6429d08`
- `real_graph_v0.svg`: `0db3d70494d5fa0f77183772632605f8697a08b35ee28dcd981433c08bfbbbde`

## Provenance notes

Washington Sea Grant / Crab Team monitoring inputs provide observed green-crab outcomes, effort metadata, and monitoring coordinates. ShoreZone and SalishSeaCast are queried by the reproduction scripts as public external context. The receipt does not vendor raw datasets or remote caches into Git.
