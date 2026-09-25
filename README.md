# Grab the Crab

**Adaptive first-response mission control for marine invasive species.**  
HackMIT 2026 · Sustainability

> **One detection. Unknown extent. Limited field capacity. Where do we search next?**

Grab the Crab turns imperfect field evidence into the next operational decision. Starting from one confirmed detection, it maintains an explicit probabilistic belief over the hidden extent of an invasion, decides **where to survey next and how much effort to spend there**, ingests the field return, updates the belief, and replans.

```
OBSERVE → INFER → DECIDE → SURVEY → LEARN → REPLAN
```

The core idea is simple:

> **Field evidence changes the belief. The changed belief changes the mission.**

## Why we built it

Part of the inspiration is personal. We come from **Liguria, Italy**, where the coastline around the Cinque Terre is not an abstract sustainability case: the UNESCO-listed Portovenere–Cinque Terre cultural landscape includes a protected marine area, and the Cinque Terre National Park actively monitors coastal biodiversity and the spread of alien species. In 2026, a coastal BioBlitz in Monterosso explicitly included monitoring the distribution of alien species along the Ligurian coast.

That made the problem feel immediate: after an invasive species is found, the hard question is not simply *“is it here?”* It is *“how far has it spread, where should we look next, and how much scarce field capacity should we spend before the next piece of evidence arrives?”*

Our prototype is calibrated on the **Washington Sea Grant Crab Team** network because it provides structured, public monitoring data for European green crab (*Carcinus maenas*) in the Salish Sea. The inspiration is local; the technical evaluation uses a dataset rich enough to test the full sequential decision loop.

Sources: [UNESCO World Heritage Centre — Portovenere, Cinque Terre and the Islands](https://whc.unesco.org/en/list/826), [Cinque Terre National Park — coastal monitoring BioBlitz, 2026](https://www.parconazionale5terre.it/dettaglio.php?id=85731), [Cinque Terre National Park — Marine Strategy and alien species](https://www.parconazionale5terre.it/dettaglio.php?id=32772).

## Sustainability, twice

Grab the Crab is designed around **two layers of sustainability**.

**1. Protect ecosystems before an incursion becomes harder to contain.**  
Invasive alien species are one of the five major direct drivers of biodiversity loss worldwide. They can alter habitats, compete with native species, prey on them, spread pathogens, and disrupt ecosystem services. IPBES/UNEP estimates that invasive alien species contributed to 60% of recorded global extinctions and imposed more than **$423B in annual global economic costs in 2019**.

**2. Make conservation response itself more resource-efficient.**  
Field response consumes people, traps, vessel trips, travel time, fuel and budget. Grab the Crab makes those trade-offs explicit: instead of automatically spending maximum effort everywhere, it chooses a site and an effort level from `{1, 3, 6}` based on current evidence. The goal is to preserve field capacity where extra effort has low marginal value and concentrate it where it matters most.

We measure the second layer primarily in **field-effort units and deployments**. Any translation into real-world dollars, vessel hours or fuel is treated as illustrative until an explicit operational cost model is validated.

Global context: [UNEP / IPBES Invasive Alien Species Assessment](https://www.unep.org/resources/report/invasive-alien-species-report).

## What it does

A confirmed first detection starts the incident. The true invasion extent remains hidden.

Grab the Crab then:

1. builds a probabilistic belief over which monitored sites may be occupied;
2. models imperfect detectability explicitly, so a non-detection is evidence but never automatic proof of absence;
3. recommends the next survey site and an effort level;
4. receives a detection or non-detection from the field simulator;
5. updates the posterior;
6. replans using the remaining budget;
7. reveals the hidden truth only at the end for evaluation.

The judge can follow the recommendation or override it. The system keeps the same hidden incident underneath, so the full trajectory can be compared after reveal.

## Real data, explicit uncertainty

The project uses Washington Sea Grant Crab Team monitoring data from the Salish Sea:

- **49 real monitoring sites** with authoritative coordinates;
- real Crab Team habitat classes;
- a **derived 118-edge connectivity graph built from those 49 real monitoring sites**;
- water-constrained route proxies generated from the SalishSeaCast ocean-model mesh;
- monitoring effort and field-protocol structure used to constrain the simulator and effort semantics.

The graph connectivity is our derived operational representation, **not an official Crab Team graph**.

## How it works

### Bayesian belief engine

`src/adaptive_response/spatial_belief.py` implements a finite-ensemble Bayesian model over ecological extent and detectability. For an occupied site, the probability of seeing no detection after effort `e` is:

```
P(no detection | occupied, effort=e) = (1 - q)^e
```

That means one empty check and six empty checks do not carry the same evidential weight.

The belief ensemble spans multiple ecological world-model families rather than a single synthetic generator: graph diffusion, spatial clustering, habitat-driven extent and fragmented/patchy extent.

### Hidden-truth firewall

The true occupancy state lives in a separate `HiddenWorld`. Planner and UI code cannot access it before reveal. That separation is structural and covered by tests.

### Graph decision layer

Monitoring sites become nodes. The observable graph state can include:

- occupancy belief;
- uncertainty;
- observed effort;
- detections;
- habitat/access context;
- frontier information;
- remaining budget and round.

The neural network does **not** learn what a non-detection means. Bayesian inference determines the evidence semantics first; planners act on the resulting belief state.

### Interchangeable planners

The same mission loop supports multiple planners:

- a transparent frontier heuristic;
- an information-gain planner;
- a resource-aware planner that separates **where** to search from **how much** effort to spend;
- a deterministic non-Bayesian Dynamic Delimitation comparator;
- a PyTorch GNN actor-critic policy that jointly selects `(site, effort)`.

This modularity matters: the project is the decision loop, not a requirement to use the most complicated planner.

## Product

The interactive mission-control UI is served by FastAPI with a hand-written JavaScript/SVG frontend.

It shows:

- real monitoring coordinates on a coastline map;
- posterior occupancy heat;
- probable invasion worlds;
- detectability belief;
- current resource budget;
- site/effort recommendations;
- evidence-driven mission updates;
- decision and outcome receipts;
- hidden truth only after reveal.

The memorable moment is intentionally causal: **a field return changes the posterior, and that changed posterior can move the next mission.**

## Results

We keep the **current live product audit** separate from our earlier frozen benchmark lanes so that changes to the interactive effort policy are not retroactively attributed to older evaluations.

### Current live product

After fixing the product effort policy so that `{1, 3, 6}` are genuinely used over time, we reran the current live planner on the same **100 frozen UI incidents** at the same **18-unit field budget**.

The current adaptive policy detected **40.5%** of true occupied extent versus **40.2%** for the fixed maximum-effort Static response, and matched or exceeded Static in **90 of 100 incidents**.

More importantly for the product behavior we set out to demonstrate, the adaptive policy no longer collapses to maximum effort: across **492 missions**, it used **e1 in 23.8%**, **e3 in 38.4%**, and **e6 in 37.8%** of actions. Campaigns ranged from **3 to 14 missions**, and **56/100 incidents** used at least two different effort levels.

We treat this 100-case result as a **product behavior and sanity audit**, not as proof of real-world effectiveness or as a new formal performance benchmark. No new formal confidence interval is claimed for this current-product audit.

### Earlier frozen benchmark lane

Earlier frozen evaluation artifacts used a different resource-aware planner configuration. On its **100-case evaluation population**, that configuration detected **41.9%** of true occupied extent versus **40.0%** for its fixed maximum-effort comparator — a **4.67% relative difference** — and matched or exceeded the comparator in **88%** of incidents.

Across the broader **pooled 280-case evaluation** associated with that frozen benchmark lane (180 formal cases + 100 evaluation cases), the mean detected-extent advantage was **+1.58 percentage points**, with a paired bootstrap 95% interval of **[+0.08, +2.99] pp**.

On the **180-case formal slice alone**, the effect remained directionally positive but the confidence interval crossed zero. We keep that result visible because the benchmark is meant to be able to prove us wrong.

A separate effort-equivalent analysis on the same frozen benchmark lane found that the fixed maximum-effort comparator required about **21% more simulated field effort** to reach the same mean detected fraction. We use that only as a resource-efficiency illustration, not as measured real-world savings.

The current live-product audit is documented in `reports/ui_effort_policy_audit.md`; reproducible benchmark artifacts live under `configs/`, `reports/` and `scripts/`.

## Validation discipline

Complete ecological ground truth does not exist for unsampled coastline, so hidden occupancy remains synthetic by necessity. We do not treat one simulator as reality.

The evaluation stack uses:

- real-data-constrained incident geometry and habitat context;
- multiple ecological world-model families;
- frozen case manifests;
- validation/test separation;
- independently seeded learned-policy training;
- paired comparisons and bootstrap intervals;
- OOD lanes for world-model and detectability shifts;
- a deterministic Dynamic Delimitation comparator with an explicit information firewall.

The correct interpretation is **robustness to simulator assumptions**, not proof of real-world effectiveness.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"

python -m uvicorn adaptive_response.web_app:app --port 8000
```

Open `http://127.0.0.1:8000`.

Optional extras:

```bash
pip install -e ".[dev,rl]"   # PyTorch GNN/RL stack
pip install -e ".[dev,llm]"  # optional natural-language mission briefings
```

## Test it

```bash
pip install -e ".[dev,ui]"
pytest
```

## Reproduce the benchmarks

```bash
# Effort-aware planner / fixed-effort / GNN-RL on the frozen R8 manifest
python scripts/run_spatial_benchmark_r8.py --rl-checkpoint <checkpoint>.pt --out reports/r8_benchmark.csv

# GNN/RL vs deterministic Dynamic Delimitation
python scripts/run_r11_dynamic_benchmark.py --rl-checkpoint <checkpoint>.pt --out reports/r11_benchmark.csv

# Conditional effort-to-parity analysis
python scripts/r11_effort_to_parity.py --gnn-csv reports/r11_benchmark_seed0.csv ... --out-csv reports/r11_effort_to_parity.csv --out-md reports/r11_effort_to_parity.md

# Retrain the learned policy
python scripts/train_spatial_gnn_policy.py --run-name my_run --seed 0
```

## Data sources

- Washington Sea Grant Crab Team monitoring data, Dryad (2017–2023, *Carcinus maenas*, Salish Sea).
- Washington Sea Grant Crab Team monitoring-program materials.
- Washington Department of Fish & Wildlife European green crab management publications.
- SalishSeaCast ocean-model mesh (UBC), used for water-constrained route proxies.
- Cinque Terre National Park / Marine Protected Area public monitoring materials for project inspiration and Mediterranean context.
- IPBES / UNEP invasive-alien-species assessment for global sustainability context.

## Team

**Pablo Ronco — concept and project direction, Bayesian/statistical backend, and product/UI.**  
Pablo originated and developed the core project direction from the early concept stage, helping steer Grab the Crab away from a generic invasive-species map and toward the post-detection decision problem that became its north star: *where should we search next, how much effort should we spend, and how should the mission change when new evidence arrives?* He also helped keep the simulator, inference engine, planners and product aligned around that decision loop as the project evolved.

On the technical side, Pablo worked on the mathematical and probabilistic backbone of the system. He developed the Bayesian belief layer that represents multiple possible invasion extents together with uncertain detectability, and made field evidence effort-aware through `P(no detection | occupied, effort=e) = (1-q)^e`. This means a weak non-detection and a strong non-detection are mathematically different observations rather than both being treated as “absence.” The backend exposes posterior occupancy probabilities, uncertainty and detectability belief to the planner while keeping the realized hidden world structurally separate, and connects that inference state to the simulator, real monitoring context and planner-facing graph state.
Pablo also led much of the product integration and UI refinement, turning the mathematical state into an interactive mission-control experience where judges can inspect belief, uncertainty, probable worlds and resource use — and visibly watch **field evidence → belief change → mission change**.

**Federico Passarelli— system architecture, backend/modeling, product/UI, and overall project direction.**
Federico worked across the core architecture of Grab the Crab, helping define how the simulator, probabilistic Bayesian belief state, observation model, planner interface and real monitoring context fit together into a single adaptive decision system. His work covered the modeling of possible invasion worlds, uncertainty and detectability, effort-aware evidence, the separation between hidden ground truth and planner-visible state, and the broader logic connecting ecological spread, field observations and sequential mission planning. He also helped shape the ecological and operational framing of the system so that the technical design remained tied to a realistic post-detection response workflow.

He also worked extensively on the product and UI, translating the backend into an interactive mission-control experience built around maps, uncertainty, possible worlds, resource choices, recommended actions and visible replanning after new evidence. Beyond individual components, Federico had a broad project-direction role throughout the hackathon: reviewing and challenging design choices, making cross-cutting decisions on what to build and prioritize, shaping evaluation and baseline strategy, and keeping the backend, UI, demo and pitch aligned around one coherent product. He also led much of the ecological framing, demo narrative and presentation structure used to communicate the system to judges.


**Francesco Demuro, Learning and Evaluation Stack.**

Single-handedly architected the entire learning and evaluation stack: the GNN actor-critic backbone and the joint (site, effort) policy, three independently-seeded training pipelines with full checkpointing for reproducibility, and the frozen R7/R8 benchmark protocol with statistically rigorous paired-bootstrap comparison tooling — including catching and fixing a subtle seed-leakage bug that had silently let the model see the answer, invalidating an entire earlier benchmark run before anyone noticed. Then, in the final push before submission, he designed and shipped the effort-aware RAMP planner end to end: a full calibration audit, a dual-population tuning process that caught and corrected an overfit configuration before it ever shipped, the statistical analysis behind the earlier frozen benchmark lane — including its +4.67% relative detection difference on the 100-case evaluation population and the paired-bootstrap analysis reported for the broader pooled evaluation — and the illustrative economic/time translation of the simulated effort gap — equivalent in scale to about $1.28M and 164 field-days per year under the stated proportional assumptions, grounded in real Washington State budget and monitoring-program data, not measured real-world savings.



---

**Grab the Crab does not try to predict the ocean. It helps decide what question to ask it next.**
