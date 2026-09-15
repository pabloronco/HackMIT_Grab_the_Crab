from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..world_models import (
    FragmentedPatchyWorldModel,
    GraphDiffusionWorldModel,
    HabitatDrivenWorldModel,
    SpatialClusterWorldModel,
    WorldModel,
    WorldModelContext,
    sample_ecological_hypotheses,
)
from ..spatial_belief import QHypothesis
from .real_graph_cases import build_real_incident_case
from .spatial_benchmark import SpatialBenchmarkCase

# Consumes the R8 frozen, planner-independent case manifest
# (docs/BENCHMARK_CASE_MANIFEST_R8.md, reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json)
# rather than generating provisional cases - this is what makes the R8
# benchmark "exact frozen cases," not a re-sample.

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_MANIFEST_PATH = REPO_ROOT / "reports" / "milestones" / "r8_benchmark_case_manifest" / "benchmark_cases.json"

FAMILY_MODELS: dict[str, WorldModel] = {
    "A_graph_diffusion": GraphDiffusionWorldModel(),
    "B_spatial_cluster": SpatialClusterWorldModel(),
    "C_habitat_driven": HabitatDrivenWorldModel(),
    "E_fragmented_patchy": FragmentedPatchyWorldModel(),
}
TRAIN_FAMILY_IDS = ("A_graph_diffusion", "B_spatial_cluster", "C_habitat_driven")

FORMAL_REPORTING_SPLITS = ("id_test", "ood_model_test", "ood_q_low", "ood_q_high")

# R8 correction (review 2026-09-15): the truth seed (spec["world_seed"], used
# for build_real_incident_case's rl_seed and later for the hidden-world draw
# in Environment.reset) must never equal a hypothesis-draw seed inside the
# belief ensemble sample_ecological_hypotheses() computes. Before this fix
# both used spec["world_seed"] directly, and sample_ecological_hypotheses's
# model_index=0/draw_index=0 seed is exactly the input seed unmodified - so
# for every truth family_id == "A_graph_diffusion" case, hypothesis draw 0
# for family A was generated with the identical (model, context, seed) as
# the truth draw, making the true hidden world exactly recoverable inside
# what is supposed to be an uncertain belief-ensemble prior. This is a
# real truth-vs-belief independence violation, not a style choice.
#
# Fix: derive a belief_seed in a disjoint integer namespace, deterministically
# from case_id + world_seed (so it's still fully reproducible), offset far
# enough above manifest world_seed's actual range (10_000-47_009, see
# reports/milestones/r8_benchmark_case_manifest/benchmark_cases.json) that no
# hypothesis draw seed (belief_seed + model_index*100_003 + draw_index, see
# world_models.sample_ecological_hypotheses) can coincide with any truth
# seed. test_r8_manifest_cases.py checks this holds for every case in the
# actual frozen manifest, not just an example.
_BELIEF_SEED_NAMESPACE_OFFSET = 10_000_000_000


def derive_belief_seed(case_id: str, world_seed: int) -> int:
    digest = hashlib.sha256(f"r8-belief-ensemble::{case_id}::{world_seed}".encode("utf-8")).hexdigest()
    return _BELIEF_SEED_NAMESPACE_OFFSET + (int(digest[:16], 16) % 1_000_000_000)


def load_frozen_manifest(path: Path = FROZEN_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_case_to_benchmark_case(
    spec: dict[str, Any],
    *,
    budget: int = 18,
    max_rounds: int = 6,
    incident_max_sites: int = 20,
    incident_preferred_min_sites: int = 12,
    draws_per_model: int = 60,
) -> SpatialBenchmarkCase:
    """Materialize one frozen manifest case spec into a runnable SpatialBenchmarkCase.

    The manifest itself never contains hidden occupancy truth or planner
    results (see benchmark_cases.py docstring) - this function is what turns
    a case_id into an actual, runnable incident + world model + belief
    support, identically for every planner that consumes it.
    """

    incident = build_real_incident_case(
        spec["incident_seed_site_id"],
        budget=budget,
        max_sites=incident_max_sites,
        preferred_min_sites=incident_preferred_min_sites,
        rl_seed=int(spec["world_seed"]),
    ).incident

    world_model = FAMILY_MODELS[spec["family_id"]]
    train_models = [FAMILY_MODELS[f] for f in TRAIN_FAMILY_IDS]
    context = WorldModelContext(tuple(incident.sites), tuple(incident.edges), spec["incident_seed_site_id"])
    belief_seed = derive_belief_seed(spec["case_id"], int(spec["world_seed"]))
    hypotheses = sample_ecological_hypotheses(
        train_models, context, draws_per_model=draws_per_model, seed=belief_seed,
    )
    q_hypotheses = [QHypothesis(float(q)) for q in spec["belief_q_support"]]

    return SpatialBenchmarkCase(
        label=spec["case_id"],
        incident=incident,
        world_model=world_model,
        q_true=float(spec["simulator_q_true"]),
        ecological_hypotheses=tuple(hypotheses),
        q_hypotheses=tuple(q_hypotheses),
        max_rounds=max_rounds,
        seed=int(spec["world_seed"]),
        group=spec["split"],
    )


def load_formal_benchmark_cases(
    *,
    splits: tuple[str, ...] = FORMAL_REPORTING_SPLITS,
    manifest_path: Path = FROZEN_MANIFEST_PATH,
    budget: int = 18,
    max_rounds: int = 6,
    incident_max_sites: int = 20,
    incident_preferred_min_sites: int = 12,
    draws_per_model: int = 60,
) -> list[SpatialBenchmarkCase]:
    """Load and materialize the frozen R8 formal reporting cases.

    Default `splits` excludes `validation` (60 cases) per
    docs/BENCHMARK_CASE_MANIFEST_R8.md: "The 60 validation cases are reserved
    for training/checkpoint selection and must not be folded into the final
    test averages."
    """

    manifest = load_frozen_manifest(manifest_path)
    specs = [case for case in manifest["cases"] if case["split"] in splits]
    return [
        manifest_case_to_benchmark_case(
            spec, budget=budget, max_rounds=max_rounds,
            incident_max_sites=incident_max_sites, incident_preferred_min_sites=incident_preferred_min_sites,
            draws_per_model=draws_per_model,
        )
        for spec in specs
    ]
