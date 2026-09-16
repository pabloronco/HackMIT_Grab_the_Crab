from __future__ import annotations

from adaptive_response.rl.r8_manifest_cases import (
    FROZEN_MANIFEST_PATH,
    TRAIN_FAMILY_IDS,
    derive_belief_seed,
    load_frozen_manifest,
)


def test_derive_belief_seed_is_deterministic() -> None:
    a = derive_belief_seed("id_test__A_graph_diffusion__001", 12345)
    b = derive_belief_seed("id_test__A_graph_diffusion__001", 12345)
    assert a == b


def test_derive_belief_seed_differs_across_cases() -> None:
    a = derive_belief_seed("id_test__A_graph_diffusion__001", 12345)
    b = derive_belief_seed("id_test__A_graph_diffusion__002", 12345)
    assert a != b


def test_no_hypothesis_draw_seed_collides_with_truth_seed_across_the_real_frozen_manifest() -> None:
    """R8 correction (review 2026-09-15): for every one of the actual 240
    frozen manifest cases, none of the hypothesis-draw seeds that
    sample_ecological_hypotheses will actually use (belief_seed +
    model_index*100_003 + draw_index, for each of the 3 train families and
    every draw) may equal that case's truth seed (world_seed). Checked
    against the real manifest file, not a synthetic example, because the
    bug this guards against is specific to the real seed/case_id values."""

    manifest = load_frozen_manifest(FROZEN_MANIFEST_PATH)
    cases = manifest["cases"]
    assert len(cases) == 240, "test assumes the frozen 240-case R8 manifest; update if it is re-frozen"

    draws_per_model = 60  # matches load_formal_benchmark_cases / run_spatial_benchmark_r8.py default
    collisions: list[str] = []

    for spec in cases:
        truth_seed = int(spec["world_seed"])
        belief_seed = derive_belief_seed(spec["case_id"], truth_seed)
        for model_index in range(len(TRAIN_FAMILY_IDS)):
            for draw_index in range(draws_per_model):
                hypothesis_seed = belief_seed + model_index * 100_003 + draw_index
                if hypothesis_seed == truth_seed:
                    collisions.append(spec["case_id"])

    assert not collisions, f"truth/belief seed collision on cases: {collisions}"


def test_belief_seed_namespace_is_disjoint_from_the_manifests_actual_world_seed_range() -> None:
    """Structural guarantee, not just empirical: belief_seed's minimum
    possible value must sit above the manifest's actual world_seed range so
    collisions are impossible by construction, not by luck of the hash."""

    manifest = load_frozen_manifest(FROZEN_MANIFEST_PATH)
    world_seeds = [int(c["world_seed"]) for c in manifest["cases"]]
    max_world_seed = max(world_seeds)

    for spec in manifest["cases"]:
        belief_seed = derive_belief_seed(spec["case_id"], int(spec["world_seed"]))
        assert belief_seed > max_world_seed
