import pytest

from adaptive_response.models import Edge, Site
from adaptive_response.world_models import (
    FragmentedPatchyWorldModel,
    GraphDiffusionWorldModel,
    HabitatDrivenWorldModel,
    SpatialClusterWorldModel,
    WorldModelContext,
    default_world_model_split,
    sample_ecological_hypotheses,
)


def site(site_id: str, x: float, habitat: float) -> Site:
    return Site(
        id=site_id,
        x=x,
        y=0.0,
        habitat_score=habitat,
        q_model=0.2,
    )


def line_context() -> WorldModelContext:
    sites = (
        site("a", 0.0, 0.5),
        site("b", 1.0, 1.0),
        site("c", 2.0, 0.0),
        site("d", 4.0, 0.6),
    )
    edges = (
        Edge("a", "b", distance=1.0),
        Edge("b", "c", distance=1.0),
        Edge("c", "d", distance=2.0),
    )
    return WorldModelContext(sites=sites, edges=edges, initial_detection="a")


def test_default_split_has_three_train_families_and_structural_holdout() -> None:
    split = default_world_model_split()

    assert [model.family_id for model in split["train"]] == [
        "A_graph_diffusion",
        "B_spatial_cluster",
        "C_habitat_driven",
    ]
    assert [model.family_id for model in split["ood_model_holdout"]] == [
        "E_fragmented_patchy"
    ]


def test_every_family_is_seed_deterministic_and_conditions_on_initial_detection() -> None:
    context = line_context()
    models = (*default_world_model_split()["train"], *default_world_model_split()["ood_model_holdout"])

    for model in models:
        first = model.sample(context, seed=91)
        second = model.sample(context, seed=91)
        assert first == second
        assert first.occupied_by_site["a"] is True
        assert set(first.occupied_by_site) == {"a", "b", "c", "d"}


def test_graph_diffusion_uses_adjacency_and_multiple_waves() -> None:
    context = line_context()
    model = GraphDiffusionWorldModel(
        transmission_range=(1.0, 1.0),
        habitat_effect_range=(0.0, 0.0),
        wave_range=(3, 3),
    )

    world = model.sample(context, seed=1)

    assert world.occupied_by_site == {"a": True, "b": True, "c": True, "d": True}
    assert world.generator_parameters["waves"] == 3


def test_spatial_cluster_with_tiny_radius_does_not_invent_remote_occupancy() -> None:
    context = line_context()
    model = SpatialClusterWorldModel(
        radius_scale_range=(1e-6, 1e-6),
        patchiness_range=(0.0, 0.0),
    )

    world = model.sample(context, seed=4)

    assert world.occupied_by_site == {"a": True, "b": False, "c": False, "d": False}


def test_habitat_driven_family_can_distinguish_high_and_low_habitat() -> None:
    context = line_context()
    model = HabitatDrivenWorldModel(
        prevalence_range=(0.5, 0.5),
        habitat_coefficient_range=(20.0, 20.0),
        local_boost_range=(0.0, 0.0),
    )

    world = model.sample(context, seed=7)

    assert world.occupied_by_site["b"] is True
    assert world.occupied_by_site["c"] is False


def test_fragmented_holdout_creates_a_remote_patch_anchor() -> None:
    context = line_context()
    model = FragmentedPatchyWorldModel(
        extra_patch_range=(1, 1),
        patch_radius_scale_range=(1e-6, 1e-6),
        dropout_range=(0.0, 0.0),
    )

    world = model.sample(context, seed=2)

    assert world.occupied_by_site["a"] is True
    assert any(world.occupied_by_site[site_id] for site_id in ("c", "d"))
    assert sum(world.occupied_by_site.values()) >= 2


def test_sampling_hypotheses_deduplicates_and_preserves_family_provenance() -> None:
    context = line_context()
    model = GraphDiffusionWorldModel(
        transmission_range=(1.0, 1.0),
        habitat_effect_range=(0.0, 0.0),
        wave_range=(3, 3),
    )

    hypotheses = sample_ecological_hypotheses(
        [model],
        context,
        draws_per_model=5,
        seed=100,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].prior_weight == pytest.approx(5.0)
    assert hypotheses[0].label == "A_graph_diffusion"
    assert all(hypotheses[0].presence_by_site.values())


def test_generated_world_converts_to_hidden_world_with_provenance() -> None:
    generated = SpatialClusterWorldModel().sample(line_context(), seed=12)

    hidden = generated.as_hidden_world()

    assert hidden.occupied_by_site == dict(generated.occupied_by_site)
    assert hidden.generator_parameters["family_id"] == "B_spatial_cluster"
    assert hidden.generator_parameters["seed"] == 12


def test_invalid_context_rejects_unknown_initial_detection() -> None:
    with pytest.raises(ValueError, match="initial_detection"):
        WorldModelContext(
            sites=(site("a", 0.0, 0.5),),
            edges=(),
            initial_detection="missing",
        )
