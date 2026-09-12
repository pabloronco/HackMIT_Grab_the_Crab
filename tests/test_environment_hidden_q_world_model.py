import pytest

from adaptive_response import (
    Edge,
    Environment,
    GeneratedWorld,
    IncidentConfig,
    MissionAction,
    MissionAllocation,
    Site,
)


class FixedWorldModel:
    family_id = "test_fixed_world"

    def sample(self, context, *, seed: int):
        return GeneratedWorld(
            family_id=self.family_id,
            occupied_by_site={
                site.id: site.id == context.initial_detection
                for site in context.sites
            },
            seed=seed,
            generator_parameters={"test": 1},
        )


def make_config(*, q_model=0.2, budget: int = 1) -> IncidentConfig:
    return IncidentConfig(
        sites=[
            Site("seed", 0.0, 0.0, 0.5, q_model),
            Site("other", 1.0, 0.0, 0.5, q_model),
        ],
        edges=[Edge("seed", "other", 1.0)],
        initial_detection="seed",
        budget=budget,
        teams=1,
        protocol="test",
        seed=11,
    )


def test_explicit_q_true_allows_non_scalar_belief_side_q_model_and_never_leaks_q() -> None:
    environment = Environment(
        make_config(q_model={"status": "OPEN"}),
        q_true=0.0,
    )
    public = environment.reset(seed=11)
    assert public.sites[0].q_model == {"status": "OPEN"}

    batch, done, _ = environment.step(
        MissionAction(
            allocations=(MissionAllocation("seed", 1),),
            total_cost=1,
        )
    )
    assert done is True
    observation = batch.observations[0]
    assert observation.detection is False
    assert "q_used" not in observation.metadata
    assert "q_true" not in observation.metadata
    assert observation.metadata["observation_model"] == "binary_effort_imperfect_detection"


def test_injected_world_model_replaces_legacy_toy_generator() -> None:
    environment = Environment(
        make_config(q_model=0.2),
        world_model=FixedWorldModel(),
        q_true=0.2,
    )
    public = environment.reset(seed=99)
    assert public.world_model_id == "test_fixed_world"

    environment.step(
        MissionAction(
            allocations=(MissionAllocation("other", 1),),
            total_cost=1,
        )
    )
    environment._allow_reveal()
    hidden = environment.reveal()
    assert hidden.occupied_by_site == {"seed": True, "other": False}
    assert hidden.generator_parameters["family_id"] == "test_fixed_world"


def test_q_true_map_must_align_exactly_with_sites() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        Environment(
            make_config(),
            q_true={"seed": 0.2},
        )


def test_world_model_cannot_contradict_confirmed_initial_detection() -> None:
    class BadWorld:
        family_id = "bad"

        def sample(self, context, *, seed: int):
            return GeneratedWorld(
                family_id=self.family_id,
                occupied_by_site={site.id: False for site in context.sites},
                seed=seed,
            )

    environment = Environment(make_config(), world_model=BadWorld(), q_true=0.2)
    with pytest.raises(ValueError, match="initial detection"):
        environment.reset()
