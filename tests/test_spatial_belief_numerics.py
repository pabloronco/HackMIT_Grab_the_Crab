import pytest

from adaptive_response import SpatialBeliefState, SpatialHypothesis
from adaptive_response.spatial_belief import _probability_with_roundoff_guard


def test_spatial_marginal_clamps_machine_epsilon_above_one() -> None:
    belief = SpatialBeliefState(
        site_ids=("a",),
        hypotheses=(
            SpatialHypothesis(presence=(True,), q=0.1),
            SpatialHypothesis(presence=(True,), q=0.3),
        ),
        # This reproduces the failure seen in the integrated R6 VOI path:
        # normalized floating-point weights can sum to 1.0000000000000002.
        weights=(0.5000000000000001, 0.5000000000000001),
    )

    assert belief.p_by_site()["a"] == 1.0
    assert belief.uncertainty_by_site()["a"] == 0.0


def test_probability_roundoff_guard_clamps_only_tiny_boundary_noise() -> None:
    assert _probability_with_roundoff_guard(
        1.0 + 2e-16,
        name="test probability",
    ) == 1.0
    assert _probability_with_roundoff_guard(
        -2e-16,
        name="test probability",
    ) == 0.0

    with pytest.raises(ValueError, match="outside"):
        _probability_with_roundoff_guard(
            1.0 + 1e-6,
            name="test probability",
        )

    with pytest.raises(ValueError, match="outside"):
        _probability_with_roundoff_guard(
            -1e-6,
            name="test probability",
        )
