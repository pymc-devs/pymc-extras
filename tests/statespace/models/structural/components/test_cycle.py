import numpy as np

from numpy.testing import assert_allclose
from pytensor import config

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import assert_pattern_repeats, simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


cycle_test_vals = zip([None, None, 3, 5, 10], [False, True, True, False, False])


def test_cycle_component_deterministic(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False
    )
    params = {"cycle": np.array([1.0, 1.0], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    assert_pattern_repeats(y, 12, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_dampening(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False, dampen=True
    )
    params = {"cycle": np.array([10.0, 10.0], dtype=config.floatX), "cycle_dampening_factor": 0.75}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=100)

    # Check that the cycle dampens to zero over time
    assert_allclose(y[-1], 0.0, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_innovations_and_cycle_length(rng):
    cycle = st.CycleComponent(
        name="cycle", estimate_cycle_length=True, innovations=True, dampen=True
    )
    params = {
        "cycle": np.array([1.0, 1.0], dtype=config.floatX),
        "cycle_length": 12.0,
        "cycle_dampening_factor": 0.95,
        "sigma_cycle": 1.0,
    }

    x, y = simulate_from_numpy_model(cycle, rng, params)

    cycle.build(verbose=False)
    _assert_basic_coords_correct(cycle)
