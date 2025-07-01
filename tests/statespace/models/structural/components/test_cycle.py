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


def test_cycle_multivariate_deterministic(rng):
    """Test multivariate cycle component with deterministic cycles."""
    cycle = st.CycleComponent(
        name="cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {"cycle": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    # Check that each variable has a cyclical pattern with the expected period
    for i in range(3):
        assert_pattern_repeats(y[:, i], 12, atol=ATOL, rtol=RTOL)

    # Check that the cycles have different amplitudes (different initial states)
    assert np.std(y[:, 0]) > 0
    assert np.std(y[:, 1]) > 0
    assert np.std(y[:, 2]) > 0
    # The second and third variables should have larger amplitudes due to larger initial states
    assert np.std(y[:, 1]) > np.std(y[:, 0])
    assert np.std(y[:, 2]) > np.std(y[:, 0])


def test_cycle_multivariate_with_dampening(rng):
    """Test multivariate cycle component with dampening."""
    cycle = st.CycleComponent(
        name="cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        dampen=True,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {
        "cycle": np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=config.floatX),
        "cycle_dampening_factor": 0.75,
    }
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=100)

    # Check that all cycles dampen to zero over time
    for i in range(3):
        assert_allclose(y[-1, i], 0.0, atol=ATOL, rtol=RTOL)

    # Check that the dampening pattern is consistent across variables
    # The variables should dampen at the same rate but with different initial amplitudes
    for i in range(1, 3):
        # The ratio of final to initial values should be similar across variables
        ratio_0 = abs(y[-1, 0] / y[0, 0]) if y[0, 0] != 0 else 0
        ratio_i = abs(y[-1, i] / y[0, i]) if y[0, i] != 0 else 0
        assert_allclose(ratio_0, ratio_i, atol=1e-2, rtol=1e-2)


def test_cycle_multivariate_with_innovations_and_cycle_length(rng):
    """Test multivariate cycle component with innovations and estimated cycle length."""
    cycle = st.CycleComponent(
        name="cycle",
        estimate_cycle_length=True,
        innovations=True,
        dampen=True,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {
        "cycle": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=config.floatX),
        "cycle_length": 12.0,
        "cycle_dampening_factor": 0.95,
        "sigma_cycle": np.array([0.5, 1.0, 1.5]),  # Different innovation variances per variable
    }
    x, y = simulate_from_numpy_model(cycle, rng, params)

    cycle.build(verbose=False)
    _assert_basic_coords_correct(cycle)

    assert cycle.coords["cycle_state"] == ["cycle_Cos", "cycle_Sin"]
    assert cycle.coords["cycle_endog"] == ["data_1", "data_2", "data_3"]

    assert cycle.k_endog == 3
    assert cycle.k_states == 6  # 2 states per variable
    assert cycle.k_posdef == 6  # 2 innovations per variable

    # Check that the data has the expected shape
    assert y.shape[1] == 3  # 3 variables

    # Check that each variable shows some variation (due to innovations)
    for i in range(3):
        assert np.std(y[:, i]) > 0
