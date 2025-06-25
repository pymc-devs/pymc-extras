import numpy as np

from numpy.testing import assert_allclose
from pytensor import config

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


def test_level_trend_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["trend_state"] == ["level", "trend"]


def test_level_trend_multiple_observed_construction():
    mod = st.LevelTrendComponent(
        order=2, innovations_order=1, observed_state_names=["data_1", "data_2", "data_3"]
    )
    mod = mod.build(verbose=False)
    assert mod.k_endog == 3
    assert mod.k_states == 6
    assert mod.k_posdef == 3

    assert mod.coords["trend_state"] == ["level", "trend"]
    assert mod.coords["trend_endog"] == ["data_1", "data_2", "data_3"]

    Z = mod.ssm["design"].eval()
    T = mod.ssm["transition"].eval()
    R = mod.ssm["selection"].eval()

    np.testing.assert_allclose(
        Z,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        T,
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        R,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_level_trend_multiple_observed(rng):
    mod = st.LevelTrendComponent(
        order=2, innovations_order=0, observed_state_names=["data_1", "data_2", "data_3"]
    )
    params = {"initial_trend": np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])}

    x, y = simulate_from_numpy_model(mod, rng, params)
    assert (np.diff(y, axis=0) == np.array([[1.0, 2.0, 3.0]])).all().all()
    assert (np.diff(x, axis=0) == np.array([[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]])).all().all()
