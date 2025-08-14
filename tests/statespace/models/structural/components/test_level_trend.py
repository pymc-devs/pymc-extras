import numpy as np
import pytensor

from numpy.testing import assert_allclose
from pytensor import config

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


def test_level_trend_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_level_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["state_level_trend"] == ["level", "trend"]


def test_level_trend_multiple_observed_construction():
    mod = st.LevelTrendComponent(
        order=2, innovations_order=1, observed_state_names=["data_1", "data_2", "data_3"]
    )
    mod = mod.build(verbose=False)
    assert mod.k_endog == 3
    assert mod.k_states == 6
    assert mod.k_posdef == 3

    assert mod.coords["state_level_trend"] == ["level", "trend"]
    assert mod.coords["endog_level_trend"] == ["data_1", "data_2", "data_3"]

    assert mod.state_names == [
        "level[data_1]",
        "trend[data_1]",
        "level[data_2]",
        "trend[data_2]",
        "level[data_3]",
        "trend[data_3]",
    ]
    assert mod.shock_names == ["level[data_1]", "level[data_2]", "level[data_3]"]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

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


def test_level_trend_multiple_shared_construction():
    mod = st.LevelTrendComponent(
        order=2, innovations_order=1, observed_state_names=["data_1", "data_2"], share_states=True
    )
    mod = mod.build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 2
    assert mod.k_posdef == 1

    assert mod.coords["state_level_trend"] == ["level", "trend"]
    assert mod.coords["endog_level_trend"] == ["data_1", "data_2"]

    assert mod.state_names == [
        "level[level_trend_shared]",
        "trend[level_trend_shared]",
    ]
    assert mod.shock_names == ["level[level_trend_shared]"]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(
        Z,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(T, np.array([[1.0, 1.0], [0.0, 1.0]]))

    np.testing.assert_allclose(R, np.array([[1.0], [0.0]]))


def test_level_trend_multiple_observed(rng):
    mod = st.LevelTrendComponent(
        order=2, innovations_order=0, observed_state_names=["data_1", "data_2", "data_3"]
    )
    params = {"initial_level_trend": np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])}

    x, y = simulate_from_numpy_model(mod, rng, params)
    assert (np.diff(y, axis=0) == np.array([[1.0, 2.0, 3.0]])).all().all()
    assert (np.diff(x, axis=0) == np.array([[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]])).all().all()


def test_level_trend_multiple_shared_observed(rng):
    mod = st.LevelTrendComponent(
        order=2,
        innovations_order=0,
        observed_state_names=["data_1", "data_2", "data_3"],
        share_states=True,
    )
    params = {"initial_level_trend": np.array([10.0, 0.1])}
    x, y = simulate_from_numpy_model(mod, rng, params)
    np.testing.assert_allclose(y[:, 0], y[:, 1])
    np.testing.assert_allclose(y[:, 0], y[:, 2])


def test_add_level_trend_with_different_observed():
    mod_1 = st.LevelTrendComponent(
        name="ll", order=2, innovations_order=[0, 1], observed_state_names=["data_1"]
    )
    mod_2 = st.LevelTrendComponent(
        name="grw", order=1, innovations_order=[1], observed_state_names=["data_2"]
    )

    mod = (mod_1 + mod_2).build(verbose=False)
    assert mod.k_endog == 2
    assert mod.k_states == 3
    assert mod.k_posdef == 2

    assert mod.coords["state_ll"] == ["level", "trend"]
    assert mod.coords["state_grw"] == ["level"]

    assert mod.state_names == ["level[data_1]", "trend[data_1]", "level[data_2]"]
    assert mod.shock_names == ["trend[data_1]", "level[data_2]"]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(
        Z,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        T,
        np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        R,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )


def test_mixed_shared_and_not_shared():
    mod_1 = st.LevelTrendComponent(
        name="individual",
        order=2,
        innovations_order=[0, 1],
        observed_state_names=["data_1", "data_2"],
    )
    mod_2 = st.LevelTrendComponent(
        name="joint",
        order=2,
        innovations_order=[1, 1],
        observed_state_names=["data_1", "data_2"],
        share_states=True,
    )

    mod = (mod_1 + mod_2).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 6
    assert mod.k_posdef == 4

    assert mod.state_names == [
        "level[data_1]",
        "trend[data_1]",
        "level[data_2]",
        "trend[data_2]",
        "level[joint_shared]",
        "trend[joint_shared]",
    ]

    assert mod.shock_names == [
        "trend[data_1]",
        "trend[data_2]",
        "level[joint_shared]",
        "trend[joint_shared]",
    ]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(
        Z, np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
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
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
