import numpy as np
import pytensor
import pytest

from numpy.testing import assert_allclose
from pytensor import config
from pytensor.graph.basic import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
    ar = st.AutoregressiveComponent(order=order).build(verbose=False)

    # Check coords
    _assert_basic_coords_correct(ar)

    lags = np.arange(len(order) if isinstance(order, list) else order, dtype="int") + 1
    if isinstance(order, list):
        lags = lags[np.flatnonzero(order)]
    assert_allclose(ar.coords["lag_auto_regressive"], lags)


def test_autoregressive_multiple_observed_build(rng):
    ar = st.AutoregressiveComponent(order=3, observed_state_names=["data_1", "data_2"])
    mod = ar.build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 6
    assert mod.k_posdef == 2

    assert mod.state_names == [
        "L1[data_1]",
        "L2[data_1]",
        "L3[data_1]",
        "L1[data_2]",
        "L2[data_2]",
        "L3[data_2]",
    ]

    assert mod.shock_names == ["auto_regressive[data_1]", "auto_regressive[data_2]"]

    params = {
        "params_auto_regressive": np.full(
            (
                2,
                sum(ar.order),
            ),
            0.5,
            dtype=config.floatX,
        ),
        "sigma_auto_regressive": np.array([0.05, 0.12]),
    }
    _, _, _, _, T, Z, R, _, Q = mod._unpack_statespace_with_placeholders()
    input_vars = explicit_graph_inputs([T, Z, R, Q])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[T, Z, R, Q],
        mode="FAST_COMPILE",
    )

    T, Z, R, Q = fn(**params)

    np.testing.assert_allclose(
        T,
        np.array(
            [
                [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        Z, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    )

    np.testing.assert_allclose(
        R, np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    )

    np.testing.assert_allclose(Q, np.diag([0.05**2, 0.12**2]))


def test_autoregressive_multiple_observed_data(rng):
    ar = st.AutoregressiveComponent(order=1, observed_state_names=["data_1", "data_2", "data_3"])
    mod = ar.build(verbose=False)

    params = {
        "params_auto_regressive": np.array([0.9, 0.8, 0.5]).reshape((3, 1)),
        "sigma_auto_regressive": np.array([0.05, 0.12, 0.22]),
        "initial_state_cov": np.eye(3),
    }

    # Recover the AR(1) coefficients from the simulated data via OLS
    x, y = simulate_from_numpy_model(mod, rng, params, steps=2000)
    for i in range(3):
        ols_coefs = np.polyfit(y[:-1, i], y[1:, i], 1)
        np.testing.assert_allclose(ols_coefs[0], params["params_auto_regressive"][i, 0], atol=1e-1)


def test_add_autoregressive_different_observed():
    mod_1 = st.AutoregressiveComponent(order=1, name="ar1", observed_state_names=["data_1"])
    mod_2 = st.AutoregressiveComponent(name="ar6", order=6, observed_state_names=["data_2"])

    mod = (mod_1 + mod_2).build(verbose=False)

    print(mod.coords)

    assert mod.k_endog == 2
    assert mod.k_states == 7
    assert mod.k_posdef == 2
    assert mod.state_names == [
        "L1[data_1]",
        "L1[data_2]",
        "L2[data_2]",
        "L3[data_2]",
        "L4[data_2]",
        "L5[data_2]",
        "L6[data_2]",
    ]

    assert mod.shock_names == ["ar1[data_1]", "ar6[data_2]"]
    assert mod.coords["lag_ar1"] == [1]
    assert mod.coords["lag_ar6"] == [1, 2, 3, 4, 5, 6]
