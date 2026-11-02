import numpy as np
import pytensor
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from pytensor import config
from pytensor.graph.traversal import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct


def _evaluate_matrices(mod, params):
    outputs = [mod.ssm["transition"], mod.ssm["design"], mod.ssm["selection"], mod.ssm["state_cov"]]
    return pytensor.function(list(explicit_graph_inputs(outputs)), outputs)(**params)


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["MA1", "MA2", "MA(1,0,1)"])
def test_moving_average_model(order):
    mod = st.MovingAverage(order=order).build(verbose=False)
    _assert_basic_coords_correct(mod)

    q = len(order) if isinstance(order, list) else order
    lags = np.arange(q, dtype="int") + 1
    if isinstance(order, list):
        lags = lags[np.flatnonzero(order)]

    assert_allclose(mod.coords["lag_moving_average"], lags)
    assert mod.k_states == q + 1
    assert mod.state_names == tuple(f"L{i}_moving_average" for i in range(q + 1))


def test_moving_average_multiple_observed_build():
    ma = st.MovingAverage(order=2, name="ma", observed_state_names=["data_1", "data_2"])
    mod = ma.build(verbose=False)
    _assert_basic_coords_correct(mod)

    assert mod.k_endog == 2
    assert mod.k_states == 6
    assert mod.k_posdef == 2
    assert mod.state_names == (
        "L0_ma[data_1]",
        "L1_ma[data_1]",
        "L2_ma[data_1]",
        "L0_ma[data_2]",
        "L1_ma[data_2]",
        "L2_ma[data_2]",
    )
    assert mod.shock_names == ("ma[data_1]", "ma[data_2]")
    assert mod.coords["endog_ma"] == ("data_1", "data_2")

    T, Z, R, Q = _evaluate_matrices(
        mod,
        {
            "params_ma": np.array([[0.5, 0.2], [-0.4, 0.1]], dtype=config.floatX),
            "sigma_ma": np.array([0.05, 0.12], dtype=config.floatX),
        },
    )

    assert_allclose(T, np.kron(np.eye(2), np.eye(3, k=1)))
    assert_allclose(Z, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))
    assert_allclose(
        R,
        np.array([[1.0, 0.0], [0.5, 0.0], [0.2, 0.0], [0.0, 1.0], [0.0, -0.4], [0.0, 0.1]]),
    )
    assert_allclose(Q, np.diag([0.05**2, 0.12**2]), rtol=1e-6)


def test_moving_average_multiple_observed_shared():
    ma = st.MovingAverage(
        order=1,
        name="latent",
        observed_state_names=["data_1", "data_2", "data_3"],
        share_states=True,
    )
    mod = ma.build(verbose=False)

    assert mod.k_endog == 3
    assert mod.k_states == 2
    assert mod.k_posdef == 1
    assert mod.state_names == ("L0_latent[shared]", "L1_latent[shared]")
    assert mod.shock_names == ("latent[shared]",)
    assert mod.coords["lag_latent"] == (1,)
    assert "endog_latent" not in mod.coords

    T, Z, R, Q = _evaluate_matrices(
        mod, {"params_latent": np.array([0.9], dtype=config.floatX), "sigma_latent": 0.5}
    )

    assert_allclose(T, np.eye(2, k=1))
    assert_allclose(Z, np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
    assert_allclose(R, np.array([[1.0], [0.9]]))
    assert_allclose(Q, np.array([[0.25]]))


def test_moving_average_fixed_coefficients():
    ma = st.MovingAverage(
        order=3, name="ma", observed_state_names=["data_1", "data_2"], coefficients=np.ones(3)
    )
    mod = ma.build(verbose=False)

    assert "params_ma" not in mod.param_info
    assert "lag_ma" not in mod.coords
    assert mod.param_names == ("sigma_ma", "P0")

    T, Z, R, Q = _evaluate_matrices(mod, {"sigma_ma": np.array([1.0, 2.0], dtype=config.floatX)})
    assert_allclose(R, np.kron(np.eye(2), np.ones((4, 1))))
    assert_allclose(Q, np.diag([1.0, 4.0]))


def test_moving_average_fixed_coefficients_wrong_length():
    with pytest.raises(ValueError, match="one entry per included lag"):
        st.MovingAverage(order=[1, 0, 1], coefficients=np.ones(3))


def test_add_moving_average_and_autoregressive():
    mod = (
        st.LevelTrend(order=1, innovations_order=0)
        + st.MovingAverage(order=2, name="ma")
        + st.Autoregressive(order=1, name="ar")
    ).build(verbose=False)

    assert mod.state_names == ("level", "L0_ma", "L1_ma", "L2_ma", "L1_ar")
    assert mod.shock_names == ("ma", "ar")

    Z = mod.ssm["design"].eval()
    assert_allclose(Z, np.array([[1.0, 1.0, 0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("order", [1, 3, [1, 0, 1]], ids=["MA1", "MA3", "MA(1,0,1)"])
def test_moving_average_matrices_match_statsmodels(order):
    theta = np.array([0.5, -0.3, 0.2])[: int(np.sum(order))]
    sigma = 1.3

    sm_mod = sm.tsa.SARIMAX(np.zeros(10), order=(0, 0, order), trend="n")
    sm_mod.update(np.r_[theta, sigma**2])

    mod = st.MovingAverage(order=order).build(verbose=False)
    T, Z, R, Q = _evaluate_matrices(
        mod,
        {
            "params_moving_average": theta.astype(config.floatX),
            "sigma_moving_average": np.array(sigma, dtype=config.floatX),
        },
    )

    assert_allclose(T, sm_mod.ssm["transition"])
    assert_allclose(Z, sm_mod.ssm["design"])
    assert_allclose(R, sm_mod.ssm["selection"])
    assert_allclose(Q, sm_mod.ssm["state_cov"])
