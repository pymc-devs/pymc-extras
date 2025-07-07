import numpy as np
import pandas as pd
import pymc as pm

from numpy.testing import assert_allclose
from pytensor import config
from pytensor import tensor as pt
from pytensor.graph.basic import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


def test_exogenous_component(rng):
    data = rng.normal(size=(100, 2)).astype(config.floatX)
    mod = st.RegressionComponent(state_names=["feature_1", "feature_2"], name="exog")

    params = {"beta_exog": np.array([1.0, 2.0], dtype=config.floatX)}
    exog_data = {"data_exog": data}
    x, y = simulate_from_numpy_model(mod, rng, params, exog_data)

    # Check that the generated data is just a linear regression
    assert_allclose(y, data @ params["beta_exog"], atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["exog_state"] == ["feature_1", "feature_2"]


def test_adding_exogenous_component(rng):
    data = rng.normal(size=(100, 2)).astype(config.floatX)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    ll = st.LevelTrendComponent(name="level")

    seasonal = st.FrequencySeasonality(name="annual", season_length=12, n=4)
    mod = reg + ll + seasonal

    assert mod.ssm["design"].eval({"data_exog": data}).shape == (100, 1, 2 + 2 + 8)
    assert_allclose(mod.ssm["design", 5, 0, :2].eval({"data_exog": data}), data[5])


def test_regression_with_multiple_observed_states(rng):
    from scipy.linalg import block_diag

    data = rng.normal(size=(100, 2)).astype(config.floatX)
    mod = st.RegressionComponent(
        state_names=["feature_1", "feature_2"],
        name="exog",
        observed_state_names=["data_1", "data_2"],
    )

    params = {"beta_exog": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)}
    exog_data = {"data_exog": data}
    x, y = simulate_from_numpy_model(mod, rng, params, exog_data)

    assert x.shape == (100, 4)  # 2 features, 2 states
    assert y.shape == (100, 2)

    # Check that the generated data are two independent linear regressions
    assert_allclose(y[:, 0], data @ params["beta_exog"][0], atol=ATOL, rtol=RTOL)
    assert_allclose(y[:, 1], data @ params["beta_exog"][1], atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    assert mod.coords["exog_state"] == [
        "feature_1[data_1]",
        "feature_2[data_1]",
        "feature_1[data_2]",
        "feature_2[data_2]",
    ]

    Z = mod.ssm["design"].eval({"data_exog": data})
    vec_block_diag = np.vectorize(block_diag, signature="(n,m),(o,p)->(q,r)")
    assert Z.shape == (100, 2, 4)
    assert np.allclose(Z, vec_block_diag(data[:, None, :], data[:, None, :]))


def test_add_regression_components_with_multiple_observed_states(rng):
    from scipy.linalg import block_diag

    data_1 = rng.normal(size=(100, 2)).astype(config.floatX)
    data_2 = rng.normal(size=(100, 1)).astype(config.floatX)

    reg1 = st.RegressionComponent(
        state_names=["a", "b"], name="exog1", observed_state_names=["data_1", "data_2"]
    )
    reg2 = st.RegressionComponent(state_names=["c"], name="exog2", observed_state_names=["data_3"])

    mod = (reg1 + reg2).build(verbose=False)
    assert mod.coords["exog1_state"] == ["a[data_1]", "b[data_1]", "a[data_2]", "b[data_2]"]
    assert mod.coords["exog2_state"] == ["c[data_3]"]

    Z = mod.ssm["design"].eval({"data_exog1": data_1, "data_exog2": data_2})
    vec_block_diag = np.vectorize(block_diag, signature="(n,m),(o,p)->(q,r)")
    assert Z.shape == (100, 3, 5)
    assert np.allclose(
        Z,
        vec_block_diag(vec_block_diag(data_1[:, None, :], data_1[:, None, :]), data_2[:, None, :]),
    )

    x0 = mod.ssm["initial_state"].eval(
        {
            "beta_exog1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX),
            "beta_exog2": np.array([5.0], dtype=config.floatX),
        }
    )
    np.testing.assert_allclose(x0, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=config.floatX))


def test_filter_scans_time_varying_design_matrix(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    mod = reg.build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.Data("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])

        mod.build_statespace_graph(y)
        x0, P0, c, d, T, Z, R, H, Q = mod.unpack_statespace()
        pm.Deterministic("Z", Z)

        prior = pm.sample_prior_predictive(draws=10)

    prior_Z = prior.prior.Z.values
    assert prior_Z.shape == (1, 10, 100, 1, 2)
    assert_allclose(prior_Z[0, :, :, 0, :], data.values[None].repeat(10, axis=0))
