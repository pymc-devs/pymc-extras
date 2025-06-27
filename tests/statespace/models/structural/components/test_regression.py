import numpy as np
import pandas as pd
import pymc as pm

from numpy.testing import assert_allclose
from pytensor import config
from pytensor import tensor as pt

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

    mod.build(verbose=False)
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
