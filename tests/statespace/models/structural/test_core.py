from functools import partial

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from numpy.testing import assert_allclose
from pytensor import config
from pytensor import tensor as pt
from scipy import linalg

from pymc_extras.statespace.models import structural as st
from tests.statespace.test_utilities import unpack_symbolic_matrices_with_params

floatX = config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 0 if floatX.endswith("64") else 1e-6


def test_add_components():
    ll = st.LevelTrendComponent(order=2)
    se = st.TimeSeasonality(name="seasonal", season_length=12)
    mod = ll + se

    ll_params = {
        "initial_level_trend": np.zeros(2, dtype=floatX),
        "sigma_level_trend": np.ones(2, dtype=floatX),
    }
    se_params = {
        "params_seasonal": np.ones(11, dtype=floatX),
        "sigma_seasonal": 1.0,
    }
    all_params = ll_params.copy()
    all_params.update(se_params)

    (ll_x0, ll_P0, ll_c, ll_d, ll_T, ll_Z, ll_R, ll_H, ll_Q) = unpack_symbolic_matrices_with_params(
        ll, ll_params
    )
    (se_x0, se_P0, se_c, se_d, se_T, se_Z, se_R, se_H, se_Q) = unpack_symbolic_matrices_with_params(
        se, se_params
    )
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, all_params)

    for property in ["param_names", "shock_names", "param_info", "coords", "param_dims"]:
        assert [x in getattr(mod, property) for x in getattr(ll, property)]
        assert [x in getattr(mod, property) for x in getattr(se, property)]

    assert (mod.observed_state_names == ll.observed_state_names) and (
        ll.observed_state_names == se.observed_state_names
    )

    ll_mats = [ll_T, ll_R, ll_Q]
    se_mats = [se_T, se_R, se_Q]
    all_mats = [T, R, Q]

    for ll_mat, se_mat, all_mat in zip(ll_mats, se_mats, all_mats):
        assert_allclose(all_mat, linalg.block_diag(ll_mat, se_mat), atol=ATOL, rtol=RTOL)

    ll_mats = [ll_x0, ll_c, ll_Z]
    se_mats = [se_x0, se_c, se_Z]
    all_mats = [x0, c, Z]
    axes = [0, 0, 1]

    for ll_mat, se_mat, all_mat, axis in zip(ll_mats, se_mats, all_mats, axes):
        assert_allclose(all_mat, np.concatenate([ll_mat, se_mat], axis=axis), atol=ATOL, rtol=RTOL)


def test_add_components_multiple_observed():
    ll = st.LevelTrendComponent(order=2, observed_state_names=["data_1", "data_2"])
    me = st.MeasurementError(name="obs", observed_state_names=["data_1", "data_2"])

    mod = (ll + me).build()

    for property in ["param_names", "shock_names", "param_info", "coords", "param_dims"]:
        assert [x in getattr(mod, property) for x in getattr(ll, property)]


@pytest.mark.skipif(floatX.endswith("32"), reason="Prior covariance not PSD at half-precision")
def test_extract_components_from_idata(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    ll = st.LevelTrendComponent()
    season = st.FrequencySeasonality(name="seasonal", season_length=12, n=2, innovations=False)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    me = st.MeasurementError("obs")
    mod = (ll + season + reg + me).build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.Data("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["state_exog"])
        initial_trend = pm.Normal("initial_level_trend", dims=["state_level_trend"])
        sigma_trend = pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
        seasonal_coefs = pm.Normal("params_seasonal", dims=["state_seasonal"])
        sigma_obs = pm.Exponential("sigma_obs", 1)

        mod.build_statespace_graph(y)

        prior = pm.sample_prior_predictive(draws=10)

    filter_prior = mod.sample_conditional_prior(prior)
    comp_prior = mod.extract_components_from_idata(filter_prior)
    comp_states = comp_prior.filtered_prior.coords["state"].values
    expected_states = ["level_trend[level]", "level_trend[trend]", "seasonal", "exog[a]", "exog[b]"]
    missing = set(comp_states) - set(expected_states)

    assert len(missing) == 0, missing


def test_extract_multiple_observed(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(
        rng.normal(size=(100, 3)), columns=["data_1", "data_2", "data_3"], index=time_idx
    )

    ll = st.LevelTrendComponent(name="trend", observed_state_names=["data_1", "data_2"])
    season = st.FrequencySeasonality(
        name="seasonal", observed_state_names=["data_1"], season_length=12, n=2, innovations=False
    )
    reg = st.RegressionComponent(
        state_names=["a", "b"], name="exog", observed_state_names=["data_2", "data_3"]
    )
    ar = st.AutoregressiveComponent(observed_state_names=["data_1", "data_2"], order=3)
    me = st.MeasurementError("obs", observed_state_names=["data_1", "data_3"])
    mod = (ll + season + reg + ar + me).build(verbose=True)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.Data("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["endog_exog", "state_exog"])
        params_auto_regressive = pm.Normal(
            "params_auto_regressive", dims=["endog_auto_regressive", "lag_auto_regressive"]
        )
        sigma_auto_regressive = pm.Normal("sigma_auto_regressive", dims=["endog_auto_regressive"])
        initial_trend = pm.Normal("initial_trend", dims=["endog_trend", "state_trend"])
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["endog_trend", "shock_trend"])
        seasonal_coefs = pm.Normal("params_seasonal", dims=["state_seasonal"])
        sigma_obs = pm.Exponential("sigma_obs", 1, dims=["endog_obs"])

        mod.build_statespace_graph(y)

        prior = pm.sample_prior_predictive(draws=10)

    filter_prior = mod.sample_conditional_prior(prior)
    comp_prior = mod.extract_components_from_idata(filter_prior)
    comp_states = comp_prior.filtered_prior.coords["state"].values

    expected_states = [
        "trend[level[data_1]]",
        "trend[trend[data_1]]",
        "trend[level[data_2]]",
        "trend[trend[data_2]]",
        "seasonal[data_1]",
        "exog[a[data_2]]",
        "exog[b[data_2]]",
        "exog[a[data_3]]",
        "exog[b[data_3]]",
        "auto_regressive[data_1]",
        "auto_regressive[data_2]",
    ]

    missing = set(comp_states) - set(expected_states)
    assert len(missing) == 0, missing


@pytest.mark.parametrize(
    "arg_type", [tuple, list, set, np.array], ids=["tuple", "list", "set", "array"]
)
def test_sequence_type_component_arguments(arg_type):
    state_names = list("ABCDEFG")
    components = [
        st.LevelTrendComponent,
        partial(st.CycleComponent, cycle_length=12),
        st.AutoregressiveComponent,
        partial(st.FrequencySeasonality, season_length=12),
        partial(st.TimeSeasonality, season_length=12),
        st.MeasurementError,
    ]

    components = [
        components[i](observed_state_names=arg_type(state_names))
        for i in np.random.choice(len(components), size=3, replace=False)
    ]
    ss_mod = sum(components[1:], start=components[0]).build(verbose=False)

    assert ss_mod.k_endog == len(state_names)
    assert sorted(ss_mod.observed_states) == sorted(list(state_names))
