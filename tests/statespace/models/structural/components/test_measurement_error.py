import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

from pytensor import tensor as pt
from pytensor.graph.traversal import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct


def test_measurement_error(rng):
    mod = st.MeasurementError("obs") + st.LevelTrend(order=2)
    mod = mod.build(verbose=False)

    assert "sigma_obs" in mod.param_names


def test_measurement_error_multiple_observed():
    mod = st.MeasurementError("obs", observed_state_names=["data_1", "data_2"])
    assert mod.k_endog == 2
    assert mod.coords["endog_obs"] == (
        "data_1",
        "data_2",
    )
    assert mod.param_info["sigma_obs"].dims == ("endog_obs",)


def test_measurement_error_share_states():
    mod = st.MeasurementError("obs", observed_state_names=["data_1", "data_2"], share_states=True)
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)

    assert mod.k_endog == 2
    assert mod.param_names == (
        "sigma_obs",
        "P0",
    )
    assert "endog_obs" not in mod.coords

    # Check that the parameter is shared across the observed states
    assert mod._param_info["sigma_obs"].shape == ()

    outputs = mod.ssm["obs_cov"]

    H = pytensor.function(list(explicit_graph_inputs([outputs])), outputs)(sigma_obs=np.array(0.5))
    np.testing.assert_allclose(H, np.diag([0.5, 0.5]) ** 2)


def test_measurement_error_shared_and_not_shared():
    shared = st.MeasurementError(
        "error_shared", observed_state_names=["data_1", "data_2"], share_states=True
    )
    individual = st.MeasurementError("error_individual", observed_state_names=["data_1", "data_2"])
    mod = (shared + individual).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.param_names == (
        "sigma_error_shared",
        "sigma_error_individual",
        "P0",
    )
    assert mod.coords["endog_error_individual"] == (
        "data_1",
        "data_2",
    )

    assert mod._param_info["sigma_error_shared"].shape == ()
    assert mod._param_info["sigma_error_individual"].shape == (2,)

    outputs = mod.ssm["obs_cov"]

    H = pytensor.function(list(explicit_graph_inputs([outputs])), outputs)(
        sigma_error_shared=np.array(0.5), sigma_error_individual=np.array([0.1, 0.9])
    )
    np.testing.assert_allclose(H, np.diag([0.5, 0.5]) ** 2 + np.diag([0.1, 0.9]) ** 2)


def test_build_with_measurement_error_subset():
    ll = st.LevelTrend(order=2, observed_state_names=["data_1", "data_2", "data_3"])
    me = st.MeasurementError("obs", observed_state_names=["data_1", "data_3"])
    mod = (ll + me).build()

    H = mod.ssm["obs_cov"]
    assert H.type.shape == (3, 3)
    np.testing.assert_allclose(
        H.eval({"sigma_obs": [1.0, 3.0]}),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
    )


def _known_error(observed=None, shared=False, name="known", data_name="variance"):
    return st.MeasurementError(
        name,
        observed_state_names=observed,
        share_states=shared,
        variance_data=data_name,
    )


@pytest.mark.parametrize("name", ["", " ", "data", 2])
def test_known_variance_bad_name(name):
    with pytest.raises(ValueError, match="variance_data"):
        st.MeasurementError(variance_data=name)


@pytest.mark.parametrize("shared", [False, True])
def test_known_variance_metadata_and_matrix(shared):
    error = _known_error(["a", "b"], shared)
    assert not error.param_names
    assert error.exog_names == ("variance",)
    assert error._data_info["variance"].is_exogenous
    assert error.k_states == error.k_posdef == 0
    values = np.array([0.0, 0.5, 2.0]) if shared else np.array([[0.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    H = error.ssm["obs_cov"].eval({"variance": values})
    expected = [np.eye(2) * v if shared else np.diag(v) for v in values]
    np.testing.assert_allclose(H, expected)
    assert "obs_cov" in error.ssm.time_varying_names


@pytest.mark.parametrize("reverse", [False, True])
def test_known_variance_composition_subset(reverse):
    trend = st.LevelTrend(order=1, observed_state_names=["a", "b", "c"])
    known = _known_error(["a", "c"])
    model = (known + trend if reverse else trend + known).build(verbose=False)
    values = np.array([[0.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    H = model.ssm["obs_cov"].eval({"variance": values})
    expected = np.zeros((3, 3, 3))
    order = list(model.observed_states)
    expected[:, order.index("a"), order.index("a")] = values[:, 0]
    expected[:, order.index("c"), order.index("c")] = values[:, 1]
    np.testing.assert_allclose(H, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_known_and_estimated_variances_add(reverse):
    known = _known_error(["a", "b"], shared=True)
    estimated = st.MeasurementError("estimated", observed_state_names=["a", "b"])
    model = known + estimated if reverse else estimated + known
    values = np.array([0.0, 0.5, 2.0])
    H = model.ssm["obs_cov"].eval({"variance": values, "sigma_estimated": [2.0, 3.0]})
    np.testing.assert_allclose(H, [np.diag([v + 4.0, v + 9.0]) for v in values])


def test_two_known_variances_add_and_names_collide():
    model = _known_error() + _known_error(name="other", data_name="other_variance")
    H = model.ssm["obs_cov"].eval({"variance": [1.0, 2.0], "other_variance": [3.0, 5.0]})
    np.testing.assert_allclose(H[:, 0, 0], [4.0, 7.0])
    with pytest.raises(ValueError, match="Duplicate"):
        _known_error() + _known_error(name="other")


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_known_variance_invalid_values(value):
    with pytest.raises(AssertionError, match=r"variance.*finite, nonnegative"):
        _known_error().ssm["obs_cov"].eval({"variance": [1.0, value]})


def _known_level_model(values, filter_type="standard"):
    model = (st.LevelTrend(order=1, innovations_order=0) + _known_error()).build(
        verbose=False, filter_type=filter_type, cov_jitter=0
    )
    with pm.Model(coords=model.coords) as pymc_model:
        pm.Data("variance", values, dims="time")
        for name, info in model.param_info.items():
            shape = info["shape"]
            value = np.eye(model.k_states) if name == "P0" else np.zeros(shape)
            pm.Deterministic(name, pt.as_tensor_variable(value))
        model.build_statespace_graph(
            pd.DataFrame(
                {"data": [0.2, -0.4, 0.8, 0.1]}, index=pd.date_range("2024-01-01", periods=4)
            )
        )
    return model, pymc_model


@pytest.mark.parametrize("filter_type", ["standard", "univariate"])
def test_known_variance_logp_oracle_and_updates(filter_type):
    variances = np.array([0.4, 0.2, 0.8, 1.1])
    model, pymc_model = _known_level_model(variances, filter_type)
    logp = pymc_model.compile_logp()

    def oracle(v):
        mean, covariance, result = 0.0, 1.0, 0.0
        for y, variance in zip([0.2, -0.4, 0.8, 0.1], v):
            residual = y - mean
            predictive = covariance + variance
            result -= 0.5 * (np.log(2 * np.pi * predictive) + residual**2 / predictive)
            gain = covariance / predictive
            mean += gain * residual
            covariance *= 1 - gain
        return result

    np.testing.assert_allclose(logp({}), oracle(variances), rtol=1e-6)
    with pymc_model:
        pm.set_data({"variance": variances * 2})
    np.testing.assert_allclose(logp({}), oracle(variances * 2), rtol=1e-6)
    with pymc_model:
        pm.set_data({"variance": [0.4, -1.0, 0.8, 1.1]})
    with pytest.raises(AssertionError, match=r"variance.*finite, nonnegative"):
        logp({})


def test_known_variance_missing_data_and_shape():
    model = (st.LevelTrend(order=1) + _known_error()).build(verbose=False)
    with pm.Model(), pytest.raises(ValueError, match="variance"):
        model._insert_data_variables([model.ssm["obs_cov"]])
    for shape in [(4, 1), (4, 3)]:
        with pm.Model():
            pm.Data("variance", np.ones(shape))
            with pytest.raises((TypeError, ValueError)):
                model._insert_data_variables([model.ssm["obs_cov"]])


@pytest.mark.parametrize("width", [1, 3])
def test_known_variance_multivariate_width(width):
    model = _known_error(["a", "b"]).build(verbose=False)
    with pm.Model():
        pm.Data("variance", np.ones((4, width)))
        try:
            [H] = model._insert_data_variables([model.ssm["obs_cov"]])
        except (TypeError, ValueError):
            return
        with pytest.raises((AssertionError, ValueError)):
            H.eval()


def test_known_variance_length_and_coordinated_update():
    model, pymc_model = _known_level_model(np.ones(4))
    logp = pymc_model.compile_logp()
    with pymc_model:
        pm.set_data(
            {"variance": np.ones(3)}, coords={"time": pd.date_range("2024-01-01", periods=3)}
        )
    with pytest.raises(AssertionError, match="time dimension"):
        logp({})
    with pymc_model:
        model.build_statespace_graph(
            pd.DataFrame({"data": np.zeros(3)}, index=pd.date_range("2024-01-01", periods=3))
        )
    assert np.isfinite(logp({}))


@pytest.mark.parametrize("kind", ["array", "list", "tuple", "series", "frame"])
def test_known_variance_scenario_forms(kind):
    model = (st.LevelTrend(order=1) + _known_error()).build(verbose=False)
    values = np.array([0.1, 0.3])
    scenario = {
        "array": values,
        "list": values.tolist(),
        "tuple": tuple(values),
        "series": pd.Series(values),
        "frame": pd.DataFrame({"v": values}),
    }[kind]
    result = model._validate_scenario_data(scenario, model.coords)
    assert np.asarray(result).shape == (2,)
    finalized = model._finalize_scenario_initialization(result, pd.RangeIndex(4, 6), model.coords)
    assert isinstance(finalized, pd.Series)
    np.testing.assert_array_equal(finalized, values)
    assert finalized.index.equals(pd.RangeIndex(4, 6))
    with pytest.raises(ValueError, match="missing required variables: variance"):
        model._validate_scenario_data({}, model.coords)
    with pytest.raises(ValueError, match="Forecasting cannot be performed"):
        model._validate_scenario_data(None, model.coords)
    with pytest.raises(ValueError, match="one-dimensional"):
        model._validate_scenario_data(np.ones((2, 2)), model.coords)


def test_known_variance_forecast(monkeypatch):
    model, pymc_model = _known_level_model(np.ones(4))
    with pymc_model:
        idata = pm.sample_prior_predictive(draws=2, random_seed=42)
    scenario = np.array([0.2, 0.7])
    from pymc_extras.statespace.core import forecast as forecast_module

    original = forecast_module.LinearGaussianStateSpace
    covariance = []

    def record_covariance(*args, **kwargs):
        covariance.append(args[8])
        return original(*args, **kwargs)

    monkeypatch.setattr(forecast_module, "LinearGaussianStateSpace", record_covariance)
    result = model.forecast(idata, periods=2, scenario=scenario, group="prior", random_seed=42)
    assert result.forecast_observed.shape[-2:] == (2, 1)
    np.testing.assert_allclose(covariance[0].eval()[-2:, 0, 0], scenario)
