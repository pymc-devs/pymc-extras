from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.core.statespace import FILTER_FACTORY, PyMCStateSpace
from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.utilities import make_default_coords
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_NAMES,
    MATRIX_NAMES,
    SMOOTHER_OUTPUT_NAMES,
)
from tests.statespace.shared_fixtures import (
    rng,
)
from tests.statespace.test_utilities import (
    fast_eval,
    load_nile_test_data,
    make_test_inputs,
)

floatX = pytensor.config.floatX
nile = load_nile_test_data()
ALL_SAMPLE_OUTPUTS = MATRIX_NAMES + FILTER_OUTPUT_NAMES + SMOOTHER_OUTPUT_NAMES


def make_statespace_mod(k_endog, k_states, k_posdef, filter_type, verbose=False, data_info=None):
    class StateSpace(PyMCStateSpace):
        def make_symbolic_graph(self):
            pass

        @property
        def data_info(self):
            return data_info

    ss = StateSpace(
        k_states=k_states,
        k_endog=k_endog,
        k_posdef=k_posdef,
        filter_type=filter_type,
        verbose=verbose,
    )
    ss._needs_exog_data = data_info is not None
    ss._exog_names = list(data_info.keys()) if data_info is not None else []

    return ss


@pytest.fixture(scope="session")
def ss_mod():
    class StateSpace(PyMCStateSpace):
        @property
        def param_names(self):
            return ["rho", "zeta"]

        @property
        def state_names(self):
            return ["a", "b"]

        @property
        def observed_states(self):
            return ["a"]

        @property
        def shock_names(self):
            return ["a"]

        @property
        def coords(self):
            return make_default_coords(self)

        def make_symbolic_graph(self):
            rho = self.make_and_register_variable("rho", ())
            zeta = self.make_and_register_variable("zeta", ())
            self.ssm["transition", 0, 0] = rho
            self.ssm["transition", 1, 0] = zeta

    Z = np.array([[1.0, 0.0]], dtype=floatX)
    R = np.array([[1.0], [0.0]], dtype=floatX)
    H = np.array([[0.1]], dtype=floatX)
    Q = np.array([[0.8]], dtype=floatX)
    P0 = np.eye(2, dtype=floatX) * 1e6

    ss_mod = StateSpace(
        k_endog=nile.shape[1], k_states=2, k_posdef=1, filter_type="standard", verbose=False
    )
    for X, name in zip(
        [Z, R, H, Q, P0],
        ["design", "selection", "obs_cov", "state_cov", "initial_state_cov"],
    ):
        ss_mod.ssm[name] = X

    return ss_mod


@pytest.fixture(scope="session")
def pymc_mod(ss_mod):
    with pm.Model(coords=ss_mod.coords) as pymc_mod:
        rho = pm.Beta("rho", 1, 1)
        zeta = pm.Deterministic("zeta", 1 - rho)

        ss_mod.build_statespace_graph(data=nile, save_kalman_filter_outputs_in_idata=True)
        names = ["x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
        for name, matrix in zip(names, ss_mod.unpack_statespace()):
            pm.Deterministic(name, matrix)

    return pymc_mod


@pytest.fixture(scope="session")
def ss_mod_no_exog(rng):
    ll = st.LevelTrendComponent(order=2, innovations_order=1)
    return ll.build()


@pytest.fixture(scope="session")
def ss_mod_no_exog_dt(rng):
    ll = st.LevelTrendComponent(order=2, innovations_order=1)
    return ll.build()


@pytest.fixture(scope="session")
def exog_data(rng):
    # simulate data
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-05-01", end="2023-05-10", freq="D"),
            "x1": rng.choice(2, size=10, replace=True).astype(float),
            "y": rng.normal(size=(10,)),
        }
    )

    df.loc[[1, 3, 9], ["y"]] = np.nan
    return df.set_index("date")


@pytest.fixture(scope="session")
def exog_ss_mod(exog_data):
    level_trend = st.LevelTrendComponent(order=1, innovations_order=[0])
    exog = st.RegressionComponent(
        name="exog",  # Name of this exogenous variable component
        k_exog=1,  # Only one exogenous variable now
        innovations=False,  # Typically fixed effect (no stochastic evolution)
        state_names=exog_data[["x1"]].columns.tolist(),
    )

    combined_model = level_trend + exog
    return combined_model.build()


@pytest.fixture(scope="session")
def exog_pymc_mod(exog_ss_mod, exog_data):
    # define pymc model
    with pm.Model(coords=exog_ss_mod.coords) as struct_model:
        P0_diag = pm.Gamma("P0_diag", alpha=2, beta=4, dims=["state"])
        P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=["state", "state_aux"])

        initial_trend = pm.Normal("initial_trend", mu=[0], sigma=[0.005], dims=["trend_state"])

        data_exog = pm.Data(
            "data_exog", exog_data["x1"].values[:, None], dims=["time", "exog_state"]
        )
        beta_exog = pm.Normal("beta_exog", mu=0, sigma=1, dims=["exog_state"])

        exog_ss_mod.build_statespace_graph(exog_data["y"])

    return struct_model


@pytest.fixture(scope="session")
def pymc_mod_no_exog(ss_mod_no_exog, rng):
    y = pd.DataFrame(rng.normal(size=(100, 1)).astype(floatX), columns=["y"])

    with pm.Model(coords=ss_mod_no_exog.coords) as m:
        initial_trend = pm.Normal("initial_trend", dims=["trend_state"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["trend_shock"])
        ss_mod_no_exog.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def pymc_mod_no_exog_dt(ss_mod_no_exog_dt, rng):
    y = pd.DataFrame(
        rng.normal(size=(100, 1)).astype(floatX),
        columns=["y"],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    with pm.Model(coords=ss_mod_no_exog_dt.coords) as m:
        initial_trend = pm.Normal("initial_trend", dims=["trend_state"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog_dt.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["trend_shock"])
        ss_mod_no_exog_dt.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def idata(pymc_mod, rng):
    with pymc_mod:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)

    idata.extend(idata_prior)
    return idata


@pytest.fixture(scope="session")
def idata_exog(exog_pymc_mod, rng):
    with exog_pymc_mod:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.extend(idata_prior)
    return idata


@pytest.fixture(scope="session")
def idata_no_exog(pymc_mod_no_exog, rng):
    with pymc_mod_no_exog:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.extend(idata_prior)
    return idata


@pytest.fixture(scope="session")
def idata_no_exog_dt(pymc_mod_no_exog_dt, rng):
    with pymc_mod_no_exog_dt:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.extend(idata_prior)
    return idata


def test_invalid_filter_name_raises():
    msg = "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
    with pytest.raises(NotImplementedError, match=msg):
        mod = make_statespace_mod(k_endog=1, k_states=5, k_posdef=1, filter_type="invalid_filter")


def test_unpack_before_insert_raises(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = make_statespace_mod(
        k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False
    )

    msg = "Cannot unpack the complete statespace system until PyMC model variables have been inserted."
    with pytest.raises(ValueError, match=msg):
        outputs = mod.unpack_statespace()


def test_unpack_matrices(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = make_statespace_mod(
        k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False
    )

    # mod is a dummy statespace, so there are no placeholders to worry about. Monkey patch subbed_ssm with the defaults
    mod.subbed_ssm = mod._unpack_statespace_with_placeholders()

    outputs = mod.unpack_statespace()
    for x, y in zip(inputs, outputs):
        assert_allclose(np.zeros_like(x), fast_eval(y))


def test_param_names_raises_on_base_class():
    mod = make_statespace_mod(
        k_endog=1, k_states=5, k_posdef=1, filter_type="standard", verbose=False
    )
    with pytest.raises(NotImplementedError):
        x = mod.param_names


def test_base_class_raises():
    with pytest.raises(NotImplementedError):
        mod = PyMCStateSpace(
            k_endog=1, k_states=5, k_posdef=1, filter_type="standard", verbose=False
        )


def test_update_raises_if_missing_variables(ss_mod):
    with pm.Model() as mod:
        rho = pm.Normal("rho")
        msg = "The following required model parameters were not found in the PyMC model: zeta"
        with pytest.raises(ValueError, match=msg):
            ss_mod._insert_random_variables()


def test_build_statespace_graph_warns_if_data_has_nans():
    # Breaks tests if it uses the session fixtures because we can't call build_statespace_graph over and over
    ss_mod = st.LevelTrendComponent(order=1, innovations_order=0).build(verbose=False)

    with pm.Model() as pymc_mod:
        initial_trend = pm.Normal("initial_trend", shape=(1,))
        P0 = pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        with pytest.warns(pm.ImputationWarning):
            ss_mod.build_statespace_graph(
                data=np.full((10, 1), np.nan, dtype=floatX), register_data=False
            )


def test_build_statespace_graph_raises_if_data_has_missing_fill():
    # Breaks tests if it uses the session fixtures because we can't call build_statespace_graph over and over
    ss_mod = st.LevelTrendComponent(order=1, innovations_order=0).build(verbose=False)

    with pm.Model() as pymc_mod:
        initial_trend = pm.Normal("initial_trend", shape=(1,))
        P0 = pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        with pytest.raises(ValueError, match="Provided data contains the value 1.0"):
            data = np.ones((10, 1), dtype=floatX)
            data[3] = np.nan
            ss_mod.build_statespace_graph(data=data, missing_fill_value=1.0, register_data=False)


def test_build_statespace_graph(pymc_mod):
    for name in [
        "filtered_state",
        "predicted_state",
        "predicted_covariance",
        "filtered_covariance",
    ]:
        assert name in [x.name for x in pymc_mod.deterministics]


def test_build_smoother_graph(ss_mod, pymc_mod):
    names = ["smoothed_state", "smoothed_covariance"]
    for name in names:
        assert name in [x.name for x in pymc_mod.deterministics]


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("matrix", ALL_SAMPLE_OUTPUTS)
def test_no_nans_in_sampling_output(group, matrix, idata):
    assert not np.any(np.isnan(idata[group][matrix].values))


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("kind", ["conditional", "unconditional"])
def test_sampling_methods(group, kind, ss_mod, idata, rng):
    f = getattr(ss_mod, f"sample_{kind}_{group}")
    test_idata = f(idata, random_seed=rng)

    if kind == "conditional":
        for output in ["filtered", "predicted", "smoothed"]:
            assert f"{output}_{group}" in test_idata
            assert not np.any(np.isnan(test_idata[f"{output}_{group}"].values))
            assert not np.any(np.isnan(test_idata[f"{output}_{group}_observed"].values))
    if kind == "unconditional":
        for output in ["latent", "observed"]:
            assert f"{group}_{output}" in test_idata
            assert not np.any(np.isnan(test_idata[f"{group}_{output}"].values))


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_sample_conditional_with_time_varying():
    class TVCovariance(PyMCStateSpace):
        def __init__(self):
            super().__init__(k_states=1, k_endog=1, k_posdef=1)

        def make_symbolic_graph(self) -> None:
            self.ssm["transition", 0, 0] = 1.0

            self.ssm["design", 0, 0] = 1.0

            sigma_cov = self.make_and_register_variable("sigma_cov", (None,))
            self.ssm["state_cov"] = sigma_cov[:, None, None] ** 2

        @property
        def param_names(self) -> list[str]:
            return ["sigma_cov"]

        @property
        def coords(self) -> dict[str, Sequence[str]]:
            return make_default_coords(self)

        @property
        def state_names(self) -> list[str]:
            return ["level"]

        @property
        def observed_states(self) -> list[str]:
            return ["level"]

        @property
        def shock_names(self) -> list[str]:
            return ["level"]

    ss_mod = TVCovariance()
    empty_data = pd.DataFrame(
        np.nan, index=pd.date_range("2020-01-01", periods=100, freq="D"), columns=["data"]
    )

    coords = ss_mod.coords
    coords["time"] = empty_data.index
    with pm.Model(coords=coords) as mod:
        log_sigma_cov = pm.Normal("log_sigma_cov", mu=0, sigma=0.1, dims=["time"])
        pm.Deterministic("sigma_cov", pm.math.exp(log_sigma_cov.cumsum()), dims=["time"])

        ss_mod.build_statespace_graph(data=empty_data)

        prior = pm.sample_prior_predictive(10)

    ss_mod.sample_unconditional_prior(prior)
    ss_mod.sample_conditional_prior(prior)


def _make_time_idx(mod, use_datetime_index=True):
    if use_datetime_index:
        mod._fit_coords["time"] = nile.index
        time_idx = nile.index
    else:
        mod._fit_coords["time"] = nile.reset_index().index
        time_idx = pd.RangeIndex(start=0, stop=nile.shape[0], step=1)

    return time_idx


@pytest.mark.parametrize("use_datetime_index", [True, False])
def test_bad_forecast_arguments(use_datetime_index, caplog):
    ss_mod = make_statespace_mod(
        k_endog=1, k_posdef=1, k_states=2, filter_type="standard", verbose=False
    )

    # Not-fit model raises
    ss_mod._fit_coords = dict()
    with pytest.raises(ValueError, match="Has this model been fit?"):
        ss_mod._get_fit_time_index()

    time_idx = _make_time_idx(ss_mod, use_datetime_index)

    # Start value not in time index
    match = (
        "Datetime start must be in the data index used to fit the model"
        if use_datetime_index
        else "Integer start must be within the range of the data index used to fit the model."
    )
    with pytest.raises(ValueError, match=match):
        start = time_idx.shift(10)[-1] if use_datetime_index else time_idx[-1] + 11
        ss_mod._validate_forecast_args(time_index=time_idx, start=start, periods=10)

    # End value cannot be inferred
    with pytest.raises(ValueError, match="Must specify one of either periods or end"):
        start = time_idx[-1]
        ss_mod._validate_forecast_args(time_index=time_idx, start=start)

    # Unnecessary args warn on verbose
    start = time_idx[-1]
    forecast_idx = pd.date_range(start=start, periods=10, freq="YS-JAN")
    scenario = pd.DataFrame(0, index=forecast_idx, columns=[0, 1, 2])

    ss_mod._validate_forecast_args(
        time_index=time_idx, start=start, periods=10, scenario=scenario, use_scenario_index=True
    )
    last_message = caplog.messages[-1]
    assert "start, end, and periods arguments are ignored" in last_message

    # Verbose=False silences warning
    ss_mod._validate_forecast_args(
        time_index=time_idx,
        start=start,
        periods=10,
        scenario=scenario,
        use_scenario_index=True,
        verbose=False,
    )
    assert len(caplog.messages) == 1


@pytest.mark.parametrize("use_datetime_index", [True, False])
def test_forecast_index(use_datetime_index):
    ss_mod = make_statespace_mod(
        k_endog=1, k_posdef=1, k_states=2, filter_type="standard", verbose=False
    )
    ss_mod._fit_coords = dict()
    time_idx = _make_time_idx(ss_mod, use_datetime_index)

    # From start and end
    start = time_idx[-1]
    delta = pd.DateOffset(years=10) if use_datetime_index else 11
    end = start + delta

    x0_index, forecast_idx = ss_mod._build_forecast_index(time_idx, start=start, end=end)
    assert start not in forecast_idx
    assert x0_index == start
    assert forecast_idx.shape == (10,)

    # From start and periods
    start = time_idx[-1]
    periods = 10

    x0_index, forecast_idx = ss_mod._build_forecast_index(time_idx, start=start, periods=periods)
    assert start not in forecast_idx
    assert x0_index == start
    assert forecast_idx.shape == (10,)

    # From integer start
    start = 10
    x0_index, forecast_idx = ss_mod._build_forecast_index(time_idx, start=start, periods=periods)
    delta = forecast_idx.freq if use_datetime_index else 1

    assert x0_index == time_idx[start]
    assert forecast_idx.shape == (10,)
    assert (forecast_idx == time_idx[start + 1 : start + periods + 1]).all()

    # From scenario index
    scenario = pd.DataFrame(0, index=forecast_idx, columns=[0, 1, 2])
    new_start, forecast_idx = ss_mod._build_forecast_index(
        time_index=time_idx, scenario=scenario, use_scenario_index=True
    )
    assert x0_index not in forecast_idx
    assert x0_index == (forecast_idx[0] - delta)
    assert forecast_idx.shape == (10,)
    assert forecast_idx.equals(scenario.index)

    # From dictionary of scenarios
    scenario = {"a": pd.DataFrame(0, index=forecast_idx, columns=[0, 1, 2])}
    x0_index, forecast_idx = ss_mod._build_forecast_index(
        time_index=time_idx, scenario=scenario, use_scenario_index=True
    )
    assert x0_index == (forecast_idx[0] - delta)
    assert forecast_idx.shape == (10,)
    assert forecast_idx.equals(scenario["a"].index)


@pytest.mark.parametrize(
    "data_type",
    [pd.Series, pd.DataFrame, np.array, list, tuple],
    ids=["series", "dataframe", "array", "list", "tuple"],
)
def test_validate_scenario(data_type):
    if data_type is pd.DataFrame:
        # Ensure dataframes have the correct column name
        data_type = partial(pd.DataFrame, columns=["column_1"])

    # One data case
    data_info = {"a": {"shape": (None, 1), "dims": ("time", "features_a")}}
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(features_a=["column_1"])

    scenario = data_type(np.zeros(10))
    scenario = ss_mod._validate_scenario_data(scenario)

    # Lists and tuples are cast to 2d arrays
    if data_type in [tuple, list]:
        assert isinstance(scenario, np.ndarray)
        assert scenario.shape == (10, 1)

    # A one-item dictionary should also work
    scenario = {"a": scenario}
    ss_mod._validate_scenario_data(scenario)

    # Now data has to be a dictionary
    data_info.update({"b": {"shape": (None, 1), "dims": ("time", "features_b")}})
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(features_a=["column_1"], features_b=["column_1"])

    scenario = {"a": data_type(np.zeros(10)), "b": data_type(np.zeros(10))}
    ss_mod._validate_scenario_data(scenario)

    # Mixed data types
    data_info.update({"a": {"shape": (None, 10), "dims": ("time", "features_a")}})
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(
        features_a=[f"column_{i}" for i in range(10)], features_b=["column_1"]
    )

    scenario = {
        "a": pd.DataFrame(np.zeros((10, 10)), columns=ss_mod._fit_coords["features_a"]),
        "b": data_type(np.arange(10)),
    }

    ss_mod._validate_scenario_data(scenario)


@pytest.mark.parametrize(
    "data_type",
    [pd.Series, pd.DataFrame, np.array, list, tuple],
    ids=["series", "dataframe", "array", "list", "tuple"],
)
@pytest.mark.parametrize("use_datetime_index", [True, False])
def test_finalize_scenario_single(data_type, use_datetime_index):
    if data_type is pd.DataFrame:
        # Ensure dataframes have the correct column name
        data_type = partial(pd.DataFrame, columns=["column_1"])

    data_info = {"a": {"shape": (None, 1), "dims": ("time", "features_a")}}
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(features_a=["column_1"])

    time_idx = _make_time_idx(ss_mod, use_datetime_index)

    scenario = data_type(np.zeros((10,)))

    scenario = ss_mod._validate_scenario_data(scenario)
    t0, forecast_idx = ss_mod._build_forecast_index(time_idx, start=time_idx[-1], periods=10)
    scenario = ss_mod._finalize_scenario_initialization(scenario, forecast_index=forecast_idx)

    assert isinstance(scenario, pd.DataFrame)
    assert scenario.index.equals(forecast_idx)
    assert scenario.columns == ["column_1"]


@pytest.mark.parametrize(
    "data_type",
    [pd.Series, pd.DataFrame, np.array, list, tuple],
    ids=["series", "dataframe", "array", "list", "tuple"],
)
@pytest.mark.parametrize("use_datetime_index", [True, False])
@pytest.mark.parametrize("use_scenario_index", [True, False])
def test_finalize_secenario_dict(data_type, use_datetime_index, use_scenario_index):
    data_info = {
        "a": {"shape": (None, 1), "dims": ("time", "features_a")},
        "b": {"shape": (None, 2), "dims": ("time", "features_b")},
    }
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(features_a=["column_1"], features_b=["column_1", "column_2"])
    time_idx = _make_time_idx(ss_mod, use_datetime_index)

    initial_index = (
        pd.date_range(start=time_idx[-1], periods=10, freq=time_idx.freq)
        if use_datetime_index
        else pd.RangeIndex(time_idx[-1], time_idx[-1] + 10, 1)
    )

    if data_type is pd.DataFrame:
        # Ensure dataframes have the correct column name
        data_type = partial(pd.DataFrame, columns=["column_1"], index=initial_index)
    elif data_type is pd.Series:
        data_type = partial(pd.Series, index=initial_index)

    scenario = {
        "a": data_type(np.zeros((10,))),
        "b": pd.DataFrame(
            np.zeros((10, 2)), columns=ss_mod._fit_coords["features_b"], index=initial_index
        ),
    }

    scenario = ss_mod._validate_scenario_data(scenario)

    if use_scenario_index and data_type not in [np.array, list, tuple]:
        t0, forecast_idx = ss_mod._build_forecast_index(
            time_idx, scenario=scenario, periods=10, use_scenario_index=True
        )
    elif use_scenario_index and data_type in [np.array, list, tuple]:
        t0, forecast_idx = ss_mod._build_forecast_index(
            time_idx, scenario=scenario, start=-1, periods=10, use_scenario_index=True
        )
    else:
        t0, forecast_idx = ss_mod._build_forecast_index(time_idx, start=time_idx[-1], periods=10)

    scenario = ss_mod._finalize_scenario_initialization(scenario, forecast_index=forecast_idx)

    assert list(scenario.keys()) == ["a", "b"]
    assert all(isinstance(value, pd.DataFrame) for value in scenario.values())
    assert all(value.index.equals(forecast_idx) for value in scenario.values())


def test_invalid_scenarios():
    data_info = {"a": {"shape": (None, 1), "dims": ("time", "features_a")}}
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(features_a=["column_1", "column_2"])

    # Omitting the data raises
    with pytest.raises(
        ValueError, match="This model was fit using exogenous data. Forecasting cannot be performed"
    ):
        ss_mod._validate_scenario_data(None)

    # Giving a list, tuple, or Series when a matrix of data is expected should always raise
    with pytest.raises(
        ValueError,
        match="Scenario data for variable 'a' has the wrong number of columns. "
        "Expected 2, got 1",
    ):
        for data_type in [list, tuple, pd.Series]:
            ss_mod._validate_scenario_data(data_type(np.zeros(10)))
            ss_mod._validate_scenario_data({"a": data_type(np.zeros(10))})

    # Providing irrevelant data raises
    with pytest.raises(
        ValueError,
        match="Scenario data provided for variable 'jk lol', which is not an exogenous " "variable",
    ):
        ss_mod._validate_scenario_data({"jk lol": np.zeros(10)})

    # Incorrect 2nd dimension of a non-dataframe
    with pytest.raises(
        ValueError,
        match="Scenario data for variable 'a' has the wrong number of columns. Expected "
        "2, got 1",
    ):
        scenario = np.zeros(10).tolist()
        ss_mod._validate_scenario_data(scenario)
        ss_mod._validate_scenario_data(tuple(scenario))

        scenario = {"a": np.zeros(10).tolist()}
        ss_mod._validate_scenario_data(scenario)
        ss_mod._validate_scenario_data({"a": tuple(scenario["a"])})

    # If a data frame is provided, it needs to have all columns
    with pytest.raises(
        ValueError, match="Scenario data for variable 'a' is missing the following column: column_2"
    ):
        scenario = pd.DataFrame(np.zeros((10, 1)), columns=["column_1"])
        ss_mod._validate_scenario_data(scenario)

    # Extra columns also raises
    with pytest.raises(
        ValueError,
        match="Scenario data for variable 'a' contains the following extra columns "
        "that are not used by the model: column_3, column_4",
    ):
        scenario = pd.DataFrame(
            np.zeros((10, 4)), columns=["column_1", "column_2", "column_3", "column_4"]
        )
        ss_mod._validate_scenario_data(scenario)

    # Wrong number of time steps raises
    data_info = {
        "a": {"shape": (None, 1), "dims": ("time", "features_a")},
        "b": {"shape": (None, 1), "dims": ("time", "features_b")},
    }
    ss_mod = make_statespace_mod(
        k_endog=1,
        k_posdef=1,
        k_states=2,
        filter_type="standard",
        verbose=False,
        data_info=data_info,
    )
    ss_mod._fit_coords = dict(
        features_a=["column_1", "column_2"], features_b=["column_1", "column_2"]
    )

    with pytest.raises(
        ValueError, match="Scenario data must have the same number of time steps for all variables"
    ):
        scenario = {
            "a": pd.DataFrame(np.zeros((10, 2)), columns=ss_mod._fit_coords["features_a"]),
            "b": pd.DataFrame(np.zeros((11, 2)), columns=ss_mod._fit_coords["features_b"]),
        }
        ss_mod._validate_scenario_data(scenario)


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.parametrize("filter_output", ["predicted", "filtered", "smoothed"])
@pytest.mark.parametrize(
    "mod_name, idata_name, start, end, periods",
    [
        ("ss_mod_no_exog", "idata_no_exog", None, None, 10),
        ("ss_mod_no_exog", "idata_no_exog", -1, None, 10),
        ("ss_mod_no_exog", "idata_no_exog", 10, None, 10),
        ("ss_mod_no_exog", "idata_no_exog", 10, 21, None),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", None, None, 10),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", -1, None, 10),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", 10, None, 10),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", 10, "2020-01-21", None),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", "2020-03-01", "2020-03-11", None),
        ("ss_mod_no_exog_dt", "idata_no_exog_dt", "2020-03-01", None, 10),
    ],
    ids=[
        "range_default",
        "range_negative",
        "range_int",
        "range_end",
        "datetime_default",
        "datetime_negative",
        "datetime_int",
        "datetime_int_end",
        "datetime_datetime_end",
        "datetime_datetime",
    ],
)
def test_forecast(filter_output, mod_name, idata_name, start, end, periods, rng, request):
    mod = request.getfixturevalue(mod_name)
    idata = request.getfixturevalue(idata_name)
    time_idx = mod._get_fit_time_index()
    is_datetime = isinstance(time_idx, pd.DatetimeIndex)

    if isinstance(start, str):
        t0 = pd.Timestamp(start)
    elif isinstance(start, int):
        t0 = time_idx[start]
    else:
        t0 = time_idx[-1]

    delta = time_idx.freq if is_datetime else 1

    forecast_idata = mod.forecast(
        idata, start=start, end=end, periods=periods, filter_output=filter_output, random_seed=rng
    )

    forecast_idx = forecast_idata.coords["time"].values
    forecast_idx = pd.DatetimeIndex(forecast_idx) if is_datetime else pd.Index(forecast_idx)

    assert forecast_idx.shape == (10,)
    assert forecast_idata.forecast_latent.dims == ("chain", "draw", "time", "state")
    assert forecast_idata.forecast_observed.dims == ("chain", "draw", "time", "observed_state")

    assert not np.any(np.isnan(forecast_idata.forecast_latent.values))
    assert not np.any(np.isnan(forecast_idata.forecast_observed.values))

    assert forecast_idx[0] == (t0 + delta)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.filterwarnings("ignore:Skipping `CheckAndRaise` Op")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
@pytest.mark.parametrize("start", [None, -1, 5])
def test_forecast_with_exog_data(rng, exog_ss_mod, idata_exog, start):
    scenario = pd.DataFrame(np.zeros((10, 1)), columns=["x1"])
    scenario.iloc[5, 0] = 1e9

    forecast_idata = exog_ss_mod.forecast(
        idata_exog, start=start, periods=10, random_seed=rng, scenario=scenario
    )

    components = exog_ss_mod.extract_components_from_idata(forecast_idata)
    level = components.forecast_latent.sel(state="LevelTrend[level]")
    betas = components.forecast_latent.sel(state=["exog[x1]"])

    scenario.index.name = "time"
    scenario_xr = (
        scenario.unstack()
        .to_xarray()
        .rename({"level_0": "state"})
        .assign_coords(state=["exog[x1]"])
    )

    regression_effect = forecast_idata.forecast_observed.isel(observed_state=0) - level
    regression_effect_expected = (betas * scenario_xr).sum(dim=["state"])

    assert_allclose(regression_effect, regression_effect_expected)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.filterwarnings("ignore:Skipping `CheckAndRaise` Op")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
def test_foreacast_valid_index(exog_pymc_mod, exog_ss_mod, exog_data):
    # Regression test for issue reported at  https://github.com/pymc-devs/pymc-extras/issues/424
    with exog_pymc_mod:
        idata = pm.sample_prior_predictive()

    # Define start date and forecast period
    start_date, n_periods = pd.to_datetime("2023-05-05"), 5

    # Extract exogenous data for the forecast period
    scenario = {
        "data_exog": pd.DataFrame(
            exog_data[["x1"]].loc[start_date:].iloc[:n_periods], columns=exog_data[["x1"]].columns
        )
    }

    # Generate the forecast
    forecasts = exog_ss_mod.forecast(idata.prior, scenario=scenario, use_scenario_index=True)
    assert "forecast_latent" in forecasts
    assert "forecast_observed" in forecasts

    assert (forecasts.coords["time"].values == scenario["data_exog"].index.values).all()
    assert not np.any(np.isnan(forecasts.forecast_latent.values))
    assert not np.any(np.isnan(forecasts.forecast_observed.values))

    assert forecasts.forecast_latent.shape[2] == n_periods
    assert forecasts.forecast_observed.shape[2] == n_periods
