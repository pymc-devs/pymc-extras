import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc.testing import mock_sample_setup_and_teardown

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models import structural as st
from tests.statespace.shared_fixtures import rng  # pylint: disable=unused-import
from tests.statespace.test_utilities import load_nile_test_data

floatX = pytensor.config.floatX
nile = load_nile_test_data()
mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)


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

        ss_mod.build_statespace_graph(data=nile)
        names = ["x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
        for name, matrix in zip(names, ss_mod._insert_random_variables()):
            pm.Deterministic(name, matrix)

    return pymc_mod


@pytest.fixture(scope="session")
def ss_mod_no_exog(rng):
    ll = st.LevelTrend(name="trend", order=2, innovations_order=1)
    return ll.build()


@pytest.fixture(scope="session")
def ss_mod_no_exog_mv(rng):
    ll = st.LevelTrend(
        name="trend", order=2, innovations_order=1, observed_state_names=["y1", "y2"]
    )
    return ll.build()


@pytest.fixture(scope="session")
def ss_mod_no_exog_dt(rng):
    ll = st.LevelTrend(name="trend", order=2, innovations_order=1)
    return ll.build()


@pytest.fixture(scope="session")
def ss_mod_time_varying():
    """A minimal model with time-varying observation intercept (uses n_timesteps)."""

    class TimeVaryingInterceptModel(PyMCStateSpace):
        def __init__(self):
            super().__init__(k_states=1, k_endog=1, k_posdef=1)

        def make_symbolic_graph(self) -> None:
            self.ssm["transition", 0, 0] = 1.0
            self.ssm["design", 0, 0] = 1.0
            self.ssm["selection", 0, 0] = 1.0
            self.ssm["state_cov", 0, 0] = 1.0

            # Time-varying observation intercept: slope * arange(n_timesteps)
            slope = self.make_and_register_variable("slope", ())
            time_trend = slope * pt.arange(self.n_timesteps)
            self.ssm["obs_intercept"] = time_trend[:, None]
            self.ssm.declare_time_varying("obs_intercept")

        @property
        def param_names(self) -> list[str]:
            return ["slope"]

        @property
        def state_names(self) -> list[str]:
            return ["level"]

        @property
        def observed_states(self) -> list[str]:
            return ["level"]

        @property
        def shock_names(self) -> list[str]:
            return ["level"]

    return TimeVaryingInterceptModel()


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
def exog_data_mv(rng):
    # simulate data
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-05-01", end="2023-05-10", freq="D"),
            "x1": rng.choice(2, size=10, replace=True).astype(float),
            "y1": rng.normal(size=(10,)),
            "y2": rng.normal(size=(10,)),
        }
    )

    df.loc[[1, 3, 9], ["y1"]] = np.nan
    df.loc[[3, 5, 7], ["y2"]] = np.nan
    return df.set_index("date")


@pytest.fixture(scope="session")
def exog_ss_mod(exog_data):
    level_trend = st.LevelTrend(name="trend", order=1, innovations_order=[0])
    exog = st.Regression(
        name="exog",  # Name of this exogenous variable component
        innovations=False,  # Typically fixed effect (no stochastic evolution)
        state_names=exog_data[["x1"]].columns.tolist(),  # Only one exogenous variable now
    )

    combined_model = level_trend + exog
    return combined_model.build()


@pytest.fixture(scope="session")
def exog_ss_mod_mv(exog_data_mv):
    level_trend = st.LevelTrend(
        name="trend", order=1, innovations_order=[0], observed_state_names=["y1", "y2"]
    )
    exog = st.Regression(
        name="exog",  # Name of this exogenous variable component
        innovations=False,  # Typically fixed effect (no stochastic evolution)
        state_names=exog_data_mv[["x1"]].columns.tolist(),  # Only one exogenous variable now
        observed_state_names=["y1", "y2"],
    )

    combined_model = level_trend + exog
    return combined_model.build()


@pytest.fixture(scope="session")
def ss_mod_multi_component(rng):
    ll = st.LevelTrend(
        name="trend", order=2, innovations_order=1, observed_state_names=["y1", "y2"]
    )
    exog = st.Regression(
        name="exog",
        innovations=True,
        state_names=["x1"],
    )
    ar = st.Autoregressive(observed_state_names=["y1"])
    cycle = st.Cycle(cycle_length=2, observed_state_names=["y1", "y2"], innovations=True)
    season = st.TimeSeasonality(season_length=2, observed_state_names=["y1"], innovations=True)

    fseason = st.FrequencySeasonality(
        season_length=2, observed_state_names=["y1"], innovations=True
    )
    measure_error = st.MeasurementError(observed_state_names=["y1", "y2"])
    return (ll + exog + ar + cycle + season + fseason + measure_error).build()


@pytest.fixture(scope="session")
def exog_pymc_mod(exog_ss_mod, exog_data):
    # define pymc model
    with pm.Model(coords=exog_ss_mod.coords) as struct_model:
        P0_diag = pm.Gamma("P0_diag", alpha=2, beta=4, dims=["state"])
        P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=["state", "state_aux"])

        initial_trend = pm.Normal("initial_trend", mu=[0], sigma=[0.005], dims=["state_trend"])

        data_exog = pm.Data(
            "data_exog", exog_data["x1"].values[:, None], dims=["time", "state_exog"]
        )
        beta_exog = pm.Normal("beta_exog", mu=0, sigma=1, dims=["state_exog"])

        exog_ss_mod.build_statespace_graph(exog_data["y"])

    return struct_model


@pytest.fixture(scope="session")
def exog_pymc_mod_mv(exog_ss_mod_mv, exog_data_mv):
    # define pymc model
    with pm.Model(coords=exog_ss_mod_mv.coords) as struct_model:
        P0_diag = pm.Gamma("P0_diag", alpha=2, beta=4, dims=["state"])
        P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=["state", "state_aux"])

        initial_trend = pm.Normal(
            "initial_trend", mu=[0], sigma=[0.005], dims=["endog_trend", "state_trend"]
        )

        data_exog = pm.Data(
            "data_exog", exog_data_mv["x1"].values[:, None], dims=["time", "state_exog"]
        )
        beta_exog = pm.Normal("beta_exog", mu=0, sigma=1, dims=["endog_exog", "state_exog"])

        exog_ss_mod_mv.build_statespace_graph(exog_data_mv[["y1", "y2"]])

    return struct_model


@pytest.fixture(scope="session")
def pymc_mod_no_exog(ss_mod_no_exog, rng):
    y = pd.DataFrame(rng.normal(size=(100, 1)).astype(floatX), columns=["y"])

    with pm.Model(coords=ss_mod_no_exog.coords) as m:
        initial_trend = pm.Normal("initial_trend", dims=["state_trend"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["shock_trend"])
        ss_mod_no_exog.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def pymc_mod_no_exog_mv(ss_mod_no_exog_mv, rng):
    y = pd.DataFrame(rng.normal(size=(100, 2)).astype(floatX), columns=["y1", "y2"])

    with pm.Model(coords=ss_mod_no_exog_mv.coords) as m:
        trend_initial = pm.Normal("initial_trend", dims=["endog_trend", "state_trend"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog_mv.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        trend_sigma = pm.Exponential("sigma_trend", 1, dims=["endog_trend", "shock_trend"])
        ss_mod_no_exog_mv.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def pymc_mod_no_exog_mv_dt(ss_mod_no_exog_mv, rng):
    y = pd.DataFrame(
        rng.normal(size=(100, 2)).astype(floatX),
        columns=["y1", "y2"],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    with pm.Model(coords=ss_mod_no_exog_mv.coords) as m:
        trend_initial = pm.Normal("initial_trend", dims=["endog_trend", "state_trend"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog_mv.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        trend_sigma = pm.Exponential("sigma_trend", 1, dims=["endog_trend", "shock_trend"])
        ss_mod_no_exog_mv.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def pymc_mod_no_exog_dt(ss_mod_no_exog_dt, rng):
    y = pd.DataFrame(
        rng.normal(size=(100, 1)).astype(floatX),
        columns=["y"],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    with pm.Model(coords=ss_mod_no_exog_dt.coords) as m:
        initial_trend = pm.Normal("initial_trend", dims=["state_trend"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_no_exog_dt.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["shock_trend"])
        ss_mod_no_exog_dt.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def pymc_mod_time_varying(ss_mod_time_varying, rng):
    """PyMC model with time-varying observation intercept."""
    n_obs = 40
    y = rng.normal(size=(n_obs, 1)).astype(floatX)

    with pm.Model(coords=ss_mod_time_varying.coords) as m:
        slope = pm.Normal("slope", mu=0, sigma=1)
        P0 = pm.Deterministic(
            "P0", pt.eye(ss_mod_time_varying.k_states) * 1.0, dims=["state", "state_aux"]
        )
        x0 = pm.Normal("x0", dims=["state"])
        ss_mod_time_varying.build_statespace_graph(y)

    return m


@pytest.fixture(scope="module")
def idata(pymc_mod, rng, mock_pymc_sample):
    with pymc_mod:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)

    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_exog(exog_pymc_mod, rng, mock_pymc_sample):
    with exog_pymc_mod:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_exog_mv(exog_pymc_mod_mv, rng, mock_pymc_sample):
    with exog_pymc_mod_mv:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_no_exog(pymc_mod_no_exog, rng, mock_pymc_sample):
    with pymc_mod_no_exog:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_no_exog_mv(pymc_mod_no_exog_mv, rng, mock_pymc_sample):
    with pymc_mod_no_exog_mv:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_no_exog_mv_dt(pymc_mod_no_exog_mv_dt, rng, mock_pymc_sample):
    with pymc_mod_no_exog_mv_dt:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_no_exog_dt(pymc_mod_no_exog_dt, rng, mock_pymc_sample):
    with pymc_mod_no_exog_dt:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata


@pytest.fixture(scope="module")
def idata_time_varying(pymc_mod_time_varying, rng, mock_pymc_sample):
    """Inference data for time-varying model."""
    with pymc_mod_time_varying:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(draws=10, random_seed=rng)
    idata.update(idata_prior)
    return idata
