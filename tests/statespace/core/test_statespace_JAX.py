import warnings

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc.model.transform.optimization import freeze_dims_and_data

from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_NAMES,
    MATRIX_NAMES,
    SMOOTHER_OUTPUT_NAMES,
)
from tests.statespace.core.test_statespace import (  # pylint: disable=unused-import
    exog_ss_mod,
    ss_mod,
)
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import load_nile_test_data

pytest.importorskip("jax")
pytest.importorskip("numpyro")


floatX = pytensor.config.floatX
nile = load_nile_test_data()
ALL_SAMPLE_OUTPUTS = MATRIX_NAMES + FILTER_OUTPUT_NAMES + SMOOTHER_OUTPUT_NAMES


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
def exog_pymc_mod(exog_ss_mod, rng):
    y = rng.normal(size=(100, 1)).astype(floatX)
    X = rng.normal(size=(100, 3)).astype(floatX)

    with pm.Model(coords=exog_ss_mod.coords) as m:
        exog_data = pm.Data("data_exog", X)
        initial_trend = pm.Normal("initial_trend", dims=["trend_state"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(exog_ss_mod.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])

        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["trend_shock"])
        exog_ss_mod.build_statespace_graph(y)

    return m


@pytest.fixture(scope="session")
def idata(pymc_mod, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pymc_mod:
            idata = pm.sample(
                draws=10,
                tune=1,
                chains=1,
                random_seed=rng,
                nuts_sampler="numpyro",
                progressbar=False,
            )
        with freeze_dims_and_data(pymc_mod):
            idata_prior = pm.sample_prior_predictive(
                samples=10, random_seed=rng, compile_kwargs={"mode": "JAX"}
            )

    idata.extend(idata_prior)
    return idata


@pytest.fixture(scope="session")
def idata_exog(exog_pymc_mod, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        with exog_pymc_mod:
            idata = pm.sample(
                draws=10,
                tune=1,
                chains=1,
                random_seed=rng,
                nuts_sampler="numpyro",
                progressbar=False,
            )
        with freeze_dims_and_data(pymc_mod):
            idata_prior = pm.sample_prior_predictive(
                samples=10, random_seed=rng, compile_kwargs={"mode": "JAX"}
            )

    idata.extend(idata_prior)
    return idata


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("matrix", ALL_SAMPLE_OUTPUTS)
def test_no_nans_in_sampling_output(ss_mod, group, matrix, idata):
    assert not np.any(np.isnan(idata[group][matrix].values))


@pytest.mark.parametrize("group", ["prior", "posterior"])
@pytest.mark.parametrize("kind", ["conditional", "unconditional"])
def test_sampling_methods(group, kind, ss_mod, idata, rng):
    f = getattr(ss_mod, f"sample_{kind}_{group}")
    with pytest.warns(UserWarning, match="The RandomType SharedVariables"):
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


@pytest.mark.parametrize("filter_output", ["predicted", "filtered", "smoothed"])
def test_forecast(filter_output, ss_mod, idata, rng):
    time_idx = idata.posterior.coords["time"].values

    with pytest.warns(UserWarning, match="The RandomType SharedVariables"):
        forecast_idata = ss_mod.forecast(
            idata, start=time_idx[-1], periods=10, filter_output=filter_output, random_seed=rng
        )

    assert forecast_idata.coords["time"].values.shape == (10,)
    assert forecast_idata.forecast_latent.dims == ("chain", "draw", "time", "state")
    assert forecast_idata.forecast_observed.dims == ("chain", "draw", "time", "observed_state")

    assert not np.any(np.isnan(forecast_idata.forecast_latent.values))
    assert not np.any(np.isnan(forecast_idata.forecast_observed.values))
