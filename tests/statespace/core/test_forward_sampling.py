import re

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_NAMES,
    MATRIX_NAMES,
    SMOOTHER_OUTPUT_NAMES,
)
from tests.statespace.test_utilities import (
    load_nile_test_data,
)

floatX = pytensor.config.floatX
nile = load_nile_test_data()
ALL_SAMPLE_OUTPUTS = MATRIX_NAMES + FILTER_OUTPUT_NAMES + SMOOTHER_OUTPUT_NAMES


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
            self.ssm.declare_time_varying("state_cov")

        @property
        def param_names(self) -> list[str]:
            return ["sigma_cov"]

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


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.filterwarnings("ignore:Skipping `CheckAndRaise` Op")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
def test_sample_filter_outputs(rng, exog_ss_mod, idata_exog):
    # Simple tests
    idata_filter_prior = exog_ss_mod.sample_filter_outputs(
        idata_exog, filter_output_names=None, group="prior"
    )

    specific_outputs = ["filtered_states", "filtered_covariances"]
    idata_filter_specific = exog_ss_mod.sample_filter_outputs(
        idata_exog, filter_output_names=specific_outputs
    )
    missing_outputs = np.setdiff1d(
        specific_outputs, [x for x in idata_filter_specific.posterior_predictive.data_vars]
    )

    assert missing_outputs.size == 0

    msg = "['filter_covariances' 'filter_states'] not a valid filter output name!"
    incorrect_outputs = ["filter_states", "filter_covariances"]
    with pytest.raises(ValueError, match=re.escape(msg)):
        exog_ss_mod.sample_filter_outputs(idata_exog, filter_output_names=incorrect_outputs)


class TestTimeVaryingTransition:
    """Tests for models with time-varying transition matrices (n_timesteps placeholder)."""

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    def test_sample_conditional_prior(self, ss_mod_time_varying, idata_time_varying):
        result = ss_mod_time_varying.sample_conditional_prior(idata_time_varying)
        assert "filtered_prior" in result
        assert "smoothed_prior" in result
        assert "predicted_prior" in result
        assert not np.any(np.isnan(result["filtered_prior"].values))

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    def test_sample_conditional_posterior(self, ss_mod_time_varying, idata_time_varying):
        result = ss_mod_time_varying.sample_conditional_posterior(idata_time_varying)
        assert "filtered_posterior" in result
        assert "smoothed_posterior" in result
        assert "predicted_posterior" in result
        assert not np.any(np.isnan(result["filtered_posterior"].values))

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    def test_sample_unconditional_prior(self, ss_mod_time_varying, idata_time_varying):
        result = ss_mod_time_varying.sample_unconditional_prior(idata_time_varying)
        assert "prior_latent" in result
        assert "prior_observed" in result
        assert not np.any(np.isnan(result["prior_latent"].values))

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    def test_sample_unconditional_posterior(self, ss_mod_time_varying, idata_time_varying):
        result = ss_mod_time_varying.sample_unconditional_posterior(idata_time_varying)
        assert "posterior_latent" in result
        assert "posterior_observed" in result
        assert not np.any(np.isnan(result["posterior_latent"].values))

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    @pytest.mark.filterwarnings("ignore:No start date provided")
    @pytest.mark.parametrize(
        "periods", [10, 50], ids=["shorter_than_training", "longer_than_training"]
    )
    def test_forecast(self, ss_mod_time_varying, idata_time_varying, periods):
        n_obs = 40  # must match pymc_mod_time_varying fixture
        result = ss_mod_time_varying.forecast(idata_time_varying, periods=periods)

        assert "forecast_latent" in result
        assert "forecast_observed" in result
        assert result["forecast_latent"].dims == ("chain", "draw", "time", "state")
        assert result["forecast_observed"].dims == ("chain", "draw", "time", "observed_state")
        assert result["forecast_latent"].shape[2] == periods
        assert not np.any(np.isnan(result["forecast_latent"].values))
        assert not np.any(np.isnan(result["forecast_observed"].values))

        # Value check: the model has y_t = d_t + Z @ x_t with Z=[[1]] and H=0 (no obs noise),
        # so forecast_observed - forecast_latent = d_t = slope * t.
        # Forecast matrices are phase-aligned: they continue from n_obs, not from 0.
        latent = result["forecast_latent"].values  # (chain, draw, time, state)
        observed = result["forecast_observed"].values  # (chain, draw, time, obs)
        slope = idata_time_varying.posterior["slope"].values  # (chain, draw)

        intercepts = observed[..., 0] - latent[..., 0]  # (chain, draw, time)
        expected = slope[..., None] * np.arange(n_obs, n_obs + periods)[None, None, :]

        assert_allclose(intercepts, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
    def test_impulse_response_function(self, ss_mod_time_varying, idata_time_varying):
        result = ss_mod_time_varying.impulse_response_function(
            idata_time_varying, n_steps=20, shock_size=1.0
        )
        assert "irf" in result
        assert result["irf"].shape[2] == 20
        assert not np.any(np.isnan(result["irf"].values))
