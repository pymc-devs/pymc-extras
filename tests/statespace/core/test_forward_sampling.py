import re

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import (
    LONG_MATRIX_NAMES,
    MATRIX_DIMS,
    MATRIX_NAMES,
    SHORT_NAME_TO_LONG,
    TIME_DIM,
)


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("matrix", MATRIX_NAMES)
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

    unconditional = ss_mod.sample_unconditional_prior(prior)
    conditional = ss_mod.sample_conditional_prior(prior)

    n_time = empty_data.shape[0]
    for name in ["prior_latent", "prior_observed"]:
        assert name in unconditional
        assert unconditional[name].sizes["time"] == n_time
        assert np.all(np.isfinite(unconditional[name].values))

    for name in ["filtered_prior", "predicted_prior", "smoothed_prior"]:
        assert name in conditional
        assert conditional[name].sizes["time"] == n_time
        assert np.all(np.isfinite(conditional[name].values))


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

    for name, values in idata_filter_prior.posterior_predictive.data_vars.items():
        assert not np.any(np.isnan(values)), f"{name} contains NaNs"

    specific_outputs = ["filtered_states", "filtered_covariances"]
    idata_filter_specific = exog_ss_mod.sample_filter_outputs(
        idata_exog, filter_output_names=specific_outputs
    )
    missing_outputs = np.setdiff1d(
        specific_outputs, [x for x in idata_filter_specific.posterior_predictive.data_vars]
    )

    assert missing_outputs.size == 0

    msg = "['filter_covariances', 'filter_states'] not a valid filter output name!"
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


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
def test_sample_statespace_matrices_keeps_its_dims(exog_ss_mod, idata_exog):
    """Every sampled matrix carries the dims MATRIX_DIMS declares for it.

    The dims are looked up against the fit coords and silently become ``None`` for any coord that is
    missing, so a coord regression would degrade the output rather than fail.
    """
    matrix_idata = exog_ss_mod.sample_statespace_matrices(
        idata_exog, matrix_names=LONG_MATRIX_NAMES, group="prior"
    )
    sampled = matrix_idata.posterior_predictive

    for short_name in MATRIX_NAMES:
        long_name = SHORT_NAME_TO_LONG[short_name]
        expected = ("chain", "draw", *MATRIX_DIMS[short_name])
        if long_name in exog_ss_mod.ssm.time_varying_names:
            expected = ("chain", "draw", TIME_DIM, *MATRIX_DIMS[short_name])

        assert sampled[long_name].dims == expected, f"{long_name} lost its dims"
        assert np.all(np.isfinite(sampled[long_name].values)), f"{long_name} is not finite"


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
@pytest.mark.parametrize(
    "mod_name, idata_name",
    [("ss_mod", "idata"), ("exog_ss_mod", "idata_exog")],
    ids=["no_exog", "with_exog"],
)
def test_sample_statespace_matrices_defaults_to_every_matrix(mod_name, idata_name, request):
    """The default names must not collide with the x0 and P0 parameters models declare."""
    ss_mod = request.getfixturevalue(mod_name)
    idata = request.getfixturevalue(idata_name)

    sampled = ss_mod.sample_statespace_matrices(
        idata, matrix_names=None, group="prior"
    ).posterior_predictive

    assert set(sampled.data_vars) == set(LONG_MATRIX_NAMES)


def test_sample_statespace_matrices_rejects_unknown_names(ss_mod, idata):
    """A mistyped name fails naming itself, rather than as a KeyError from inside pymc."""
    with pytest.raises(ValueError, match="not a valid statespace matrix name"):
        ss_mod.sample_statespace_matrices(idata, matrix_names="tranistion", group="prior")

    # A valid name alongside an invalid one must not mask the mistake.
    with pytest.raises(ValueError, match="not a valid statespace matrix name"):
        ss_mod.sample_statespace_matrices(
            idata, matrix_names=["transition", "bogus"], group="prior"
        )
