import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import pymc_extras.statespace.models.structural as st

from pymc_extras.statespace.core.fit_recovery import (
    coords_from_idata,
    data_from_idata,
    dims_from_idata,
    exog_from_idata,
    verify_group,
)
from pymc_extras.statespace.utils.constants import MISSING_FILL

floatX = pytensor.config.floatX


@pytest.fixture(scope="module")
def exog_fit():
    """A fitted model with exogenous data, its inference data, and what it was fit on."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    data = pd.DataFrame(rng.normal(size=(40, 1)).astype(floatX), columns=["y"], index=index)
    exog = pd.DataFrame(rng.normal(size=(40, 2)).astype(floatX), columns=["a", "b"], index=index)

    mod = (
        st.LevelTrend(order=2, innovations_order=1)
        + st.Regression(name="exog", state_names=["a", "b"], innovations=False)
    ).build(verbose=False)

    with pm.Model(coords=mod.coords):
        pm.Normal("initial_level_trend", dims=["state_level_trend"])
        pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
        pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
        pm.Normal("beta_exog", dims=["state_exog"])
        pm.Data("data_exog", exog.values)
        mod.build_statespace_graph(data)
        idata = pm.sample_prior_predictive(draws=5, random_seed=1)

    return mod, idata, data, exog


def test_recovered_coords_cover_every_dim_the_model_uses(exog_fit):
    """Template and idata are complements, so together they cover the whole system."""
    mod, idata, data, _ = exog_fit

    recovered = coords_from_idata(mod, idata, "observed_data")

    assert set(mod.coords) <= set(recovered)
    assert len(recovered["time"]) == len(data)


def test_time_comes_only_from_the_idata(exog_fit):
    """The template cannot know the dataset's time index."""
    mod, idata, data, _ = exog_fit

    assert "time" not in mod.coords
    assert len(coords_from_idata(mod, idata, "observed_data")["time"]) == len(data)


def test_recovered_time_keeps_its_timestamps(exog_fit):
    """``_get_fit_time_index`` tests ``isinstance(time[0], pd.Timestamp)`` to recover the freq.

    A datetime coord read straight off the idata is ``datetime64``, which fails that test and
    silently costs the forecast index its frequency.
    """
    mod, idata, *_ = exog_fit

    time = coords_from_idata(mod, idata, "observed_data")["time"]

    assert isinstance(time[0], pd.Timestamp)
    assert pd.DatetimeIndex(time).inferred_freq == "D"


@pytest.mark.parametrize("dim", ["shock", "shock_aux", "observed_state_aux"])
def test_matrix_dims_come_only_from_the_template(exog_fit, dim):
    """These are attached to no variable, so no idata group carries them.

    They are the dims ``MATRIX_DIMS`` needs for R, Q and H; recovering from the idata alone
    would leave those matrices correctly shaped with their dims silently dropped.
    """
    mod, idata, *_ = exog_fit

    assert dim not in {name for group in idata.groups for name in idata[group].coords}
    assert dim in coords_from_idata(mod, idata, "observed_data")


def test_a_merged_forecast_does_not_redefine_the_fit_time(exog_fit):
    """xarray only aligns a group against its ancestors, so sibling groups may disagree.

    Merging a forecast back into the idata is ordinary, and it lands a shorter ``time`` in a
    group that sorts after ``observed_data``.
    """
    mod, idata, *_ = exog_fit
    merged = idata.copy()
    merged["/forecast"] = idata["/observed_data"].dataset.isel(time=slice(0, 3))

    assert len(merged["/forecast"].coords["time"]) == 3
    assert len(coords_from_idata(mod, merged, "observed_data")["time"]) == 40


@pytest.mark.parametrize(
    ("recover", "group"),
    [(coords_from_idata, "posterior"), (dims_from_idata, "posterior_predictive")],
    ids=["coords", "dims"],
)
def test_a_missing_group_names_what_is_there(exog_fit, recover, group):
    """A trimmed idata should say what it holds, not recover a silently incomplete answer."""
    mod, idata, *_ = exog_fit

    with pytest.raises(ValueError, match=f"no {group!r} group"):
        recover(mod, idata, group)


def test_explicit_coords_win(exog_fit):
    """The override exists for idata that carries no time index of its own."""
    mod, idata, *_ = exog_fit

    recovered = coords_from_idata(mod, idata, "observed_data", coords={"time": np.arange(7)})

    assert len(recovered["time"]) == 7


def test_recovered_dims_match_the_fit(exog_fit):
    """Only the parameters are recovered -- they are all ``build_dummy_graph`` asks about."""
    mod, idata, *_ = exog_fit

    recovered = dims_from_idata(mod, idata, "prior")

    assert recovered == {
        "initial_level_trend": ["state_level_trend"],
        "sigma_level_trend": ["shock_level_trend"],
        "beta_exog": ["state_exog"],
    }


def test_generated_dims_are_dropped(exog_fit):
    """``P0`` is registered without dims, so PyMC generates P0_dim_0 and P0_dim_1 for it."""
    mod, idata, *_ = exog_fit

    assert "P0_dim_0" in idata["/prior"].coords
    assert "P0" not in dims_from_idata(mod, idata, "prior")


def test_a_user_dim_that_only_resembles_the_generated_form_survives():
    """Only ``{name}_dim_{integer}`` is dropped, not every dim whose name looks like it."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(size=(12, 1)).astype(floatX),
        columns=["y"],
        index=pd.date_range("2020-01-01", periods=12, freq="D"),
    )
    mod = st.LevelTrend(order=2, innovations_order=1).build(verbose=False)

    with pm.Model(coords=mod.coords | {"initial_level_trend_dim_zero": ["a", "b"]}):
        pm.Normal("initial_level_trend", dims=["initial_level_trend_dim_zero"])
        pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
        pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
        mod.build_statespace_graph(data)
        idata = pm.sample_prior_predictive(draws=3, random_seed=1)

    recovered = dims_from_idata(mod, idata, "prior")

    assert recovered["initial_level_trend"] == ["initial_level_trend_dim_zero"]


def test_recovered_data_matches_the_fit(exog_fit):
    """Values and index round-trip; the sentinel stands where the fit saw ``nan``."""
    mod, idata, data, exog = exog_fit

    recovered = data_from_idata(idata, "constant_data")

    assert recovered.shape == data.shape
    assert recovered.index.equals(data.index)
    assert recovered.index.freq == data.index.freq
    np.testing.assert_allclose(recovered.values, data.values)


def test_recovered_exog_matches_the_fit(exog_fit):
    """The fixture registers its exogenous data without dims, so none should be recovered."""
    mod, idata, data, exog = exog_fit

    recovered = exog_from_idata(mod, idata, "constant_data")

    assert recovered.keys() == {"data_exog"}
    entry = recovered["data_exog"]
    assert entry["name"] == "data_exog"
    assert entry["dims"] is None
    np.testing.assert_allclose(entry["value"], exog.values)


def test_recovered_exog_keeps_declared_dims():
    """Dims the user did give survive, unlike the generated ones."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=15, freq="D")
    data = pd.DataFrame(rng.normal(size=(15, 1)).astype(floatX), columns=["y"], index=index)
    exog = pd.DataFrame(rng.normal(size=(15, 2)).astype(floatX), columns=["a", "b"], index=index)

    mod = (
        st.LevelTrend(order=2, innovations_order=1)
        + st.Regression(name="exog", state_names=["a", "b"], innovations=False)
    ).build(verbose=False)

    with pm.Model(coords=mod.coords):
        pm.Normal("initial_level_trend", dims=["state_level_trend"])
        pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
        pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
        pm.Normal("beta_exog", dims=["state_exog"])
        pm.Data("data_exog", exog.values, dims=["time", "state_exog"])
        mod.build_statespace_graph(data)
        idata = pm.sample_prior_predictive(draws=3, random_seed=1)

    recovered = exog_from_idata(mod, idata, "constant_data")

    assert recovered["data_exog"]["dims"] == ["time", "state_exog"]


def test_recovered_exog_can_be_registered(exog_fit):
    """The result is ``pm.Data`` keywords, so it must splat straight back into a model."""
    mod, idata, data, exog = exog_fit
    recovered = exog_from_idata(mod, idata, "constant_data")

    with pm.Model(coords=coords_from_idata(mod, idata, "observed_data")):
        registered = {name: pm.Data(**entry) for name, entry in recovered.items()}

    assert registered["data_exog"].get_value().shape == exog.shape


def test_missing_observed_data_names_what_the_group_holds(exog_fit):
    mod, idata, *_ = exog_fit

    with pytest.raises(ValueError, match="holds no 'data' variable"):
        data_from_idata(idata, "prior")


@pytest.mark.parametrize("group", ["posterior_predictive", "observed_data", "Posterior", ""])
def test_verify_group_rejects_groups_parameters_cannot_come_from(group):
    """It guards the public ``group=`` argument, so it must reject near-misses too."""
    with pytest.raises(ValueError, match='must be one of "prior" or "posterior"'):
        verify_group(group)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
def test_recovered_data_carries_the_sentinel_where_the_fit_saw_nan():
    """The substitution happens before the data is recorded, so ``nan`` cannot come back.

    That is only safe because the filter derives its missing mask from the sentinel rather than
    from ``nan``, which this pins by filtering both and comparing the log-likelihood. Only the
    raw fit warns about imputation, since the recovered frame has no ``nan`` left to impute.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=25, freq="D")
    values = rng.normal(size=(25, 1)).astype(floatX)
    values[[3, 11, 12]] = np.nan
    raw = pd.DataFrame(values, columns=["y"], index=index)

    def fit_and_score(frame):
        mod = st.LevelTrend(order=2, innovations_order=1).build(verbose=False)
        with pm.Model(coords=mod.coords) as pymc_mod:
            pm.Normal("initial_level_trend", dims=["state_level_trend"])
            pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
            pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
            mod.build_statespace_graph(frame)
            idata = pm.sample_prior_predictive(draws=3, random_seed=1)
            return idata, pymc_mod.compile_logp()(pymc_mod.initial_point())

    idata, raw_logp = fit_and_score(raw)
    recovered = data_from_idata(idata, "constant_data")

    assert np.isnan(recovered.values).sum() == 0
    assert (recovered.values[[3, 11, 12]] == MISSING_FILL).all()
    assert fit_and_score(recovered)[1] == raw_logp
