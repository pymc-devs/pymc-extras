from functools import partial

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from numpy.testing import assert_allclose
from pytensor.compile import SharedVariable
from pytensor.graph.traversal import graph_inputs

from tests.statespace.test_utilities import (
    load_nile_test_data,
    make_statespace_mod,
)

nile = load_nile_test_data()


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
    with pytest.raises(ValueError, match=r"Has this model been fit?"):
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
def test_finalize_scenario_dict(data_type, use_datetime_index, use_scenario_index):
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
        ValueError,
        match=r"This model was fit using exogenous data. Forecasting cannot be performed",
    ):
        ss_mod._validate_scenario_data(None)

    # Giving a list, tuple, or Series when a matrix of data is expected should always raise
    with pytest.raises(
        ValueError,
        match=r"Scenario data for variable 'a' has the wrong number of columns. Expected 2, got 1",
    ):
        for data_type in [list, tuple, pd.Series]:
            ss_mod._validate_scenario_data(data_type(np.zeros(10)))
            ss_mod._validate_scenario_data({"a": data_type(np.zeros(10))})

    # Providing irrevelant data raises
    with pytest.raises(
        ValueError,
        match="Scenario data provided for variable 'jk lol', which is not an exogenous variable",
    ):
        ss_mod._validate_scenario_data({"jk lol": np.zeros(10)})

    # Incorrect 2nd dimension of a non-dataframe
    with pytest.raises(
        ValueError,
        match=r"Scenario data for variable 'a' has the wrong number of columns. Expected 2, got 1",
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
        ("ss_mod_no_exog_mv", "idata_no_exog_mv", None, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv", -1, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv", 10, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv", 10, 21, None),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", None, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", -1, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", 10, None, 10),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", 10, "2020-01-21", None),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", "2020-03-01", "2020-03-11", None),
        ("ss_mod_no_exog_mv", "idata_no_exog_mv_dt", "2020-03-01", None, 10),
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
        "multivariate_default",
        "multivariate_negative",
        "multivariate_int",
        "multivariate_end",
        "multivariate_datetime_default",
        "multivariate_datetime_negative",
        "multivariate_datetime_int",
        "multivariate_datetime_int_end",
        "multivariate_datetime_datetime_end",
        "multivariate_datetime_datetime",
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
    level = components.forecast_latent.sel(state="trend[level]")
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
@pytest.mark.parametrize("start", [None, -1, 5])
def test_forecast_with_exog_data_mv(rng, exog_ss_mod_mv, idata_exog_mv, start):
    scenario = pd.DataFrame(np.zeros((10, 1)), columns=["x1"])
    scenario.iloc[5, 0] = 1e9

    forecast_idata = exog_ss_mod_mv.forecast(
        idata_exog_mv, start=start, periods=10, random_seed=rng, scenario=scenario
    )

    components = exog_ss_mod_mv.extract_components_from_idata(forecast_idata)
    level_y1 = components.forecast_latent.sel(state="trend[level[y1]]")
    level_y2 = components.forecast_latent.sel(state="trend[level[y2]]")
    betas_y1 = components.forecast_latent.sel(state=["exog[x1[y1]]"])
    betas_y2 = components.forecast_latent.sel(state=["exog[x1[y2]]"])

    scenario.index.name = "time"
    scenario_xr_y1 = (
        scenario.unstack()
        .to_xarray()
        .rename({"level_0": "state"})
        .assign_coords(state=["exog[x1[y1]]"])
    )

    scenario_xr_y2 = (
        scenario.unstack()
        .to_xarray()
        .rename({"level_0": "state"})
        .assign_coords(state=["exog[x1[y2]]"])
    )

    regression_effect_y1 = forecast_idata.forecast_observed.isel(observed_state=0) - level_y1
    regression_effect_expected_y1 = (betas_y1 * scenario_xr_y1).sum(dim=["state"])

    regression_effect_y2 = forecast_idata.forecast_observed.isel(observed_state=1) - level_y2
    regression_effect_expected_y2 = (betas_y2 * scenario_xr_y2).sum(dim=["state"])

    np.testing.assert_allclose(regression_effect_y1, regression_effect_expected_y1)
    np.testing.assert_allclose(regression_effect_y2, regression_effect_expected_y2)


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.filterwarnings("ignore:Skipping `CheckAndRaise` Op")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
def test_build_forecast_model(rng, exog_ss_mod, exog_pymc_mod, exog_data, idata_exog):
    data_before_build_forecast_model = {d.name: d.get_value() for d in exog_pymc_mod.data_vars}

    scenario = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-05-11", end="2023-05-20", freq="D"),
            "x1": rng.choice(2, size=10, replace=True).astype(float),
        }
    )
    scenario.set_index("date", inplace=True)

    time_index = exog_ss_mod._get_fit_time_index()
    t0, forecast_index = exog_ss_mod._build_forecast_index(
        time_index=time_index,
        start=exog_data.index[-1],
        end=scenario.index[-1],
        scenario=scenario,
    )

    # Fetched before the forecast model mutates any shared data.
    predicted = exog_ss_mod.sample_filter_outputs(
        idata_exog,
        filter_output_names=["predicted_states", "predicted_covariances"],
        group="posterior",
    ).posterior_predictive

    test_forecast_model = exog_ss_mod._build_forecast_model(
        time_index=time_index,
        t0=t0,
        forecast_index=forecast_index,
        scenario=scenario,
        filter_output="predicted",
        mvn_method="svd",
    )

    frozen_shared_inputs = [
        inpt
        for inpt in graph_inputs([test_forecast_model.x0_slice, test_forecast_model.P0_slice])
        if isinstance(inpt, SharedVariable)
        and not isinstance(inpt.get_value(), np.random.Generator)
    ]

    assert (
        len(frozen_shared_inputs) == 0
    )  # check there are no non-random generator SharedVariables in the frozen inputs

    unfrozen_shared_inputs = [
        inpt
        for inpt in graph_inputs([test_forecast_model.forecast_combined])
        if isinstance(inpt, SharedVariable)
        and not isinstance(inpt.get_value(), np.random.Generator)
    ]

    # Check that there is one (in this case) unfrozen shared input and it corresponds to the exogenous data
    assert len(unfrozen_shared_inputs) == 1
    assert unfrozen_shared_inputs[0].name == "data_exog"

    data_after_build_forecast_model = {d.name: d.get_value() for d in test_forecast_model.data_vars}

    with test_forecast_model:
        dummy_obs_data = np.zeros((len(forecast_index), exog_ss_mod.k_endog))
        pm.set_data(
            {"data_exog": scenario} | {"data": dummy_obs_data},
            coords={"data_time": np.arange(len(forecast_index))},
        )
        idata_forecast = pm.sample_posterior_predictive(
            idata_exog, var_names=["x0_slice", "P0_slice"]
        )

    np.testing.assert_allclose(
        unfrozen_shared_inputs[0].get_value(), scenario["x1"].values.reshape((-1, 1))
    )  # ensure the replaced data matches the exogenous data

    for k in data_before_build_forecast_model.keys():
        assert (  # check that the data needed to init the forecasts doesn't change
            data_before_build_forecast_model[k].mean() == data_after_build_forecast_model[k].mean()
        )

    # Check that the frozen states and covariances correctly match the sliced index
    np.testing.assert_allclose(
        predicted["predicted_covariances"].sel(time=t0).mean(("chain", "draw")).values,
        idata_forecast.posterior_predictive["P0_slice"].mean(("chain", "draw")).values,
    )
    np.testing.assert_allclose(
        predicted["predicted_states"].sel(time=t0).mean(("chain", "draw")).values,
        idata_forecast.posterior_predictive["x0_slice"].mean(("chain", "draw")).values,
    )


@pytest.mark.filterwarnings("ignore:Provided data contains missing values")
@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
@pytest.mark.filterwarnings("ignore:No time index found on the supplied data.")
@pytest.mark.filterwarnings("ignore:Skipping `CheckAndRaise` Op")
@pytest.mark.filterwarnings("ignore:No frequency was specific on the data's DateTimeIndex.")
def test_forecast_valid_index(exog_pymc_mod, exog_ss_mod, exog_data):
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
    forecasts = exog_ss_mod.forecast(
        idata, scenario=scenario, use_scenario_index=True, group="prior"
    )
    assert "forecast_latent" in forecasts
    assert "forecast_observed" in forecasts

    assert (forecasts.coords["time"].values == scenario["data_exog"].index.values).all()
    assert not np.any(np.isnan(forecasts.forecast_latent.values))
    assert not np.any(np.isnan(forecasts.forecast_observed.values))

    assert forecasts.forecast_latent.shape[2] == n_periods
    assert forecasts.forecast_observed.shape[2] == n_periods
