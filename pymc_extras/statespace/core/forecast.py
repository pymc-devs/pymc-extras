import logging

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.util import RandomState
from pytensor.graph.replace import graph_replace
from xarray import DataTree

from pymc_extras.statespace.core.fit_recovery import (
    coords_from_idata,
    data_from_idata,
    dims_from_idata,
    exog_from_idata,
    verify_group,
)
from pymc_extras.statespace.filters.distributions import LinearGaussianStateSpace
from pymc_extras.statespace.filters.utilities import scan_sequence_names
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    FILTER_OUTPUT_TYPES,
    MATRIX_NAMES,
    OBS_STATE_DIM,
    SHORT_NAME_TO_LONG,
    TIME_DIM,
)

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace

_log = logging.getLogger("pymc.experimental.statespace")


def _validate_filter_arg(filter_arg):
    if filter_arg.lower() not in FILTER_OUTPUT_TYPES:
        raise ValueError(
            f"filter_output should be one of {', '.join(FILTER_OUTPUT_TYPES)}, received {filter_arg}"
        )


def _validate_forecast_args(
    time_index: pd.RangeIndex | pd.DatetimeIndex,
    start: int | pd.Timestamp,
    periods: int | None = None,
    end: int | pd.Timestamp = None,
    scenario: pd.DataFrame | np.ndarray | None = None,
    use_scenario_index: bool = False,
    verbose: bool = True,
):
    if isinstance(start, pd.Timestamp) and start not in time_index:
        raise ValueError("Datetime start must be in the data index used to fit the model.")
    elif isinstance(start, int):
        if abs(start) > len(time_index):
            raise ValueError(
                "Integer start must be within the range of the data index used to fit the model."
            )
    if periods is None and end is None and not use_scenario_index:
        raise ValueError("Must specify one of either periods or end unless use_scenario_index=True")
    if periods is not None and end is not None:
        raise ValueError("Must specify exactly one of either periods or end")
    if scenario is None and use_scenario_index:
        raise ValueError("use_scenario_index=True requires a scenario to be provided.")
    if scenario is not None and use_scenario_index:
        if isinstance(scenario, dict):
            first_df = next(
                (df for df in scenario.values() if isinstance(df, pd.DataFrame | pd.Series)),
                None,
            )
            if first_df is None:
                raise ValueError(
                    "use_scenario_index=True requires a scenario to be a DataFrame or Series."
                )
        elif not isinstance(scenario, pd.DataFrame | pd.Series):
            raise ValueError(
                "use_scenario_index=True requires a scenario to be a DataFrame or Series."
            )
    if use_scenario_index and any(arg is not None for arg in [start, end, periods]) and verbose:
        _log.warning(
            "start, end, and periods arguments are ignored when use_scenario_index is True. Pass only "
            "one or the other to avoid this warning, or pass verbose = False."
        )


def _get_fit_time_index(
    ss_mod: "PyMCStateSpace", idata: DataTree
) -> pd.RangeIndex | pd.DatetimeIndex:
    time_index = coords_from_idata(ss_mod, idata, "observed_data").get(TIME_DIM, None)
    if time_index is None:
        raise ValueError(
            "No time dimension found on coordinates used to fit the model. Has this model been fit?"
        )

    if isinstance(time_index[0], pd.Timestamp):
        time_index = pd.DatetimeIndex(time_index)
        time_index.freq = time_index.inferred_freq
    else:
        time_index = np.array(time_index)

    return time_index


def _validate_scenario_data(
    ss_mod: "PyMCStateSpace",
    scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None,
    coords: dict[str, Sequence],
    name: str | None = None,
    verbose=True,
):
    """
    Validate the scenario data provided to the forecast method by checking that it has the correct shape and
    dimensions.

    Parameters
    ----------
    scenario
    name
    verbose

    Returns
    -------
    scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray]
        Scenario data, validated and potentially modified.

    """
    if not ss_mod._needs_exog_data:
        return scenario

    var_to_dims = {key: info["dims"][1:] for key, info in ss_mod.data_info.items()}

    if any(len(dims) > 1 for dims in var_to_dims.values()):
        raise NotImplementedError(">2d exogenous data is not yet supported.")
    var_to_coords = {var: coords[dim[0]] for var, dim in var_to_dims.items() if dim[0] in coords}

    if ss_mod._needs_exog_data and scenario is None:
        exog_str = ",".join(ss_mod.data_names)
        suffix = "s" if len(exog_str) > 1 else ""
        raise ValueError(
            f"This model was fit using exogenous data. Forecasting cannot be performed without "
            f"providing scenario data for the following variable{suffix}: {exog_str}"
        )

    if isinstance(scenario, dict):
        for name, data in scenario.items():
            if name not in ss_mod.data_names:
                raise ValueError(
                    f"Scenario data provided for variable '{name}', which is not an exogenous variable "
                    f"used to fit the model."
                )

            # Recursively call this function to trigger the non-dictionary branch of the checks on each object
            # inside the dictionary
            scenario[name] = ss_mod._validate_scenario_data(data, coords, name)

        # The provided dictionary might be a mix of numpy arrays and dataframes if the user is truly horrible.
        # For checking shapes, the first object will always be good enough. But we also need to make sure all the
        # indices agree, so we grab the first dataframe (which might not exist, but that's OK)
        first_scenario = next(iter(scenario.values()))
        first_df = next((df for df in scenario.values() if isinstance(df, pd.DataFrame)), None)

        if not all(data.shape[0] == first_scenario.shape[0] for data in scenario.values()):
            raise ValueError(
                "Scenario data must have the same number of time steps for all variables."
            )

        if first_df is not None and not all(
            df.index.equals(first_df.index)
            for df in scenario.values()
            if isinstance(df, pd.DataFrame)
        ):
            raise ValueError("Scenario data must have the same index for all variables.")

        return scenario

    elif isinstance(scenario, pd.Series | pd.DataFrame | np.ndarray | list | tuple):
        # A user might be lazy and pass a simple list when there is only one exogenous variable.
        if isinstance(scenario, list | tuple) or (
            isinstance(scenario, np.ndarray) and scenario.ndim == 1
        ):
            scenario = np.array(scenario).reshape(-1, 1)

        if name is None:
            # name should only be None on the first non-recursive call. We only arrive to this branch in that case
            # if a non-dictionary was passed, which in turn should only happen if only a single exogenous data
            # needs to be set.
            if len(ss_mod.data_names) > 1:
                raise ValueError(
                    "Multiple exogenous variables were used to fit the model. Provide a dictionary of "
                    "scenario data instead."
                )
            name = ss_mod.data_names[0]

        # Omit dataframe from this basic shape check so we can give more detailed information about missing columns
        # in the next check
        if not isinstance(scenario, pd.DataFrame | pd.Series) and scenario.shape[1] != len(
            var_to_coords[name]
        ):
            raise ValueError(
                f"Scenario data for variable '{name}' has the wrong number of columns. Expected "
                f"{len(var_to_coords[name])}, got {scenario.shape[1]}"
            )

        if isinstance(scenario, pd.Series):
            if len(var_to_coords[name]) > 1:
                raise ValueError(
                    f"Scenario data for variable '{name}' has the wrong number of columns. Expected "
                    f"{len(var_to_coords[name])}, got 1"
                )

        if isinstance(scenario, pd.DataFrame):
            expected_cols = var_to_coords[name]
            cols = scenario.columns
            missing_columns = sorted(list(set(expected_cols) - set(cols)))
            if len(missing_columns) > 0:
                suffix = "s" if len(missing_columns) > 1 else ""
                raise ValueError(
                    f"Scenario data for variable '{name}' is missing the following column{suffix}: "
                    f"{', '.join(missing_columns)}"
                )

            extra_columns = sorted(list(set(cols) - set(expected_cols)))
            if len(extra_columns) > 0:
                suffix = "s" if len(extra_columns) > 1 else ""
                verb = "is" if len(extra_columns) == 1 else "are"
                raise ValueError(
                    f"Scenario data for variable '{name}' contains the following extra column{suffix} "
                    f"that {verb} not used by the model: "
                    f"{', '.join(extra_columns)}"
                )

            if not (a == b for a, b in zip(expected_cols, cols)) and verbose:
                _log.warning(
                    f"Scenario data for {name} has a different column order than the data used to fit the "
                    f"model. Columns will be automatically re-ordered. Ensure consistent ordering to avoid "
                    f"silent errors."
                )
                scenario = scenario[expected_cols]

        return scenario


def _build_forecast_index(
    time_index: pd.RangeIndex | pd.DatetimeIndex,
    start: int | pd.Timestamp | None = None,
    end: int | pd.Timestamp = None,
    periods: int | None = None,
    use_scenario_index: bool = False,
    scenario: pd.DataFrame | np.ndarray | None = None,
) -> tuple[int | pd.Timestamp, pd.RangeIndex | pd.DatetimeIndex]:
    """
    Construct a pandas Index for the requested forecast horizon.

    Parameters
    ----------
    time_index: pd.RangeIndex or pd.DatetimeIndex
        Index of the data used to fit the model
    start: int or pd.Timestamp, optional
        Date from which to begin forecasting. If using a datetime index, integer start will be interpreted
        as a positional index. Otherwise, start must be found inside the time_index
    end: int or pd.Timestamp, optional
        Date at which to end forecasting. If using a datetime index, end must be a timestamp.
    periods: int, optional
        Number of periods to forecast
    scenario:  pd.DataFrame, np.ndarray, optional
        Scenario data to use for forecasting. If provided, the index of the scenario data will be used as the
        forecast index. If provided, start, end, and periods will be ignored.
    use_scenario_index: bool, default False
        If True, the index of the scenario data will be used as the forecast index.


    Returns
    -------
    start: int | pd.TimeStamp
        The starting date index or time step from which to generate the forecasts.

    forecast_index: pd.DatetimeIndex or pd.RangeIndex
        Index for the forecast results
    """

    def get_or_create_index(x, time_index, start=None):
        if isinstance(x, pd.DataFrame | pd.Series):
            return x.index
        elif isinstance(x, dict):
            return get_or_create_index(next(iter(x.values())), time_index, start)
        elif isinstance(x, np.ndarray | list | tuple):
            if start is None:
                raise ValueError(
                    "Provided scenario has no index and no start date was provided. This combination "
                    "is ambiguous. Please provide a start date, or add an index to the scenario."
                )
            is_datetime_index = isinstance(time_index, pd.DatetimeIndex)
            n = x.shape[0] if isinstance(x, np.ndarray) else len(x)

            if isinstance(start, int):
                start = time_index[start]
            if is_datetime_index:
                return pd.date_range(start, periods=n, freq=time_index.freq)
            return pd.RangeIndex(start, n + start, step=1, dtype="int")

        else:
            raise ValueError(f"{type(x)} is not a valid type for scenario data.")

    x0_idx = None

    if use_scenario_index:
        forecast_index = get_or_create_index(scenario, time_index, start)
        is_datetime = isinstance(forecast_index, pd.DatetimeIndex)

        # If the user provided an index, we want to take it as-is (without removing the start value). Instead,
        # step one back and use this as the start value.
        delta = forecast_index.freq if is_datetime else 1
        x0_idx = forecast_index[0] - delta

    else:
        # Otherwise, build an index. It will be a DateTime index if we have all the necessary information, otherwise
        # use a range index.
        is_datetime = isinstance(time_index, pd.DatetimeIndex)
        forecast_index = None

        if is_datetime:
            freq = time_index.freq
            if isinstance(start, int):
                start = time_index[start]
            if isinstance(end, int):
                raise ValueError(
                    "end must be a timestamp if using a datetime index. To specify a number of "
                    "timesteps from the start date, use the periods argument instead."
                )
            if end is not None:
                forecast_index = pd.date_range(start, end=end, freq=freq)
            if periods is not None:
                # date_range includes both the start and end date, but we're going to pop off the start later
                # (it will be interpreted as x0). So we need to add 1 to the periods so the user gets "periods"
                # number of forecasts back
                forecast_index = pd.date_range(start, periods=periods + 1, freq=freq)

        else:
            # If the user provided a positive integer as start, directly interpret it as the start time. If its
            # negative, interpret it as a positional index.
            if start < 0:
                start = time_index[start]
            if end is not None:
                # end is inclusive, matching the datetime branch; the start is popped off below.
                forecast_index = pd.RangeIndex(start, end + 1, step=1, dtype="int")
            if periods is not None:
                forecast_index = pd.RangeIndex(start, start + periods + 1, step=1, dtype="int")

    if is_datetime:
        if forecast_index.freq != time_index.freq:
            raise ValueError(
                "The frequency of the forecast index must match the frequency on the data used "
                f"to fit the model. Got {forecast_index.freq}, expected {time_index.freq}"
            )

    if x0_idx is None:
        x0_idx, forecast_index = forecast_index[0], forecast_index[1:]
    if x0_idx in forecast_index:
        raise ValueError("x0_idx should not be in the forecast index")
    if x0_idx not in time_index:
        raise ValueError("start must be in the data index used to fit the model.")

    # The starting value should not be included in the forecast index. It will be used only to define x0 and P0,
    # and no forecast will be associated with it.
    return x0_idx, forecast_index


def _finalize_scenario_initialization(
    ss_mod: "PyMCStateSpace",
    scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None,
    forecast_index: pd.RangeIndex | pd.DatetimeIndex,
    coords: dict[str, Sequence],
    name=None,
):
    if not ss_mod.data_info:
        return scenario

    var_to_dims = {key: info["dims"][1:] for key, info in ss_mod.data_info.items()}

    if any(len(dims) > 1 for dims in var_to_dims.values()):
        raise NotImplementedError(">2d exogenous data is not yet supported.")
    var_to_coords = {var: coords[dim[0]] for var, dim in var_to_dims.items() if dim[0] in coords}

    if scenario is None:
        return scenario

    if isinstance(scenario, dict):
        for name, data in scenario.items():
            scenario[name] = ss_mod._finalize_scenario_initialization(
                data, forecast_index, coords, name
            )
        return scenario

    # This was already checked as valid
    name = ss_mod.data_names[0] if name is None else name

    # Small tidying up in the case we just have a single scenario that's already a dataframe.
    if isinstance(scenario, pd.DataFrame | pd.Series):
        if isinstance(scenario, pd.Series):
            scenario = scenario.to_frame(name=var_to_coords[name][0])
        if not scenario.index.equals(forecast_index):
            scenario.index = forecast_index

    # lists and tuples were handled during validation, along with shape check, so just cast arrays to dataframes
    # with the correct index and columns
    if isinstance(scenario, np.ndarray):
        scenario = pd.DataFrame(scenario, index=forecast_index, columns=var_to_coords[name])

    return scenario


def _build_forecast_model(
    ss_mod: "PyMCStateSpace",
    idata,
    group,
    time_index,
    t0,
    forecast_index,
    scenario,
    filter_output,
    mvn_method,
):
    filter_time_dim = TIME_DIM
    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    fit_dims = dims_from_idata(ss_mod, idata, group)
    fit_exog = exog_from_idata(ss_mod, idata, "constant_data")
    temp_coords = fit_coords.copy()

    trajectory_dims = None
    if all([dim in temp_coords for dim in [filter_time_dim, ALL_STATE_DIM, OBS_STATE_DIM]]):
        trajectory_dims = [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]

    t0_idx = np.flatnonzero(time_index == t0)[0]

    temp_coords["data_time"] = time_index
    temp_coords[TIME_DIM] = forecast_index

    mu_dims, cov_dims = None, None
    if all([dim in fit_coords for dim in [TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM]]):
        mu_dims = ["data_time", ALL_STATE_DIM]
        cov_dims = ["data_time", ALL_STATE_DIM, ALL_STATE_AUX_DIM]

    with pm.Model(coords=temp_coords) as forecast_model:
        unpinned_matrices, grouped_outputs = ss_mod._kalman_filter_outputs_from_dummy_graph(
            data_from_idata(idata, "constant_data"),
            coords=fit_coords,
            dims=fit_dims,
            exog=fit_exog,
            data_dims=["data_time", OBS_STATE_DIM],
        )

        group_idx = FILTER_OUTPUT_TYPES.index(filter_output)
        mu, cov = grouped_outputs[group_idx]

        sub_dict = {
            data_var: pt.as_tensor_variable(data_var.get_value(), name="data")
            for data_var in forecast_model.data_vars
        }

        missing_data_vars = np.setdiff1d(
            ar1=[*ss_mod.data_names, "data"], ar2=[k.name for k, _ in sub_dict.items()]
        )
        if missing_data_vars.size > 0:
            raise ValueError(f"{missing_data_vars} data used for fitting not found!")

        mu_frozen, cov_frozen = graph_replace([mu, cov], replace=sub_dict, strict=True)

        x0 = pm.Deterministic(
            "x0_slice", mu_frozen[t0_idx], dims=mu_dims[1:] if mu_dims is not None else None
        )
        P0 = pm.Deterministic(
            "P0_slice", cov_frozen[t0_idx], dims=cov_dims[1:] if cov_dims is not None else None
        )

        # Build for the full timeline (training + forecast) so that time-varying matrices
        # continue at the correct phase, then slice to keep only the forecast portion.
        n_train = len(time_index)
        n_total = n_train + len(forecast_index)

        full_matrices = ss_mod._insert_constant_timestep(unpinned_matrices, n_total)
        _, _, *forecast_matrices = full_matrices

        # For exogenous-data-driven matrices the time dimension comes from the
        # data shared variable, not from the n_timesteps symbolic.  Replace the
        # shared variables with concatenated training + scenario tensors so the
        # [n_train:] slice below yields the correct forecast portion.
        # TODO: Is there a way to handle this in a fully symbolic way, without having to
        #  run the full scan on training data to get the system's state at the start date?
        if scenario is not None and ss_mod._needs_exog_data:
            exog_replace = {}
            for name in ss_mod.data_names:
                if name not in scenario:
                    continue
                forecast_data = scenario[name]
                train_val = fit_exog[name]["value"]
                fc_val = (
                    forecast_data.values
                    if isinstance(forecast_data, pd.DataFrame)
                    else np.asarray(forecast_data)
                )
                combined = np.concatenate([train_val, fc_val], axis=0)
                exog_replace[forecast_model[name]] = pt.as_tensor_variable(combined, name=name)
            if exog_replace:
                forecast_matrices = graph_replace(
                    forecast_matrices, replace=exog_replace, strict=False
                )

        forecast_names = MATRIX_NAMES[2:]  # c, d, T, Z, R, H, Q
        time_varying_names = ss_mod.ssm.time_varying_names
        # Start one step early: the transition into the first forecast period uses the
        # matrices of the last training period, and its observation is discarded.
        forecast_matrices = [
            m[n_train - 1 :] if SHORT_NAME_TO_LONG[name] in time_varying_names else m
            for m, name in zip(forecast_matrices, forecast_names)
        ]

        _ = LinearGaussianStateSpace(
            "forecast",
            x0,
            P0,
            *forecast_matrices,
            steps=len(forecast_index),
            dims=trajectory_dims,
            sequence_names=scan_sequence_names(ss_mod.ssm.time_varying_names),
            k_endog=ss_mod.k_endog,
            append_x0=False,
            method=mvn_method,
        )

    return forecast_model


def forecast(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    start: int | pd.Timestamp | None = None,
    periods: int | None = None,
    end: int | pd.Timestamp = None,
    scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None = None,
    use_scenario_index: bool = False,
    filter_output="smoothed",
    random_seed: RandomState | None = None,
    verbose: bool = True,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    group: str = "posterior",
    **kwargs,
) -> DataTree:
    _validate_filter_arg(filter_output)
    verify_group(group)

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    time_index = ss_mod._get_fit_time_index(idata)

    if start is None and verbose:
        _log.warning(
            "No start date provided. Using the last date in the data index. To silence this warning, "
            "explicitly pass a start date or set verbose = False"
        )
        start = time_index[-1]

    if ss_mod._needs_exog_data and not isinstance(scenario, dict):
        if len(ss_mod.data_names) > 1:
            raise ValueError(
                "Model needs more than one exogenous data to do forecasting. In this case, you must "
                "pass a dictionary of scenario data."
            )
        [data_name] = ss_mod.data_names
        scenario = {data_name: scenario}

    scenario_coords = coords_from_idata(ss_mod, idata, "constant_data")
    scenario: dict = ss_mod._validate_scenario_data(scenario, scenario_coords, verbose=verbose)

    ss_mod._validate_forecast_args(
        time_index=time_index,
        start=start,
        end=end,
        periods=periods,
        scenario=scenario,
        use_scenario_index=use_scenario_index,
        verbose=verbose,
    )

    t0, forecast_index = ss_mod._build_forecast_index(
        time_index=time_index,
        start=start,
        end=end,
        periods=periods,
        scenario=scenario,
        use_scenario_index=use_scenario_index,
    )
    scenario = ss_mod._finalize_scenario_initialization(scenario, forecast_index, scenario_coords)

    forecast_model = ss_mod._build_forecast_model(
        idata=idata,
        group=group,
        time_index=time_index,
        t0=t0,
        forecast_index=forecast_index,
        scenario=scenario,
        filter_output=filter_output,
        mvn_method=mvn_method,
    )

    with forecast_model:
        if scenario is not None:
            dummy_obs_data = np.zeros((len(forecast_index), ss_mod.k_endog))
            pm.set_data(
                scenario | {"data": dummy_obs_data},
                coords={"data_time": np.arange(len(forecast_index))},
            )

    forecast_model.rvs_to_initial_values = {
        k: None for k in forecast_model.rvs_to_initial_values.keys()
    }
    frozen_model = freeze_dims_and_data(forecast_model)

    with frozen_model:
        idata_forecast = pm.sample_posterior_predictive(
            idata[group],
            var_names=["forecast_latent", "forecast_observed"],
            random_seed=random_seed,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

    return idata_forecast.posterior_predictive
