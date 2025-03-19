import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc import ImputationWarning, modelcontext
from pytensor.tensor.sharedvar import TensorSharedVariable

from pymc_extras.statespace.utils.constants import (
    MISSING_FILL,
    OBS_STATE_DIM,
    TIME_DIM,
)

NO_TIME_INDEX_WARNING = (
    "No time index found on the supplied data. A simple range index will be automatically "
    "generated."
)
NO_FREQ_INFO_WARNING = "No frequency was specific on the data's DateTimeIndex."


def get_data_dims(data):
    if not isinstance(data, pt.TensorVariable | TensorSharedVariable):
        return

    data_name = getattr(data, "name", None)
    if not data_name:
        return

    pm_mod = modelcontext(None)
    data_dims = pm_mod.named_vars_to_dims.get(data_name, None)
    return data_dims


def _validate_data_shape(data_shape, n_obs, obs_coords=None, check_col_names=False, col_names=None):
    if col_names is None:
        col_names = []

    if len(data_shape) != 2:
        raise ValueError("Data must be a 2d matrix")

    if data_shape[-1] != n_obs:
        raise ValueError(
            f"Shape of data does not match model output. Expected {n_obs} columns, "
            f"found {data_shape[-1]}."
        )

    if check_col_names:
        missing_cols = set(obs_coords) - set(col_names)
        if len(missing_cols) > 0:
            raise ValueError(
                "Columns of DataFrame provided as data do not match state names. The following states were"
                f'not found: {", ".join(missing_cols)}. This may result in unexpected results in complex'
                f"statespace models"
            )


def preprocess_tensor_data(data, n_obs, obs_coords=None):
    data_shape = data.shape.eval()
    _validate_data_shape(data_shape, n_obs, obs_coords)
    if obs_coords is not None:
        warnings.warn(NO_TIME_INDEX_WARNING)
    index = np.arange(data_shape[0], dtype="int")

    return data.eval(), index


def preprocess_numpy_data(data, n_obs, obs_coords=None):
    _validate_data_shape(data.shape, n_obs, obs_coords)
    if obs_coords is not None:
        warnings.warn(NO_TIME_INDEX_WARNING)

    index = np.arange(data.shape[0], dtype="int")

    return data, index


def preprocess_pandas_data(data, n_obs, obs_coords=None, check_column_names=False):
    if isinstance(data, pd.Series):
        if data.name is None:
            data.name = "data"
        data = data.to_frame()

    col_names = data.columns
    _validate_data_shape(data.shape, n_obs, obs_coords, check_column_names, col_names)

    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.freq is None:
            warnings.warn(NO_FREQ_INFO_WARNING)
            data.index.freq = data.index.inferred_freq

        index = data.index
        return data.values, index

    elif isinstance(data.index, pd.RangeIndex):
        if obs_coords is not None:
            warnings.warn(NO_TIME_INDEX_WARNING)
        return preprocess_numpy_data(data.values, n_obs, obs_coords)

    elif isinstance(data.index, pd.MultiIndex):
        if obs_coords is not None:
            warnings.warn(NO_TIME_INDEX_WARNING)

        raise NotImplementedError("MultiIndex panel data is not currently supported.")

    else:
        if obs_coords is not None:
            warnings.warn(NO_TIME_INDEX_WARNING)

        index = data.index
        if not np.issubdtype(index.dtype, np.integer):
            raise IndexError("Provided index is not an integer index.")

        index_diff = index.to_series().diff().dropna().values
        if not (index_diff == 1).all():
            raise IndexError("Provided index is not monotonic increasing.")

        return preprocess_numpy_data(data.values, n_obs, obs_coords)


def add_data_to_active_model(values, index, data_dims=None):
    pymc_mod = modelcontext(None)
    if data_dims is None:
        data_dims = [TIME_DIM, OBS_STATE_DIM]
    time_dim = data_dims[0]

    if time_dim not in pymc_mod.coords:
        pymc_mod.add_coord(time_dim, index)
    else:
        found_time = pymc_mod.coords[time_dim]
        if found_time is None:
            pymc_mod.coords.update({time_dim: index})
        elif not np.array_equal(found_time, tuple(index)):
            raise ValueError(
                "Provided data has a different time index than the model. Please ensure that the time values "
                "set on coords matches that of the exogenous data."
            )

    # If the data has just one column, we need to specify the shape as (None, 1), or else the JAX backend will
    # raise a broadcasting error.
    data_shape = None
    if values.shape[-1] == 1:
        data_shape = (None, 1)

    data = pm.Data("data", values, dims=data_dims, shape=data_shape)

    return data


def mask_missing_values_in_data(values, missing_fill_value=None):
    if missing_fill_value is None:
        missing_fill_value = MISSING_FILL

    masked_values = np.ma.masked_invalid(values)
    filled_values = masked_values.filled(missing_fill_value)
    nan_mask = masked_values.mask

    if np.any(nan_mask):
        if np.any(values == missing_fill_value):
            raise ValueError(
                f"Provided data contains the value {missing_fill_value}, which is used as a missing value marker. "
                f"Please manually change the missing_fill_value to avoid this collision."
            )

        impute_message = (
            "Provided data contains missing values and"
            " will be automatically imputed as hidden states"
            " during Kalman filtering."
        )

        warnings.warn(impute_message, ImputationWarning)

    return filled_values, nan_mask


def register_data_with_pymc(
    data, n_obs, obs_coords, register_data=True, missing_fill_value=None, data_dims=None
):
    if isinstance(data, pt.TensorVariable | TensorSharedVariable):
        values, index = preprocess_tensor_data(data, n_obs, obs_coords)
    elif isinstance(data, np.ndarray):
        values, index = preprocess_numpy_data(data, n_obs, obs_coords)
    elif isinstance(data, pd.DataFrame | pd.Series):
        values, index = preprocess_pandas_data(data, n_obs, obs_coords)
    else:
        raise ValueError("Data should be one of pytensor tensor, numpy array, or pandas dataframe")

    data, nan_mask = mask_missing_values_in_data(values, missing_fill_value)

    if register_data:
        data = add_data_to_active_model(data, index, data_dims)
    else:
        data = pytensor.shared(data, name="data")
    return data, nan_mask
