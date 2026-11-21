import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pymc import modelcontext
from pymc.exceptions import ImputationWarning
from pytensor.tensor.sharedvar import TensorSharedVariable

from pymc_extras.statespace.utils.constants import (
    MISSING_FILL,
    OBS_STATE_DIM,
    OBSERVED_DATA_NAME,
    RESERVED_DIMS,
    TIME_DIM,
)

NO_TIME_INDEX_WARNING = (
    "No time index found on the supplied data. A simple range index will be automatically "
    "generated."
)
NO_FREQ_INFO_WARNING = (
    "No frequency was specified on the data's DatetimeIndex. One will be inferred from the index "
    "if possible, and forecasting needs a regular frequency."
)


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
                f"not found: {', '.join(missing_cols)}. This may result in unexpected results in complex"
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

    return data.values, resolve_time_index(data.index, obs_coords)


def resolve_time_index(index, obs_coords=None):
    """
    Validate the time index of observed data and return the index the model uses.

    Parameters
    ----------
    index : pd.Index
        Time index of the data.
    obs_coords : sequence of str, optional
        Names of the observed states. When given, an index that carries no dates gets a warning.

    Returns
    -------
    index : pd.DatetimeIndex or numpy array
        The index itself if it holds dates, with its frequency inferred when missing, and an integer
        range otherwise.
    """
    if isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            warnings.warn(NO_FREQ_INFO_WARNING)
            index.freq = pd.DatetimeIndex(index, freq="infer").freq
        return index

    if obs_coords is not None:
        warnings.warn(NO_TIME_INDEX_WARNING)

    if isinstance(index, pd.MultiIndex):
        raise NotImplementedError("MultiIndex panel data is not currently supported.")

    if not isinstance(index, pd.RangeIndex):
        if not np.issubdtype(index.dtype, np.integer):
            raise IndexError("Provided index is not an integer index.")

        index_diff = index.to_series().diff().dropna().values
        if not (index_diff == 1).all():
            raise IndexError("Provided index is not monotonic increasing.")

    return np.arange(len(index), dtype="int")


def preprocess_xarray_data(data, n_obs, obs_coords=None):
    """
    Lay out a DataArray as ``(time, *batch, observed_state)``.

    Parameters
    ----------
    data : xr.DataArray
        Observed data with ``time`` and ``observed_state`` dims, plus any batch dims.
    n_obs : int
        Number of observed states the model expects.
    obs_coords : sequence of str, optional
        Names of the observed states. When given, ``data`` is reordered to match them if it labels
        its observed states.

    Returns
    -------
    values : numpy array
        The data with its dims transposed.
    index : pandas Index or numpy array
        Time index of the data, a range index when ``data`` has no ``time`` coord.
    batch_coords : dict of str to numpy array
        Coords of each batch dim in order, a range when ``data`` has no coord for it.
    """
    missing_dims = {TIME_DIM, OBS_STATE_DIM} - set(data.dims)
    if missing_dims:
        raise ValueError(
            f"DataArray data must have {TIME_DIM!r} and {OBS_STATE_DIM!r} dims, got {data.dims}"
        )
    reserved_batch_dims = sorted(RESERVED_DIMS.intersection(data.dims) - {TIME_DIM, OBS_STATE_DIM})
    if reserved_batch_dims:
        raise ValueError(
            f"DataArray data cannot have the dims {reserved_batch_dims}, which statespace models "
            f"reserve. Rename them to use them as batch dims."
        )
    if data.sizes[OBS_STATE_DIM] != n_obs:
        raise ValueError(
            f"Shape of data does not match model output. Expected {n_obs} observed states, "
            f"found {data.sizes[OBS_STATE_DIM]}."
        )

    if obs_coords is not None and OBS_STATE_DIM in data.coords:
        labels = list(data.indexes[OBS_STATE_DIM])
        if set(labels) != set(obs_coords):
            raise ValueError(
                f"DataArray data labels its observed states {labels}, but the model's are "
                f"{list(obs_coords)}"
            )
        data = data.sel({OBS_STATE_DIM: list(obs_coords)})

    data = data.transpose(TIME_DIM, ..., OBS_STATE_DIM)
    batch_dims = data.dims[1:-1]

    def coord_values(dim):
        if dim in data.coords:
            return data.indexes[dim]
        return pd.RangeIndex(data.sizes[dim])

    batch_coords = {dim: np.asarray(coord_values(dim)) for dim in batch_dims}
    return data.values, resolve_time_index(coord_values(TIME_DIM), obs_coords), batch_coords


def add_data_to_active_model(values, index, data_dims=None, batch_coords=None):
    pymc_mod = modelcontext(None)
    if batch_coords is None:
        batch_coords = {}
    if data_dims is None:
        data_dims = [TIME_DIM, *batch_coords, OBS_STATE_DIM]
    time_dim = data_dims[0]

    for dim, labels in batch_coords.items():
        if dim not in pymc_mod.coords:
            pymc_mod.add_coord(dim, labels)
        elif pymc_mod.coords[dim] is None:
            pymc_mod.coords.update({dim: labels})
        elif not np.array_equal(pymc_mod.coords[dim], tuple(labels)):
            raise ValueError(
                f"Provided data has different {dim!r} coords than the model. Please ensure that "
                f"the {dim!r} values set on coords match those of the data."
            )

    if isinstance(index, pd.Index):
        index = index.rename(time_dim)

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

    # Every dim but time is fixed, so that a single observed state is known to have size 1. The JAX
    # backend raises a broadcasting error otherwise.
    data_shape = (None, *values.shape[1:])

    data = pm.Data(OBSERVED_DATA_NAME, values, dims=data_dims, shape=data_shape)

    return data


def update_data_in_active_model(values, index, data_dims=None):
    """
    Point the active model's observed data variable at new values.

    Parameters
    ----------
    values : numpy array
        Data to write, with missing values already filled.
    index : pandas Index or numpy array
        Time index of the data, written to the time coordinate.
    data_dims : sequence of str, optional
        Dims of the observed data variable. Default None, meaning the package defaults.

    Returns
    -------
    data : TensorVariable
        The model's observed data variable.
    """
    pymc_mod = modelcontext(None)
    if data_dims is None:
        data_dims = [TIME_DIM, OBS_STATE_DIM]
    time_dim = data_dims[0]

    if isinstance(index, pd.Index):
        index = index.rename(time_dim)

    pymc_mod.set_data(OBSERVED_DATA_NAME, values, coords={time_dim: index})

    return pymc_mod[OBSERVED_DATA_NAME]


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


def prepare_data_for_pymc(data, n_obs, obs_coords, missing_fill_value=None):
    """
    Validate observed data and fill its missing values with a sentinel.

    Parameters
    ----------
    data : pytensor tensor, numpy array, pandas DataFrame or Series, or xr.DataArray
        Observed data to prepare. Only a DataArray can carry batch dims.
    n_obs : int
        Number of observed states the model expects.
    obs_coords : sequence of str, optional
        Names of the observed states, used to check the columns of a DataFrame.
    missing_fill_value : float, optional
        Value substituted for missing observations. Default None, meaning the package default.

    Returns
    -------
    filled_values : numpy array
        Data with missing values replaced by the sentinel.
    index : pandas Index or numpy array
        Time index of the data.
    nan_mask : numpy array
        Boolean mask that is True where the data was missing.
    batch_coords : dict of str to numpy array
        Coords of each batch dim of the data, empty unless ``data`` is a DataArray with batch dims.
    """
    batch_coords = {}
    if isinstance(data, xr.DataArray):
        values, index, batch_coords = preprocess_xarray_data(data, n_obs, obs_coords)
    elif isinstance(data, pt.TensorVariable | TensorSharedVariable):
        values, index = preprocess_tensor_data(data, n_obs, obs_coords)
    elif isinstance(data, np.ndarray):
        values, index = preprocess_numpy_data(data, n_obs, obs_coords)
    elif isinstance(data, pd.DataFrame | pd.Series):
        values, index = preprocess_pandas_data(data, n_obs, obs_coords)
    else:
        raise ValueError(
            "Data should be one of pytensor tensor, numpy array, pandas dataframe, or xarray "
            "DataArray"
        )

    filled_values, nan_mask = mask_missing_values_in_data(values, missing_fill_value)

    return filled_values, index, nan_mask, batch_coords


def register_data_with_pymc(data, n_obs, obs_coords, missing_fill_value=None, data_dims=None):
    filled_values, index, nan_mask, batch_coords = prepare_data_for_pymc(
        data, n_obs, obs_coords, missing_fill_value
    )
    if batch_coords:
        raise NotImplementedError("Post-estimation does not support data with batch dims yet")
    data_variable = add_data_to_active_model(filled_values, index, data_dims)

    return data_variable, nan_mask
