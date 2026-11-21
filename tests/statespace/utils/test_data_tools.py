import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from numpy.testing import assert_allclose

from pymc_extras.statespace.utils.data_tools import (
    NO_FREQ_INFO_WARNING,
    NO_TIME_INDEX_WARNING,
    add_data_to_active_model,
    preprocess_xarray_data,
    register_data_with_pymc,
)


def make_data(dims=("series", "time", "observed_state"), shape=(2, 5, 3), **coords):
    return xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape), dims=dims, coords=coords
    )


def test_xarray_data_is_laid_out_time_first_with_observed_states_in_model_order():
    data = make_data(observed_state=["a", "b", "c"], series=["x", "y"], time=np.arange(10, 15))

    with pytest.warns(UserWarning, match=NO_TIME_INDEX_WARNING):
        values, index, batch_coords = preprocess_xarray_data(
            data, n_obs=3, obs_coords=["c", "a", "b"]
        )

    expected = data.sel(observed_state=["c", "a", "b"]).transpose(
        "time", "series", "observed_state"
    )
    assert_allclose(values, expected.values)
    assert_allclose(index, np.arange(5))
    assert list(batch_coords) == ["series"]
    assert list(batch_coords["series"]) == ["x", "y"]


def test_xarray_data_without_coords_gets_ranges():
    _, index, batch_coords = preprocess_xarray_data(make_data(), n_obs=3)

    assert_allclose(index, np.arange(5))
    assert_allclose(batch_coords["series"], np.arange(2))


def test_xarray_datetime_index_gets_a_frequency():
    data = make_data(time=pd.date_range("2000-01-01", periods=5, freq="D").to_numpy())

    with pytest.warns(UserWarning, match=NO_FREQ_INFO_WARNING):
        _, index, _ = preprocess_xarray_data(data, n_obs=3)

    assert index.freq == "D"


@pytest.mark.parametrize(
    "data, obs_coords, error, match",
    [
        (make_data(dims=("series", "date", "observed_state")), None, ValueError, "'time'"),
        (make_data(dims=("state", "time", "observed_state")), None, ValueError, "reserve"),
        (make_data(shape=(2, 5, 4)), None, ValueError, "Expected 3 observed states"),
        (make_data(observed_state=["a", "b", "z"]), ["a", "b", "c"], ValueError, "labels"),
        (make_data(time=[0, 2, 1, 3, 4]), None, IndexError, "monotonic"),
    ],
    ids=["missing_time", "reserved_batch_dim", "wrong_n_obs", "wrong_labels", "unordered_time"],
)
def test_invalid_xarray_data_raises(data, obs_coords, error, match):
    with pytest.raises(error, match=match):
        preprocess_xarray_data(data, n_obs=3, obs_coords=obs_coords)


def test_batch_coords_must_match_the_model():
    with pm.Model(coords={"series": ["x", "y"]}):
        with pytest.raises(ValueError, match="different 'series' coords"):
            add_data_to_active_model(
                np.zeros((5, 2, 3)),
                index=np.arange(5),
                batch_coords={"series": np.array(["y", "x"])},
            )


def test_batch_coords_fill_a_length_only_model_coord():
    with pm.Model() as model:
        model.add_coord("series", length=2)
        add_data_to_active_model(
            np.zeros((5, 2, 3)), index=np.arange(5), batch_coords={"series": np.array(["x", "y"])}
        )

    assert list(model.coords["series"]) == ["x", "y"]


def test_post_estimation_data_rejects_batch_dims():
    with pm.Model(), pytest.raises(NotImplementedError, match="batch dims"):
        register_data_with_pymc(make_data(), n_obs=3, obs_coords=None)
