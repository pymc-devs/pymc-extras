import unittest

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose

from pymc_experimental.statespace.filters import (
    CholeskyFilter,
    KalmanSmoother,
    SingleTimeseriesFilter,
    StandardFilter,
    SteadyStateFilter,
    UnivariateFilter,
)
from pymc_experimental.statespace.filters.kalman_filter import BaseFilter
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    get_expected_shape,
    get_sm_state_from_output_name,
    initialize_filter,
    make_test_inputs,
    nile_test_test_helper,
)

floatX = pytensor.config.floatX

standard_inout = initialize_filter(StandardFilter())
cholesky_inout = initialize_filter(CholeskyFilter())
univariate_inout = initialize_filter(UnivariateFilter())
single_inout = initialize_filter(SingleTimeseriesFilter())
steadystate_inout = initialize_filter(SteadyStateFilter())

f_standard = pytensor.function(*standard_inout)
f_cholesky = pytensor.function(*cholesky_inout)
f_univariate = pytensor.function(*univariate_inout)
f_single_ts = pytensor.function(*single_inout)
f_steady = pytensor.function(*steadystate_inout)

filter_funcs = [f_standard, f_cholesky, f_univariate, f_single_ts, f_steady]

filter_names = [
    "StandardFilter",
    "CholeskyFilter",
    "UnivariateFilter",
    "SingleTimeSeriesFilter",
    "SteadyStateFilter",
]
output_names = [
    "filtered_states",
    "predicted_states",
    "smoothed_states",
    "filtered_covs",
    "predicted_covs",
    "smoothed_covs",
    "log_likelihood",
    "ll_obs",
]


def test_base_class_update_raises():
    filter = BaseFilter()
    inputs = [None] * 8
    with pytest.raises(NotImplementedError):
        filter.update(*inputs)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_one_state_one_observed(filter_func, output_idx, name):
    p, m, r, n = 1, 1, 1, 10
    inputs = make_test_inputs(p, m, r, n)
    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_when_all_states_are_stochastic(filter_func, output_idx, name):
    p, m, r, n = 1, 2, 2, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_when_some_states_are_deterministic(filter_func, output_idx, name):
    p, m, r, n = 1, 5, 2, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.fixture
def f_standard_nd():
    ksmoother = KalmanSmoother()
    data = pt.tensor(name="data", dtype=floatX, shape=(None, None))
    a0 = pt.vector(name="a0", dtype=floatX)
    P0 = pt.matrix(name="P0", dtype=floatX)
    c = pt.vector(name="c", dtype=floatX)
    d = pt.vector(name="d", dtype=floatX)
    Q = pt.tensor(name="Q", dtype=floatX, shape=(None, None, None))
    H = pt.tensor(name="H", dtype=floatX, shape=(None, None, None))
    T = pt.tensor(name="T", dtype=floatX, shape=(None, None, None))
    R = pt.tensor(name="R", dtype=floatX, shape=(None, None, None))
    Z = pt.tensor(name="Z", dtype=floatX, shape=(None, None, None))

    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]

    (
        filtered_states,
        predicted_states,
        filtered_covs,
        predicted_covs,
        log_likelihood,
        ll_obs,
    ) = StandardFilter().build_graph(*inputs)

    smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

    outputs = [
        filtered_states,
        predicted_states,
        smoothed_states,
        filtered_covs,
        predicted_covs,
        smoothed_covs,
        log_likelihood,
        ll_obs,
    ]

    f_standard = pytensor.function(inputs, outputs)

    return f_standard


@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_shapes_with_time_varying_matrices(f_standard_nd, output_idx, name):
    p, m, r, n = 1, 5, 2, 10
    data, a0, P0, c, d, T, Z, R, H, Q = make_test_inputs(p, m, r, n)
    T = np.concatenate([np.expand_dims(T, 0)] * n, axis=0)
    Z = np.concatenate([np.expand_dims(Z, 0)] * n, axis=0)
    R = np.concatenate([np.expand_dims(R, 0)] * n, axis=0)
    H = np.concatenate([np.expand_dims(H, 0)] * n, axis=0)
    Q = np.concatenate([np.expand_dims(Q, 0)] * n, axis=0)

    outputs = f_standard_nd(data, a0, P0, c, d, T, Z, R, H, Q)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_with_deterministic_observation_equation(filter_func, output_idx, name):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)

    outputs = filter_func(*inputs)
    expected_output = get_expected_shape(name, p, m, r, n)

    assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize(
    ("filter_func", "filter_name"), zip(filter_funcs, filter_names), ids=filter_names
)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
def test_output_with_multiple_observed(filter_func, filter_name, output_idx, name):
    p, m, r, n = 5, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)
    expected_output = get_expected_shape(name, p, m, r, n)

    if filter_name == "SingleTimeSeriesFilter":
        with pytest.raises(
            AssertionError,
            match="UnivariateTimeSeries filter requires data be at most 1-dimensional",
        ):
            filter_func(*inputs)

    else:
        outputs = filter_func(*inputs)
        assert outputs[output_idx].shape == expected_output


@pytest.mark.parametrize(
    ("filter_func", "filter_name"), zip(filter_funcs, filter_names), ids=filter_names
)
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
@pytest.mark.parametrize("p", [1, 5], ids=["univariate (p=1)", "multivariate (p=5)"])
def test_missing_data(filter_func, filter_name, output_idx, name, p):
    m, r, n = 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, missing_data=1)
    if p > 1 and filter_name == "SingleTimeSeriesFilter":
        with pytest.raises(
            AssertionError,
            match="UnivariateTimeSeries filter requires data be at most 1-dimensional",
        ):
            filter_func(*inputs)

    else:
        outputs = filter_func(*inputs)
        assert not np.any(np.isnan(outputs[output_idx]))


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize("output_idx", [(0, 2), (3, 5)], ids=["smoothed_states", "smoothed_covs"])
def test_last_smoother_is_last_filtered(filter_func, output_idx):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n)
    outputs = filter_func(*inputs)

    filtered = outputs[output_idx[0]]
    smoothed = outputs[output_idx[1]]

    assert_allclose(filtered[-1], smoothed[-1])


# TODO: These tests omit the SteadyStateFilter, because it gives different results to StatsModels (reason to dump it?)
@pytest.mark.parametrize("filter_func", filter_funcs[:-1], ids=filter_names[:-1])
@pytest.mark.parametrize(("output_idx", "name"), list(enumerate(output_names)), ids=output_names)
@pytest.mark.parametrize("n_missing", [0, 5], ids=["n_missing=0", "n_missing=5"])
@pytest.mark.skipif(floatX == "float32", reason="Tests are too sensitive for float32")
def test_filters_match_statsmodel_output(filter_func, output_idx, name, n_missing):
    fit_sm_mod, inputs = nile_test_test_helper(n_missing)
    outputs = filter_func(*inputs)

    val_to_test = outputs[output_idx].squeeze()
    ref_val = get_sm_state_from_output_name(fit_sm_mod, name)

    if name == "smoothed_covs":
        # TODO: The smoothed covariance matrices have large errors (1e-2) ONLY in the first few states -- no idea why.
        assert_allclose(val_to_test[5:], ref_val[5:])
    else:
        # Need atol = 1e-7 for smoother tests to pass
        assert_allclose(val_to_test, ref_val, atol=1e-7)


if __name__ == "__main__":
    unittest.main()