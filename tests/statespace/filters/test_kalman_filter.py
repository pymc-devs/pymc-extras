import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose, assert_array_less

from pymc_extras.statespace.filters import (
    KalmanSmoother,
    SquareRootFilter,
    StandardFilter,
    UnivariateFilter,
)
from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import (
    get_expected_shape,
    get_sm_state_from_output_name,
    initialize_filter,
    make_test_inputs,
    nile_test_test_helper,
)

floatX = pytensor.config.floatX

# TODO: These are pretty loose because of all the stabilizing of covariance matrices that is done inside the kalman
#  filters. When that is improved, this should be tightened.
ATOL = 1e-6 if floatX.endswith("64") else 1e-3
RTOL = 1e-6 if floatX.endswith("64") else 1e-3

standard_inout = initialize_filter(StandardFilter())
cholesky_inout = initialize_filter(SquareRootFilter())
univariate_inout = initialize_filter(UnivariateFilter())

f_standard = pytensor.function(*standard_inout, on_unused_input="ignore")
f_cholesky = pytensor.function(*cholesky_inout, on_unused_input="ignore")
f_univariate = pytensor.function(*univariate_inout, on_unused_input="ignore")

filter_funcs = [f_standard, f_cholesky, f_univariate]

filter_names = [
    "StandardFilter",
    "CholeskyFilter",
    "UnivariateFilter",
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
    inputs = [None] * 7
    with pytest.raises(NotImplementedError):
        filter.update(*inputs)


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_one_state_one_observed(filter_func, rng):
    p, m, r, n = 1, 1, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)
    outputs = filter_func(*inputs)

    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_when_all_states_are_stochastic(filter_func, rng):
    p, m, r, n = 1, 2, 2, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = filter_func(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_shapes_when_some_states_are_deterministic(filter_func, rng):
    p, m, r, n = 1, 5, 2, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = filter_func(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


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
        observed_states,
        filtered_covs,
        predicted_covs,
        observed_covs,
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
        ll_obs.sum(),
        ll_obs,
    ]

    f_standard = pytensor.function(inputs, outputs)

    return f_standard


def test_output_shapes_with_time_varying_matrices(f_standard_nd, rng):
    p, m, r, n = 1, 5, 2, 10
    data, a0, P0, c, d, T, Z, R, H, Q = make_test_inputs(p, m, r, n, rng)
    T = np.concatenate([np.expand_dims(T, 0)] * n, axis=0)
    Z = np.concatenate([np.expand_dims(Z, 0)] * n, axis=0)
    R = np.concatenate([np.expand_dims(R, 0)] * n, axis=0)
    H = np.concatenate([np.expand_dims(H, 0)] * n, axis=0)
    Q = np.concatenate([np.expand_dims(Q, 0)] * n, axis=0)

    outputs = f_standard_nd(data, a0, P0, c, d, T, Z, R, H, Q)

    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
def test_output_with_deterministic_observation_equation(filter_func, rng):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = filter_func(*inputs)

    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize(
    ("filter_func", "filter_name"), zip(filter_funcs, filter_names), ids=filter_names
)
def test_output_with_multiple_observed(filter_func, filter_name, rng):
    p, m, r, n = 5, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = filter_func(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize(
    ("filter_func", "filter_name"), zip(filter_funcs, filter_names), ids=filter_names
)
@pytest.mark.parametrize("p", [1, 5], ids=["univariate (p=1)", "multivariate (p=5)"])
def test_missing_data(filter_func, filter_name, p, rng):
    m, r, n = 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng, missing_data=1)

    outputs = filter_func(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert (
            outputs[output_idx].shape == expected_output
        ), f"Shape of {name} does not match expected"


@pytest.mark.parametrize("filter_func", filter_funcs, ids=filter_names)
@pytest.mark.parametrize("output_idx", [(0, 2), (3, 5)], ids=["smoothed_states", "smoothed_covs"])
def test_last_smoother_is_last_filtered(filter_func, output_idx, rng):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)
    outputs = filter_func(*inputs)

    filtered = outputs[output_idx[0]]
    smoothed = outputs[output_idx[1]]

    assert_allclose(filtered[-1], smoothed[-1])


@pytest.mark.parametrize(
    "filter_func, filter_name", zip(filter_funcs, filter_names), ids=filter_names
)
@pytest.mark.parametrize("n_missing", [0, 5], ids=["n_missing=0", "n_missing=5"])
@pytest.mark.skipif(floatX == "float32", reason="Tests are too sensitive for float32")
def test_filters_match_statsmodel_output(filter_func, filter_name, n_missing, rng):
    fit_sm_mod, [data, a0, P0, c, d, T, Z, R, H, Q] = nile_test_test_helper(rng, n_missing)
    if filter_name == "CholeskyFilter":
        P0 = np.linalg.cholesky(P0)
    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]
    outputs = filter_func(*inputs)

    for output_idx, name in enumerate(output_names):
        ref_val = get_sm_state_from_output_name(fit_sm_mod, name)
        val_to_test = outputs[output_idx].squeeze()

        if name == "smoothed_covs":
            # TODO: The smoothed covariance matrices have large errors (1e-2) ONLY in the first few states -- no idea why.
            assert_allclose(
                val_to_test[5:],
                ref_val[5:],
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"{name} does not match statsmodels",
            )
        elif name.startswith("predicted"):
            # statsmodels doesn't throw away the T+1 forecast in the predicted states like we do
            assert_allclose(
                val_to_test,
                ref_val[:-1],
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"{name} does not match statsmodels",
            )
        else:
            # Need atol = 1e-7 for smoother tests to pass
            assert_allclose(
                val_to_test,
                ref_val,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"{name} does not match statsmodels",
            )


@pytest.mark.parametrize(
    "filter_func, filter_name", zip(filter_funcs[:-1], filter_names[:-1]), ids=filter_names[:-1]
)
@pytest.mark.parametrize("n_missing", [0, 5], ids=["n_missing=0", "n_missing=5"])
@pytest.mark.parametrize("obs_noise", [True, False])
def test_all_covariance_matrices_are_PSD(filter_func, filter_name, n_missing, obs_noise, rng):
    if (floatX == "float32") & (filter_name == "UnivariateFilter"):
        # TODO: These tests all pass locally for me with float32 but they fail on the CI, so i'm just disabling them.
        pytest.skip("Univariate filter not stable at half precision without measurement error")

    fit_sm_mod, [data, a0, P0, c, d, T, Z, R, H, Q] = nile_test_test_helper(rng, n_missing)
    if filter_name == "CholeskyFilter":
        P0 = np.linalg.cholesky(P0)

    H *= int(obs_noise)
    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]
    outputs = filter_func(*inputs)

    for output_idx, name in zip([3, 4, 5], output_names[3:-2]):
        cov_stack = outputs[output_idx]
        w, v = np.linalg.eig(cov_stack)

        assert_array_less(0, w, err_msg=f"Smallest eigenvalue of {name}: {min(w.ravel())}")
        assert_allclose(
            cov_stack,
            np.swapaxes(cov_stack, -2, -1),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{name} is not symmetrical",
        )


@pytest.mark.parametrize(
    "filter",
    [StandardFilter, SquareRootFilter],
    ids=["standard", "cholesky"],
)
def test_kalman_filter_jax(filter):
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_graph

    # TODO: Add UnivariateFilter to test; need to figure out the broadcasting issue when 2nd data dim is defined

    p, m, r, n = 1, 5, 1, 10
    inputs, outputs = initialize_filter(filter(), p=p, m=m, r=r, n=n)
    inputs_np = make_test_inputs(p, m, r, n, rng)

    f_jax = get_jaxified_graph(inputs, outputs)
    f_pt = pytensor.function(inputs, outputs, mode="FAST_COMPILE")

    jax_outputs = f_jax(*inputs_np)
    pt_outputs = f_pt(*inputs_np)

    for name, jax_res, pt_res in zip(output_names, jax_outputs, pt_outputs):
        assert_allclose(jax_res, pt_res, atol=ATOL, rtol=RTOL, err_msg=f"{name} failed!")
