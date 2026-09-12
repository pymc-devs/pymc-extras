from collections.abc import Callable
from functools import cache

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose, assert_array_less

from pymc_extras.statespace.filters import (
    DisturbanceSmoother,
    RTSSmoother,
    SquareRootFilter,
    StandardFilter,
    UnivariateFilter,
)
from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.utils.constants import (
    LONG_NAME_TO_SHORT,
    MATRIX_NAMES,
    MISSING_FILL,
)
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import (
    get_expected_shape,
    get_sm_state_from_output_name,
    initialize_filter,
    make_test_inputs,
    nile_test_test_helper,
    statsmodels_loglike_obs,
)

floatX = pytensor.config.floatX

# TODO: These are pretty loose because of all the stabilizing of covariance matrices that is done inside the kalman
#  filters. When that is improved, this should be tightened.
ATOL = 1e-6 if floatX.endswith("64") else 1e-3
RTOL = 1e-6 if floatX.endswith("64") else 1e-3


@cache
def get_filter_function(filter_name: str) -> Callable:
    """
    Compile and return a filter function given its name, caching the result to make tests as fast as possible
    """
    match filter_name:
        case "StandardFilter":
            filter_inout = initialize_filter(StandardFilter())
        case "CholeskyFilter":
            filter_inout = initialize_filter(SquareRootFilter())
        case "UnivariateFilter":
            filter_inout = initialize_filter(UnivariateFilter())
        case _:
            raise ValueError(f"Unknown filter name: {filter_name}")

    filter_func = pytensor.function(*filter_inout, on_unused_input="ignore")
    return filter_func


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


@pytest.mark.parametrize("filter_name", filter_names)
def test_output_shapes_one_state_one_observed(filter_name, rng):
    p, m, r, n = 1, 1, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)
    outputs = get_filter_function(filter_name)(*inputs)

    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_output_shapes_when_all_states_are_stochastic(filter_name, rng):
    p, m, r, n = 1, 2, 2, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = get_filter_function(filter_name)(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_output_shapes_when_some_states_are_deterministic(filter_name, rng):
    p, m, r, n = 1, 5, 2, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = get_filter_function(filter_name)(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.fixture
def f_standard_nd():
    time_varying_names = ("transition", "design", "selection", "obs_cov", "state_cov")
    ksmoother = RTSSmoother(time_varying_names=time_varying_names)
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
    ) = filter_outputs = StandardFilter(time_varying_names=time_varying_names).build_graph(*inputs)

    smoothed_states, smoothed_covs = ksmoother.build_graph(
        data, (a0, P0, c, d, T, Z, R, H, Q), filter_outputs
    )

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


@pytest.mark.parametrize(
    ("time_varying_names", "expected_seq_names"),
    [
        ((), []),
        (("transition",), ["T"]),
        (("design", "obs_intercept"), ["d", "Z"]),
        (
            ("transition", "design", "selection", "obs_cov", "state_cov"),
            ["T", "Z", "R", "H", "Q"],
        ),
    ],
    ids=["none", "one", "declared_out_of_order", "many"],
)
def test_time_varying_matrices_become_scan_sequences_in_matrix_order(
    time_varying_names, expected_seq_names, rng
):
    """``scan`` receives sequences in matrix order, not the order the model declared them."""
    p, m, r, n = 1, 5, 2, 10
    inputs = list(make_test_inputs(p, m, r, n, rng))

    # inputs are [data, a0, P0, c, d, T, Z, R, H, Q]; a time-varying matrix needs a leading time axis.
    index_of = dict(zip(MATRIX_NAMES[2:], range(3, 10), strict=True))
    for long_name in time_varying_names:
        i = index_of[LONG_NAME_TO_SHORT[long_name]]
        inputs[i] = np.repeat(np.expand_dims(inputs[i], 0), n, axis=0)

    kfilter = StandardFilter(time_varying_names=time_varying_names)
    assert kfilter.seq_names == expected_seq_names
    assert kfilter.non_seq_names == [n for n in MATRIX_NAMES[2:] if n not in expected_seq_names]

    # Building proves the split is usable: a mis-ordered sequence fails on a shape mismatch.
    kfilter.build_graph(*[pt.as_tensor_variable(x) for x in inputs])


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
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_output_with_deterministic_observation_equation(filter_name, rng):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = get_filter_function(filter_name)(*inputs)

    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_output_with_multiple_observed(filter_name, rng):
    p, m, r, n = 5, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)

    outputs = get_filter_function(filter_name)(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
@pytest.mark.parametrize("p", [1, 5], ids=["univariate (p=1)", "multivariate (p=5)"])
def test_missing_data(filter_name, p, rng):
    m, r, n = 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng, missing_data=1)

    outputs = get_filter_function(filter_name)(*inputs)
    for output_idx, name in enumerate(output_names):
        expected_output = get_expected_shape(name, p, m, r, n)
        assert outputs[output_idx].shape == expected_output, (
            f"Shape of {name} does not match expected"
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_missing_value_with_nonzero_obs_intercept(filter_name, rng):
    """
    With non-zero observation intercept ``d``, masking must zero ``d`` at missing rows so the
    innovation does not become ``-d`` and contaminate the log-likelihood. Verify by comparing
    against the equivalent ``(y - d, 0)`` parameterization, under which the filter is invariant.
    """
    p, m, r, n = 3, 5, 1, 10
    data, a0, P0, c, d, T, Z, R, H, Q = make_test_inputs(p, m, r, n, rng, missing_data=2)

    d_nonzero = np.array([1.5, -0.7, 2.1], dtype=floatX)

    # Reference: absorb d into the data (NaN entries stay NaN under subtraction).
    data_absorbed = data - d_nonzero
    out_ref = get_filter_function(filter_name)(
        data_absorbed, a0, P0, c, np.zeros_like(d_nonzero), T, Z, R, H, Q
    )
    out_d = get_filter_function(filter_name)(data, a0, P0, c, d_nonzero, T, Z, R, H, Q)

    for idx, name in enumerate(output_names):
        assert_allclose(
            out_d[idx],
            out_ref[idx],
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"{name} differs between (d, y) and (0, y - d) with missing observations",
        )


@pytest.mark.parametrize("filter_name", filter_names)
def test_missing_value_with_nondiagonal_obs_cov(filter_name, rng):
    """
    With non-diagonal ``H`` and a missing observation at position ``j``, the cross-covariances
    ``H[:, j]`` and ``H[j, :]`` cannot influence any observed quantity. Verify by comparing
    against a run where those rows and columns have been zeroed by hand — the two must agree.
    """
    p, m, r, n = 2, 5, 1, 10
    data, a0, P0, c, d, T, Z, R, H, Q = make_test_inputs(p, m, r, n, rng)
    data[:, 1] = np.nan

    H_full = np.array([[1.0, 0.4], [0.4, 1.0]], dtype=floatX)
    H_zeroed = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=floatX)

    out_full = get_filter_function(filter_name)(data, a0, P0, c, d, T, Z, R, H_full, Q)
    out_zeroed = get_filter_function(filter_name)(data, a0, P0, c, d, T, Z, R, H_zeroed, Q)

    for idx, name in enumerate(output_names):
        assert_allclose(
            out_full[idx],
            out_zeroed[idx],
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"{name} depends on H entries at masked positions",
        )


LOGLIKE_TEST_K_ENDOG = 3


def loglike_test_matrices(obs_cov: str):
    """
    Build the model filtered by :func:`test_loglike_matches_statsmodels`. A dense ``Z`` makes every
    series load on every state, so a masked row of ``F`` carries cross-covariances that the update
    has to clear -- the identity default of :func:`make_test_inputs` would hide a regression there.
    """
    p = m = r = LOGLIKE_TEST_K_ENDOG
    _, a0, P0, c, d, T, _, R, _, Q = make_test_inputs(p, m, r, 1, rng=None)

    Z = np.eye(p, m, dtype=floatX) + 0.3
    H = np.eye(p, dtype=floatX) + (0.4 if obs_cov == "dense" else 0.0)

    return [a0, P0, c, d, T, Z, R, H, Q]


@cache
def get_loglike_function(filter_name: str, obs_cov: str) -> Callable:
    """Compile the per-timestep log-likelihood alone, which ``get_filter_function`` does not expose."""
    filter_cls = {
        "StandardFilter": StandardFilter,
        "CholeskyFilter": SquareRootFilter,
        "UnivariateFilter": UnivariateFilter,
    }[filter_name]

    a0, P0, c, d, T, Z, R, H, Q = loglike_test_matrices(obs_cov)

    data = pt.tensor(name="data", dtype=floatX, shape=(None, LOGLIKE_TEST_K_ENDOG))
    matrices = [pt.as_tensor_variable(x) for x in [a0, P0, c, d, T, Z, R, H, Q]]
    ll_obs = filter_cls().build_graph(data, *matrices)[-1]

    return pytensor.function([data], ll_obs)


@pytest.mark.parametrize(
    ("filter_name", "obs_cov"),
    [
        ("StandardFilter", "dense"),
        ("UnivariateFilter", "diagonal"),
        ("CholeskyFilter", "dense"),
    ],
)
def test_loglike_matches_statsmodels(filter_name, obs_cov, rng):
    """
    Each timestep scores the observed components of ``y_t``, so the per-timestep log-likelihood must
    match statsmodels -- normalizing constant included -- however much of the observation is missing.
    """
    n = 20
    data = rng.normal(size=(n, LOGLIKE_TEST_K_ENDOG)).astype(floatX)
    data[3, 0] = np.nan
    data[7, [0, 2]] = np.nan
    data[11, :] = np.nan

    ll_obs = get_loglike_function(filter_name, obs_cov)(data)
    expected = statsmodels_loglike_obs(data, *loglike_test_matrices(obs_cov))

    assert_allclose(ll_obs, expected, atol=ATOL, rtol=RTOL)


def test_square_root_filter_takes_a_covariance_for_P0(rng):
    """
    Every filter takes ``P0`` as a covariance; the square-root filter factors it itself.

    The shared inputs use ``P0 = I``, where a factor and a covariance coincide, so only a
    non-identity ``P0`` separates the two.
    """
    p, m, r, n = 1, 3, 1, 20
    data, a0, _, c, d, T, Z, R, H, Q = make_test_inputs(p, m, r, n, rng)
    P0 = np.diag([2.0, 3.0, 4.0]).astype(floatX)

    standard = get_filter_function("StandardFilter")(data, a0, P0, c, d, T, Z, R, H, Q)
    square_root = get_filter_function("CholeskyFilter")(data, a0, P0, c, d, T, Z, R, H, Q)

    for name, expected, actual in zip(output_names, standard, square_root, strict=True):
        assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=name)


class TestDisturbanceSmootherMatchesRTS:
    @classmethod
    def setup_class(cls):
        inputs, _ = initialize_filter(StandardFilter(cov_jitter=0.0))
        data, *matrices = inputs
        filter_outputs = StandardFilter(cov_jitter=0.0).build_graph(data, *matrices)
        rts_states, rts_covs = RTSSmoother(cov_jitter=0.0).build_graph(
            data, matrices, filter_outputs
        )
        dk_states, dk_covs = DisturbanceSmoother(cov_jitter=0.0).build_graph(
            data, matrices, filter_outputs
        )
        cls.smoother_fn = pytensor.function(
            inputs, [filter_outputs[4][-1], dk_states, rts_states, dk_covs, rts_covs]
        )

    @pytest.mark.parametrize("stochastic_states", [4, 1], ids=["full_rank_P", "singular_P"])
    @pytest.mark.parametrize("n_missing", [0, 5], ids=["complete", "missing"])
    def test_matches_rts(self, stochastic_states, n_missing, rng):
        """
        The disturbance smoother never inverts ``P``, so it must agree with the RTS form even where
        ``P`` is singular. A diagonal transition keeps process noise out of the noiseless states, so
        ``P`` stays rank-deficient at every step rather than filling in.
        """
        m, p, n = 4, 1, 30
        T = np.diag(np.linspace(0.5, 0.9, m)).astype(floatX)
        R = np.eye(m, dtype=floatX)[:, :stochastic_states]
        Q = np.eye(stochastic_states, dtype=floatX) * 0.3
        Z = (rng.normal(size=(p, m)) * 0.5).astype(floatX)
        H = np.eye(p, dtype=floatX) * 0.4
        a0, c, d = np.zeros(m, dtype=floatX), np.zeros(m, dtype=floatX), np.zeros(p, dtype=floatX)
        P0 = (R @ Q @ R.T).astype(floatX)

        y = rng.normal(size=(n, p)).astype(floatX)
        y[rng.choice(n, n_missing, replace=False), 0] = np.nan

        P_last, dk_states, rts_states, dk_covs, rts_covs = self.smoother_fn(
            y, a0, P0, c, d, T, Z, R, H, Q
        )

        assert np.linalg.matrix_rank(P_last) == stochastic_states
        assert_allclose(dk_states, rts_states, atol=1e-6)
        assert_allclose(dk_covs, rts_covs, atol=1e-6)


@pytest.mark.parametrize("filter_name", filter_names)
@pytest.mark.parametrize("output_idx", [(0, 2), (3, 5)], ids=["smoothed_states", "smoothed_covs"])
def test_last_smoother_is_last_filtered(filter_name, output_idx, rng):
    p, m, r, n = 1, 5, 1, 10
    inputs = make_test_inputs(p, m, r, n, rng)
    outputs = get_filter_function(filter_name)(*inputs)

    filtered = outputs[output_idx[0]]
    smoothed = outputs[output_idx[1]]

    assert_allclose(filtered[-1], smoothed[-1])


@pytest.mark.parametrize("filter_name", filter_names)
@pytest.mark.parametrize("n_missing", [0, 5], ids=["n_missing=0", "n_missing=5"])
@pytest.mark.skipif(floatX == "float32", reason="Tests are too sensitive for float32")
def test_filters_match_statsmodel_output(filter_name, n_missing, rng):
    fit_sm_mod, [data, a0, P0, c, d, T, Z, R, H, Q] = nile_test_test_helper(rng, n_missing)
    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]
    outputs = get_filter_function(filter_name)(*inputs)

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


@pytest.mark.parametrize("filter_name", filter_names)
@pytest.mark.parametrize("n_missing", [0, 5], ids=["n_missing=0", "n_missing=5"])
@pytest.mark.parametrize("obs_noise", [True, False])
def test_all_covariance_matrices_are_PSD(filter_name, n_missing, obs_noise, rng):
    if (floatX == "float32") & (filter_name == "UnivariateFilter"):
        # TODO: These tests all pass locally for me with float32 but they fail on the CI, so i'm just disabling them.
        pytest.skip("Univariate filter not stable at half precision without measurement error")

    fit_sm_mod, [data, a0, P0, c, d, T, Z, R, H, Q] = nile_test_test_helper(rng, n_missing)

    H *= int(obs_noise)
    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]
    outputs = get_filter_function(filter_name)(*inputs)

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


# -------------------- ConvergentFilter --------------------
# Tests comparing ConvergentFilter outputs and gradients to StandardFilter.
# ConvergentFilter requires stationary parameters and no missing data, so it
# can't join the shared parametrized suite above.


from pymc_extras.statespace.filters import ConvergentFilter


def _make_stationary_system(m, p, n_shocks, n, rng):
    """Build a valid stable stationary system for ConvergentFilter testing."""
    T_np = rng.standard_normal((m, m)) * 0.3
    T_np = T_np / (np.abs(np.linalg.eigvals(T_np)).max() * 1.5)
    Z_np = rng.standard_normal((p, m)) * 0.5
    H_root = rng.standard_normal((p, p)) * 0.3
    H_np = H_root @ H_root.T + 0.2 * np.eye(p)
    Q_root = rng.standard_normal((n_shocks, n_shocks)) * 0.3
    Q_np = Q_root @ Q_root.T + 0.1 * np.eye(n_shocks)
    R_np = rng.standard_normal((m, n_shocks)) * 0.5
    c_np = rng.standard_normal(m) * 0.05
    d_np = rng.standard_normal(p) * 0.05
    a0_np = rng.standard_normal(m) * 0.1
    P0_np = np.eye(m) * 0.5
    a = rng.multivariate_normal(a0_np, P0_np)
    data_np = np.empty((n, p), dtype=floatX)
    for t in range(n):
        w = R_np @ rng.multivariate_normal(np.zeros(n_shocks), Q_np)
        eps = rng.multivariate_normal(np.zeros(p), H_np)
        a = T_np @ a + c_np + w
        data_np[t] = Z_np @ a + d_np + eps
    return [
        data_np.astype(floatX),
        a0_np.astype(floatX),
        P0_np.astype(floatX),
        c_np.astype(floatX),
        d_np.astype(floatX),
        T_np.astype(floatX),
        Z_np.astype(floatX),
        R_np.astype(floatX),
        H_np.astype(floatX),
        Q_np.astype(floatX),
    ]


GRAD_NAMES = ["loss", "d_a0", "d_P0", "d_c", "d_d", "d_T", "d_Z", "d_R", "d_H", "d_Q"]

# Gradients of parameters that are themselves symmetric matrices are only determined up to their
# symmetric part, so the analytic and autodiff versions can differ in the antisymmetric half without
# actually disagreeing.
SYMMETRIC_GRADS = {"d_P0", "d_H", "d_Q"}


def assert_results_match(out_std, out_conv, names, err_prefix=""):
    for name, std, conv in zip(names, out_std, out_conv, strict=True):
        std, conv = np.asarray(std, float), np.asarray(conv, float)
        if name in SYMMETRIC_GRADS:
            std, conv = 0.5 * (std + std.T), 0.5 * (conv + conv.T)
        assert_allclose(conv, std, atol=ATOL, rtol=RTOL, err_msg=f"{err_prefix}{name} mismatch")


def _make_local_level_system(n, rng):
    """A unit-root (non-stationary) but observable and controllable local level. Its Riccati
    recursion still converges to a steady-state gain, so ConvergentFilter applies -- convergence
    requires detectability and stabilizability, not stationarity."""
    sigma_level, sigma_obs = 0.4, 0.7
    T_np = np.array([[1.0]], dtype=floatX)
    Z_np = np.array([[1.0]], dtype=floatX)
    R_np = np.array([[1.0]], dtype=floatX)
    Q_np = np.array([[sigma_level**2]], dtype=floatX)
    H_np = np.array([[sigma_obs**2]], dtype=floatX)
    c_np = np.zeros(1, dtype=floatX)
    d_np = np.zeros(1, dtype=floatX)
    a0_np = np.zeros(1, dtype=floatX)
    P0_np = np.array([[1.0]], dtype=floatX)
    level = rng.standard_normal()
    data_np = np.empty((n, 1), dtype=floatX)
    for t in range(n):
        level = level + sigma_level * rng.standard_normal()
        data_np[t] = level + sigma_obs * rng.standard_normal()
    return [data_np, a0_np, P0_np, c_np, d_np, T_np, Z_np, R_np, H_np, Q_np]


N_FORWARD_OUTPUTS = len(output_names)


class TestConvergentFilter:
    """Compare ConvergentFilter with StandardFilter through one compiled function per filter."""

    @classmethod
    def setup_class(cls):
        cls.standard = cls._compile(StandardFilter())
        cls.convergent = cls._compile(ConvergentFilter())
        cls.convergent_never_converging = cls._compile(ConvergentFilter(tol=0.0))

    @staticmethod
    def _compile(kfilter) -> Callable:
        """Compile with unknown shapes, returning every forward output, the loss, and its gradients."""
        inputs, outputs = initialize_filter(kfilter)
        loss = outputs[-1].sum()
        grads = pt.grad(loss, inputs[1:])
        return pytensor.function(inputs, [*outputs, loss, *grads], on_unused_input="ignore")

    @staticmethod
    def _forward_and_gradients(fn, vals):
        results = fn(*vals)
        return results[:N_FORWARD_OUTPUTS], results[N_FORWARD_OUTPUTS:]

    @pytest.mark.parametrize(
        "m,p,n_shocks,n",
        [(5, 2, 5, 100), (10, 3, 10, 200)],
        ids=["small", "medium"],
    )
    def test_forward_matches_standard(self, m, p, n_shocks, n, rng):
        """ConvergentFilter forward outputs should match StandardFilter to numerical precision."""
        vals = _make_stationary_system(m, p, n_shocks, n, rng)
        out_std, _ = self._forward_and_gradients(self.standard, vals)
        out_conv, _ = self._forward_and_gradients(self.convergent, vals)

        # The tail path only runs once the Riccati recursion converges. Without this the comparison
        # could pass vacuously, with ConvergentFilter having degenerated into StandardFilter.
        predicted_covs = out_conv[output_names.index("predicted_covs")]
        assert_allclose(predicted_covs[-1], predicted_covs[n // 2], atol=ATOL, rtol=RTOL)

        assert_results_match(out_std, out_conv, output_names, err_prefix="ConvergentFilter ")

    @pytest.mark.parametrize(
        "m,p,n_shocks,n",
        [(5, 2, 5, 100), (10, 3, 10, 200)],
        ids=["small", "medium"],
    )
    def test_gradient_matches_standard(self, m, p, n_shocks, n, rng):
        """ConvergentFilter's analytic gradients should match StandardFilter's autodiff gradients
        for every model parameter."""
        vals = _make_stationary_system(m, p, n_shocks, n, rng)
        _, grads_std = self._forward_and_gradients(self.standard, vals)
        _, grads_conv = self._forward_and_gradients(self.convergent, vals)

        assert_results_match(grads_std, grads_conv, GRAD_NAMES, err_prefix="ConvergentFilter ")

    def test_asserts_nan_symbolic_data(self, rng):
        """For fully symbolic data, NaN should be caught by a runtime Assert op."""
        m, p, n_shocks, n = 3, 2, 3, 30
        vals = _make_stationary_system(m, p, n_shocks, n, rng)
        # Inject NaN at runtime
        vals[0][5, 0] = np.nan

        with pytest.raises(AssertionError, match="missing data"):
            self.convergent(*vals)

    def test_singular_H_gradient_matches_standard(self, rng):
        """A measurement-error-free model has singular H. The tail backward routes its one
        H-coupled term through F^{-1}, so every gradient -- d_H included -- matches StandardFilter
        even when H is singular."""
        m, p, n_shocks, n = 4, 3, 4, 120
        vals = _make_stationary_system(m, p, n_shocks, n, rng)
        # Perfectly observe one series: zero its measurement-noise row and column.
        vals[8][0, :] = 0.0
        vals[8][:, 0] = 0.0

        _, grads_std = self._forward_and_gradients(self.standard, vals)
        _, grads_conv = self._forward_and_gradients(self.convergent, vals)
        assert_results_match(grads_std, grads_conv, GRAD_NAMES, err_prefix="singular H: ")

    def test_local_level_matches_standard(self, rng):
        """A unit-root local level converges to a steady-state gain. ConvergentFilter forward
        outputs and gradients should match StandardFilter even though the system is
        non-stationary."""
        vals = _make_local_level_system(250, rng)

        out_std = self.standard(*vals)
        out_conv = self.convergent(*vals)
        for std, conv in zip(out_std, out_conv, strict=True):
            assert_allclose(np.asarray(conv), np.asarray(std), atol=ATOL, rtol=RTOL)

    def test_k_equals_n_gradient_matches_standard(self, rng):
        """With tol=0 the until clause never fires, so the Riccati never converges. The split is
        then capped at n-1 (a one-step tail -- a single tail step is exactly the Kalman step), and
        the gradient of this degenerate no-convergence case must still match StandardFilter."""
        m, p, n_shocks, n = 4, 2, 4, 40
        vals = _make_stationary_system(m, p, n_shocks, n, rng)

        _, grads_std = self._forward_and_gradients(self.standard, vals)
        _, grads_conv = self._forward_and_gradients(self.convergent_never_converging, vals)
        assert_results_match(grads_std, grads_conv, GRAD_NAMES, err_prefix="tol=0: ")


def test_convergent_filter_rejects_time_varying_params():
    """Declaring any matrix time-varying should raise ValueError at build time."""
    data = pt.matrix("data")
    a0 = pt.vector("a0")
    P0 = pt.matrix("P0")
    c = pt.vector("c")
    d = pt.vector("d")
    T = pt.matrix("T")
    Z = pt.matrix("Z")
    R = pt.matrix("R")
    H = pt.matrix("H")
    Q = pt.matrix("Q")
    with pytest.raises(ValueError, match=r"time-invariant.*\['transition'\]"):
        ConvergentFilter(time_varying_names=["transition"])


def test_convergent_filter_rejects_nan_constant_data():
    """NaN in a TensorConstant data tensor should raise ValueError at build time."""
    n, p = 10, 2
    data_arr = np.zeros((n, p), dtype=floatX)
    data_arr[3, 0] = np.nan
    data = pt.as_tensor(data_arr)
    a0 = pt.vector("a0")
    P0 = pt.matrix("P0")
    c = pt.vector("c")
    d = pt.vector("d")
    T = pt.matrix("T")
    Z = pt.matrix("Z")
    R = pt.matrix("R")
    H = pt.matrix("H")
    Q = pt.matrix("Q")
    with pytest.raises(ValueError, match="missing data"):
        ConvergentFilter().build_graph(data, a0, P0, c, d, T, Z, R, H, Q)


def test_convergent_filter_rejects_missing_fill_sentinel(rng):
    """The statespace core replaces NaN with missing_fill_value before the filter runs, so the
    sentinel -- not just NaN -- must be rejected. This mirrors the real PyMC path, where data is a
    shared variable holding the pre-filled values."""
    m, p, n_shocks, n = 3, 2, 3, 30
    vals = _make_stationary_system(m, p, n_shocks, n, rng)
    data_np = vals[0].copy()
    data_np[5, 0] = MISSING_FILL  # a missing observation, pre-filled by the statespace core
    shared_data = pytensor.shared(data_np, name="data")

    _, a0, P0, c, d, T, Z, R, H, Q = (pt.as_tensor_variable(v) for v in vals)
    with pytest.raises(ValueError, match="missing data"):
        ConvergentFilter().build_graph(
            pt.as_tensor_variable(shared_data), a0, P0, c, d, T, Z, R, H, Q
        )


def test_convergent_filter_builds_and_runs_at_float32(rng):
    """The filter follows ``pytensor.config.floatX``; nothing in its graph is pinned to float64."""
    m, p, n_shocks, n = 3, 2, 3, 60
    vals = _make_stationary_system(m, p, n_shocks, n, rng)

    with pytensor.config.change_flags(floatX="float32"):
        dtype = pytensor.config.floatX
        shapes = {
            "data": (n, p),
            "a0": (m,),
            "P0": (m, m),
            "c": (m,),
            "d": (p,),
            "T": (m, m),
            "Z": (p, m),
            "R": (m, n_shocks),
            "H": (p, p),
            "Q": (n_shocks, n_shocks),
        }
        inputs = [pt.tensor(name, dtype=dtype, shape=shape) for name, shape in shapes.items()]

        *_, ll_obs = ConvergentFilter().build_graph(*inputs)
        loss = ll_obs.sum()
        fn = pytensor.function(inputs, [loss, *pt.grad(loss, inputs[1:])], on_unused_input="ignore")

    results = fn(*[np.asarray(v, dtype=dtype) for v in vals])
    for name, value in zip(GRAD_NAMES, results, strict=True):
        assert np.all(np.isfinite(value)), f"{name} is not finite at float32"
