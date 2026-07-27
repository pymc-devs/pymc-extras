import numpy as np
import pytest

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pytensor.tensor.optimize import LRUCache1
from scipy.optimize import OptimizeResult

from pymc_extras.inference.pathfinder.bfgs_sample import (
    get_neg_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
)
from pymc_extras.inference.pathfinder.lbfgs import (
    LBFGS,
    LBFGSConfig,
    LBFGSStatus,
    LBFGSStreamingCallback,
    _as_numpy_value_grad,
    _check_lbfgs_curvature_condition,
)
from tests.inference.pathfinder.conftest import (
    JITTER,
    MAXCOR,
    MAXITER,
    NUM_ELBO_DRAWS,
    STREAMING_EPSILON,
    run_streaming_lbfgs,
)
from tests.inference.pathfinder.equivalence_models import make_ard_regression


@pytest.mark.parametrize(
    "s, z, epsilon, expected",
    [
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1e-8, True),
        (np.array([1e-12, 0.0]), np.array([1.0, 0.0]), 1.0, False),
        (np.array([0.5, 0.0]), np.array([1.0, 0.0]), 0.5, True),  # s·z == epsilon·‖z‖²
    ],
    ids=["positive_curvature", "below_threshold", "boundary"],
)
def test_check_lbfgs_curvature_condition(s, z, epsilon, expected):
    assert _check_lbfgs_curvature_condition(s, z, epsilon) is expected


def test_as_numpy_value_grad_converts_and_preserves_dtype():
    """Foreign array outputs (e.g. MLX) become numpy without upcasting the computation's dtype."""

    class FakeArray:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=np.float32)

        def __array__(self, dtype=None, copy=None):
            return self._data if dtype is None else self._data.astype(dtype)

    wrapped = _as_numpy_value_grad(lambda x: (FakeArray(1.5), FakeArray([0.1, 0.2])))
    value, grad = wrapped(np.zeros(2))

    assert isinstance(value, np.ndarray)
    assert isinstance(grad, np.ndarray)
    assert value.dtype == np.float32
    assert grad.dtype == np.float32
    np.testing.assert_allclose(value, 1.5)
    np.testing.assert_allclose(grad, [0.1, 0.2])


def test_set_default_maxcor_keeps_explicit_value():
    config = LBFGSConfig(maxcor=7)
    assert config.set_default_maxcor(1000).maxcor == 7


@pytest.mark.parametrize(
    "N, expected",
    [
        (2, 5),  # ceil(3·ln2)=3, floored to 5
        (1000, 21),  # ceil(3·ln1000)=21
    ],
    ids=["small_N_floored", "large_N"],
)
def test_set_default_maxcor_heuristic(N, expected):
    assert LBFGSConfig().set_default_maxcor(N).maxcor == expected


@pytest.mark.parametrize(
    "nit, status, update_count, expected",
    [
        (1, 0, 1, LBFGSStatus.INIT_FAILED),
        (5, 0, 1, LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT),
        (10, 1, 5, LBFGSStatus.MAX_ITER_REACHED),
        (10, 2, 5, LBFGSStatus.NON_FINITE),
        (20, 0, 5, LBFGSStatus.LOW_UPDATE_PCT),  # 5·3 = 15 < 20
        (10, 0, 5, LBFGSStatus.CONVERGED),  # 5·3 = 15 not < 10
    ],
    ids=["init_failed", "init_low_pct", "max_iter", "non_finite", "low_update_pct", "converged"],
)
def test_classify_status(nit, status, update_count, expected):
    result = OptimizeResult(nit=nit, status=status)
    assert LBFGS._classify_status(result, update_count) is expected


@pytest.mark.parametrize("vectorize", [False, True])
def test_elbo_call_count(vectorize):
    """sample_logp_fn is called exactly once per accepted LBFGS step."""
    model = make_ard_regression()
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    sample_logp_fn = make_pathfinder_sample_fn(model, N, MAXCOR, jacobian=True, vectorize=vectorize)

    call_count = [0]

    def counting_fn(x, g, alpha, s_win, z_win, u):
        call_count[0] += 1
        return sample_logp_fn(x, g, alpha, s_win, z_win, u)

    cb, _, _ = run_streaming_lbfgs(model, counting_fn, seed=0)

    assert call_count[0] == cb.step_count
    assert cb.step_count > 0


def test_infinite_logp_step_tolerance():
    """A step that yields non-finite logP is scored as -inf and skipped; the path succeeds."""
    model = make_ard_regression()
    neg_logp_dlogp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    sample_logp_fn = make_pathfinder_sample_fn(model, N, MAXCOR, jacobian=True)

    fail_at_step = [1]
    call_count = [0]

    def failing_fn(x, g, alpha, s_win, z_win, u):
        idx = call_count[0]
        call_count[0] += 1
        phi, logQ, logP, diag = sample_logp_fn(x, g, alpha, s_win, z_win, u)
        if idx == fail_at_step[0]:
            # Mimic a degenerate step: the linalg ops return NaN logP on failure, never raise.
            logP = np.full_like(np.asarray(logP), np.nan)
        return phi, logQ, logP, diag

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    rng = np.random.default_rng(77)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    cached_fn = LRUCache1(neg_logp_dlogp, copy_x=True)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=failing_fn,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng,
        J=MAXCOR,
        epsilon=STREAMING_EPSILON,
    )

    from scipy.optimize import minimize as scipy_minimize

    scipy_minimize(
        cached_fn,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"maxcor": MAXCOR, "maxiter": MAXITER, "ftol": 1e-5, "gtol": 1e-8, "maxls": 1000},
    )

    assert cb.step_count >= 2
    assert cb.any_valid
    assert cb.best_step_idx != fail_at_step[0]
