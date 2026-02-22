"""
Robustness tests: BFGS init retry, short-history fallback, infinite-logP step tolerance.
"""

from unittest.mock import patch

import numpy as np
import pytest

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pytensor.compile.mode import Mode

from pymc_extras.inference.pathfinder.lbfgs import (
    LBFGSInitFailed,
    LBFGSStatus,
    _CachedValueGrad,
)
from pymc_extras.inference.pathfinder.pathfinder import (
    DEFAULT_LINKER,
    FAILED_PATH_STATUS,
    LBFGSStreamingCallback,
    PathInvalidLogP,
    PathStatus,
    get_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
    make_single_pathfinder_fn,
)
from tests.pathfinder.equivalence_models import MODEL_FACTORIES, make_ard_regression

COMPILE_KWARGS = {"mode": Mode(linker=DEFAULT_LINKER)}

NUM_DRAWS = 50
NUM_ELBO_DRAWS = 100
MAXCOR = 6
MAXITER = 100
JITTER = 2.0
EPSILON = 1e-8


def _make_single_fn(
    model,
    maxiter: int = MAXITER,
    maxcor: int = MAXCOR,
    max_init_retries: int = 10,
):
    return make_single_pathfinder_fn(
        model=model,
        num_draws=NUM_DRAWS,
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=1e-5,
        gtol=1e-8,
        maxls=1000,
        num_elbo_draws=NUM_ELBO_DRAWS,
        jitter=JITTER,
        epsilon=EPSILON,
        max_init_retries=max_init_retries,
        compile_kwargs=COMPILE_KWARGS,
    )


def _run_path(fn, seed: int = 42):
    return fn(seed)


def _build_logp_and_neg(model, compile_kwargs=COMPILE_KWARGS):
    logp_dlogp = get_logp_dlogp_of_ravel_inputs(model, jacobian=True, **compile_kwargs)

    def neg_logp_dlogp_func(x):
        v, dv = logp_dlogp(x)
        return -v, -dv

    return None, neg_logp_dlogp_func


# ---------------------------------------------------------------------------
# BFGS init retry
# ---------------------------------------------------------------------------


def _make_failing_lbfgs_patcher(fail_k: int, real_neg_logp_dlogp_func):
    """Patch LBFGS to raise LBFGSInitFailed for first fail_k invocations."""
    call_count = [0]

    class PatchedLBFGS:
        def __init__(self, *args, **kwargs):
            from pymc_extras.inference.pathfinder.lbfgs import LBFGS as RealLBFGS

            self._real = RealLBFGS(*args, **kwargs)

        def minimize_streaming(self, callback, x0):
            call_count[0] += 1
            if call_count[0] <= fail_k:
                raise LBFGSInitFailed(LBFGSStatus.INIT_FAILED)
            return self._real.minimize_streaming(callback, x0)

    return patch("pymc_extras.inference.pathfinder.pathfinder.LBFGS", PatchedLBFGS), call_count


def test_retry_succeeds():
    """Path succeeds after K LBFGSInitFailed attempts when max_init_retries >= K."""
    model = make_ard_regression()
    fail_k = 3
    max_init_retries = 5

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k, None)

    with patcher:
        fn = _make_single_fn(model, max_init_retries=max_init_retries)
        result = fn(42)

    assert result.path_status not in FAILED_PATH_STATUS
    assert result.samples is not None
    assert call_count[0] == fail_k + 1


def test_retry_exhausted():
    """Path returns LBFGS_FAILED after all max_init_retries are exhausted."""
    model = make_ard_regression()
    max_init_retries = 2
    fail_k = max_init_retries + 1

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k, None)

    with patcher:
        fn = _make_single_fn(model, max_init_retries=max_init_retries)
        result = fn(99)

    assert result.path_status == PathStatus.LBFGS_FAILED
    assert call_count[0] == max_init_retries + 1


def test_no_retry_on_non_init_failure():
    """LBFGSException (non-init) is NOT retried."""
    from pymc_extras.inference.pathfinder.lbfgs import LBFGSException

    model = make_ard_regression()
    call_count = [0]

    class FailWithLBFGSException:
        def __init__(self, *args, **kwargs):
            pass

        def minimize_streaming(self, callback, x0):
            call_count[0] += 1
            raise LBFGSException("non-init failure", LBFGSStatus.LBFGS_FAILED)

    with patch("pymc_extras.inference.pathfinder.pathfinder.LBFGS", FailWithLBFGSException):
        fn = _make_single_fn(model, max_init_retries=5)
        result = fn(7)

    assert result.path_status == PathStatus.LBFGS_FAILED
    assert call_count[0] == 1


def test_progress_callback_retry():
    """progress_callback receives 'retry N' status on each retry attempt."""
    model = make_ard_regression()
    fail_k = 2
    max_init_retries = 3

    status_updates = []

    def cb(info):
        if "status" in info and info["status"] is not None:
            status_updates.append(info["status"])

    patcher, _ = _make_failing_lbfgs_patcher(fail_k, None)

    with patcher:
        fn = _make_single_fn(model, max_init_retries=max_init_retries)
        fn(11, progress_callback=cb)

    retry_statuses = [s for s in status_updates if s.startswith("retry")]
    assert len(retry_statuses) == fail_k
    terminal_statuses = [s for s in status_updates if s in ("ok", "elbo@0")]
    assert len(terminal_statuses) >= 1


# ---------------------------------------------------------------------------
# Short-history fallback (maxiter << J)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["ard_regression", "bpca_small"])
def test_short_history_fallback(model_name):
    """Streaming handles partial windows (L < J) via zero-padding without crashing."""
    model = MODEL_FACTORIES[model_name]()

    for maxiter in (1, 2, 3):
        fn = _make_single_fn(model, maxiter=maxiter)
        result = _run_path(fn, seed=99)
        assert result.path_status in list(PathStatus)
        if result.path_status not in FAILED_PATH_STATUS and result.samples is not None:
            N = DictToArrayBijection.map(model.initial_point()).data.shape[0]
            assert result.samples.shape == (1, NUM_DRAWS, N)


# ---------------------------------------------------------------------------
# Infinite-logP step tolerance
# ---------------------------------------------------------------------------


def test_infinite_logp_step_tolerance():
    """A step where sample_logp_fn raises is silently skipped; path still succeeds."""
    model = make_ard_regression()
    _, neg_logp_dlogp_func = _build_logp_and_neg(model)
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    sample_logp_fn = make_pathfinder_sample_fn(
        model, N, MAXCOR, jacobian=True, compile_kwargs=COMPILE_KWARGS
    )

    fail_at_step = [1]
    call_count = [0]

    def failing_fn(x, g, alpha, s_win, z_win, u):
        idx = call_count[0]
        call_count[0] += 1
        if idx == fail_at_step[0]:
            raise PathInvalidLogP("synthetic failure for test")
        return sample_logp_fn(x, g, alpha, s_win, z_win, u)

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    rng = np.random.default_rng(77)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    cached_fn = _CachedValueGrad(neg_logp_dlogp_func)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=failing_fn,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng,
        J=MAXCOR,
        epsilon=EPSILON,
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
