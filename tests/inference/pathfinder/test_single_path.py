from unittest.mock import patch

import numpy as np
import pytest

from pymc.blocking import DictToArrayBijection

from pymc_extras.inference.pathfinder.lbfgs import LBFGSInitFailed, LBFGSStatus
from pymc_extras.inference.pathfinder.results import FAILED_PATH_STATUS, PathStatus
from tests.inference.pathfinder.conftest import NUM_DRAWS, make_single_fn
from tests.inference.pathfinder.equivalence_models import MODEL_FACTORIES, make_ard_regression


def _make_failing_lbfgs_patcher(fail_k: int):
    """Patch LBFGS to raise LBFGSInitFailed for the first fail_k invocations."""
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

    return patch("pymc_extras.inference.pathfinder.single_path.LBFGS", PatchedLBFGS), call_count


def test_retry_succeeds():
    """Path succeeds after K LBFGSInitFailed attempts when max_init_retries >= K."""
    model = make_ard_regression()
    fail_k = 3
    max_init_retries = 5

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k)

    with patcher:
        fn = make_single_fn(model, max_init_retries=max_init_retries)
        result = fn(42)

    assert result.path_status not in FAILED_PATH_STATUS
    assert result.samples is not None
    assert call_count[0] == fail_k + 1


def test_retry_exhausted():
    """Path returns LBFGS_FAILED after all max_init_retries are exhausted."""
    model = make_ard_regression()
    max_init_retries = 2
    fail_k = max_init_retries + 1

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k)

    with patcher:
        fn = make_single_fn(model, max_init_retries=max_init_retries)
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

    with patch("pymc_extras.inference.pathfinder.single_path.LBFGS", FailWithLBFGSException):
        fn = make_single_fn(model, max_init_retries=5)
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

    patcher, _ = _make_failing_lbfgs_patcher(fail_k)

    with patcher:
        fn = make_single_fn(model, max_init_retries=max_init_retries)
        fn(11, progress_callback=cb)

    retry_statuses = [s for s in status_updates if s.startswith("retry")]
    assert len(retry_statuses) == fail_k
    terminal_statuses = [s for s in status_updates if s in ("ok", "elbo@0")]
    assert len(terminal_statuses) >= 1


def test_shared_single_fn_is_leak_free():
    """A path's result must not depend on what other seeds ran on the same compiled fn.

    Paths share one ``single_pathfinder_fn`` (the per-path ``.copy(share_memory=False)`` of the
    compiled functions was removed, since paths run either serially or in isolated processes). The
    compiled functions and their I/O containers must therefore carry no cross-path state: reusing
    one fn for several seeds must yield bit-identical results to running each seed alone on its own
    freshly built fn. This is the correctness guarantee that replaces the removed copies — if
    serial reuse is leak-free, process-isolated parallel execution is safe a fortiori.
    """
    model = make_ard_regression()
    seeds = [3, 7, 11, 19]

    # Reference: each seed as the sole call on its own freshly built fn.
    refs = {s: make_single_fn(model)(s) for s in seeds}

    # Reuse one fn across all seeds, in reverse order, so any retained state from an earlier
    # seed would perturb a later one.
    shared = make_single_fn(model)
    out = {s: shared(s) for s in reversed(seeds)}

    compared = 0
    for s in seeds:
        r, ref = out[s], refs[s]
        assert r.path_status == ref.path_status
        if ref.samples is None:
            assert r.samples is None
            continue
        np.testing.assert_array_equal(r.samples, ref.samples)
        np.testing.assert_array_equal(r.logP, ref.logP)
        np.testing.assert_array_equal(r.logQ, ref.logQ)
        np.testing.assert_array_equal(r.lbfgs_niter, ref.lbfgs_niter)
        np.testing.assert_array_equal(r.inv_hessian_diag, ref.inv_hessian_diag)
        compared += 1
    assert compared >= 2, "need >=2 successful paths to meaningfully exercise fn reuse"


@pytest.mark.parametrize("model_name", ["ard_regression", "bpca_small"])
def test_short_history_fallback(model_name):
    """Streaming handles partial windows (L < J) via zero-padding without crashing."""
    model = MODEL_FACTORIES[model_name]()

    for maxiter in (1, 2, 3):
        fn = make_single_fn(model, maxiter=maxiter)
        # Reaching the assertions below means streaming did not raise on the partial window.
        result = fn(99)
        if result.path_status not in FAILED_PATH_STATUS and result.samples is not None:
            N = DictToArrayBijection.map(model.initial_point()).data.shape[0]
            assert result.samples.shape == (1, NUM_DRAWS, N)
