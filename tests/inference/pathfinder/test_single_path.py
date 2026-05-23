from unittest.mock import patch

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
