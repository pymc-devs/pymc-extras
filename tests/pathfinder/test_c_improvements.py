"""
T-C1, T-C2: acceptance tests for Section C misc improvements.

T-C1: BFGS init retry — LBFGSInitFailed triggers re-jitter up to max_init_retries times.
T-C2: Parallel vs serial output parity — thread and serial modes produce equivalent per-path results.
"""

import threading

from unittest.mock import patch

import numpy as np

from pytensor.compile.mode import Mode

from pymc_extras.inference.pathfinder.lbfgs import (
    LBFGSInitFailed,
    LBFGSStatus,
)
from pymc_extras.inference.pathfinder.pathfinder import (
    DEFAULT_LINKER,
    FAILED_PATH_STATUS,
    PathStatus,
    make_single_pathfinder_fn,
    multipath_pathfinder,
)
from tests.pathfinder.equivalence_models import make_iso_gaussian

COMPILE_KWARGS = {"mode": Mode(linker=DEFAULT_LINKER)}

NUM_DRAWS = 20
NUM_ELBO_DRAWS = 10
MAXCOR = 6
MAXITER = 50
JITTER = 2.0
EPSILON = 1e-8


def _make_single_fn(model, max_init_retries: int = 10):
    return make_single_pathfinder_fn(
        model=model,
        num_draws=NUM_DRAWS,
        maxcor=MAXCOR,
        maxiter=MAXITER,
        ftol=1e-5,
        gtol=1e-8,
        maxls=1000,
        num_elbo_draws=NUM_ELBO_DRAWS,
        jitter=JITTER,
        epsilon=EPSILON,
        max_init_retries=max_init_retries,
        compile_kwargs=COMPILE_KWARGS,
    )


# ---------------------------------------------------------------------------
# T-C1: BFGS init retry
# ---------------------------------------------------------------------------


class _FailFirstK:
    """Wraps a value_grad_fn so that the first K calls return non-finite values."""

    def __init__(self, fn, k: int):
        self._fn = fn
        self._k = k
        self._call_count = 0
        self.lock = threading.Lock()

    def __call__(self, x):
        with self.lock:
            idx = self._call_count
            self._call_count += 1
        if idx < self._k:
            return np.inf, np.full_like(x, np.nan)
        return self._fn(x)


def _make_failing_lbfgs_patcher(fail_k: int, real_neg_logp_dlogp_func):
    """
    Patch LBFGS so that minimize() / minimize_streaming() raise LBFGSInitFailed
    for the first `fail_k` invocations, then succeed on subsequent invocations.
    """
    call_count = [0]

    original_minimize = None
    original_minimize_streaming = None

    class PatchedLBFGS:
        def __init__(self, *args, **kwargs):
            # Store the real LBFGS so we can delegate on success
            from pymc_extras.inference.pathfinder.lbfgs import LBFGS as RealLBFGS

            self._real = RealLBFGS(*args, **kwargs)

        def minimize_streaming(self, callback, x0):
            call_count[0] += 1
            if call_count[0] <= fail_k:
                raise LBFGSInitFailed(LBFGSStatus.INIT_FAILED)
            return self._real.minimize_streaming(callback, x0)

    return patch("pymc_extras.inference.pathfinder.pathfinder.LBFGS", PatchedLBFGS), call_count


def test_tc1_retry_succeeds():
    """Path succeeds after K LBFGSInitFailed attempts when max_init_retries >= K."""
    model = make_iso_gaussian()
    fail_k = 3
    max_init_retries = 5  # >= fail_k

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k, None)

    with patcher:
        fn = _make_single_fn(model, max_init_retries=max_init_retries)
        result = fn(42)

    assert (
        result.path_status not in FAILED_PATH_STATUS
    ), f"Expected success after retries, got path_status={result.path_status}"
    assert result.samples is not None
    # LBFGS was called fail_k times (failures) + 1 (success) = fail_k + 1
    assert call_count[0] == fail_k + 1, f"Expected {fail_k + 1} LBFGS calls, got {call_count[0]}"


def test_tc1_retry_exhausted():
    """Path returns LBFGS_FAILED after all max_init_retries are exhausted."""
    model = make_iso_gaussian()
    max_init_retries = 2
    fail_k = max_init_retries + 1  # always fails

    patcher, call_count = _make_failing_lbfgs_patcher(fail_k, None)

    with patcher:
        fn = _make_single_fn(model, max_init_retries=max_init_retries)
        result = fn(99)

    assert (
        result.path_status == PathStatus.LBFGS_FAILED
    ), f"Expected LBFGS_FAILED after retries exhausted, got {result.path_status}"
    # LBFGS was called max_init_retries + 1 times (all failed)
    assert (
        call_count[0] == max_init_retries + 1
    ), f"Expected {max_init_retries + 1} LBFGS calls, got {call_count[0]}"


def test_tc1_no_retry_on_non_init_failure():
    """LBFGSException (non-init) is NOT retried — only LBFGSInitFailed is."""
    from pymc_extras.inference.pathfinder.lbfgs import LBFGSException

    model = make_iso_gaussian()
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
    # Should have been called exactly once — no retries for non-init failures
    assert call_count[0] == 1, f"Expected 1 LBFGS call (no retry), got {call_count[0]}"


def test_tc1_progress_callback_retry():
    """progress_callback receives 'retry N' status on each retry attempt."""
    model = make_iso_gaussian()
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

    # Should have received "retry 1" and "retry 2" (fail_k=2 failures)
    retry_statuses = [s for s in status_updates if s.startswith("retry")]
    assert (
        len(retry_statuses) == fail_k
    ), f"Expected {fail_k} retry status updates, got {retry_statuses}"
    # Final status should be "ok" or "elbo@0"
    terminal_statuses = [s for s in status_updates if s in ("ok", "elbo@0")]
    assert len(terminal_statuses) >= 1, f"Expected terminal ok/elbo@0, got {status_updates}"


# ---------------------------------------------------------------------------
# T-C2: Parallel vs serial output parity
# ---------------------------------------------------------------------------


def test_tc2_thread_vs_serial_parity():
    """Thread and serial execution produce equivalent per-path results (same seeds)."""
    model = make_iso_gaussian()
    num_paths = 3
    random_seed = 42

    common_kwargs = dict(
        model=model,
        num_paths=num_paths,
        num_draws=NUM_DRAWS,
        num_draws_per_path=NUM_DRAWS,
        maxcor=MAXCOR,
        maxiter=MAXITER,
        ftol=1e-5,
        gtol=1e-8,
        maxls=1000,
        num_elbo_draws=NUM_ELBO_DRAWS,
        jitter=JITTER,
        epsilon=EPSILON,
        importance_sampling=None,
        progressbar=False,
        max_init_retries=0,
        random_seed=random_seed,
        compile_kwargs=COMPILE_KWARGS,
        display_summary=False,
    )

    result_serial = multipath_pathfinder(**common_kwargs, concurrent=None)
    result_thread = multipath_pathfinder(**common_kwargs, concurrent="thread")

    # Both should yield the same number of successful paths
    assert result_serial.num_paths == result_thread.num_paths

    # Per-path lbfgs_niter and path_status should match (paths are independently seeded)
    serial_statuses = sorted(str(s) for s in result_serial.path_status)
    thread_statuses = sorted(str(s) for s in result_thread.path_status)
    assert (
        serial_statuses == thread_statuses
    ), f"Path statuses differ: serial={serial_statuses}, thread={thread_statuses}"

    serial_niters = sorted(int(n) for n in result_serial.lbfgs_niter if n is not None)
    thread_niters = sorted(int(n) for n in result_thread.lbfgs_niter if n is not None)
    assert (
        serial_niters == thread_niters
    ), f"lbfgs_niter differ: serial={serial_niters}, thread={thread_niters}"


def test_tc2_default_concurrent_is_process():
    """fit_pathfinder and multipath_pathfinder default to concurrent='process'.

    'thread' is not a supported option: pytensor's Function.copy(share_memory=False)
    only replaces I/O storage, not intermediate op buffers, so concurrent thread calls
    corrupt each other's in-flight state.  Processes have fully independent memory.
    """
    import inspect

    from pymc_extras.inference.pathfinder.pathfinder import fit_pathfinder

    mp_sig = inspect.signature(multipath_pathfinder)
    fp_sig = inspect.signature(fit_pathfinder)

    assert (
        mp_sig.parameters["concurrent"].default == "process"
    ), "multipath_pathfinder should default concurrent='process'"
    assert (
        fp_sig.parameters["concurrent"].default == "process"
    ), "fit_pathfinder should default concurrent='process'"


def test_tc2_progress_callback_called_per_path():
    """progress_callback is called for each path with status updates."""
    model = make_iso_gaussian()
    callback_calls = {i: [] for i in range(3)}

    callbacks = [(lambda i: (lambda info: callback_calls[i].append(info)))(i) for i in range(3)]

    from pymc_extras.inference.pathfinder.pathfinder import make_generator

    fn = _make_single_fn(model)
    seeds = [10, 20, 30]

    results = list(
        make_generator(
            concurrent=None,
            fn=fn,
            seeds=seeds,
            progress_callbacks=callbacks,
        )
    )

    assert len(results) == 3
    # Each callback should have been called at least once (for "running" + terminal status)
    for i in range(3):
        assert len(callback_calls[i]) >= 1, f"Callback for path {i} was never called"
        # First call should set status="running"
        first_statuses = [c.get("status") for c in callback_calls[i] if "status" in c]
        assert (
            "running" in first_statuses
        ), f"Path {i} callback never received status='running': {callback_calls[i]}"
