import contextlib
import sys

import numpy as np
import pytest

from rich.file_proxy import FileProxy

import pymc_extras as pmx

from pymc_extras.inference.pathfinder import multipath as multipath_mod
from pymc_extras.inference.pathfinder.multipath import (
    REFRESH_INTERVAL,
    _make_multipath_progress,
    _make_progress_callback,
    _make_refresher,
    _resolve_mp_context,
)
from tests.inference.pathfinder.equivalence_models import make_ard_regression


def test_concurrent_results(eight_schools_model):
    # Serial and parallel execution of the same seed must agree to within sampling noise.
    with eight_schools_model:
        idata_serial = pmx.fit(
            method="pathfinder", num_paths=10, jitter=12.0, random_seed=41, parallel=False
        )
        idata_parallel = pmx.fit(
            method="pathfinder", num_paths=10, jitter=12.0, random_seed=41, parallel=True
        )

    np.testing.assert_allclose(
        idata_serial.posterior.mu.data.mean(),
        idata_parallel.posterior.mu.data.mean(),
        atol=0.4,
    )
    np.testing.assert_allclose(
        idata_serial.posterior.tau.data.mean(),
        idata_parallel.posterior.tau.data.mean(),
        atol=0.4,
    )


def _per_path_means(posterior):
    """Stack each path's (chain's) per-variable posterior means into one row per path, in chain
    order."""
    rows = []
    for c in range(posterior.sizes["chain"]):
        rows.append(
            np.concatenate(
                [
                    np.atleast_1d(posterior[v].values[c].mean(axis=0).ravel())
                    for v in sorted(posterior.data_vars)
                ]
            )
        )
    return np.asarray(rows)


def test_parallel_paths_match_serial_per_path():
    """Each chain gets the same per-path approximation under parallel and serial execution (same
    seed, within cross-process BLAS noise) -- guarding against state leaking across the shared
    compiled functions."""
    if sys.platform == "win32":
        pytest.skip("non-deterministic on Windows CI workers")

    model = make_ard_regression()
    kw = dict(
        method="pathfinder",
        num_paths=4,
        num_draws=80,
        num_draws_per_path=80,
        num_elbo_draws=8,
        jitter=2.0,
        random_seed=13,
        importance_sampling=None,
        progressbar=False,
    )
    with model:
        serial = pmx.fit(parallel=False, **kw).posterior
        parallel = pmx.fit(parallel=True, **kw).posterior

    assert serial.sizes["chain"] == parallel.sizes["chain"] == kw["num_paths"]
    np.testing.assert_allclose(
        _per_path_means(serial), _per_path_means(parallel), rtol=1e-3, atol=1e-3
    )


def _small_serial_fit(model, **overrides):
    """Run a small serial pathfinder fit on ``model``, with per-test ``overrides`` applied to the
    shared config below."""
    kwargs = dict(
        method="pathfinder",
        parallel=False,
        num_paths=2,
        num_draws=20,
        num_draws_per_path=20,
        num_elbo_draws=4,
        random_seed=1,
        progressbar=False,
    )
    kwargs.update(overrides)
    with model:
        return pmx.fit(**kwargs)


@pytest.mark.parametrize(
    "system, expected", [("Darwin", "spawn"), ("Linux", None)], ids=["macos", "linux"]
)
def test_default_context_spawns_on_macos(monkeypatch, system, expected):
    """macOS defaults to spawn; elsewhere the choice is left to pymc's resolver."""
    seen = []
    monkeypatch.setattr(multipath_mod.platform, "system", lambda: system)
    monkeypatch.setattr(
        multipath_mod,
        "_initialize_multiprocessing_context",
        lambda mp_ctx, *, mode=None, quiet=False: seen.append(mp_ctx),
    )

    _resolve_mp_context(None, mode=None)

    assert seen == [expected]


@pytest.mark.skipif(sys.platform == "win32", reason="fork start method is unavailable on Windows")
@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "serial"])
def test_fork_caps_blas_to_one_thread_only_while_workers_run(monkeypatch, parallel):
    """A parallel fit under fork caps BLAS to one thread in the parent so forked workers inherit
    the cap; a serial fit keeps the parent's full BLAS width."""
    requested = []
    monkeypatch.setattr(
        multipath_mod,
        "threadpool_limits",
        lambda limits=None: (requested.append(limits), contextlib.nullcontext())[1],
    )

    _small_serial_fit(make_ard_regression(), parallel=parallel, mp_ctx="fork")

    assert requested == ([1] if parallel else [])


def test_compile_mode_threaded_to_mp_context(monkeypatch):
    """The compile ``mode`` must reach the multiprocessing-context resolver so a JAX backend can
    dodge a fork+JAX deadlock; resolution is unconditional, so a serial fit exercises it."""
    seen = []
    real = multipath_mod._initialize_multiprocessing_context

    def spy(mp_ctx, *, mode=None, quiet=False):
        seen.append(mode)
        return real(mp_ctx, mode=mode, quiet=quiet)

    monkeypatch.setattr(multipath_mod, "_initialize_multiprocessing_context", spy)

    _small_serial_fit(make_ard_regression(), compile_kwargs={"mode": "NUMBA"})

    assert seen == ["NUMBA"]


def test_blas_limiter_wraps_path_execution(monkeypatch):
    """Paths must run inside ``joined_blas_limiter()``; a recording limiter makes the wrap
    observable even on fork, where the real limiter is a no-op."""
    entered = []
    real = multipath_mod.setup_cores_blas_cores

    @contextlib.contextmanager
    def recording_limiter():
        entered.append("enter")
        yield
        entered.append("exit")

    def patched(blas_cores, chains, cores, mp_ctx):
        _, eff_cores, per_worker = real(blas_cores, chains, cores, mp_ctx)
        return recording_limiter, eff_cores, per_worker

    monkeypatch.setattr(multipath_mod, "setup_cores_blas_cores", patched)

    _small_serial_fit(make_ard_regression())

    assert entered == ["enter", "exit"]


def test_interrupt_keeps_completed_paths(monkeypatch):
    """A Ctrl-C mid-run keeps the paths that already finished instead of discarding them all."""
    real_make_generator = multipath_mod.make_generator

    def interrupt_after_one(*args, **kwargs):
        gen = real_make_generator(*args, **kwargs)
        yield next(gen)
        gen.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(multipath_mod, "make_generator", interrupt_after_one)

    idata = _small_serial_fit(make_ard_regression(), num_paths=3, importance_sampling=None)

    # importance_sampling=None makes each completed path its own chain; only one finished.
    assert idata.posterior.sizes["chain"] == 1


def test_interrupt_before_any_path_propagates(monkeypatch):
    """An interrupt before any path completes aborts the run rather than fabricating an empty
    result."""

    def interrupt_immediately(*args, **kwargs):
        # Must be a generator: the raise has to fire during iteration (inside the drain's
        # try/except), not at the make_generator() call site, or the empty-results re-raise
        # branch isn't the thing under test.
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr(multipath_mod, "make_generator", interrupt_immediately)

    with pytest.raises(KeyboardInterrupt):
        _small_serial_fit(make_ard_regression())


def test_parallel_path_order_is_deterministic(monkeypatch):
    """A fixed seed gives identical output regardless of path completion order: results are
    reassembled by chain index, not arrival order."""
    real = multipath_mod.make_generator

    def run(reorder):
        def patched(*args, **kwargs):
            yield from reorder(list(real(*args, **kwargs)))

        monkeypatch.setattr(multipath_mod, "make_generator", patched)
        return _small_serial_fit(make_ard_regression(), num_paths=4, importance_sampling=None)

    forward = run(lambda pairs: pairs).posterior
    backward = run(lambda pairs: pairs[::-1]).posterior

    for var in forward.data_vars:
        np.testing.assert_array_equal(forward[var].values, backward[var].values)


def test_live_progress_display_starts_no_thread_and_no_stream_proxies():
    """Workers fork while the display is live, so it must own no refresh thread and no
    stdout/stderr proxies for a child to inherit locked (pymc#8421)."""
    with _make_multipath_progress(progressbar=True) as progress:
        assert progress.live._refresh_thread is None
        assert not isinstance(sys.stdout, FileProxy)
        assert not isinstance(sys.stderr, FileProxy)


def _new_task():
    progress = _make_multipath_progress(progressbar=False)
    task_id = progress.add_task(
        "path 0",
        start=False,
        status="",
        elbo="",
        speed=0.0,
        speed_unit="it/s",
        total=1000,
        completed=0,
    )
    return progress, task_id


def test_progress_callback_starts_the_clock_on_first_message():
    """A queued path's elapsed time counts from its first message, not from the run's start."""
    progress, task_id = _new_task()
    cb = _make_progress_callback(progress, task_id)
    assert progress.tasks[0].start_time is None

    cb({"status": "running"})

    assert progress.tasks[0].start_time is not None


def test_progress_callback_formats_fields():
    progress, task_id = _new_task()
    cb = _make_progress_callback(progress, task_id)

    cb({"status": "running", "iteration": 7, "best_elbo": 1.23456})
    task = progress.tasks[0]
    assert task.fields["status"] == "running"
    assert task.completed == 7
    assert task.fields["elbo"] == "1.235"

    cb({"best_elbo": np.inf})  # non-finite renders as a dash
    assert progress.tasks[0].fields["elbo"] == "—"


def test_serial_paths_redraw_after_every_message():
    """With Rich's refresh thread off, the executor is the only thing that redraws."""
    messages = [{"iteration": 1}, {"best_elbo": 0.5}, {"status": "ok"}]

    def fake_path(seed, cb):
        for info in messages:
            cb(info)
        return seed

    redraws = []
    results = list(
        multipath_mod._execute_serially(
            fake_path,
            seeds=[7, 8],
            progress_callbacks=[lambda info: None] * 2,
            refresh=lambda: redraws.append(1),
        )
    )

    assert results == [(0, 7), (1, 8)]
    assert len(redraws) == 2 * len(messages)


def test_parallel_fit_redraws_from_the_parent(monkeypatch):
    redraws = []
    monkeypatch.setattr(
        multipath_mod, "_make_refresher", lambda progress: lambda: redraws.append(1)
    )

    _small_serial_fit(make_ard_regression(), parallel=True, num_paths=2)

    assert redraws


def test_refresher_throttles_to_the_refresh_interval(monkeypatch):
    progress, _ = _new_task()
    redraws = []
    monkeypatch.setattr(progress, "refresh", lambda: redraws.append(1))
    clock = iter([1.0, 1.0 + REFRESH_INTERVAL / 2, 1.0 + 2 * REFRESH_INTERVAL])
    monkeypatch.setattr(multipath_mod.time, "monotonic", lambda: next(clock))
    refresh = _make_refresher(progress)

    refresh()
    refresh()
    refresh()

    assert len(redraws) == 2


def test_progress_callback_stops_task_on_terminal_status():
    progress, task_id = _new_task()
    cb = _make_progress_callback(progress, task_id)

    cb({"status": "ok"})

    assert progress.tasks[0].stop_time is not None
