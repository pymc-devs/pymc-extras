import contextlib

import numpy as np

import pymc_extras as pmx

from pymc_extras.inference.pathfinder import multipath as multipath_mod
from pymc_extras.inference.pathfinder.multipath import (
    _make_multipath_progress,
    _make_progress_callback,
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
    """Stack each path's (chain's) per-variable posterior means into one row per path, then
    sort the rows by a content key so the two execution modes can be aligned despite differing
    path completion order."""
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
    rows = np.asarray(rows)
    return rows[np.argsort(rows[:, 0])]


def test_parallel_paths_match_serial_per_path():
    """Process-isolated parallel paths must produce the same per-path approximations as serial
    execution — not merely finite output.

    With ``importance_sampling=None`` each path is a separate ``chain``, so we compare per-path
    posterior means between modes (aligned by content, since parallel paths complete out of order).
    The paths share one set of compiled functions, so any state leaking across paths through those
    functions would make a path's approximation depend on its co-runners, diverging between serial
    and parallel; identical seeds otherwise pin each path's result, so the two modes must agree
    tightly.
    """
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


def test_compile_mode_threaded_to_mp_context(monkeypatch):
    """The compile ``mode`` must reach ``_initialize_multiprocessing_context`` so a JAX backend
    forces a non-fork start method -- fork + JAX can deadlock, which pymc.sample guards against the
    same way. Resolution happens once, unconditionally, so a serial fit exercises the threading
    without spawning workers.
    """
    seen = []
    real = multipath_mod._initialize_multiprocessing_context

    def spy(mp_ctx, *, mode=None, quiet=False):
        seen.append(mode)
        return real(mp_ctx, mode=mode, quiet=quiet)

    monkeypatch.setattr(multipath_mod, "_initialize_multiprocessing_context", spy)

    model = make_ard_regression()
    with model:
        pmx.fit(
            method="pathfinder",
            parallel=False,
            num_paths=2,
            num_draws=20,
            num_draws_per_path=20,
            num_elbo_draws=4,
            random_seed=1,
            compile_kwargs={"mode": "NUMBA"},
            progressbar=False,
        )

    assert seen == ["NUMBA"]


def test_blas_limiter_wraps_path_execution(monkeypatch):
    """Paths must run inside ``joined_blas_limiter()``, as pymc.sample / pymc.smc do. A recording
    limiter is substituted so the wrap is observable even on fork (where the real limiter is a
    no-op); a serial fit suffices since the limiter wraps both modes.
    """
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

    model = make_ard_regression()
    with model:
        pmx.fit(
            method="pathfinder",
            parallel=False,
            num_paths=2,
            num_draws=20,
            num_draws_per_path=20,
            num_elbo_draws=4,
            random_seed=1,
            progressbar=False,
        )

    assert entered == ["enter", "exit"]


def _new_task():
    progress = _make_multipath_progress(progressbar=False)
    task_id = progress.add_task(
        "path 0", status="", elbo="", speed=0.0, speed_unit="it/s", total=1000, completed=0
    )
    return progress, task_id


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


def test_progress_callback_stops_task_on_terminal_status():
    progress, task_id = _new_task()
    cb = _make_progress_callback(progress, task_id)

    cb({"status": "ok"})

    assert progress.tasks[0].stop_time is not None
