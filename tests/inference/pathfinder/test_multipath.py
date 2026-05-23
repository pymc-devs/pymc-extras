import numpy as np

import pymc_extras as pmx

from pymc_extras.inference.pathfinder.multipath import (
    _make_multipath_progress,
    _make_progress_callback,
)


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
