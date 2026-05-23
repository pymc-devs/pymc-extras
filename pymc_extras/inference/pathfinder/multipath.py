import contextlib
import logging
import multiprocessing as mp
import time

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np

from pymc import Model
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.sampling.mcmc import setup_cores_blas_cores
from pymc.sampling.parallel import _cpu_count, _initialize_multiprocessing_context
from pymc.util import RandomSeed, _get_seeds_per_chain
from rich.console import Console
from rich.progress import TextColumn, TimeElapsedColumn
from rich.table import Column
from threadpoolctl import threadpool_limits

from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig, LBFGSStatus
from pymc_extras.inference.pathfinder.results import (
    MultiPathfinderResult,
    PathfinderConfig,
    PathfinderResult,
    PathStatus,
)
from pymc_extras.inference.pathfinder.single_path import (
    SinglePathfinderFn,
    make_single_pathfinder_fn,
)

logger = logging.getLogger(__name__)


def multipath_pathfinder(
    model: Model,
    num_paths: int,
    num_draws: int,
    num_draws_per_path: int,
    num_elbo_draws: int,
    jitter: float,
    lbfgs_config: LBFGSConfig,
    importance_sampling: Literal["psis", "psir", "identity"] | None,
    progressbar: bool,
    parallel: bool = True,
    cores: int | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
    mp_ctx: mp.context.BaseContext | str | None = None,
    random_seed: RandomSeed = None,
    max_init_retries: int = 10,
    jacobian_correction: bool = True,
    vectorize_logp: bool = True,
    compile_kwargs: dict[str, Any] = {},
) -> MultiPathfinderResult:
    """Fit Pathfinder variational inference with multiple paths on the PyMC/PyTensor backend.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_paths : int, optional
        Number of independent paths to run. Increase this when increasing the jitter value.
        Default 4.
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation. Default 1000.
    num_draws_per_path : int, optional
        Number of samples to draw per path. Default 1000.
    num_elbo_draws : int, optional
        Number of draws for the Evidence Lower Bound (ELBO) estimation. Default 10.
    jitter : float, optional
        Amount of jitter to apply to initial points. Pathfinder can be highly sensitive to this
        value; increase num_paths when increasing it. Default 2.0.
    lbfgs_config : LBFGSConfig
        L-BFGS configuration. For details, including default arguments, see :class:`LBFGSConfig`.
    importance_sampling : str or None, optional
        Method to apply based on log importance weights (logP - logQ):

        - "psis" : Pareto Smoothed Importance Sampling; usually most stable.
        - "psir" : Pareto Smoothed Importance Resampling; less stable than PSIS.
        - "identity" : apply log importance weights directly without resampling.
        - None : no importance sampling; return raw samples of shape
          (num_paths, num_draws_per_path, N). The other methods return shape (num_draws, N).

        Default "psis".
    progressbar : bool, optional
        Whether to display a progress bar. Enabling it likely increases computation time.
        Default False.
    random_seed : RandomSeed, optional
        Random seed for reproducibility.
    parallel : bool, optional
        If True, spawn a separate worker process per path for true parallelism. If False, run
        paths serially in the main process. Default True.
    cores : int, optional
        Number of paths to run in parallel. If None, set to min(4, cpu_count, num_paths),
        mirroring pm.sample. Default None.
    blas_cores : int or "auto" or None, optional
        Total number of threads BLAS/OpenMP should use per worker. "auto" matches the total to
        ``cores``; None keeps default BLAS behavior. Default "auto".
    max_init_retries : int, optional
        Maximum number of re-jitter retries per path when LBFGSInitFailed is raised. Default 10.
    jacobian_correction : bool, optional
        Whether to add the log-determinant-of-Jacobian correction term to ``model.logp`` to
        account for value-var transforms (e.g. ``log``, ``logit``). With the correction,
        ``logp`` is the joint density on unconstrained coordinates, which is what L-BFGS
        optimizes and what importance sampling needs. Default True.
    vectorize_logp : bool, optional
        If True, use ``vectorize_graph`` to batch ``model.logp`` across the num_draws axis for
        ELBO and importance-sampling evaluation; if False, fall back to ``pytensor.map``. This
        trades high memory with parallel compute (True) against low memory with sequential
        compute (False); prefer True unless the model is memory bound. Default True.
    compile_kwargs : dict, optional
        Additional keyword arguments for the PyTensor compiler. If not provided, a performant
        default is used.

    Returns
    -------
    MultiPathfinderResult
        Samples and other information from the multi-path Pathfinder run.
    """

    *path_seeds, choice_seed = _get_seeds_per_chain(random_seed, num_paths + 1)

    pathfinder_config = PathfinderConfig(
        num_draws=num_draws_per_path,
        maxcor=lbfgs_config.maxcor,
        maxiter=lbfgs_config.maxiter,
        ftol=lbfgs_config.ftol,
        gtol=lbfgs_config.gtol,
        maxls=lbfgs_config.maxls,
        num_elbo_draws=num_elbo_draws,
        jitter=jitter,
        epsilon=lbfgs_config.epsilon,
    )

    compile_start = time.time()
    single_pathfinder_fn = make_single_pathfinder_fn(
        model,
        num_draws=num_draws_per_path,
        num_elbo_draws=num_elbo_draws,
        jitter=jitter,
        lbfgs_config=lbfgs_config,
        max_init_retries=max_init_retries,
        jacobian_correction=jacobian_correction,
        vectorize_logp=vectorize_logp,
        compile_kwargs=compile_kwargs,
    )
    compile_end = time.time()

    results = []
    compute_start = time.time()
    try:
        # Per-path progress bar (one row per path, updated in real time)
        progress = _make_multipath_progress(progressbar)

        # Create one task per path and build per-path progress callbacks
        task_ids = []
        path_callbacks: list[Callable | None] = []
        with progress:
            for i in range(num_paths):
                tid = progress.add_task(
                    f"Path {i + 1}",
                    status="queued",
                    lbfgs_steps=0,
                    steps_per_sec="—",
                    best_elbo="—",
                    best_ind="—",
                    current_elbo="—",
                    step_size="—",
                    total=None,
                )
                task_ids.append(tid)
                path_callbacks.append(_make_progress_callback(progress, tid))

            # parallel=True gives true parallelism via separate worker processes
            # parallel=False is serial.
            generator = make_generator(
                parallel=parallel,
                fn=single_pathfinder_fn,
                seeds=path_seeds,
                cores=cores,
                blas_cores=blas_cores,
                progress_callbacks=path_callbacks,
                mp_ctx=mp_ctx,
            )

            for result in generator:
                try:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        results.append(result)
                except Exception as e:
                    logger.warning("Unexpected error in a path: %s", str(e))
                    results.append(
                        PathfinderResult(
                            path_status=PathStatus.PATH_FAILED,
                            lbfgs_status=LBFGSStatus.LBFGS_FAILED,
                        )
                    )
    except (KeyboardInterrupt, StopIteration) as e:
        # The user may abort early; MultiPathfinderResult still keeps the results gathered so far.
        if isinstance(e, StopIteration):
            logger.info(str(e))
    finally:
        compute_end = time.time()
        if results:
            mpr = (
                MultiPathfinderResult.from_path_results(results)
                .with_pathfinder_config(config=pathfinder_config)
                .with_importance_sampling(
                    num_draws=num_draws, method=importance_sampling, random_seed=choice_seed
                )
                .with_timing(
                    compile_time=compile_end - compile_start,
                    compute_time=compute_end - compute_start,
                )
            )
            if mpr.all_paths_failed:
                raise ValueError(
                    "All paths failed. Consider decreasing the jitter or reparameterizing "
                    "the model."
                )
        else:
            raise ValueError(
                "BUG: Failed to iterate!"
                "Please report this issue at: "
                "https://github.com/pymc-devs/pymc-extras/issues "
                "with your code to reproduce the issue and the following details:\n"
                f"pathfinder_config: \n{pathfinder_config}\n"
                f"compile_kwargs: {compile_kwargs}\n"
                f"jacobian_correction: {jacobian_correction}\n"
                f"vectorize_logp: {vectorize_logp}\n"
                f"num_paths: {num_paths}\n"
                f"num_draws: {num_draws}\n"
            )

    return mpr


def _default_cores(num_paths: int, cores: int | None) -> int:
    """Default cores for parallel pathfinder, mirroring pm.sample."""
    if cores is not None:
        return min(cores, num_paths)
    return min(4, _cpu_count(), num_paths)


class _PipeCallback:
    """Picklable progress callback that relays updates through a multiprocessing Pipe."""

    def __init__(self, conn: Any, idx: int) -> None:
        self.conn = conn
        self.idx = idx

    def __call__(self, info: dict) -> None:
        # Progress relay is best-effort: a closed or broken pipe must not crash the worker.
        try:
            self.conn.send((self.idx, info))
        except Exception:
            pass


def _run_path(
    fn_pickled: bytes,
    seed: int,
    path_id: int,
    progress_conn: Any,
    blas_cores: int | None,
    mp_start_method: str,
) -> PathfinderResult:
    """Worker: unpickle fn, run with threadpool_limits when blas_cores set (non-fork)."""
    import cloudpickle

    from pytensor.compile.compilelock import lock_ctx

    ctx = (
        threadpool_limits(limits=blas_cores)
        if mp_start_method != "fork" and blas_cores is not None
        else contextlib.nullcontext()
    )
    with ctx:
        with lock_ctx(timeout=-1):
            fn = cloudpickle.loads(fn_pickled)
        cb = _PipeCallback(progress_conn, path_id)
        try:
            return fn(seed, cb)
        finally:
            # Best-effort sentinel so the listener can stop; a broken pipe here is harmless.
            try:
                progress_conn.send((path_id, None))  # sentinel
            except Exception:
                pass


def _execute_concurrently(
    fn: SinglePathfinderFn,
    seeds: list[int],
    cores: int,
    blas_cores: int | None,
    progress_callbacks: list[Callable | None] | None = None,
    mp_ctx: mp.context.BaseContext | str | None = None,
) -> Iterator[PathfinderResult]:
    """Execute pathfinder runs concurrently via ProcessPoolExecutor.

    Uses Pipe instead of Manager().Queue() to avoid spawn bootstrapping issues
    when mp_ctx is 'spawn' and the main module is still loading.
    """
    import threading

    from concurrent.futures import ProcessPoolExecutor, as_completed

    import cloudpickle

    mp_ctx = _initialize_multiprocessing_context(mp_ctx, quiet=True)
    fn_pickled = cloudpickle.dumps(fn, protocol=-1)

    # One pipe per worker; avoids Manager() which spawns a process and triggers
    # "An attempt has been made to start a new process before bootstrapping" when
    # the main module is still loading (e.g. script run without if __name__ guard).
    n_workers = len(seeds)
    parent_conns = []
    child_conns = []
    for _ in range(n_workers):
        parent, child = mp_ctx.Pipe(duplex=False)
        parent_conns.append(parent)
        child_conns.append(child)

    sentinel_count: list[int] = [0]
    sentinel_lock = threading.Lock()

    def _listener() -> None:
        from multiprocessing.connection import wait

        while True:
            ready = wait(parent_conns, timeout=0.1)
            for conn in ready:
                try:
                    idx, info = conn.recv()
                except EOFError:
                    # Worker closed its end before the sentinel; nothing left to read here.
                    continue
                if info is None:
                    with sentinel_lock:
                        sentinel_count[0] += 1
                    if sentinel_count[0] >= n_workers:
                        return
                    continue
                if (
                    progress_callbacks
                    and idx < len(progress_callbacks)
                    and progress_callbacks[idx] is not None
                ):
                    progress_callbacks[idx](info)

    listener = threading.Thread(target=_listener, daemon=True)
    listener.start()

    def _run_executor():
        with ProcessPoolExecutor(max_workers=cores, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(
                    _run_path,
                    fn_pickled,
                    seed,
                    i,
                    child_conns[i],
                    blas_cores,
                    mp_ctx.get_start_method(),
                ): i
                for i, seed in enumerate(seeds)
            }
            for f in as_completed(futures):
                yield f.result()

    try:
        yield from _run_executor()
    except RuntimeError as e:
        if "bootstrapping" in str(e) and mp_ctx.get_start_method() == "spawn":
            raise RuntimeError(
                "Pathfinder with mp_ctx='spawn' requires the entry point to be "
                "guarded with `if __name__ == '__main__':`. When using spawn, "
                "wrap your fit_pathfinder call (or the function that contains it) "
                "in that block. See: "
                "https://docs.python.org/3/library/multiprocessing.html"
                "#the-spawn-and-forkserver-start-methods"
            ) from e
        raise
    finally:
        for c in child_conns:
            # Best-effort cleanup; a pipe already closed by its worker is fine.
            try:
                c.close()
            except Exception:
                pass
        listener.join(timeout=5)


def _execute_serially(
    fn: SinglePathfinderFn,
    seeds: list[int],
    progress_callbacks: list[Callable | None] | None = None,
) -> Iterator[PathfinderResult]:
    """Execute pathfinder runs serially."""
    callbacks = progress_callbacks or [None] * len(seeds)
    for seed, cb in zip(seeds, callbacks):
        yield fn(seed, cb)


def make_generator(
    parallel: bool,
    fn: SinglePathfinderFn,
    seeds: list[int],
    cores: int | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
    progress_callbacks: list[Callable | None] | None = None,
    mp_ctx: mp.context.BaseContext | str | None = None,
) -> Iterator[PathfinderResult]:
    """Generator for executing pathfinder runs concurrently or serially."""
    if parallel:
        num_paths = len(seeds)
        effective_cores = _default_cores(num_paths, cores)
        _, _, num_blas_per_worker = setup_cores_blas_cores(
            blas_cores,
            num_paths,
            effective_cores,
            _initialize_multiprocessing_context(mp_ctx, quiet=True),
        )
        yield from _execute_concurrently(
            fn,
            seeds,
            cores=effective_cores,
            blas_cores=num_blas_per_worker,
            progress_callbacks=progress_callbacks,
            mp_ctx=mp_ctx,
        )
    else:
        yield from _execute_serially(fn, seeds, progress_callbacks)


def _make_multipath_progress(progressbar: bool) -> CustomProgress:
    return CustomProgress(
        TextColumn("{task.description}", table_column=Column("Path", min_width=7, no_wrap=True)),
        TextColumn(
            "{task.fields[status]}", table_column=Column("Status", min_width=10, no_wrap=True)
        ),
        TextColumn(
            "{task.fields[lbfgs_steps]}", table_column=Column("Steps", min_width=6, no_wrap=True)
        ),
        TextColumn(
            "{task.fields[steps_per_sec]}",
            table_column=Column("Steps/s", min_width=8, no_wrap=True),
        ),
        TextColumn(
            "{task.fields[best_ind]}", table_column=Column("Best step", min_width=9, no_wrap=True)
        ),
        TextColumn(
            "{task.fields[best_elbo]}", table_column=Column("Best ELBO", min_width=12, no_wrap=True)
        ),
        TextColumn(
            "{task.fields[current_elbo]}",
            table_column=Column("Cur ELBO", min_width=12, no_wrap=True),
        ),
        TextColumn(
            "{task.fields[step_size]}",
            table_column=Column("Step size", min_width=10, no_wrap=True),
        ),
        TimeElapsedColumn(table_column=Column("Elapsed", min_width=8, no_wrap=True)),
        include_headers=True,
        console=Console(theme=default_progress_theme),
        disable=not progressbar,
    )


def _make_progress_callback(progress: CustomProgress, task_id: int) -> Callable[[dict], None]:
    def cb(info: dict) -> None:
        fields: dict[str, Any] = {}
        if "status" in info and info["status"] is not None:
            fields["status"] = info["status"]
        if "lbfgs_steps" in info:
            fields["lbfgs_steps"] = info["lbfgs_steps"]
        if "best_elbo" in info:
            val = info["best_elbo"]
            fields["best_elbo"] = (
                f"{val:.3f}" if val is not None and np.isfinite(float(val)) else "—"
            )
        if "best_ind" in info:
            val = info["best_ind"]
            fields["best_ind"] = (
                str(int(val)) if val is not None and np.isfinite(float(val)) else "—"
            )
        if "current_elbo" in info:
            val = info["current_elbo"]
            fields["current_elbo"] = (
                f"{val:.3f}" if val is not None and np.isfinite(float(val)) else "—"
            )
        if "step_size" in info:
            val = info["step_size"]
            fields["step_size"] = (
                f"{val:.2e}" if val is not None and np.isfinite(float(val)) else "—"
            )
        if "steps_per_sec" in info:
            val = info["steps_per_sec"]
            fields["steps_per_sec"] = (
                f"{val:.1f}/s" if val is not None and np.isfinite(float(val)) else "—"
            )
        if fields:
            progress.update(task_id, **fields)
        if info.get("status") in ("ok", "elbo@0"):
            progress.stop_task(task_id)

    return cb
