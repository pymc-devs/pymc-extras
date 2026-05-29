import contextlib
import logging
import multiprocessing as mp
import time

from collections.abc import Callable, Iterator
from multiprocessing.connection import wait
from typing import Any, Literal

import cloudpickle
import numpy as np

from pymc import Model
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.sampling.mcmc import setup_cores_blas_cores
from pymc.sampling.parallel import (
    ExceptionWithTraceback,
    _cpu_count,
    _initialize_multiprocessing_context,
)
from pymc.util import RandomSeed, _get_seeds_per_chain
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, TaskID, TextColumn, TimeElapsedColumn
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
    compile_kwargs: dict[str, Any] | None = None,
) -> MultiPathfinderResult:
    """Fit Pathfinder variational inference with multiple paths on the PyMC/PyTensor backend.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_paths : int
        Number of independent paths to run. Increase this when increasing the jitter value.
    num_draws : int
        Total number of samples to draw from the fitted approximation.
    num_draws_per_path : int
        Number of samples to draw per path.
    num_elbo_draws : int
        Number of draws for the Evidence Lower Bound (ELBO) estimation.
    jitter : float
        Amount of jitter to apply to initial points. Pathfinder can be highly sensitive to this
        value; increase num_paths when increasing it.
    lbfgs_config : LBFGSConfig
        L-BFGS configuration. For details, including default arguments, see :class:`LBFGSConfig`.
    importance_sampling : str or None
        Method to apply based on log importance weights (logP - logQ):

        - "psis" : Pareto Smoothed Importance Sampling; usually most stable.
        - "psir" : Pareto Smoothed Importance Resampling; less stable than PSIS.
        - "identity" : apply log importance weights directly without resampling.
        - None : no importance sampling; return raw samples of shape
          (num_paths, num_draws_per_path, N). The other methods return shape (num_draws, N).
    progressbar : bool
        Whether to display a progress bar. Enabling it likely increases computation time.
    parallel : bool, optional
        If True, spawn a separate worker process per path for true parallelism. If False, run
        paths serially in the main process. Default True.
    cores : int, optional
        Number of paths to run in parallel. If None, set to min(4, cpu_count, num_paths),
        mirroring pm.sample. Default None.
    blas_cores : int or "auto" or None, optional
        Total number of threads BLAS/OpenMP should use per worker. "auto" matches the total to
        ``cores``; None keeps default BLAS behavior. Default "auto".
    mp_ctx : str or multiprocessing.Context, optional
        Multiprocessing context for parallel path execution (e.g. ``"spawn"``, ``"fork"``).
        Default None.
    random_seed : RandomSeed, optional
        Random seed for reproducibility. Default None.
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
        Additional keyword arguments for the PyTensor compiler. Default None.

    Returns
    -------
    MultiPathfinderResult
        Samples and other information from the multi-path Pathfinder run.
    """

    compile_kwargs = compile_kwargs or {}

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

    compute_start = time.time()

    # Resolve the multiprocessing context once, threading the compile mode through so a JAX backend
    # forces a non-fork start method -- fork + JAX can deadlock. Mirrors pymc.sample.
    mp_ctx = _initialize_multiprocessing_context(
        mp_ctx, mode=compile_kwargs.get("mode"), quiet=True
    )

    # One progress row per path, updated in real time.
    progress = _make_multipath_progress(progressbar)
    task_ids: list[TaskID] = []
    path_callbacks: list[Callable | None] = []
    with progress:
        for i in range(num_paths):
            tid = progress.add_task(
                f"Path {i + 1}",
                status="queued",
                elbo="—",
                speed=0.0,
                speed_unit="it/s",
                total=lbfgs_config.maxiter,
                completed=0,
            )
            task_ids.append(tid)
            path_callbacks.append(_make_progress_callback(progress, tid))

        # parallel=True runs each path in its own worker process; parallel=False is serial.
        generator = make_generator(
            parallel=parallel,
            fn=single_pathfinder_fn,
            seeds=path_seeds,
            cores=cores,
            blas_cores=blas_cores,
            progress_callbacks=path_callbacks,
            mp_ctx=mp_ctx,
        )
        results = list(generator)
    compute_end = time.time()

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
            "All paths failed. Consider decreasing the jitter or reparameterizing the model."
        )

    return mpr


def _default_cores(num_paths: int, cores: int | None) -> int:
    """Default cores for parallel pathfinder, mirroring pm.sample."""
    if cores is not None:
        return min(cores, num_paths)
    return min(4, _cpu_count(), num_paths)


def _run_pathfinder_process(
    msg_pipe: Any,
    fn: "SinglePathfinderFn | bytes",
    fn_is_pickled: bool,
    seed: int,
    mp_start_method: str,
    blas_cores: int | None,
) -> None:
    """Worker entry point: run one path, relaying progress and the result over ``msg_pipe``.

    ``fn`` is the live single-path function (fork) or cloudpickled bytes (spawn).
    """
    ctx = (
        threadpool_limits(limits=blas_cores)
        if mp_start_method != "fork" and blas_cores is not None
        else contextlib.nullcontext()
    )
    with ctx:
        try:
            if fn_is_pickled:
                fn = cloudpickle.loads(fn)

            def progress_callback(info: dict) -> None:
                # Best-effort: a closed pipe (parent gone) must not crash the worker.
                try:
                    msg_pipe.send(("progress", info))
                except Exception:
                    pass

            msg_pipe.send(("done", fn(seed, progress_callback)))
        except KeyboardInterrupt:
            pass
        except BaseException as e:
            try:
                msg_pipe.send(("error", ExceptionWithTraceback(e, e.__traceback__)))
            except Exception:
                pass
        finally:
            msg_pipe.close()


def _execute_concurrently(
    fn: SinglePathfinderFn,
    seeds: list[int],
    cores: int,
    blas_cores: int | None,
    progress_callbacks: list[Callable | None] | None = None,
    mp_ctx: mp.context.BaseContext | str | None = None,
) -> Iterator[PathfinderResult]:
    """Run paths in worker processes, at most ``cores`` alive at once.

    Workers are spawned as slots free, so the process count never exceeds ``cores`` regardless of
    the number of paths. A worker that errors or dies yields a failed PathfinderResult rather than
    aborting the whole fit.
    """
    # mp_ctx is already resolved (mode-aware, JAX fork-safe) by multipath_pathfinder.
    start_method = mp_ctx.get_start_method()
    # Fork inherits the fn via memory; non-fork ships it pickled once, shared by all workers.
    fn_is_pickled = start_method != "fork"
    fn_payload = cloudpickle.dumps(fn, protocol=-1) if fn_is_pickled else fn

    pending = list(enumerate(seeds))  # (chain, seed), oldest first
    active: dict[Any, tuple[int, Any]] = {}  # parent_conn -> (chain, process)

    def _spawn(chain: int, seed: int) -> None:
        parent_conn, child_conn = mp_ctx.Pipe()
        proc = mp_ctx.Process(
            daemon=True,
            name=f"pathfinder_path_{chain}",
            target=_run_pathfinder_process,
            args=(child_conn, fn_payload, fn_is_pickled, seed, start_method, blas_cores),
        )
        proc.start()
        child_conn.close()  # the parent keeps only its end of the pipe
        active[parent_conn] = (chain, proc)

    def _fill() -> None:
        while pending and len(active) < cores:
            chain, seed = pending.pop(0)
            _spawn(chain, seed)

    def _failed_path() -> PathfinderResult:
        return PathfinderResult(
            path_status=PathStatus.PATH_FAILED, lbfgs_status=LBFGSStatus.LBFGS_FAILED
        )

    try:
        _fill()
        while active:
            for conn in wait(list(active)):
                chain, proc = active[conn]
                try:
                    kind, payload = conn.recv()
                except EOFError:
                    # Worker died without a message (e.g. a hard crash); count it as a failure.
                    kind, payload = "error", None

                if kind == "progress":
                    if progress_callbacks and progress_callbacks[chain] is not None:
                        progress_callbacks[chain](payload)
                    continue

                # Terminal message: finalize this worker and start a replacement before yielding.
                del active[conn]
                conn.close()
                proc.join()
                _fill()

                if kind == "done":
                    yield payload
                else:
                    if payload is not None:
                        logger.warning("Pathfinder path %d failed: %s", chain, payload)
                    yield _failed_path()
    finally:
        for conn, (_, proc) in active.items():
            with contextlib.suppress(Exception):
                conn.close()
            if proc.is_alive():
                proc.terminate()
            proc.join()


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
            mp_ctx,
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
    # Per-path lifecycle status, an L-BFGS progress bar, and the best ELBO seen so far -- the value
    # pathfinder actually selects on.
    return CustomProgress(
        BarColumn(bar_width=20, table_column=Column("Progress")),
        TextColumn(
            "{task.fields[status]}", table_column=Column("Status", min_width=10, no_wrap=True)
        ),
        MofNCompleteColumn(table_column=Column("Iter", min_width=9, no_wrap=True)),
        TextColumn(
            "{task.fields[speed]:0.2f} {task.fields[speed_unit]}",
            table_column=Column("Speed", min_width=10, no_wrap=True),
        ),
        TextColumn(
            "{task.fields[elbo]}", table_column=Column("Best ELBO", min_width=11, no_wrap=True)
        ),
        TimeElapsedColumn(table_column=Column("Elapsed", min_width=8, no_wrap=True)),
        include_headers=True,
        console=Console(theme=default_progress_theme),
        disable=not progressbar,
    )


def _make_progress_callback(progress: CustomProgress, task_id: TaskID) -> Callable[[dict], None]:
    def cb(info: dict) -> None:
        fields: dict[str, Any] = {}
        update_kwargs: dict[str, Any] = {}
        if info.get("status") is not None:
            fields["status"] = info["status"]
        if "iteration" in info:
            completed = int(info["iteration"])
            update_kwargs["completed"] = completed
            # Step pace off rich's task clock, flipping to s/step when slower than 1/s (pymc style).
            elapsed = progress.tasks[task_id].elapsed or 0.0
            if elapsed > 0.25:
                speed = completed / elapsed
                if speed >= 1.0:
                    fields["speed"], fields["speed_unit"] = speed, "it/s"
                else:
                    fields["speed"], fields["speed_unit"] = 1.0 / speed, "s/it"
        if "best_elbo" in info:
            val = info["best_elbo"]
            fields["elbo"] = format(val, ".3f") if val is not None and np.isfinite(val) else "—"
        if fields or update_kwargs:
            progress.update(task_id, **update_kwargs, **fields)
        if info.get("status") in ("ok", "elbo@0"):
            # L-BFGS converges well short of maxiter, so snap the bar to the achieved step count
            # (as better_optimize does) instead of leaving it stuck near empty.
            if "lbfgs_steps" in info:
                n = int(info["lbfgs_steps"])
                progress.update(task_id, total=n, completed=n)
            progress.stop_task(task_id)

    return cb
