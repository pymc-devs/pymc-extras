import logging

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pymc.util import _get_seeds_per_chain

from pymc_extras.inference.pathfinder.bfgs_sample import (
    get_neg_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
)
from pymc_extras.inference.pathfinder.lbfgs import (
    LBFGS,
    LBFGSConfig,
    LBFGSException,
    LBFGSInitFailed,
    LBFGSStatus,
    LBFGSStreamingCallback,
)
from pymc_extras.inference.pathfinder.results import (
    PathException,
    PathfinderResult,
    PathInvalidLogP,
    PathInvalidLogQ,
    PathStatus,
    SingleStepPathException,
)

logger = logging.getLogger(__name__)


class SinglePathfinderFn(Protocol):
    def __call__(
        self, random_seed: int, progress_callback: Callable | None = None
    ) -> PathfinderResult: ...


def make_single_pathfinder_fn(
    model,
    num_draws: int,
    num_elbo_draws: int,
    jitter: float,
    lbfgs_config: LBFGSConfig,
    max_init_retries: int = 10,
    jacobian_correction: bool = True,
    vectorize_logp: bool = True,
    compile_kwargs: dict[str, Any] | None = None,
) -> SinglePathfinderFn:
    """Build a seedable single-path pathfinder function.

    The returned function executes a compiled graph performing the local approximation and
    sampling steps of the Pathfinder algorithm for one path.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_draws : int
        Number of samples to draw from the single-path approximation.
    num_elbo_draws : int
        Number of draws for the Evidence Lower Bound (ELBO) estimation.
    jitter : float
        Amount of jitter to apply to the initial point. Pathfinder can be highly sensitive to
        this value.
    lbfgs_config : LBFGSConfig
        L-BFGS configuration. For details, including default arguments, see :class:`LBFGSConfig`.
    max_init_retries : int, optional
        Maximum number of re-jitter retries when LBFGSInitFailed is raised (i.e. the initial
        point yields a non-finite value/gradient or the first step is rejected). Each retry uses
        a different jitter seed. Default 10.
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
    callable
        A seedable single-path function accepting ``(random_seed, progress_callback=None)``.
    """

    compile_kwargs = compile_kwargs or {}

    maxcor = lbfgs_config.maxcor
    maxiter = lbfgs_config.maxiter
    ftol = lbfgs_config.ftol
    gtol = lbfgs_config.gtol
    maxls = lbfgs_config.maxls
    epsilon = lbfgs_config.epsilon

    logp_dlogp_kwargs = {"jacobian": jacobian_correction, **compile_kwargs}

    neg_logp_dlogp_func = get_neg_logp_dlogp_of_ravel_inputs(model, **logp_dlogp_kwargs)

    # initial point
    # TODO: remove make_initial_points function when feature request is implemented:
    #  https://github.com/pymc-devs/pymc/issues/7555
    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    N = x_base.shape[0]

    sample_logp_func = make_pathfinder_sample_fn(
        model=model,
        N=N,
        J=maxcor,
        jacobian=jacobian_correction,
        compile_kwargs=compile_kwargs,
        vectorize=vectorize_logp,
    )

    def _check_lbfgs_status(status):
        if status in {LBFGSStatus.INIT_FAILED, LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT}:
            raise LBFGSInitFailed(status)
        elif status == LBFGSStatus.LBFGS_FAILED:
            raise LBFGSException()

    def _make_result(
        psi, logP_psi, logQ_psi, lbfgs_niter, elbo_argmax, inv_hessian_diag, lbfgs_status
    ):
        if np.all(~np.isfinite(logQ_psi)):
            raise PathInvalidLogQ()
        path_status = PathStatus.ELBO_ARGMAX_AT_ZERO if elbo_argmax == 0 else PathStatus.SUCCESS
        return PathfinderResult(
            samples=psi,
            logP=logP_psi,
            logQ=logQ_psi,
            lbfgs_niter=lbfgs_niter,
            elbo_argmax=elbo_argmax,
            inv_hessian_diag=inv_hessian_diag,
            lbfgs_status=lbfgs_status,
            path_status=path_status,
        )

    def single_pathfinder_fn(
        random_seed: int, progress_callback: Callable | None = None
    ) -> PathfinderResult:
        if progress_callback is not None:
            progress_callback({"status": "running"})

        local_lbfgs = LBFGS(neg_logp_dlogp_func, maxcor, maxiter, ftol, gtol, maxls, epsilon)
        lbfgs_status = LBFGSStatus.LBFGS_FAILED  # default before LBFGS runs

        try:
            # Derive base seeds once from random_seed. init_seed is an int so we can
            # safely offset it per retry attempt (avoids Generator arithmetic issues).
            _base_init, elbo_seed, final_seed = _get_seeds_per_chain(random_seed, 3)

            for attempt in range(max_init_retries + 1):
                try:
                    init_seed = _base_init + attempt  # different jitter per retry
                    rng = np.random.default_rng(init_seed)
                    jitter_value = rng.uniform(-jitter, jitter, size=x_base.shape)
                    x0 = x_base + jitter_value

                    # Fresh NumPy RNG each attempt → reproducible regardless
                    # of how many ELBO steps the previous attempt took.
                    elbo_rng = np.random.default_rng(elbo_seed + attempt)

                    streaming_cb = LBFGSStreamingCallback(
                        value_grad_fn=neg_logp_dlogp_func,
                        x0=x0,
                        sample_logp_fn=sample_logp_func,
                        num_elbo_draws=num_elbo_draws,
                        rng=elbo_rng,
                        J=maxcor,
                        epsilon=epsilon,
                        progress_callback=progress_callback,
                    )

                    lbfgs_niter, lbfgs_status = local_lbfgs.minimize_streaming(streaming_cb, x0)
                    _check_lbfgs_status(lbfgs_status)

                    if lbfgs_niter < 2:
                        raise SingleStepPathException()

                    if not streaming_cb.any_valid:
                        raise PathInvalidLogP()

                    elbo_argmax = streaming_cb.best_step_idx
                    best_state = streaming_cb.best_state

                    final_rng = np.random.default_rng(final_seed)
                    u_final = final_rng.standard_normal((num_draws, N))
                    if progress_callback is not None:
                        progress_callback({"status": "sampling"})

                    sample_out = sample_logp_func(
                        best_state["x"],
                        best_state["g"],
                        best_state["alpha"],
                        best_state["s_win"],
                        best_state["z_win"],
                        u_final,
                    )
                    phi_final, logQ_psi_flat, logP_psi_flat, inv_hessian_diag = sample_out
                    phi_final = np.asarray(phi_final)
                    logQ_psi_flat = np.asarray(logQ_psi_flat)
                    logP_psi_flat = np.asarray(logP_psi_flat)
                    inv_hessian_diag = np.asarray(inv_hessian_diag)[None]
                    # Add batch dim L=1 to match downstream expectations
                    psi = phi_final[None]  # (1, M, N)
                    logP_psi = logP_psi_flat[None]  # (1, M)
                    logQ_psi = logQ_psi_flat[None]  # (1, M)

                    break  # success — exit retry loop

                except (LBFGSInitFailed, SingleStepPathException) as e:
                    if attempt < max_init_retries:
                        logger.debug(
                            "%s on attempt %d/%d, retrying with different jitter...",
                            type(e).__name__,
                            attempt + 1,
                            max_init_retries,
                        )
                        if progress_callback is not None:
                            progress_callback({"status": f"retry {attempt + 1}"})
                    else:
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "status": "lbfgs_fail"
                                    if isinstance(e, LBFGSInitFailed)
                                    else "failed"
                                }
                            )
                        if isinstance(e, LBFGSInitFailed):
                            return PathfinderResult(
                                lbfgs_status=e.status,
                                path_status=PathStatus.LBFGS_FAILED,
                            )
                        return PathfinderResult(
                            lbfgs_status=lbfgs_status,
                            path_status=PathStatus.SINGLE_STEP,
                        )

            result = _make_result(
                psi=psi,
                logP_psi=logP_psi,
                logQ_psi=logQ_psi,
                lbfgs_niter=lbfgs_niter,
                elbo_argmax=elbo_argmax,
                inv_hessian_diag=inv_hessian_diag,
                lbfgs_status=lbfgs_status,
            )
            if progress_callback is not None:
                status_str = (
                    "elbo@0" if result.path_status == PathStatus.ELBO_ARGMAX_AT_ZERO else "ok"
                )
                progress_callback({"status": status_str, "lbfgs_steps": int(lbfgs_niter)})
            return result

        except LBFGSException as e:
            if progress_callback is not None:
                progress_callback({"status": "lbfgs_fail"})
            return PathfinderResult(
                lbfgs_status=e.status,
                path_status=PathStatus.LBFGS_FAILED,
            )
        except PathException as e:
            if progress_callback is not None:
                progress_callback({"status": "failed"})
            return PathfinderResult(
                lbfgs_status=lbfgs_status,
                path_status=e.status,
            )

    return single_pathfinder_fn
