import logging
import multiprocessing as mp

from typing import Any, Literal

import pymc as pm
import xarray as xr

from packaging import version
from pymc import Model
from pymc.blocking import DictToArrayBijection
from pymc.model import modelcontext
from pymc.util import RandomSeed, _get_seeds_per_chain

from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data
from pymc_extras.inference.pathfinder.bfgs_sample import get_jaxified_logp_of_ravel_inputs
from pymc_extras.inference.pathfinder.idata import convert_flat_trace_to_idata
from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig
from pymc_extras.inference.pathfinder.multipath import multipath_pathfinder

logger = logging.getLogger(__name__)


def fit_pathfinder(
    model=None,
    num_paths: int = 4,  # I
    num_draws: int = 1000,  # R
    num_draws_per_path: int = 1000,  # M
    num_elbo_draws: int = 10,  # K
    max_init_retries: int = 10,
    jitter: float = 2.0,
    lbfgs_config: LBFGSConfig | None = None,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
    progressbar: bool = True,
    parallel: bool = True,
    cores: int | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
    mp_ctx: mp.context.BaseContext | str | None = None,
    random_seed: RandomSeed | None = None,
    jacobian_correction: bool = True,
    vectorize_logp: bool = True,
    vectorize_postprocessing: bool = True,
    compile_kwargs: dict[str, Any] | None = None,
    initvals: dict[str, Any] | None = None,
) -> xr.DataTree:
    """Fit Pathfinder variational inference (multi-path, PyMC/PyTensor backend).

    For the blackjax-backed single-path variant, see :func:`fit_blackjax_pathfinder`.

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
    max_init_retries : int, optional
        Maximum number of re-jitter retries per path when the initial point fails. Default 10.
    jitter : float, optional
        Amount of jitter to apply to initial points. Pathfinder can be highly sensitive to this
        value; increase num_paths when increasing it. Default 2.0.
    lbfgs_config : LBFGSConfig, optional
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
        Whether to display a progress bar. Disabling it likely reduces computation time.
        Default True.
    parallel : bool, optional
        If True, spawn a separate worker process per path for true parallelism (matching PyMC's
        approach for parallel chains). If False, run paths serially in the main process, which is
        useful for debugging. Default True.
    cores : int, optional
        Number of paths to run in parallel. If None, set to min(4, cpu_count, num_paths),
        mirroring pm.sample. Default None.
    blas_cores : int or "auto" or None, optional
        Total number of threads BLAS/OpenMP should use per worker. "auto" matches the total to
        ``cores``; None keeps default BLAS behavior. Default "auto".
    mp_ctx : str or multiprocessing.Context, optional
        Multiprocessing context for parallel path execution (e.g. ``"spawn"``, ``"fork"``).
    random_seed : RandomSeed, optional
        Random seed for reproducibility.
    jacobian_correction : bool, optional
        Whether to add the log-determinant-of-Jacobian correction term to ``model.logp`` to
        account for value-var transforms (e.g. ``log``, ``logit``). With the correction,
        ``logp`` is the joint density on unconstrained coordinates, which is what L-BFGS
        optimizes and what importance sampling needs. Disabling it generally produces very high
        pareto-k values. Default True.
    vectorize_logp : bool, optional
        If True, use ``vectorize_graph`` to batch ``model.logp`` across the num_draws axis for
        ELBO and importance-sampling evaluation; if False, fall back to ``pytensor.map``. This
        trades high memory with parallel compute (True) against low memory with sequential
        compute (False); prefer True unless the model is memory bound. Default True.
    vectorize_postprocessing : bool, optional
        If True, use ``vectorize_graph`` to batch the Deterministic post-processing subgraph
        across all draws in one call; if False, iterate draws with ``pytensor.scan``. Set to
        False when memory is a concern, e.g. with large intermediate computations. Default True.
    compile_kwargs : dict, optional
        Additional keyword arguments for the PyTensor compiler. Default None.
    initvals : dict, optional
        Initial values for the model parameters, as name-to-ndarray pairs. Partial
        initialization is permitted. If None, the model's default initial values are used.
        Default None.

    Returns
    -------
    :class:`~xarray.DataTree`
        The inference data containing the results of the Pathfinder algorithm.

    References
    ----------
    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel
    quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """

    model = modelcontext(model)
    compile_kwargs = compile_kwargs or {}

    if initvals is not None:
        model = pm.model.fgraph.clone_model(model)  # Create a clone of the model
        for (
            rv_name,
            ivals,
        ) in initvals.items():  # Set the initial values for the variables in the clone
            model.set_initval(model.named_vars[rv_name], ivals)

    valid_importance_sampling = {"psis", "psir", "identity", None}

    if importance_sampling is not None:
        importance_sampling = importance_sampling.lower()

    if importance_sampling not in valid_importance_sampling:
        raise ValueError(f"Invalid importance sampling method: {importance_sampling}")

    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    if lbfgs_config is None:
        lbfgs_config = LBFGSConfig()
    lbfgs_config = lbfgs_config.set_default_maxcor(N)

    mp_result = multipath_pathfinder(
        model,
        num_paths=num_paths,
        num_draws=num_draws,
        num_draws_per_path=num_draws_per_path,
        num_elbo_draws=num_elbo_draws,
        jitter=jitter,
        lbfgs_config=lbfgs_config,
        importance_sampling=importance_sampling,
        progressbar=progressbar,
        parallel=parallel,
        cores=cores,
        blas_cores=blas_cores,
        mp_ctx=mp_ctx,
        max_init_retries=max_init_retries,
        random_seed=random_seed,
        jacobian_correction=jacobian_correction,
        vectorize_logp=vectorize_logp,
        compile_kwargs=compile_kwargs,
    )
    pathfinder_samples = mp_result.samples

    logger.info("Transforming variables...")

    idata = convert_flat_trace_to_idata(
        pathfinder_samples,
        inference_backend="pymc",
        model=model,
        importance_sampling=importance_sampling,
        vectorize=vectorize_postprocessing,
    )

    idata = add_data_to_inference_data(idata, progressbar, model, compile_kwargs)

    from pymc_extras.inference.pathfinder.idata import add_pathfinder_to_inference_data

    idata = add_pathfinder_to_inference_data(idata=idata, result=mp_result, model=model)
    return idata


def fit_blackjax_pathfinder(
    model: Model | None = None,
    *,
    num_draws: int = 1000,
    num_elbo_draws: int = 10,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
    jacobian_correction: bool = True,
    lbfgs_config: LBFGSConfig | None = None,
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu",
    random_seed: RandomSeed | None = None,
) -> xr.DataTree:
    """
    Fit Pathfinder via the blackjax library (single-path, JAX-backed).

    blackjax's Pathfinder is single-path only and does not support multi-path aggregation,
    jitter retries, or PyTensor-side compilation knobs. For the multi-path PyMC implementation
    use :func:`fit_pathfinder` instead.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation. Default 1000.
    num_elbo_draws : int, optional
        Number of draws for the Evidence Lower Bound (ELBO) estimation. Default 10.
    importance_sampling : str, None, optional
        Method to apply sampling based on log importance weights. See :func:`fit_pathfinder`.
    jacobian_correction : bool, optional
        Whether to add the log-determinant-of-Jacobian correction term to ``model.logp``.
        Default True. See :func:`fit_pathfinder` for details.
    lbfgs_config : LBFGSConfig, optional
        L-BFGS configuration. For details, including default arguments, see :class:`LBFGSConfig`.
    postprocessing_backend : str, optional
        Backend for postprocessing transformations, either ``"cpu"`` or ``"gpu"``.
        Default ``"cpu"``.
    random_seed : RandomSeed, optional
        Random seed for reproducibility.

    Returns
    -------
    :class:`~xarray.DataTree`
        The inference data containing the results of the Pathfinder algorithm.
    """
    import blackjax
    import jax

    if version.parse(blackjax.__version__).major < 1:
        raise ImportError("fit_blackjax_pathfinder requires blackjax 1.0 or above")

    model = modelcontext(model)
    if lbfgs_config is None:
        lbfgs_config = LBFGSConfig()
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]
    lbfgs_config = lbfgs_config.set_default_maxcor(N)

    _, pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 3)
    x0, _ = DictToArrayBijection.map(model.initial_point())
    logp_func = get_jaxified_logp_of_ravel_inputs(model, jacobian=jacobian_correction)

    pathfinder_state, _ = blackjax.vi.pathfinder.approximate(
        rng_key=jax.random.key(pathfinder_seed),
        logdensity_fn=logp_func,
        initial_position=x0,
        num_samples=num_elbo_draws,
        maxiter=lbfgs_config.maxiter,
        maxcor=lbfgs_config.maxcor,
        maxls=lbfgs_config.maxls,
        ftol=lbfgs_config.ftol,
        gtol=lbfgs_config.gtol,
    )
    pathfinder_samples, _ = blackjax.vi.pathfinder.sample(
        rng_key=jax.random.key(sample_seed),
        state=pathfinder_state,
        num_samples=num_draws,
    )

    logger.info("Transforming variables...")
    idata = convert_flat_trace_to_idata(
        pathfinder_samples,
        postprocessing_backend=postprocessing_backend,
        inference_backend="blackjax",
        model=model,
        importance_sampling=importance_sampling,
    )
    idata = add_data_to_inference_data(idata, progressbar=False, model=model, compile_kwargs={})
    return idata
