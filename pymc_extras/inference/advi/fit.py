from __future__ import annotations

import numpy as np
import xarray as xr

from pymc import Model, modelcontext
from xarray import DataTree

from pymc_extras.inference.advi.schedules import ScalarOrSchedule
from pymc_extras.inference.advi.training import Trainer


def fit_advi(
    model: Model | None = None,
    *,
    n_steps: int = 10_000,
    n_particles: int = 1,
    draws: int = 1_000,
    learning_rate: ScalarOrSchedule | None = None,
    clip_norm: float | None = 10.0,
    path_derivative_gradient: bool = True,
    convergence_window: int | None = 200,
    relative_tolerance: float = 1e-3,
    random_seed=None,
    backend: str | None = None,
    compile_kwargs: dict | None = None,
) -> DataTree:
    """Fit a model with automatic differentiation variational inference (ADVI).

    Fits a mean-field normal approximation to the model posterior in the unconstrained
    space, then returns posterior draws from the fitted guide. A one-shot wrapper around
    :class:`~pymc_extras.inference.advi.training.Trainer` with its default guide; use the
    trainer directly to keep training, change the learning rate between runs, or sample
    more than once.

    Parameters
    ----------
    model : Model, optional
        The PyMC model to fit. If None, the model is inferred from context.
    n_steps : int, optional
        Maximum number of optimization steps, by default 10_000. Training may stop
        earlier, controlled by ``convergence_window`` and ``relative_tolerance``.
    n_particles : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by default 1.
    draws : int, optional
        Number of posterior draws to sample from the fitted guide, by default 1_000.
    learning_rate : float or callable, optional
        Learning rate, or a schedule mapping the step number to one, for the clipped Adam
        optimizer compiled into the step function. Defaults to a
        :func:`~pymc_extras.inference.advi.schedules.linear_onecycle_schedule` peaking at
        0.008 over ``n_steps``.
    clip_norm : float, optional
        Clip gradients to this global norm, by default 10. None disables clipping.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True. It is an unbiased variance reduction (it changes
        only the gradient, not the ELBO); numpyro's ``Trace_ELBO`` does not offer it.
    convergence_window : int, optional
        Number of steps per convergence window, by default 200. Set to None to always
        run for ``n_steps``.
    relative_tolerance : float, optional
        Relative loss change between consecutive windows under which training stops,
        by default 1e-3.
    random_seed : optional
        Seed for the guide initialization, the training draws, and the posterior draws.
    backend : str, optional
        PyTensor backend to compile the training and sampling functions with
        (e.g. "numba", "jax", "c"). Mutually exclusive with ``compile_kwargs["mode"]``.
    compile_kwargs : dict, optional
        Additional kwargs passed to pytensor compilation.

    Returns
    -------
    DataTree
        Posterior draws from the fitted guide, with the negative loss history in the
        ``fit`` group (as ``elbo``).
    """
    model = modelcontext(model)

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        init_seed, train_seed, sampling_seed = (int(s) for s in rng.integers(2**30, size=3))
    else:
        init_seed = train_seed = sampling_seed = None

    trainer = Trainer(
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        n_particles=n_particles,
        path_derivative_gradient=path_derivative_gradient,
        convergence_window=convergence_window,
        relative_tolerance=relative_tolerance,
        model=model,
        backend=backend,
        compile_kwargs=compile_kwargs,
        random_seed=init_seed,
    )
    state = trainer.fit(n_steps, random_seed=train_seed)
    idata = trainer.sample_posterior(draws, random_seed=sampling_seed)
    idata["fit"] = DataTree(dataset=xr.Dataset({"elbo": ("step", -state.loss_history)}))
    return idata
