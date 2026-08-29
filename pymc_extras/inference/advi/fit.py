from __future__ import annotations

import numpy as np
import xarray as xr

from pymc import Model, modelcontext
from xarray import DataTree

from pymc_extras.inference.advi.optimizers import GradientTransformation
from pymc_extras.inference.advi.training import Trainer


def fit_advi(
    model: Model | None = None,
    *,
    n_steps: int = 10_000,
    n_particles: int = 1,
    draws: int = 1_000,
    optimizer: GradientTransformation | None = None,
    path_derivative_gradient: bool = True,
    random_seed=None,
    backend: str | None = None,
    compile_kwargs: dict | None = None,
) -> DataTree:
    """Fit a model with automatic differentiation variational inference (ADVI).

    Fits a mean-field normal approximation to the model posterior in the unconstrained
    space, then returns posterior draws from the fitted guide. A one-shot wrapper around
    :class:`~pymc_extras.inference.advi.training.Trainer` with its default guide; use the
    trainer directly to keep training, change the optimizer between runs, or sample
    more than once.

    Parameters
    ----------
    model : Model, optional
        The PyMC model to fit. If None, the model is inferred from context.
    n_steps : int, optional
        Number of optimization steps, by default 10_000.
    n_particles : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by default 1.
    draws : int, optional
        Number of posterior draws to sample from the fitted guide, by default 1_000.
    optimizer : GradientTransformation, optional
        An optax-like optimizer (actual optax optimizers are compatible). By default,
        :func:`clipped_adam` is used.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True. It is an unbiased variance reduction (it changes
        only the gradient, not the ELBO); numpyro's ``Trace_ELBO`` does not offer it.
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
        optimizer=optimizer,
        n_particles=n_particles,
        path_derivative_gradient=path_derivative_gradient,
        backend=backend,
        compile_kwargs=compile_kwargs,
        random_seed=init_seed,
    )
    state = trainer.fit(n_steps, model=model, random_seed=train_seed)
    idata = trainer.sample_posterior(draws, model=model, random_seed=sampling_seed)
    idata["fit"] = DataTree(dataset=xr.Dataset({"elbo": ("step", -state.loss_history)}))
    return idata
