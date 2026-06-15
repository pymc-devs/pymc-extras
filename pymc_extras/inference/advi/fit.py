from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from pymc import Model, modelcontext
from xarray import DataTree

from pymc_extras.inference.advi.autoguide import AutoDiagonalNormal, AutoGuideModel
from pymc_extras.inference.advi.optimizers import (
    GradientTransformation,
    apply_updates,
    clipped_adam,
    linear_onecycle_schedule,
)
from pymc_extras.inference.advi.training import SVIModule, SVIState, SVITrainer


class ADVIModule(SVIModule):
    """SVIModule with batteries included.

    Uses a mean-field normal guide (:func:`AutoDiagonalNormal`), a clipped Adam
    optimizer, and a window-based convergence check that stops training when the
    mean loss over the last ``convergence_window`` steps is within
    ``relative_tolerance`` of the mean over the window before it.

    Parameters
    ----------
    guide_factory : callable, optional
        Function mapping a model to an :class:`AutoGuideModel`. Defaults to
        :func:`AutoDiagonalNormal`.
    optimizer : GradientTransformation, optional
        An optax-like optimizer (actual optax optimizers are compatible).
        Defaults to :func:`clipped_adam`.
    convergence_window : int, optional
        Number of steps per convergence window, by default 200. Set to None to
        disable early stopping.
    relative_tolerance : float, optional
        Relative loss change between consecutive windows under which training
        stops, by default 1e-3.
    random_seed : optional
        Seed passed to the guide factory.
    """

    def __init__(
        self,
        guide_factory: Callable[[Model], AutoGuideModel] | None = None,
        optimizer: GradientTransformation | None = None,
        convergence_window: int | None = 200,
        relative_tolerance: float = 1e-3,
        random_seed=None,
    ):
        self.guide_factory = guide_factory
        self.optimizer = optimizer if optimizer is not None else clipped_adam(0.008)
        self.convergence_window = convergence_window
        self.relative_tolerance = relative_tolerance
        self.random_seed = random_seed

    def configure_guide(self, model: Model) -> AutoGuideModel:
        if self.guide_factory is not None:
            return self.guide_factory(model)
        return AutoDiagonalNormal(model, random_seed=self.random_seed)

    def configure_optimizer(self, params: dict[str, np.ndarray]) -> tuple[Any, dict[str, Any]]:
        return self.optimizer, self.optimizer.init(params)

    def apply_gradients(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        optimizer: GradientTransformation,
        optimizer_state: Any,
    ) -> tuple[dict[str, np.ndarray], Any]:
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
        return apply_updates(params, updates), new_optimizer_state

    def should_stop(self, state: SVIState, loss: float) -> bool:
        window = self.convergence_window
        if window is None:
            return False
        history = state.loss_history
        if state.step % window != 0 or len(history) < 2 * window:
            return False
        recent = np.mean(history[-window:])
        previous = np.mean(history[-2 * window : -window])
        return bool(abs(recent - previous) < self.relative_tolerance * (abs(previous) + 1e-8))


def fit_advi(
    model: Model | None = None,
    *,
    n_steps: int = 10_000,
    draws_per_step: int = 1,
    draws: int = 1_000,
    optimizer: GradientTransformation | None = None,
    path_derivative_gradient: bool = True,
    convergence_window: int | None = 200,
    relative_tolerance: float = 1e-3,
    random_seed=None,
    backend: str | None = None,
    compile_kwargs: dict | None = None,
) -> DataTree:
    """Fit a model with automatic differentiation variational inference (ADVI).

    Fits a mean-field normal approximation to the model posterior in the unconstrained
    space, then returns posterior draws from the fitted guide.

    Parameters
    ----------
    model : Model, optional
        The PyMC model to fit. If None, the model is inferred from context.
    n_steps : int, optional
        Maximum number of optimization steps, by default 10_000. Training may stop
        earlier, controlled by ``convergence_window`` and ``relative_tolerance``.
    draws_per_step : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by default 1
        (numpyro's ``num_particles`` default; the path-derivative estimator keeps single-draw
        gradients well behaved, and more draws per step rarely beat more steps).
    draws : int, optional
        Number of posterior draws to sample from the fitted guide, by default 1_000.
    optimizer : GradientTransformation, optional
        An optax-like optimizer (actual optax optimizers are compatible). By default,
        clipped Adam on a :func:`linear_onecycle_schedule` peaking at 0.008 over
        ``n_steps`` is compiled *into* the step function (fast path); passing an explicit
        optimizer uses the Python-side update loop instead.
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

    module = ADVIModule(
        optimizer=optimizer,
        convergence_window=convergence_window,
        relative_tolerance=relative_tolerance,
        random_seed=init_seed,
    )
    trainer = SVITrainer(
        module,
        path_derivative_gradient=path_derivative_gradient,
        backend=backend,
        compile_kwargs=compile_kwargs,
    )
    if optimizer is None:
        # Default fast path: clipped Adam compiled into the step function
        state = trainer.fit_jitted(
            n_steps=n_steps,
            draws_per_step=draws_per_step,
            model=model,
            learning_rate=linear_onecycle_schedule(
                transition_steps=n_steps, peak_value=0.008, pct_start=0.2
            ),
            random_seed=train_seed,
        )
    else:
        state = trainer.fit(
            n_steps=n_steps,
            draws_per_step=draws_per_step,
            model=model,
            random_seed=train_seed,
        )
    idata = trainer.sample_posterior(
        draws=draws, state=state, model=model, random_seed=sampling_seed
    )
    idata["fit"] = DataTree(
        dataset=xr.Dataset({"elbo": ("step", -np.asarray(state.loss_history, dtype=float))})
    )
    return idata
