import time

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm

from arviz_base import dict_to_dataset
from pymc import Model, modelcontext
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import resolve_backend_compile_kwargs
from pytensor import config as pytensor_config
from pytensor.tensor.random.type import RandomType
from rich.console import Console
from rich.progress import (
    BarColumn,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Column
from rich.theme import Theme
from xarray import DataTree

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.compile import (
    TrainingFn,
    compile_sampling_fn,
    compile_svi_step_fn,
    compile_svi_training_fn,
)
from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data


def _reseed_function_rngs(fn, random_seed) -> None:
    """Reseed the RNG inputs of a compiled function.

    Operates on the compiled function's input storage instead of its shared variables:
    some backends (JAX) replace RNG shared variables with internal copies at compile
    time, so reseeding the user-facing shared variables would have no effect.
    """
    rng_containers = [
        container for container in fn.input_storage if isinstance(container.type, RandomType)
    ]
    if not rng_containers:
        return

    seed_seqs = np.random.SeedSequence(random_seed).spawn(len(rng_containers))
    for container, seed_seq in zip(rng_containers, seed_seqs):
        new_rng = np.random.Generator(np.random.PCG64(seed_seq))
        if not isinstance(container.storage[0], np.random.Generator):
            # The backend converted the rng into its own representation (e.g. JAX), and
            # will not do so again for a raw Generator after compilation
            from pytensor.link.jax.dispatch import jax_typify

            new_rng = jax_typify(new_rng)
        container.storage[0] = new_rng


def compute_step_speed(elapsed: float, step: int) -> tuple[float, str]:
    """Compute sampling speed and appropriate unit (draws/s or s/draw)."""
    speed = step / max(elapsed, 1e-6)

    if speed > 1 or speed == 0:
        unit = "steps/s"
    else:
        unit = "s/step"
        speed = 1 / speed

    return speed, unit


def make_advi_progress_bar(theme: Theme) -> CustomProgress:
    columns: list[ProgressColumn] = [
        TextColumn("{task.fields[step]}", table_column=Column("Step", ratio=1))
    ]

    columns += [
        TextColumn("{task.fields[loss]:.4f}", table_column=Column("ELBO", ratio=1)),
        TextColumn(
            "{task.fields[training_speed]:0.2f} {task.fields[speed_unit]}",
            table_column=Column("Training Speed", ratio=1),
        ),
        TimeElapsedColumn(table_column=Column("Elapsed", ratio=1)),
        TimeRemainingColumn(table_column=Column("Remaining", ratio=1)),
    ]

    return CustomProgress(
        BarColumn(
            table_column=Column("Progress", ratio=2),
            complete_style=Style.parse("rgb(31,119,180)"),
            finished_style=Style.parse("rgb(44,160,44)"),
        ),
        *columns,
        console=Console(theme=theme),
        include_headers=True,
    )


@dataclass
class SVIState:
    """Holds the current state of SVI training."""

    params: dict[str, np.ndarray]
    optimizer_state: Any
    step: int = 0
    loss_history: list[float] = field(default_factory=list)


class SVIModule(ABC):
    """
    Abstract base class for SVI training, following a PyTorch-Lightning style pattern.

    Users subclass this to define their guide, optimizer, and customize training hooks.

    Example:
    -------
    >>> class MyModule(SVIModule):
    ...     def configure_guide(self, model):
    ...         return AutoDiagonalNormal(model)
    ...
    ...     def configure_optimizer(self, params):
    ...         optimizer = adam(0.01)
    ...         return optimizer, optimizer.init(params)
    ...
    ...     def apply_gradients(self, params, grads, optimizer, optimizer_state):
    ...         updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    ...         return apply_updates(params, updates), optimizer_state
    ...
    ...     def on_epoch_end(self, state, loss):
    ...         if state.step % 100 == 0:
    ...             print(f"Step {state.step}: loss = {loss:.4f}")
    """

    @abstractmethod
    def configure_guide(self, model: Model) -> AutoGuideModel:
        """
        Create and return the guide for variational inference.

        Parameters
        ----------
        model : Model
            The PyMC model being fit.

        Returns
        -------
        AutoGuideModel
            The guide model with parameters to optimize.
        """
        ...

    @abstractmethod
    def configure_optimizer(self, params: dict[str, np.ndarray]) -> tuple[Any, dict[str, Any]]:
        """
        Configure the optimizer and its state.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Dictionary mapping parameter names to their initial values.

        Returns
        -------
        optimizer : Any
            The optimizer object (e.g., from optax, or a custom optimizer).
        optimizer_state : dict[str, Any]
            Initial optimizer state for each parameter.
        """
        ...

    @abstractmethod
    def apply_gradients(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        optimizer: Any,
        optimizer_state: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Apply gradients to update parameters.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Current parameter values.
        grads : dict[str, np.ndarray]
            Gradients for each parameter.
        optimizer : Any
            The optimizer object.
        optimizer_state : dict[str, Any]
            Current optimizer state.

        Returns
        -------
        new_params : dict[str, np.ndarray]
            Updated parameter values.
        new_optimizer_state : dict[str, Any]
            Updated optimizer state.
        """
        ...

    def on_fit_start(self, state: SVIState) -> None:
        """Called at the beginning of fit."""
        pass

    def on_fit_end(self, state: SVIState) -> None:
        """Called at the end of fit."""
        pass

    def on_epoch_start(self, state: SVIState) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, state: SVIState, loss: float) -> None:
        """Called at the end of each epoch with the current loss."""
        pass

    def should_stop(self, state: SVIState, loss: float) -> bool:
        """
        Override to implement early stopping logic.

        Parameters
        ----------
        state : SVIState
            Current training state.
        loss : float
            Current loss value.

        Returns
        -------
        bool
            True to stop training early, False to continue.
        """
        return False


class SVITrainer:
    """
    Trainer for stochastic variational inference.

    Handles compilation and the training loop, delegating configuration
    and customization to the SVIModule.

    Parameters
    ----------
    module : SVIModule
        The module defining the guide, optimizer, and hooks.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True.
    backend : str, optional
        PyTensor backend to compile the training and sampling functions with
        (e.g. "numba", "jax", "c"). Mutually exclusive with ``compile_kwargs["mode"]``.
    compile_kwargs : dict, optional
        Additional kwargs passed to pytensor compilation.

    Example
    -------
    >>> trainer = SVITrainer(MyModule())
    >>> state = trainer.fit(model, n_steps=1000, draws_per_step=1)
    >>> final_params = state.params
    """

    def __init__(
        self,
        module: SVIModule,
        path_derivative_gradient: bool = True,
        backend: str | None = None,
        compile_kwargs: dict | None = None,
    ):
        self.module = module
        self.path_derivative_gradient = path_derivative_gradient
        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)

        self._training_fn: TrainingFn | None = None
        self._training_draws: int | None = None
        self._step_fn: TrainingFn | None = None
        self._step_draws: int | None = None
        self._step_shared_params: dict | None = None
        self._sampling_fn: TrainingFn | None = None
        self._sampling_draws: int | None = None
        self._guide: AutoGuideModel | None = None
        self._optimizer: Any = None
        self._param_names: list[str] | None = None

    def _configure_guide(self, model: Model) -> None:
        if self._guide is not None:
            return
        # Sacrificial detached model context: a guide built naively with a plain Model()
        # inside the user's model context lands here instead of writing into their model
        with Model(model=None):
            self._guide = self.module.configure_guide(model)
        self._param_names = [p.name for p in self._guide.params]

    def _compile_training_fn(self, model: Model, draws_per_step: int) -> None:
        """Compile the training function, reusing a previous one when draws match.

        ``draws_per_step`` is baked into the compiled function as a constant, because
        backends like JAX cannot handle inputs that determine random variable shapes.
        """
        if self._training_fn is not None and self._training_draws == draws_per_step:
            return
        self._training_fn = compile_svi_training_fn(
            model,
            self._guide,
            draws=draws_per_step,
            path_derivative_gradient=self.path_derivative_gradient,
            **self.compile_kwargs,
        )
        self._training_draws = draws_per_step

    def _compile_sampling_fn(self, model: Model, draws: int) -> None:
        """Compile the posterior sampling function, reusing a previous one when draws match."""
        if self._sampling_fn is not None and self._sampling_draws == draws:
            return
        self._sampling_fn = compile_sampling_fn(
            model=model,
            guide=self._guide,
            draws=draws,
            **self.compile_kwargs,
        )
        self._sampling_draws = draws

    def fit(
        self,
        n_steps: int,
        draws_per_step: int = 1,
        model: Model | None = None,
        state: SVIState | None = None,
        random_seed=None,
    ) -> SVIState:
        """
        Fit the model using SVI.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        draws_per_step : int, optional
            Number of MC draws per step for gradient estimation, by default 1.
        model : Model
            The PyMC model to fit. If None, the model is inferred from context.
        state : SVIState, optional
            Previous state to resume training from. If None, starts fresh.
        random_seed : optional
            Seed for the guide draws used to estimate the gradients.

        Returns
        -------
        SVIState
            The final training state containing optimized parameters.
        """
        if model is None:
            model = modelcontext(None)

        self._configure_guide(model)
        self._compile_training_fn(model, draws_per_step)

        if random_seed is not None:
            _reseed_function_rngs(self._training_fn, random_seed)

        if state is None:
            init_params = {p.name: v for p, v in self._guide.params_init_values.items()}
            self._optimizer, optimizer_state = self.module.configure_optimizer(init_params)
            state = SVIState(
                params=init_params,
                optimizer_state=optimizer_state,
                step=0,
                loss_history=[],
            )

        self.module.on_fit_start(state)
        progress = make_advi_progress_bar(theme=default_progress_theme)

        try:
            with progress:
                task = progress.add_task(
                    "Fitting",
                    step=0,
                    total=n_steps,
                    loss=np.inf,
                    training_speed=0,
                    speed_unit="steps/s",
                )
                # Mutated in place each step: rebuilding it on the SVIState would be
                # quadratic in n_steps
                loss_history = list(state.loss_history)
                progress_every = max(1, n_steps // 1_000)
                # Set after the first step so the one-time graph compilation triggered by
                # that first call is excluded from the steps/s estimate
                start_time = None

                for step in range(n_steps):
                    self.module.on_epoch_start(state)

                    # The compiled function uses trust_input=True, so scalar params must be
                    # passed as 0d arrays, not python/numpy scalars
                    params = {name: np.asarray(value) for name, value in state.params.items()}
                    outputs = self._training_fn(**params)
                    if start_time is None:
                        start_time = time.perf_counter()
                    # Backends may return their own array types (e.g. JAX); convert once here
                    # so the optimizer update runs on numpy arrays
                    loss, *grads = (np.asarray(out) for out in outputs)
                    grad_dict = dict(zip(self._param_names, grads))

                    new_params, new_optimizer_state = self.module.apply_gradients(
                        state.params, grad_dict, self._optimizer, state.optimizer_state
                    )

                    loss_history.append(loss)
                    state = SVIState(
                        params=new_params,
                        optimizer_state=new_optimizer_state,
                        step=state.step + 1,
                        loss_history=loss_history,
                    )

                    self.module.on_epoch_end(state, loss)

                    if self.module.should_stop(state, loss):
                        break

                    if step % progress_every == 0:
                        elapsed = time.perf_counter() - start_time
                        speed, unit = compute_step_speed(elapsed, step)
                        progress.update(
                            task,
                            completed=step,
                            step=step,
                            loss=loss,
                            training_speed=speed,
                            speed_unit=unit,
                        )

                progress.update(
                    task,
                    completed=n_steps,
                    step=step + 1,
                    loss=loss,
                    training_speed=speed,
                    speed_unit=unit,
                    refresh=True,
                )
        except KeyboardInterrupt:
            pass

        self.module.on_fit_end(state)

        return state

    def fit_jitted(
        self,
        n_steps: int,
        draws_per_step: int = 1,
        model: Model | None = None,
        learning_rate: float | Callable[[int], float] = 0.008,
        clip_norm: float | None = 10.0,
        random_seed=None,
    ) -> SVIState:
        """
        Fit the model with the optimizer update compiled into the step function.

        Unlike :meth:`fit`, the guide parameters and the (clipped) Adam state live in
        shared variables updated in place by the compiled function, so nothing
        round-trips through Python per step. The module's ``configure_optimizer`` and
        ``apply_gradients`` are not used; ``should_stop`` and the epoch hooks still are.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        draws_per_step : int, optional
            Number of MC draws per step for gradient estimation, by default 1.
        model : Model
            The PyMC model to fit. If None, the model is inferred from context.
        learning_rate : float or callable, optional
            Learning rate, or a schedule mapping the step number to one.
        clip_norm : float, optional
            Clip gradients to this global norm, by default 10. None disables clipping.
        random_seed : optional
            Seed for the guide draws used to estimate the gradients.

        Returns
        -------
        SVIState
            The final training state containing optimized parameters.
        """
        if model is None:
            model = modelcontext(None)

        self._configure_guide(model)

        if self._step_fn is None or self._step_draws != draws_per_step:
            self._step_fn, self._step_shared_params = compile_svi_step_fn(
                model,
                self._guide,
                draws=draws_per_step,
                path_derivative_gradient=self.path_derivative_gradient,
                clip_norm=clip_norm,
                **self.compile_kwargs,
            )
            self._step_draws = draws_per_step

        if random_seed is not None:
            _reseed_function_rngs(self._step_fn, random_seed)

        lr_dtype = np.dtype(pytensor_config.floatX)
        schedule = learning_rate if callable(learning_rate) else (lambda step: learning_rate)

        loss_history: list = []
        state = SVIState(
            params={name: shared.get_value() for name, shared in self._step_shared_params.items()},
            optimizer_state=None,
            step=0,
            loss_history=loss_history,
        )

        self.module.on_fit_start(state)
        progress = make_advi_progress_bar(theme=default_progress_theme)
        progress_every = max(1, n_steps // 1_000)

        try:
            with progress:
                task = progress.add_task(
                    "Fitting",
                    step=0,
                    total=n_steps,
                    loss=np.inf,
                    training_speed=0,
                    speed_unit="steps/s",
                )
                speed, unit = 0.0, "steps/s"
                loss = np.inf
                # Set after the first step so the one-time graph compilation triggered by
                # that first call is excluded from the steps/s estimate
                start_time = None
                for step in range(n_steps):
                    self.module.on_epoch_start(state)

                    loss = np.asarray(self._step_fn(np.asarray(schedule(step), dtype=lr_dtype)))
                    if start_time is None:
                        start_time = time.perf_counter()
                    loss_history.append(loss)
                    state = SVIState(
                        params=state.params,
                        optimizer_state=None,
                        step=state.step + 1,
                        loss_history=loss_history,
                    )

                    self.module.on_epoch_end(state, loss)

                    if self.module.should_stop(state, loss):
                        break

                    if step % progress_every == 0:
                        elapsed = time.perf_counter() - start_time
                        speed, unit = compute_step_speed(elapsed, step)
                        progress.update(
                            task,
                            completed=step,
                            step=step,
                            loss=loss,
                            training_speed=speed,
                            speed_unit=unit,
                        )

                progress.update(
                    task,
                    completed=n_steps,
                    step=state.step,
                    loss=loss,
                    training_speed=speed,
                    speed_unit=unit,
                    refresh=True,
                )
        except KeyboardInterrupt:
            pass

        state = SVIState(
            params={
                name: shared.get_value().copy() for name, shared in self._step_shared_params.items()
            },
            optimizer_state=None,
            step=state.step,
            loss_history=loss_history,
        )
        self.module.on_fit_end(state)

        return state

    def sample_posterior(
        self, draws: int, state: SVIState, model: Model | None = None, random_seed=None
    ) -> DataTree:
        """
        Sample from the guide posterior using the trained parameters.

        Parameters
        ----------
        draws: int
            Number of posterior samples to draw.
        state : SVIState
            The training state containing optimized parameters.
        model : Model | None
            The PyMC model. If None, the model is inferred from context.
        random_seed : optional
            Seed for the posterior draws.

        Returns
        -------
        DataTree
            Samples from the guide posterior for each latent variable.
        """
        if self._guide is None:
            raise RuntimeError("The trainer has not been fitted yet.")

        if model is None:
            model = modelcontext(None)

        self._compile_sampling_fn(model, draws)

        if random_seed is not None:
            _reseed_function_rngs(self._sampling_fn, random_seed)

        params = {name: np.asarray(value) for name, value in state.params.items()}
        samples = self._sampling_fn(**params)
        posterior = {
            rv.name: np.expand_dims(sample, axis=0)
            for rv, sample in zip(
                (rv for rv in model.rvs_to_values.keys() if rv not in model.observed_RVs), samples
            )
        }

        model_coords, model_dims = coords_and_dims_for_inferencedata(model)
        posterior_dataset = dict_to_dataset(
            posterior, coords=model_coords, dims=model_dims, inference_library=pm
        )

        idata = DataTree.from_dict({"posterior": posterior_dataset})
        # Forward the chosen backend so model deterministics are computed on the same
        # backend as the rest of the fit, not pytensor's default.
        idata = add_data_to_inference_data(
            idata=idata, progressbar=False, model=model, compile_kwargs=self.compile_kwargs
        )

        return idata
