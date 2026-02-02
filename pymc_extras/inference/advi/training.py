from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import arviz as az
import numpy as np
import pymc as pm

from arviz import dict_to_dataset
from pymc import Model, compile, modelcontext
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import rewrite_pregrad
from pytensor import tensor as pt
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

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph
from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data


def compute_step_speed(elapsed: float, step: int) -> tuple[float, str]:
    """Compute sampling speed and appropriate unit (draws/s or s/draw)."""
    speed = step / max(elapsed, 1e-6)

    if speed > 1 or speed == 0:
        unit = "steps/s"
    else:
        unit = "s/step"
        speed = 1 / speed

    return speed, unit


class TrainingFn(Protocol):
    def __call__(self, draws: int, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_training_fn(
    model: Model, guide: AutoGuideModel, stick_the_landing: bool = True, **compile_kwargs
) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params

    logp, logq = get_logp_logq(model, guide, stick_the_landing=stick_the_landing)

    scalar_negative_elbo = advi_objective(logp, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    negative_elbo_grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=params)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_loss_dloss = compile(
        inputs=[draws, *params], outputs=[negative_elbo, *negative_elbo_grads], **compile_kwargs
    )

    return f_loss_dloss


def compile_sampling_fn(model: Model, guide: AutoGuideModel, **compile_kwargs) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params

    parameterized_value_vars = [
        guide.model[rv.name] for rv in model.rvs_to_values.keys() if rv not in model.observed_RVs
    ]
    transformed_vars = [
        transform.backward(parameterized_var)
        if (transform := model.rvs_to_transforms[rv]) is not None
        else parameterized_var
        for rv, parameterized_var in zip(model.rvs_to_values.keys(), parameterized_value_vars)
    ]

    sampled_rvs_draws = vectorize_random_graph(transformed_vars, batch_draws=draws)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_sample = compile(inputs=[draws, *params], outputs=sampled_rvs_draws, **compile_kwargs)

    return f_sample


def make_advi_progress_bar(theme: Theme) -> CustomProgress:
    columns: list[ProgressColumn] = [
        TextColumn("{task.fields[step]}", table_column=Column("Step", ratio=1))
    ]

    columns += [
        TextColumn("{task.fields[loss]:.4f}", table_column=Column("ELBO", ratio=1)),
        TextColumn(
            "{task.fields[sampling_speed]:0.2f} {task.fields[speed_unit]}",
            table_column=Column("Sampling Speed", ratio=1),
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
    ...         return Adam(lr=0.01), {name: {} for name in params}
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
    stick_the_landing : bool, optional
        Whether to use the STL gradient estimator, by default True.
    compile_kwargs : dict, optional
        Additional kwargs passed to pytensor compilation.

    Example
    -------
    >>> trainer = SVITrainer(MyModule())
    >>> state = trainer.fit(model, n_steps=1000, draws_per_step=10)
    >>> final_params = state.params
    """

    def __init__(
        self,
        module: SVIModule,
        stick_the_landing: bool = True,
        compile_kwargs: dict | None = None,
    ):
        self.module = module
        self.stick_the_landing = stick_the_landing
        self.compile_kwargs = compile_kwargs or {}

        self._training_fn: TrainingFn | None = None
        self._sampling_fn: TrainingFn | None = None
        self._guide: AutoGuideModel | None = None
        self._optimizer: Any = None
        self._param_names: list[str] | None = None

    def _compile(self, model: Model) -> None:
        """Compile the training function."""
        self._guide = self.module.configure_guide(model)
        self._training_fn = compile_svi_training_fn(
            model,
            self._guide,
            stick_the_landing=self.stick_the_landing,
            **self.compile_kwargs,
        )

        self._sampling_fn = compile_sampling_fn(
            model=model,
            guide=self._guide,
            **self.compile_kwargs,
        )

        self._param_names = [p.name for p in self._guide.params]

    def fit(
        self,
        n_steps: int,
        draws_per_step: int = 10,
        model: Model | None = None,
        state: SVIState | None = None,
    ) -> SVIState:
        """
        Fit the model using SVI.

        Parameters
        ----------
        n_steps : int
            Number of optimization steps.
        draws_per_step : int, optional
            Number of MC draws per step for gradient estimation, by default 10.
        model : Model
            The PyMC model to fit. If None, the model is inferred from context.
        state : SVIState, optional
            Previous state to resume training from. If None, starts fresh.

        Returns
        -------
        SVIState
            The final training state containing optimized parameters.
        """
        if model is None:
            model = modelcontext(None)

        if self._training_fn is None:
            self._compile(model)

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
                    sampling_speed=0,
                    speed_unit="steps/s",
                )
                for step in range(n_steps):
                    self.module.on_epoch_start(state)

                    loss, *grads = self._training_fn(np.array(draws_per_step), **state.params)
                    grad_dict = {name: grad for name, grad in zip(self._param_names, grads)}

                    new_params, new_optimizer_state = self.module.apply_gradients(
                        state.params, grad_dict, self._optimizer, state.optimizer_state
                    )

                    state = SVIState(
                        params=new_params,
                        optimizer_state=new_optimizer_state,
                        step=state.step + 1,
                        loss_history=[*state.loss_history, loss],
                    )

                    self.module.on_epoch_end(state, loss)

                    if self.module.should_stop(state, loss):
                        break

                    elapsed = progress.tasks[0].elapsed or 0.0
                    speed, unit = compute_step_speed(elapsed, step)
                    progress.update(
                        task,
                        completed=step,
                        step=step,
                        loss=loss,
                        sampling_speed=speed,
                        speed_unit=unit,
                    )

                progress.update(
                    task,
                    completed=n_steps,
                    step=step + 1,
                    loss=loss,
                    sampling_speed=speed,
                    speed_unit=unit,
                    refresh=True,
                )
        except KeyboardInterrupt:
            pass

        self.module.on_fit_end(state)

        return state

    def sample_posterior(
        self, draws: int, state: SVIState, model: Model | None = None
    ) -> az.InferenceData:
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

        Returns
        -------
        dict[str, np.ndarray]
            Samples from the guide posterior for each latent variable.
        """
        if self._guide is None or self._sampling_fn is None:
            raise RuntimeError("The trainer has not been fitted yet.")

        if model is None:
            model = modelcontext(None)

        samples = self._sampling_fn(np.array(draws), **state.params)
        posterior = {
            rv.name: np.expand_dims(sample, axis=0)
            for rv, sample in zip(
                (rv for rv in model.rvs_to_values.keys() if rv not in model.observed_RVs), samples
            )
        }

        model_coords, model_dims = coords_and_dims_for_inferencedata(model)
        posterior_dataset = dict_to_dataset(
            posterior, coords=model_coords, dims=model_dims, library=pm
        )

        idata = az.InferenceData(posterior=posterior_dataset)
        idata = add_data_to_inference_data(idata=idata, progressbar=False, model=model)

        return idata
