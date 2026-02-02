from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor import tensor as pt

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class TrainingFn(Protocol):
    def __call__(self, draws: int, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_training_fn(
    model: Model, guide: AutoGuideModel, stick_the_landing: bool = True, **compile_kwargs
) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params

    logp, logq = get_logp_logq(model, guide)

    scalar_negative_elbo = advi_objective(logp, logq, stick_the_landing=stick_the_landing)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    negative_elbo_grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=params)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_loss_dloss = compile(
        inputs=[draws, *params], outputs=[negative_elbo, *negative_elbo_grads], **compile_kwargs
    )

    return f_loss_dloss


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

        self._compiled_fn: TrainingFn | None = None
        self._guide: AutoGuideModel | None = None
        self._optimizer: Any = None
        self._param_names: list[str] | None = None

    def _compile(self, model: Model) -> None:
        """Compile the training function."""
        self._guide = self.module.configure_guide(model)
        self._compiled_fn = compile_svi_training_fn(
            model,
            self._guide,
            stick_the_landing=self.stick_the_landing,
            **self.compile_kwargs,
        )
        self._param_names = [p.name for p in self._guide.params]

    def _params_dict_to_tuple(self, params: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
        """Convert params dict to tuple in correct order."""
        return tuple(params[name] for name in self._param_names)

    def _params_tuple_to_dict(self, params: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
        """Convert params tuple to dict."""
        return dict(zip(self._param_names, params))

    def _grads_tuple_to_dict(self, grads: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
        """Convert grads tuple to dict."""
        return dict(zip(self._param_names, grads))

    def fit(
        self,
        model: Model,
        n_steps: int,
        draws_per_step: int = 10,
        state: SVIState | None = None,
    ) -> SVIState:
        """
        Fit the model using SVI.

        Parameters
        ----------
        model : Model
            The PyMC model to fit.
        n_steps : int
            Number of optimization steps.
        draws_per_step : int, optional
            Number of MC draws per step for gradient estimation, by default 10.
        state : SVIState, optional
            Previous state to resume training from. If None, starts fresh.

        Returns
        -------
        SVIState
            The final training state containing optimized parameters.
        """
        if self._compiled_fn is None:
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

        for _ in range(n_steps):
            self.module.on_epoch_start(state)

            params_tuple = self._params_dict_to_tuple(state.params)
            outputs = self._compiled_fn(draws_per_step, *params_tuple)

            loss = float(outputs[0])
            grads_tuple = outputs[1:]
            grads = self._grads_tuple_to_dict(grads_tuple)

            new_params, new_optimizer_state = self.module.apply_gradients(
                state.params, grads, self._optimizer, state.optimizer_state
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

        self.module.on_fit_end(state)

        return state
