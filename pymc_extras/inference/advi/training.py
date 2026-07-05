import time

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

from pymc_extras.inference.advi.autoguide import AutoDiagonalNormal, AutoGuideModel
from pymc_extras.inference.advi.compile import (
    TrainingFn,
    compile_sampling_fn,
    compile_svi_step_fn,
    compile_svi_training_fn,
)
from pymc_extras.inference.advi.optimizers import (
    GradientTransformation,
    ScalarOrSchedule,
    apply_updates,
    linear_onecycle_schedule,
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


# One training step: maps (step number, state) to (loss, new state).
StepFn = Callable[[int, SVIState], tuple[np.ndarray, SVIState]]


class Trainer:
    """
    Trainer for stochastic variational inference.

    Follows the design of PyTorch Lightning's ``Trainer`` (and pymc-devs/pymc#8333):
    the model owns the math, the trainer owns the training loop, and there are no
    user-facing callbacks — everything, including convergence-based early stopping,
    is configured at construction, and :meth:`fit` just runs.

    Parameters
    ----------
    guide : AutoGuideModel or callable, optional
        The guide to fit: an :class:`AutoGuideModel`, or a factory mapping the model
        to one. By default an :func:`AutoDiagonalNormal` guide is built from the
        model (mean-field ADVI).
    optimizer : GradientTransformation, optional
        An optax-like optimizer (actual optax optimizers are compatible). By default
        a clipped-Adam update on ``learning_rate`` is compiled *into* the step
        function, so no parameters or gradients round-trip through Python per step
        (fast path); passing an explicit optimizer switches to a Python-side update
        loop.
    learning_rate : float or callable, optional
        Learning rate, or a schedule mapping the step number to one, for the default
        compiled optimizer. Defaults to a :func:`linear_onecycle_schedule` peaking at
        0.008 over the ``n`` steps of each :meth:`fit` call. Ignored when
        ``optimizer`` is given (set the learning rate on the optimizer itself).
    clip_norm : float, optional
        Clip gradients to this global norm in the default compiled optimizer, by
        default 10. None disables clipping. Ignored when ``optimizer`` is given.
    n_particles : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by
        default 1.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True.
    convergence_window : int, optional
        Number of steps per convergence window, by default 200. Training stops early
        when the mean loss over the last window is within ``relative_tolerance`` of
        the mean over the window before it. Set to None to disable early stopping.
    relative_tolerance : float, optional
        Relative loss change between consecutive windows under which training stops,
        by default 1e-3.
    model : Model, optional
        The PyMC model to fit. If None, the model is taken from the context stack
        when :meth:`fit` or :meth:`sample_posterior` is called.
    backend : str, optional
        PyTensor backend to compile the training and sampling functions with
        (e.g. "numba", "jax", "c"). Mutually exclusive with ``compile_kwargs["mode"]``.
    compile_kwargs : dict, optional
        Additional kwargs passed to pytensor compilation.
    random_seed : optional
        Seed for the default guide's initialization. Seeds for the training and
        posterior draws are passed to :meth:`fit` and :meth:`sample_posterior`.

    Example
    -------
    >>> with pm.Model() as model:
    ...     mu = pm.Normal("mu", 0, 1)
    ...     pm.Normal("y", mu, 1, observed=[0.5, 1.5])
    ...     trainer = Trainer()
    ...     state = trainer.fit(10_000)
    ...     idata = trainer.sample_posterior(1_000)
    """

    def __init__(
        self,
        *,
        guide: AutoGuideModel | Callable[[Model], AutoGuideModel] | None = None,
        optimizer: GradientTransformation | None = None,
        learning_rate: ScalarOrSchedule | None = None,
        clip_norm: float | None = 10.0,
        n_particles: int = 1,
        path_derivative_gradient: bool = True,
        convergence_window: int | None = 200,
        relative_tolerance: float = 1e-3,
        model: Model | None = None,
        backend: str | None = None,
        compile_kwargs: dict | None = None,
        random_seed=None,
    ):
        self.guide = guide
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.n_particles = n_particles
        self.path_derivative_gradient = path_derivative_gradient
        self.convergence_window = convergence_window
        self.relative_tolerance = relative_tolerance
        self.model = model
        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        self.random_seed = random_seed
        self.state: SVIState | None = None

        self._guide: AutoGuideModel | None = guide if isinstance(guide, AutoGuideModel) else None
        self._param_names: list[str] | None = None
        self._training_fn: TrainingFn | None = None
        self._step_fn: TrainingFn | None = None
        self._step_shared_params: dict | None = None
        self._sampling_fn: TrainingFn | None = None
        self._sampling_draws: int | None = None

    def _resolve_guide(self, model: Model) -> None:
        if self._guide is None:
            # Sacrificial detached model context: a guide built naively with a plain
            # Model() inside the user's model context lands here instead of writing
            # into their model
            with Model(model=None):
                if callable(self.guide):
                    self._guide = self.guide(model)
                else:
                    self._guide = AutoDiagonalNormal(model, random_seed=self.random_seed)
        if self._param_names is None:
            self._param_names = [p.name for p in self._guide.params]

    def _compile_training_fn(self, model: Model) -> None:
        """Compile the training function, reusing a previous one if available.

        ``n_particles`` is baked into the compiled function as a constant, because
        backends like JAX cannot handle inputs that determine random variable shapes.
        """
        if self._training_fn is not None:
            return
        self._training_fn = compile_svi_training_fn(
            model,
            self._guide,
            draws=self.n_particles,
            path_derivative_gradient=self.path_derivative_gradient,
            **self.compile_kwargs,
        )

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

    def _should_stop(self, step: int, loss_history: list[float]) -> bool:
        """Window-based convergence check, see ``convergence_window``."""
        window = self.convergence_window
        if window is None or step % window != 0 or len(loss_history) < 2 * window:
            return False
        recent = np.mean(loss_history[-window:])
        previous = np.mean(loss_history[-2 * window : -window])
        return bool(abs(recent - previous) < self.relative_tolerance * (abs(previous) + 1e-8))

    def _make_python_step(
        self, model: Model, state: SVIState | None, random_seed
    ) -> tuple[StepFn, SVIState, Callable[[SVIState], SVIState]]:
        """Set up the Python-side update loop for a user-provided optax-like optimizer."""
        self._compile_training_fn(model)
        if random_seed is not None:
            _reseed_function_rngs(self._training_fn, random_seed)

        if state is None:
            init_params = {p.name: v for p, v in self._guide.params_init_values.items()}
            state = SVIState(
                params=init_params,
                optimizer_state=self.optimizer.init(init_params),
                step=0,
                loss_history=[],
            )
        # Mutated in place each step and shared by every per-step state: rebuilding it
        # on each new SVIState would be quadratic in the number of steps
        loss_history = list(state.loss_history)
        state = SVIState(state.params, state.optimizer_state, state.step, loss_history)

        optimizer = self.optimizer
        param_names = self._param_names

        def step_fn(step: int, state: SVIState) -> tuple[np.ndarray, SVIState]:
            # The compiled function uses trust_input=True, so scalar params must be
            # passed as 0d arrays, not python/numpy scalars
            params = {name: np.asarray(value) for name, value in state.params.items()}
            outputs = self._training_fn(**params)
            # Backends may return their own array types (e.g. JAX); convert once here
            # so the optimizer update runs on numpy arrays
            loss, *grads = (np.asarray(out) for out in outputs)
            updates, optimizer_state = optimizer.update(
                dict(zip(param_names, grads)), state.optimizer_state, state.params
            )
            new_state = SVIState(
                apply_updates(state.params, updates), optimizer_state, state.step + 1, loss_history
            )
            return loss, new_state

        return step_fn, state, lambda final_state: final_state

    def _make_compiled_step(
        self, model: Model, n: int, state: SVIState | None, random_seed
    ) -> tuple[StepFn, SVIState, Callable[[SVIState], SVIState]]:
        """Set up the fast path: clipped-Adam updates compiled into the step function.

        The guide parameters and the optimizer state live in shared variables updated
        in place by the compiled function, so nothing round-trips through Python per
        step. The function's only input is the learning rate; a resumed ``state`` only
        restores the parameters (the Adam moments are not part of ``SVIState``).
        """
        if self._step_fn is None:
            self._step_fn, self._step_shared_params = compile_svi_step_fn(
                model,
                self._guide,
                draws=self.n_particles,
                path_derivative_gradient=self.path_derivative_gradient,
                clip_norm=self.clip_norm,
                **self.compile_kwargs,
            )

        if random_seed is not None:
            _reseed_function_rngs(self._step_fn, random_seed)

        shared_params = self._step_shared_params
        if state is not None:
            for name, shared in shared_params.items():
                shared.set_value(np.asarray(state.params[name]))
            loss_history = list(state.loss_history)
            start_step = state.step
        else:
            loss_history = []
            start_step = 0

        learning_rate = self.learning_rate
        if learning_rate is None:
            learning_rate = linear_onecycle_schedule(
                transition_steps=n, peak_value=0.008, pct_start=0.2
            )
        schedule = learning_rate if callable(learning_rate) else (lambda step: learning_rate)
        lr_dtype = np.dtype(pytensor_config.floatX)

        state = SVIState(
            params={name: shared.get_value() for name, shared in shared_params.items()},
            optimizer_state=None,
            step=start_step,
            loss_history=loss_history,
        )

        compiled_step = self._step_fn

        def step_fn(step: int, state: SVIState) -> tuple[np.ndarray, SVIState]:
            loss = np.asarray(compiled_step(np.asarray(schedule(step), dtype=lr_dtype)))
            return loss, SVIState(state.params, None, state.step + 1, loss_history)

        def finalize(final_state: SVIState) -> SVIState:
            # The per-step states carry stale params; read the trained values out of
            # the shared variables once at the end
            return SVIState(
                params={name: shared.get_value().copy() for name, shared in shared_params.items()},
                optimizer_state=None,
                step=final_state.step,
                loss_history=loss_history,
            )

        return step_fn, state, finalize

    def fit(
        self,
        n: int = 10_000,
        *,
        state: SVIState | None = None,
        random_seed=None,
    ) -> SVIState:
        """
        Fit the model using SVI for ``n`` steps.

        With the default compiled optimizer the guide parameters and the Adam state
        live in shared variables updated in place by the compiled step function; with
        an explicit ``optimizer`` each step round-trips the parameters and gradients
        through a Python-side update.

        Parameters
        ----------
        n : int, optional
            Maximum number of optimization steps, by default 10_000. Training may
            stop earlier, controlled by ``convergence_window`` and
            ``relative_tolerance``.
        state : SVIState, optional
            Previous state to resume training from. If None, starts fresh. With the
            default compiled optimizer only the parameters are restored, not the
            Adam moments.
        random_seed : optional
            Seed for the guide draws used to estimate the gradients.

        Returns
        -------
        SVIState
            The final training state containing optimized parameters. Also stored on
            the trainer, where :meth:`sample_posterior` picks it up by default.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"n must be a positive integer (the number of fit steps), got {n!r}")

        model = modelcontext(self.model)
        self._resolve_guide(model)

        if self.optimizer is None:
            step_fn, state, finalize = self._make_compiled_step(model, n, state, random_seed)
        else:
            step_fn, state, finalize = self._make_python_step(model, state, random_seed)
        loss_history = state.loss_history

        progress = make_advi_progress_bar(theme=default_progress_theme)
        progress_every = max(1, n // 1_000)

        try:
            with progress:
                task = progress.add_task(
                    "Fitting",
                    step=0,
                    total=n,
                    loss=np.inf,
                    training_speed=0,
                    speed_unit="steps/s",
                )
                speed, unit = 0.0, "steps/s"
                loss = np.inf
                # Set after the first step so the one-time graph compilation triggered by
                # that first call is excluded from the steps/s estimate
                start_time = None

                for step in range(n):
                    loss, state = step_fn(step, state)
                    if start_time is None:
                        start_time = time.perf_counter()
                    loss_history.append(loss)

                    if self._should_stop(state.step, loss_history):
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
                    completed=n,
                    step=state.step,
                    loss=loss,
                    training_speed=speed,
                    speed_unit=unit,
                    refresh=True,
                )
        except KeyboardInterrupt:
            pass

        state = finalize(state)
        self.state = state

        return state

    def sample_posterior(
        self,
        draws: int = 1_000,
        *,
        state: SVIState | None = None,
        random_seed=None,
    ) -> DataTree:
        """
        Sample from the guide posterior using the trained parameters.

        Parameters
        ----------
        draws : int, optional
            Number of posterior samples to draw, by default 1_000.
        state : SVIState, optional
            The training state containing optimized parameters. Defaults to the state
            of the last :meth:`fit` call.
        random_seed : optional
            Seed for the posterior draws.

        Returns
        -------
        DataTree
            Samples from the guide posterior for each latent variable.
        """
        if state is None:
            state = self.state
        if state is None or self._guide is None:
            raise RuntimeError("The trainer has not been fitted yet.")

        model = modelcontext(self.model)
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
        idata = add_data_to_inference_data(idata=idata, progressbar=False, model=model)

        return idata
