import time

from collections.abc import Callable
from dataclasses import dataclass

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
)
from pymc_extras.inference.advi.schedules import ScalarOrSchedule, linear_onecycle_schedule
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


@dataclass(frozen=True)
class SVIState:
    """A complete snapshot of a :class:`Trainer`.

    Everything needed to resume training exactly, so that
    ``Trainer(...).load_state(state).fit(n)`` continues a run the same way a second
    ``fit(n)`` on the original trainer would.
    """

    params: dict[str, np.ndarray]
    optimizer_state: dict[str, np.ndarray]
    step: int
    loss_history: np.ndarray


class Trainer:
    """
    Trainer for stochastic variational inference.

    The trainer *is* the training state: the guide parameters and the Adam moments live
    in shared variables inside the compiled step function, and :meth:`fit` always
    continues from wherever they are. Calling ``fit(n)`` twice runs ``2 * n`` steps.
    Use :meth:`reset` to start over and :meth:`load_state` to adopt a snapshot.

    Configuration splits along the same line. Everything compiled into the step function
    (``guide``, ``n_particles``, ``path_derivative_gradient``, ``clip_norm``) is fixed at
    construction and exposed read-only. Everything that is per-run policy
    (``learning_rate``, ``convergence_window``, ``relative_tolerance``) is read afresh by
    each :meth:`fit` call and can be changed between them, or overridden for a single
    call by passing ``learning_rate`` to :meth:`fit`.

    Parameters
    ----------
    guide : AutoGuideModel or callable, optional
        The guide to fit: an :class:`AutoGuideModel`, or a factory mapping the model
        to one. By default an :func:`AutoDiagonalNormal` guide is built from the
        model (mean-field ADVI).
    learning_rate : float or callable, optional
        Learning rate, or a schedule mapping the step number within a :meth:`fit` call
        to one. Defaults to a :func:`linear_onecycle_schedule` peaking at 0.008 over the
        ``n`` steps of each call, so a follow-up ``fit`` is a warm restart: the
        parameters and Adam moments carry over, the learning rate ramps again.
    clip_norm : float, optional
        Clip gradients to this global norm, by default 10. None disables clipping.
    n_particles : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by
        default 1.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True.
    convergence_window : int, optional
        Number of steps per convergence window, by default 200. A :meth:`fit` call stops
        early when the mean loss over its last window is within ``relative_tolerance`` of
        the mean over the window before it. Only steps taken by the current call count,
        so a later ``fit`` always gets a fresh chance to make progress. Set to None to
        disable early stopping.
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
    ...     trainer.fit(10_000)
    ...     trainer.fit(5_000, learning_rate=1e-4)  # 5_000 more, at a fixed rate
    ...     idata = trainer.sample_posterior(1_000)
    """

    def __init__(
        self,
        *,
        guide: AutoGuideModel | Callable[[Model], AutoGuideModel] | None = None,
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
        # Per-run policy: read afresh by every fit call
        self.learning_rate = learning_rate
        self.convergence_window = convergence_window
        self.relative_tolerance = relative_tolerance

        self.model = model
        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        self.random_seed = random_seed

        # Compiled into the step function on the first fit, hence read-only
        self._guide_factory = guide
        self._clip_norm = clip_norm
        self._n_particles = n_particles
        self._path_derivative_gradient = path_derivative_gradient

        self._guide: AutoGuideModel | None = guide if isinstance(guide, AutoGuideModel) else None
        self._step_fn: TrainingFn | None = None
        self._shared_params: dict | None = None
        self._shared_optimizer_state: dict | None = None
        self._init_state: SVIState | None = None
        self._sampling_fn: TrainingFn | None = None
        self._sampling_draws: int | None = None
        self._loss_history: list[float] = []
        self._step = 0

    @property
    def guide(self) -> AutoGuideModel | None:
        """The guide being fit, once resolved. Compiled in, so it cannot be replaced."""
        return self._guide

    @property
    def clip_norm(self) -> float | None:
        """Global gradient norm clip. Compiled in, so it cannot be changed after a fit."""
        return self._clip_norm

    @property
    def n_particles(self) -> int:
        """Guide draws per step. Compiled in, so it cannot be changed after a fit."""
        return self._n_particles

    @property
    def path_derivative_gradient(self) -> bool:
        """Whether the path-derivative estimator is used. Compiled in, so it is fixed."""
        return self._path_derivative_gradient

    @property
    def step(self) -> int:
        """Total number of optimization steps taken so far."""
        return self._step

    @property
    def state(self) -> SVIState:
        """Snapshot of the current training state, read out of the shared variables."""
        if self._step_fn is None:
            raise RuntimeError("The trainer has not been fitted yet.")
        return SVIState(
            params={
                name: shared.get_value().copy() for name, shared in self._shared_params.items()
            },
            optimizer_state={
                name: shared.get_value().copy()
                for name, shared in self._shared_optimizer_state.items()
            },
            step=self._step,
            loss_history=np.asarray(self._loss_history, dtype=float),
        )

    def load_state(self, state: SVIState) -> None:
        """Adopt a snapshot, so that the next :meth:`fit` continues from it."""
        self._compile_step_fn(modelcontext(self.model))
        for name, shared in self._shared_params.items():
            shared.set_value(np.asarray(state.params[name]))
        for name, shared in self._shared_optimizer_state.items():
            shared.set_value(np.asarray(state.optimizer_state[name]))
        self._step = state.step
        self._loss_history = list(state.loss_history)

    def reset(self) -> None:
        """Restore the parameters, optimizer state, and loss history to their initial values."""
        if self._step_fn is not None:
            self.load_state(self._init_state)

    def _resolve_guide(self, model: Model) -> None:
        if self._guide is not None:
            return
        # Sacrificial detached model context: a guide built naively with a plain
        # Model() inside the user's model context lands here instead of writing
        # into their model
        with Model(model=None):
            if callable(self._guide_factory):
                self._guide = self._guide_factory(model)
            else:
                self._guide = AutoDiagonalNormal(model, random_seed=self.random_seed)

    def _compile_step_fn(self, model: Model) -> None:
        """Compile the step function once. Its shared variables hold all training state."""
        if self._step_fn is not None:
            return
        self._resolve_guide(model)
        self._step_fn, self._shared_params, self._shared_optimizer_state = compile_svi_step_fn(
            model,
            self._guide,
            draws=self._n_particles,
            path_derivative_gradient=self._path_derivative_gradient,
            clip_norm=self._clip_norm,
            **self.compile_kwargs,
        )
        # The shared variables still hold their initial values, so this is exactly
        # what reset() restores
        self._init_state = self.state

    def _compile_sampling_fn(self, model: Model, draws: int) -> None:
        """Compile the posterior sampling function, reusing a previous one when draws match."""
        if self._sampling_fn is not None and self._sampling_draws == draws:
            return
        self._resolve_guide(model)
        self._sampling_fn = compile_sampling_fn(
            model=model,
            guide=self._guide,
            draws=draws,
            **self.compile_kwargs,
        )
        self._sampling_draws = draws

    def _should_stop(self, losses: list) -> bool:
        """Window-based convergence check over the current fit call, see ``convergence_window``."""
        window = self.convergence_window
        if window is None or len(losses) % window != 0 or len(losses) < 2 * window:
            return False
        recent = np.mean(losses[-window:])
        previous = np.mean(losses[-2 * window : -window])
        return bool(abs(recent - previous) < self.relative_tolerance * (abs(previous) + 1e-8))

    def _resolve_learning_rates(self, n: int, learning_rate: ScalarOrSchedule | None) -> list:
        """Materialize the learning rate for every step of a fit call.

        Resolved up front so that no python-level schedule call (nor, for the default
        schedule, an ``np.interp``) happens inside the step loop. The compiled function
        uses ``trust_input=True``, so these must be 0d arrays rather than scalars.
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        if learning_rate is None:
            learning_rate = linear_onecycle_schedule(
                transition_steps=n, peak_value=0.008, pct_start=0.2
            )
        dtype = np.dtype(pytensor_config.floatX)
        if callable(learning_rate):
            return [np.asarray(learning_rate(step), dtype=dtype) for step in range(n)]
        return [np.asarray(learning_rate, dtype=dtype)] * n

    def fit(
        self,
        n: int = 10_000,
        *,
        learning_rate: ScalarOrSchedule | None = None,
        random_seed=None,
    ) -> SVIState:
        """
        Run ``n`` more optimization steps, continuing from the current state.

        The guide parameters and the Adam moments live in shared variables updated in
        place by the compiled step function, so nothing round-trips through Python per
        step and repeated calls pick up where the previous one left off. Call
        :meth:`reset` first to start over.

        Parameters
        ----------
        n : int, optional
            Maximum number of optimization steps to take, by default 10_000. The call
            may stop earlier, controlled by ``convergence_window`` and
            ``relative_tolerance``.
        learning_rate : float or callable, optional
            Learning rate, or a schedule mapping the step number within this call to
            one, overriding the trainer's for this call only.
        random_seed : optional
            Seed for the guide draws used to estimate the gradients.

        Returns
        -------
        SVIState
            A snapshot of the trainer after the call, the same value as
            :attr:`state`.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"n must be a positive integer (the number of fit steps), got {n!r}")

        model = modelcontext(self.model)
        self._compile_step_fn(model)

        if random_seed is not None:
            _reseed_function_rngs(self._step_fn, random_seed)

        learning_rates = self._resolve_learning_rates(n, learning_rate)
        step_fn = self._step_fn
        start_step = self._step
        losses: list = []

        progress = make_advi_progress_bar(theme=default_progress_theme)
        progress_every = max(1, n // 1_000)

        try:
            with progress:
                task = progress.add_task(
                    "Fitting",
                    step=start_step,
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

                for i in range(n):
                    loss = step_fn(learning_rates[i])
                    if start_time is None:
                        start_time = time.perf_counter()
                    losses.append(loss)

                    if self._should_stop(losses):
                        break

                    if i % progress_every == 0:
                        elapsed = time.perf_counter() - start_time
                        speed, unit = compute_step_speed(elapsed, i)
                        progress.update(
                            task,
                            completed=i,
                            step=start_step + i,
                            # Backends may return their own scalar types (e.g. JAX);
                            # convert here rather than once per step
                            loss=float(loss),
                            training_speed=speed,
                            speed_unit=unit,
                        )

                progress.update(
                    task,
                    completed=n,
                    step=start_step + len(losses),
                    loss=float(loss),
                    training_speed=speed,
                    speed_unit=unit,
                    refresh=True,
                )
        except KeyboardInterrupt:
            pass

        self._loss_history.extend(np.asarray(losses, dtype=float).tolist())
        self._step += len(losses)

        return self.state

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
            Snapshot to sample from. Defaults to the trainer's current state.
        random_seed : optional
            Seed for the posterior draws.

        Returns
        -------
        DataTree
            Samples from the guide posterior for each latent variable.
        """
        model = modelcontext(self.model)
        if state is None:
            state = self.state

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
