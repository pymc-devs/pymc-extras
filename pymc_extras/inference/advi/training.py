import time

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor

from arviz_base import dict_to_dataset
from pymc import Model, modelcontext
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import resolve_backend_compile_kwargs
from pymc.variational.minibatch_rv import MinibatchRandomVariable
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import ancestors
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
from pymc_extras.inference.advi.optimizers import GradientTransformation, clipped_adam
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
    ``Trainer(...).fit(n, state=state)`` continues a run the same way a second
    ``fit(n)`` on the original trainer would.
    """

    params: dict[str, np.ndarray]
    optimizer_state: dict[str, np.ndarray]
    step: int
    loss_history: np.ndarray


class Trainer:
    """
    Trainer for stochastic variational inference.

    The trainer owns the training loop: the guide parameters and the optimizer state
    live in shared variables inside the compiled step function. :meth:`fit` continues
    from the current state, and resumes from a specific snapshot when passed a previous
    :class:`SVIState`; the last state is kept on the trainer, where
    :meth:`sample_posterior` picks it up.

    Configuration splits along the same line. Everything compiled into the step function
    (``guide``, ``optimizer``, ``n_particles``, ``path_derivative_gradient``) is fixed at
    construction and exposed read-only.

    There is no convergence-based early stopping. ``fit(n)`` runs ``n`` steps. To spend
    less, ask for fewer steps. To decide as you go, fit in chunks and look at
    ``state.loss_history`` between them.

    Parameters
    ----------
    guide : AutoGuideModel or callable, optional
        The guide to fit: an :class:`AutoGuideModel`, or a factory mapping the model
        to one. By default an :func:`AutoDiagonalNormal` guide is built from the
        model (mean-field ADVI).
    optimizer : GradientTransformation, optional
        An optax-like optimizer (actual optax optimizers are compatible). By default
        a :func:`clipped_adam` optimizer is used.
    n_particles : int, optional
        Number of guide draws per step used to estimate the ELBO gradient, by
        default 1.
    path_derivative_gradient : bool, optional
        Whether to use the lower-variance path-derivative ("sticking the landing")
        gradient estimator, by default True.
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
    ...     idata = trainer.sample_posterior(1_000)
    """

    def __init__(
        self,
        *,
        guide: AutoGuideModel | Callable[[Model], AutoGuideModel] | None = None,
        optimizer: GradientTransformation | None = None,
        n_particles: int = 1,
        path_derivative_gradient: bool = True,
        backend: str | None = None,
        compile_kwargs: dict | None = None,
        random_seed=None,
    ):
        self._optimizer = optimizer

        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        self.random_seed = random_seed

        # Compiled into the step function on the first fit, hence read-only
        self._guide_factory = guide
        self._n_particles = n_particles
        self._path_derivative_gradient = path_derivative_gradient

        self._guide: AutoGuideModel | None = guide if isinstance(guide, AutoGuideModel) else None
        self._fit_model: Model | None = None
        self._stream_shareds: dict[str, SharedVariable] = {}
        self._logp_scalings: dict[str, float] = {}
        self._step_fn: TrainingFn | None = None
        self._shared_params: dict | None = None
        self._shared_optimizer_state: dict | None = None
        self._sampling_fn: TrainingFn | None = None
        self._sampling_draws: int | None = None
        self._loss_history: list[float] = []
        self._step = 0
        self.state: SVIState | None = None

    @property
    def guide(self) -> AutoGuideModel | None:
        """The guide being fit, once resolved. Compiled in, so it cannot be replaced."""
        return self._guide

    @property
    def optimizer(self) -> GradientTransformation | None:
        """The optimizer being used. Compiled in, so it cannot be changed after a fit."""
        return self._optimizer

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

    def _snapshot(self) -> SVIState:
        """Read the current training state out of the shared variables."""
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

    def _restore(self, state: SVIState) -> None:
        """Write a snapshot back into the shared variables, step counter, and history."""
        for name, shared in self._shared_params.items():
            shared.set_value(np.asarray(state.params[name]))
        for name, shared in self._shared_optimizer_state.items():
            shared.set_value(np.asarray(state.optimizer_state[name]))
        self._step = state.step
        self._loss_history = list(state.loss_history)

    def _build_guide(self, model: Model) -> AutoGuideModel:
        # Sacrificial detached model context: a guide built naively with a plain
        # Model() inside the user's model context lands here instead of writing
        # into their model
        with Model(model=None):
            if callable(self._guide_factory):
                return self._guide_factory(model)
            return AutoDiagonalNormal(model, random_seed=self.random_seed)

    def _compile_step_fn(
        self, model: Model, guide: AutoGuideModel, optimizer: GradientTransformation
    ) -> tuple[TrainingFn, dict[str, SharedVariable], dict[str, SharedVariable]]:
        """Compile the step function, returning it and its shared variables."""
        return compile_svi_step_fn(
            model,
            guide,
            optimizer,
            draws=self._n_particles,
            path_derivative_gradient=self._path_derivative_gradient,
            logp_scalings=self._logp_scalings_for(model),
            **self.compile_kwargs,
        )

    def _compile_sampling_fn(self, model: Model, guide: AutoGuideModel, draws: int) -> TrainingFn:
        """Compile the posterior sampling function."""
        return compile_sampling_fn(
            model=model,
            guide=guide,
            draws=draws,
            **self.compile_kwargs,
        )

    def _prepare_data_stream(
        self,
        model: Model,
        first_batch: dict[str, Any],
        observeds: list | None,
        total_size: int | None,
    ) -> tuple[Model, dict[str, SharedVariable], dict[str, float]]:
        """Bind the model to a data stream, observing the variables named in ``observeds``.

        A free RV in ``observeds`` is observed *once* with :func:`pymc.observe`,
        through a shared variable initialized from the first batch, so later batches
        stream in with a ``set_value`` instead of a model transform and recompile. A
        ``pm.Data`` variable in ``observeds`` needs no transform and is streamed with
        ``set_data`` like any other batch key.

        When ``total_size`` (the dataset row count ``N``) is known, the
        log-likelihood of each variable in ``observeds`` is rescaled by
        ``N / batch_rows`` so the minibatch estimate is unbiased for the full-data
        one; variables already carrying their own ``total_size`` in the model are
        left alone.

        Returns the bound model, the shared variables the stream writes into, and the
        per-name likelihood scalings; the caller caches them on the trainer.
        """
        stream_shareds: dict[str, SharedVariable] = {}
        logp_scalings: dict[str, float] = {}

        def likelihood_scale(name: str) -> float | None:
            if total_size is None or name not in first_batch:
                return None
            value = np.asarray(first_batch[name])
            return total_size / (value.shape[0] if value.ndim else 1)

        to_observe = {}
        for var in observeds or []:
            name = var if isinstance(var, str) else var.name
            var = model[name]
            if isinstance(var, SharedVariable):
                # A pm.Data placeholder: rescale the observed RVs it is the
                # observation of
                if (scale := likelihood_scale(name)) is not None:
                    for rv in model.observed_RVs:
                        if isinstance(rv.owner.op, MinibatchRandomVariable):
                            # The model already rescales this likelihood: a total_size
                            # passed to the observed RV wraps it in a
                            # MinibatchRandomVariable whose logp carries the
                            # N / batch_size factor, so adding ours would scale twice
                            continue
                        if var in ancestors([model.rvs_to_values[rv]]):
                            logp_scalings[rv.name] = scale
                continue
            if name not in first_batch:
                raise ValueError(
                    f"{name!r} is listed in observeds but the data stream's first "
                    f"batch has no entry for it."
                )
            value = np.asarray(first_batch[name], dtype=var.dtype)
            to_observe[name] = stream_shareds[name] = pytensor.shared(value, name=f"{name}_data")
            if (scale := likelihood_scale(name)) is not None:
                logp_scalings[name] = scale
        if to_observe:
            model = pm.observe(model, to_observe)

        # When total_size is known, also rescale any pm.Data variables that appear
        # in the batch but were not explicitly listed in observeds.  Without this,
        # a torch-style DataLoader with __len__ would trigger no rescaling unless
        # the user also passed observeds, which is surprising when the model already
        # declares the observations via pm.Data.
        if total_size is not None:
            for name in first_batch:
                if name in logp_scalings:
                    continue  # already handled via observeds or a free RV above
                var = model[name]
                if not isinstance(var, SharedVariable):
                    continue
                for rv in model.observed_RVs:
                    if isinstance(rv.owner.op, MinibatchRandomVariable):
                        continue
                    if var in ancestors([model.rvs_to_values[rv]]):
                        value = np.asarray(first_batch[name])
                        scale = total_size / (value.shape[0] if value.ndim else 1)
                        logp_scalings[rv.name] = scale

        return model, stream_shareds, logp_scalings

    def _logp_scalings_for(self, model: Model) -> dict | None:
        """Resolve the cached per-name likelihood scalings against the fit model."""
        if not self._logp_scalings:
            return None
        return {model[name]: scale for name, scale in self._logp_scalings.items()}

    def _apply_batch(self, model: Model, batch: dict[str, Any]) -> None:
        """Stream one batch into the model, ahead of the next training step."""
        for name, value in batch.items():
            shared = self._stream_shareds.get(name)
            if shared is not None:
                shared.set_value(np.asarray(value, dtype=shared.type.dtype))
            else:
                model.set_data(name, value)

    def fit(
        self,
        n: int = 10_000,
        data: Iterable[dict[str, Any]] | None = None,
        *,
        state: SVIState | None = None,
        model: Model | None = None,
        observeds: list | None = None,
        random_seed=None,
    ) -> SVIState:
        """
        Run ``n`` optimization steps.

        The guide parameters and the optimizer state live in shared variables updated in
        place by the compiled step function, so nothing round-trips through Python per
        step. Repeated calls continue from the current state; pass a previous
        :class:`SVIState` to resume from a specific snapshot. The final state is stored
        on the trainer.

        Parameters
        ----------
        n : int, optional
            Maximum number of optimization steps, by default 10_000. Training may
            stop earlier if the ``data`` iterator runs out.
        data : iterable of dict, optional
            A stream of batches, one per step, each a dictionary mapping variable
            names to data. Every step, each entry is reassigned on the model with
            ``set_data`` before the gradient update, so the model trains on one
            batch at a time. The batch axis is assumed to be the first (leftmost)
            axis of each array. Names listed in ``observeds`` may instead refer to
            free RVs, see below. If the iterable supports ``len``, that is taken
            as the total dataset row count ``N`` (as for a torch-style dataloader
            that yields minibatches of a dataset of ``N`` rows).
        state : SVIState, optional
            Previous state to resume training from. If None, continues from the
            current state.
        model : Model, optional
            The PyMC model to fit. If None, the model is taken from the context
            stack.
        observeds : list of str or variable, optional
            Variables whose entry in the ``data`` dictionaries is an observation.
            A variable that is a ``pm.Data`` placeholder is streamed into as usual;
            one that is instead a random variable is first converted to an observed
            RV with :func:`pymc.observe` (once, before compilation; the values then
            stream through a shared variable). Requires ``data``. When ``N`` is
            known from ``len(data)``, each observation's log-likelihood is rescaled
            by ``N / batch_rows``, making the minibatch ELBO an unbiased estimate
            of the full-data one; without it batches are treated as the full
            dataset and the posterior will be too wide. Variables that already
            carry a ``total_size`` in the model are left alone.
        random_seed : optional
            Seed for the guide draws used to estimate the gradients.

        Returns
        -------
        SVIState
            The final training state, also stored on the trainer.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"n must be a positive integer (the number of fit steps), got {n!r}")

        model = modelcontext(model)

        stream = None
        if data is not None:
            try:
                total_size = len(data)
            except TypeError:
                total_size = None
            stream = iter(data)
            first_batch = next(stream, None)
            if first_batch is None:
                raise ValueError("the data iterator yielded no batches")
            if self._fit_model is None:
                model, self._stream_shareds, self._logp_scalings = self._prepare_data_stream(
                    model, first_batch, observeds, total_size
                )
                self._fit_model = model
            else:
                model = self._fit_model
            self._apply_batch(model, first_batch)
        elif observeds:
            raise ValueError("observeds requires a data iterator to stream the observations from")
        elif self._fit_model is not None:
            # A previous fit bound this trainer to a stream-observed model; the guide
            # and compiled functions belong to it, not to the original model
            model = self._fit_model

        if self._step_fn is None:
            if self._guide is None:
                self._guide = self._build_guide(model)
            if self._optimizer is None:
                self._optimizer = clipped_adam()
            self._step_fn, self._shared_params, self._shared_optimizer_state = (
                self._compile_step_fn(model, self._guide, self._optimizer)
            )
        if state is not None:
            self._restore(state)

        if random_seed is not None:
            _reseed_function_rngs(self._step_fn, random_seed)

        start_step = self._step
        step_fn = self._step_fn
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
                    if stream is not None and i > 0:
                        # The first batch was applied before compilation; training
                        # stops early if the stream runs out
                        batch = next(stream, None)
                        if batch is None:
                            break
                        self._apply_batch(model, batch)

                    loss = step_fn()
                    if start_time is None:
                        start_time = time.perf_counter()
                    losses.append(loss)

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

        self.state = self._snapshot()
        return self.state

    def sample_posterior(
        self,
        draws: int = 1_000,
        *,
        state: SVIState | None = None,
        random_seed=None,
        model: Model | None = None,
    ) -> DataTree:
        """
        Sample from the guide posterior using the trained parameters.

        Parameters
        ----------
        draws : int, optional
            Number of posterior samples to draw, by default 1_000.
        state : SVIState, optional
            Snapshot to sample from. Defaults to the state of the last
            :meth:`fit` call.
        random_seed : optional
            Seed for the posterior draws.
        model : Model, optional
            Model context to sample within. Defaults to the active model context.

        Returns
        -------
        DataTree
            Samples from the guide posterior for each latent variable.
        """
        model = modelcontext(model)
        if state is None:
            state = self.state
        if state is None:
            raise RuntimeError("The trainer has not been fitted yet.")

        # When a data stream was used, the guide and compiled functions belong to the
        # stream-observed model, whose observed RVs are excluded from the posterior.
        fit_model = self._fit_model if self._fit_model is not None else model
        if self._sampling_fn is None or self._sampling_draws != draws:
            if self._guide is None:
                self._guide = self._build_guide(fit_model)
            self._sampling_fn = self._compile_sampling_fn(fit_model, self._guide, draws)
            self._sampling_draws = draws

        if random_seed is not None:
            _reseed_function_rngs(self._sampling_fn, random_seed)

        params = {name: np.asarray(value) for name, value in state.params.items()}
        samples = self._sampling_fn(**params)
        # Name the draws from the same list the sampling function was built from, so the
        # two cannot drift apart into naming a variable's draws after its sibling.
        posterior = {
            rv.name: np.expand_dims(sample, axis=0)
            for rv, sample in zip(fit_model.free_RVs, samples, strict=True)
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
