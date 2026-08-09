import time

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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
from pymc_extras.inference.advi.optimizers import (
    GradientTransformation,
    clipped_adam,
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


StepFn = Callable[[int, SVIState], tuple[np.ndarray, SVIState]]


class Trainer:
    """
    Trainer for stochastic variational inference.

    Follows the design of PyTorch Lightning's ``Trainer`` (and pymc-devs/pymc#8333):
    the model owns the math, the trainer owns the training loop, and :meth:`fit`
    just runs.

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
    .. code-block:: python

        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            pm.Normal("y", mu, 1, observed=[0.5, 1.5])
            trainer = Trainer()
            state = trainer.fit(10_000)
            idata = trainer.sample_posterior(1_000)
    """

    def __init__(
        self,
        *,
        guide: AutoGuideModel | Callable[[Model], AutoGuideModel] | None = None,
        optimizer: GradientTransformation | None = None,
        n_particles: int = 1,
        path_derivative_gradient: bool = True,
        model: Model | None = None,
        backend: str | None = None,
        compile_kwargs: dict | None = None,
        random_seed=None,
    ):
        self.guide = guide
        self.optimizer = optimizer
        self.n_particles = n_particles
        self.path_derivative_gradient = path_derivative_gradient
        self.model = model
        self.compile_kwargs = resolve_backend_compile_kwargs(backend, compile_kwargs)
        self.random_seed = random_seed
        self.state: SVIState | None = None

        self._guide: AutoGuideModel | None = guide if isinstance(guide, AutoGuideModel) else None
        self._fit_model: Model | None = None
        self._stream_shareds: dict[str, SharedVariable] = {}
        self._logp_scalings: dict[str, float] = {}
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

    def _make_compiled_step(
        self, model: Model, state: SVIState | None, random_seed
    ) -> tuple[StepFn, SVIState, Callable[[SVIState], SVIState]]:
        """Set up the compiled step: optimizer updates baked into the PyTensor graph.

        The guide parameters and the optimizer state live in shared variables updated
        in place by the compiled function, so nothing round-trips through Python per
        step.  A resumed ``state`` only restores the parameters (the optimizer
        moments are not part of ``SVIState``).
        """
        if self._step_fn is None:
            self._step_fn, self._step_shared_params = compile_svi_step_fn(
                model,
                self._guide,
                self.optimizer,
                draws=self.n_particles,
                path_derivative_gradient=self.path_derivative_gradient,
                logp_scalings=self._logp_scalings_for(model),
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

        state = SVIState(
            params={name: shared.get_value() for name, shared in shared_params.items()},
            optimizer_state=None,
            step=start_step,
            loss_history=loss_history,
        )

        compiled_step = self._step_fn

        def step_fn(step: int, state: SVIState) -> tuple[np.ndarray, SVIState]:
            loss = np.asarray(compiled_step())
            return loss, SVIState(state.params, None, state.step + 1, loss_history)

        def finalize(final_state: SVIState) -> SVIState:
            return SVIState(
                params={name: shared.get_value().copy() for name, shared in shared_params.items()},
                optimizer_state=None,
                step=final_state.step,
                loss_history=loss_history,
            )

        return step_fn, state, finalize

    def _prepare_data_stream(
        self,
        model: Model,
        first_batch: dict[str, Any],
        observeds: list | None,
        total_size: int | None,
    ) -> Model:
        """Bind the model to a data stream, observing the variables named in ``observeds``.

        A free RV in ``observeds`` is observed *once* with :func:`pymc.observe`,
        through a shared variable initialized from the first batch, so later batches
        stream in with a ``set_value`` instead of a model transform and recompile. A
        ``pm.Data`` variable in ``observeds`` needs no transform and is streamed with
        ``set_data`` like any other batch key. The resulting model is cached: the
        guide, the compiled functions, and :meth:`sample_posterior` all use it.

        When ``total_size`` (the dataset row count ``N``) is known, the
        log-likelihood of each variable in ``observeds`` is rescaled by
        ``N / batch_rows`` so the minibatch estimate is unbiased for the full-data
        one; variables already carrying their own ``total_size`` in the model are
        left alone.
        """
        if self._fit_model is not None:
            return self._fit_model

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
                            self._logp_scalings[rv.name] = scale
                continue
            if name not in first_batch:
                raise ValueError(
                    f"{name!r} is listed in observeds but the data stream's first "
                    f"batch has no entry for it."
                )
            value = np.asarray(first_batch[name], dtype=var.dtype)
            to_observe[name] = self._stream_shareds[name] = pytensor.shared(
                value, name=f"{name}_data"
            )
            if (scale := likelihood_scale(name)) is not None:
                self._logp_scalings[name] = scale
        if to_observe:
            model = pm.observe(model, to_observe)

        # When total_size is known, also rescale any pm.Data variables that appear
        # in the batch but were not explicitly listed in observeds.  Without this,
        # a torch-style DataLoader with __len__ would trigger no rescaling unless
        # the user also passed observeds, which is surprising when the model already
        # declares the observations via pm.Data.
        if total_size is not None:
            for name in first_batch:
                if name in self._logp_scalings:
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
                        self._logp_scalings[rv.name] = scale

        self._fit_model = model
        return model

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
        observeds: list | None = None,
        state: SVIState | None = None,
        random_seed=None,
    ) -> SVIState:
        """
        Fit the model using SVI for ``n`` steps.

        Parameters and gradients round-trip through a Python-side optimizer update
        each step.

        Parameters
        ----------
        n : int, optional
            Maximum number of optimization steps, by default 10_000. Training may
            stop earlier if the ``data`` iterator runs out.
        data : iterable of dict, optional
            A stream of batches, one per step, each a dictionary mapping variable
            names to data. Every step, each entry is reassigned on the model with
            ``set_data`` before the gradient update, so the model trains on one
            batch at a time. Names listed in ``observeds`` may instead refer to
            free RVs, see below. If the iterable supports ``len``, that is taken
            as the total dataset row count ``N`` (as for a torch-style dataloader
            that yields minibatches of a dataset of ``N`` rows).
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
            model = self._prepare_data_stream(model, first_batch, observeds, total_size)
            self._apply_batch(model, first_batch)
        elif observeds:
            raise ValueError("observeds requires a data iterator to stream the observations from")
        elif self._fit_model is not None:
            # A previous fit bound this trainer to a stream-observed model; the guide
            # and compiled functions belong to it, not to the original model
            model = self._fit_model

        self._resolve_guide(model)

        if self.optimizer is None:
            self.optimizer = clipped_adam()
        step_fn, state, finalize = self._make_compiled_step(model, state, random_seed)
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
                    if stream is not None and step > 0:
                        # The first batch was applied before compilation; training
                        # stops early if the stream runs out
                        batch = next(stream, None)
                        if batch is None:
                            break
                        self._apply_batch(model, batch)

                    loss, state = step_fn(step, state)
                    if start_time is None:
                        start_time = time.perf_counter()
                    loss_history.append(loss)

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

        # The guide was built on the fit model, which differs from the user's model
        # when fit streamed observations into it with pm.observe
        model = self._fit_model if self._fit_model is not None else modelcontext(self.model)
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
