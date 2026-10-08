from __future__ import annotations

import dataclasses
import logging
import warnings

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm

from arviz_base import dict_to_dataset
from pymc.blocking import DictToArrayBijection
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.progress_bar import ProgressBarOptions
from pymc.util import RandomSeed, _get_seeds_per_chain
from xarray import DataTree

from pymc_extras.inference.advi import (
    AutoDiagonalNormal,
    AutoGuideModel,
    AutoLowRankMultivariateNormal,
    Trainer,
)
from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data
from pymc_extras.inference.mlx_mclmc.progress import MCLMCProgressBarManager
from pymc_extras.inference.mlx_mclmc.settings import AdaptationSettings

if TYPE_CHECKING:
    from pymc_extras.inference.mlx_mclmc.kernel import Metric, TunedParameters

_log = logging.getLogger(__name__)

# MLX raises this when a fused graph exceeds Metal's argument-buffer limit. Hand-written special
# functions such as gammaln expand into large graphs, so a model can hit it through no fault of
# its own; running the step unfused is the documented way out.
_METAL_FUSION_LIMIT = "Too many inputs/outputs fused"

# The step-size controller clamps at 1e-12, so anything at that floor means it gave up.
_COLLAPSED_STEP_SIZE = 1e-11

# Unadjusted MCLMC should not diverge at all on a well-behaved target, so the bar is low.
_MAX_DIVERGING_FRACTION = 0.01

_ADVI_GUIDES = ("mean_field", "low_rank")

# Maps fitted guide parameters to the flat mean, scale, and low-rank factor (None for mean-field).
ApproximationReader = Callable[
    [dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray | None]
]


def fit_mlx_mclmc(
    draws: int = 1000,
    *,
    tune: int = 1000,
    burn_in: int = 500,
    chains: int = 4,
    model: pm.Model | None = None,
    integrator: str = "mclachlan",
    adaptation: AdaptationSettings = AdaptationSettings(),
    initial_point: dict[str, np.ndarray] | np.ndarray | None = None,
    include_transformed: bool = False,
    compile_step: bool = True,
    random_seed: RandomSeed = None,
    compile_kwargs: dict | None = None,
    progressbar: bool | ProgressBarOptions = True,
) -> DataTree:
    """
    Sample a model with unadjusted MCLMC on the Apple Silicon GPU.

    Microcanonical Langevin Monte Carlo evolves an isokinetic Hamiltonian: the momentum is held
    on the unit sphere and partially refreshed each step, and that refreshment is what
    decorrelates the chain. The integrator's discretization error is never corrected by a
    Metropolis accept step. Leaving that step out is what makes the sampler cheap, and also what
    makes it approximate. Draws carry a bias of order :math:`\\epsilon^4` in the step size,
    which ``desired_energy_var`` controls, so treat the result as an approximation to the
    posterior rather than an exact sample from it.

    The log-density is compiled to MLX and run on Metal, so the model graph must be float32. Set
    ``pytensor.config.floatX = "float32"`` before building the model.

    Parameters
    ----------
    draws : int
        Number of draws to keep per chain. Default is 1000.
    tune : int
        Number of integrator steps given to the adaptation, which runs on every chain at once and
        settles the step size, the metric, and ``L``. Default is 1000.
    burn_in : int
        Number of sampling steps to run and discard before the kept draws. The chains start
        where their adaptation ended, so this only needs to cover what the final parameter change
        of warmup unsettles. Default is 500.
    chains : int
        Number of chains, run simultaneously as the leading array axis through both warmup and
        sampling. Default is 4.
    model : pm.Model, optional
        Defaults to the model on the context stack.
    integrator : str
        Either ``"mclachlan"``, which takes 2 gradient evaluations per step, or
        ``"velocity_verlet"``, which takes 1. Default is ``"mclachlan"``.
    adaptation : AdaptationSettings
        How warmup adapts the step size, the metric, and ``L``. Defaults to
        :class:`~pymc_extras.inference.mlx_mclmc.settings.AdaptationSettings`.
    initial_point : dict or array, optional
        Starting point for the adaptation, given either as unconstrained value-variable arrays
        keyed by name or as a flat vector in ``model.value_vars`` order. Defaults to the model's
        own initial point.
    include_transformed : bool
        Whether to add an ``unconstrained_posterior`` group holding the draws in the space the
        sampler moved in. Default is False.
    compile_step : bool
        Whether to fuse the sampler step with ``mx.compile``. A fused step that exceeds Metal's
        argument-buffer limit is retried unfused, so this is a way to skip that first attempt
        rather than a requirement. Default is True.
    random_seed : int, optional
    compile_kwargs : dict, optional
        Extra keyword arguments for the PyTensor function that maps draws back to model space.
    progressbar : bool or str
        True draws one bar covering every chain, sectioned into the three warmup phases, the
        burn-in, and the kept draws, with the draw count restarting at each section. It shows the
        median step size across chains and the number of NaN steps, the chain-steps that came
        out NaN or infinite and were reverted.
        ``"split"`` or ``"split+stats"`` draws one bar per chain, up to 16 chains. False hides it.
        The bar advances every 64 steps, when the lazy MLX graph is forced. Default is True.

    Returns
    -------
    idata : DataTree
        Posterior draws, per-step energy errors and divergence flags under ``sample_stats``, and
        the adapted sampler parameters in the posterior group's attributes, ``L`` and
        ``step_size`` as one value per chain.

    References
    ----------
    .. [1] Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023). Microcanonical
       Hamiltonian Monte Carlo. Journal of Machine Learning Research, 24(311), 1-34.
    .. [2] Robnik, J., & Seljak, U. (2024). Fluctuation without dissipation: Microcanonical
       Langevin Monte Carlo. arXiv:2303.18221.
    """
    # Both modules import mlx at load time, and mlx only installs on Apple Silicon. Importing
    # them here keeps this module, and its docstring, importable everywhere else.
    from pymc_extras.inference.mlx_mclmc.kernel import warmup_and_sample, warmup_schedule
    from pymc_extras.inference.mlx_mclmc.logp import (
        MLXLogp,
        check_model_is_sampleable,
        draws_to_datasets,
    )

    model = pm.modelcontext(model)
    check_model_is_sampleable(model)

    seed = int(_get_seeds_per_chain(random_seed, 1)[0])
    logdensity_fn = MLXLogp(model)

    if initial_point is None:
        start = logdensity_fn.flat_initial_point()
    elif isinstance(initial_point, dict):
        start = np.concatenate(
            [
                np.asarray(initial_point[name], dtype="float32").ravel()
                for name in logdensity_fn.names
            ]
        )
    else:
        start = np.asarray(initial_point, dtype="float32").ravel()

    initial_metric = None
    if adaptation.advi_steps:
        approximation = _fit_approximation(
            model, logdensity_fn, start, settings=adaptation, seed=seed
        )
        if approximation is not None:
            start, initial_metric = approximation

    _log.info("Sampling %d chains of %d draws in %d dimensions", chains, draws, logdensity_fn.dim)
    sampler_kwargs = dict(
        num_tune=tune,
        draws=draws,
        discard=burn_in,
        chains=chains,
        settings=adaptation,
        integrator=integrator,
        seed=seed,
        initial_metric=initial_metric,
    )
    schedule = warmup_schedule(tune, adaptation)
    sections = [schedule.step_size, schedule.metric + schedule.readjust, schedule.L, burn_in, draws]

    # The fused attempt fails when its step is first forced, before any progress is reported, so
    # one bar carries over to the unfused retry.
    with MCLMCProgressBarManager(
        chains=chains, sections=sections, progressbar=progressbar
    ) as progress:

        def run(compile_step):
            return warmup_and_sample(
                logdensity_fn,
                start,
                compile_step=compile_step,
                progress=progress.update,
                **sampler_kwargs,
            )

        try:
            output, tuned = run(compile_step)
        except RuntimeError as exc:
            if not (compile_step and _METAL_FUSION_LIMIT in str(exc)):
                raise
            warnings.warn(
                "The fused sampler step exceeded Metal's argument-buffer limit, so MCLMC is "
                "falling back to an unfused step, which is slower. Pass compile_step=False to skip "
                "this attempt.",
                RuntimeWarning,
                stacklevel=2,
            )
            output, tuned = run(compile_step=False)

    # The kernel stacks draws first; InferenceData wants chains first.
    flat_draws = np.asarray(output.samples, dtype="float32").transpose(1, 0, 2)
    # The kernel reports diagnostics for the burn-in steps too; drop those so sample_stats lines
    # up with the posterior's draw axis.
    energy_errors = np.asarray(output.energy_errors, dtype="float32").T[:, burn_in:]
    diverging = np.asarray(output.diverging).T[:, burn_in:]
    _warn_if_adaptation_failed(tuned, diverging)

    posterior, unconstrained_posterior = draws_to_datasets(
        flat_draws,
        model,
        include_transformed=include_transformed,
        compile_kwargs=compile_kwargs,
    )
    posterior.attrs |= {
        "L": tuned.L,
        "step_size": tuned.step_size,
        "integrator": integrator,
        "num_tuning_steps": tuned.num_tuning_steps,
    }

    idata = DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": dict_to_dataset(
                {"energy_error": energy_errors, "diverging": diverging},
                coords={},
                dims={},
                inference_library=pm,
            ),
        }
    )
    if unconstrained_posterior is not None:
        idata["unconstrained_posterior"] = DataTree(dataset=unconstrained_posterior)

    return add_data_to_inference_data(
        idata, progressbar=False, model=model, compile_kwargs=compile_kwargs
    )


def _fit_approximation(
    model: pm.Model, logdensity_fn, start: np.ndarray, settings: AdaptationSettings, seed: int
) -> tuple[np.ndarray, Metric] | None:
    """
    Fit ADVI on MLX from ``start``.

    Returns
    -------
    mean : np.ndarray
        The approximation's mean, flat in the sampler's coordinate order.
    metric : Metric
        Its covariance, as an inverse mass matrix.

    Returns None instead, with a warning, when the fit comes back non-finite.
    """
    import mlx.core as mx

    from pymc_extras.inference.mlx_mclmc.kernel import Metric, metric_from_low_rank_covariance

    if settings.advi_guide not in _ADVI_GUIDES:
        raise ValueError(f"advi_guide must be one of {_ADVI_GUIDES}, got {settings.advi_guide!r}")

    # The guide draws its noise with sizes taken from the model's dims, and a compiled MLX
    # function can only take a shape from a constant.
    frozen = freeze_dims_and_data(model)
    if settings.advi_guide == "mean_field":
        guide, read_approximation = _mean_field_guide(frozen, logdensity_fn, start)
    else:
        guide, read_approximation = _low_rank_guide(
            frozen, logdensity_fn, start, rank=settings.advi_rank
        )

    trainer = Trainer(
        guide=guide,
        optimizer=settings.advi_optimizer,
        n_particles=settings.advi_particles,
        backend="mlx",
        random_seed=seed,
    )
    # The trainer seeds its functions before compiling, so the MLX linker's copy of each shared
    # generator is the one that is meant to be used, and its warning about the copy is noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The RandomType SharedVariables")
        params = trainer.fit(settings.advi_steps, model=frozen, random_seed=seed + 1).params

    mean, scale, factor = read_approximation(params)
    parts = [mean, scale] if factor is None else [mean, scale, factor]
    if not all(np.isfinite(part).all() for part in parts):
        warnings.warn(
            "ADVI returned a non-finite approximation, so MCLMC is starting from the initial "
            "point instead. A non-finite gradient during the ADVI fit is the usual cause.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    if factor is None:
        metric = Metric(scale=mx.array(scale.astype("float32")))
    else:
        metric = metric_from_low_rank_covariance(scale, factor)

    return mean.astype("float32"), metric


def _flat_slices(logdensity_fn) -> dict[str, slice]:
    """Each value variable's slice of the sampler's flat vector."""
    offsets = np.cumsum([0, *logdensity_fn.sizes])
    return {
        name: slice(begin, end)
        for name, begin, end in zip(logdensity_fn.names, offsets[:-1], offsets[1:], strict=True)
    }


def _mean_field_guide(
    frozen: pm.Model, logdensity_fn, start: np.ndarray
) -> tuple[AutoGuideModel, ApproximationReader]:
    """The mean-field guide centered on ``start``, and a reader for its fitted parameters."""
    guide = AutoDiagonalNormal(frozen)
    rv_names = {frozen.rvs_to_values[rv].name: rv.name for rv in frozen.free_RVs}
    slices = _flat_slices(logdensity_fn)

    init_values = dict(guide.params_init_values)
    for name, shape in zip(logdensity_fn.names, logdensity_fn.shapes, strict=True):
        loc = guide[f"{rv_names[name]}_loc"]
        init_values[loc] = start[slices[name]].reshape(shape).astype(loc.dtype)

    def read_approximation(params):
        def flat(suffix):
            return np.concatenate(
                [np.ravel(params[f"{rv_names[name]}_{suffix}"]) for name in logdensity_fn.names]
            )

        return flat("loc"), np.exp(flat("scale")), None

    return dataclasses.replace(guide, params_init_values=init_values), read_approximation


def _low_rank_guide(
    frozen: pm.Model, logdensity_fn, start: np.ndarray, rank: int | None
) -> tuple[AutoGuideModel, ApproximationReader]:
    """The low-rank guide centered on ``start``, and a reader for its fitted parameters."""
    guide = AutoLowRankMultivariateNormal(frozen, rank=rank)
    slices = _flat_slices(logdensity_fn)

    # The guide packs its flat mean in initial-point order, which need not be the sampler's.
    _, point_map_info = DictToArrayBijection.map(frozen.initial_point())
    positions = np.arange(start.size)
    guide_order = np.concatenate([positions[slices[name]] for name, *_ in point_map_info])
    sampler_order = np.argsort(guide_order)

    init_values = dict(guide.params_init_values)
    loc = guide["loc"]
    init_values[loc] = start[guide_order].astype(loc.dtype)

    def read_approximation(params):
        return (
            np.asarray(params["loc"])[sampler_order],
            np.exp(np.asarray(params["cov_diag_unconstrained"]))[sampler_order],
            np.asarray(params["cov_factor"])[sampler_order],
        )

    # replace keeps the guide's class, which carries the low-rank guide's closed-form logq.
    return dataclasses.replace(guide, params_init_values=init_values), read_approximation


def _warn_if_adaptation_failed(tuned: TunedParameters, diverging: np.ndarray) -> None:
    """
    Warn when the divergence rate or the adapted parameters say the draws are not usable.

    The sampler reverts a step whose log-density comes back non-finite, so a diverging chain
    keeps producing draws rather than nans. A collapsed step size means the chains never moved
    at all. Both leave something that looks like an ordinary posterior.
    """
    diverging_fraction = float(diverging.mean())
    if diverging_fraction > _MAX_DIVERGING_FRACTION:
        warnings.warn(
            f"MCLMC reverted {diverging_fraction:.1%} of steps as divergent. The draws are "
            "unreliable; the log-density is likely returning nan in the region the chains "
            "reached.",
            RuntimeWarning,
            stacklevel=3,
        )

    step_size, L = np.asarray(tuned.step_size), np.asarray(tuned.L)
    if not np.isfinite(L).all() or (step_size <= _COLLAPSED_STEP_SIZE).any():
        warnings.warn(
            f"MCLMC adaptation collapsed to step_size={np.array2string(step_size, precision=3)}, "
            f"L={np.array2string(L, precision=3)}. The draws are not usable; check that the "
            "posterior is proper.",
            RuntimeWarning,
            stacklevel=3,
        )
