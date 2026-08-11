import logging
import warnings

import numpy as np
import pymc as pm

from arviz_base import dict_to_dataset
from pymc.util import RandomSeed, _get_seeds_per_chain
from xarray import DataTree

from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data
from pymc_extras.inference.mlx_mclmc.kernel import TunedParameters, warmup_and_sample
from pymc_extras.inference.mlx_mclmc.logp import (
    MLXLogp,
    check_model_is_sampleable,
    draws_to_datasets,
)

_log = logging.getLogger(__name__)

# MLX raises this when a fused graph exceeds Metal's argument-buffer limit. Hand-written special
# functions such as gammaln expand into large graphs, so a model can hit it through no fault of
# its own; running the step unfused is the documented way out.
_METAL_FUSION_LIMIT = "Too many inputs/outputs fused"

# The step-size controller clamps at 1e-12, so anything at that floor means it gave up.
_COLLAPSED_STEP_SIZE = 1e-11

# Unadjusted MCLMC should not diverge at all on a well-behaved target, so the bar is low.
_MAX_DIVERGING_FRACTION = 0.01


def fit_mlx_mclmc(
    draws: int = 1000,
    *,
    tune: int = 1000,
    burn_in: int = 500,
    chains: int = 4,
    model: pm.Model | None = None,
    integrator: str = "mclachlan",
    diagonal_preconditioning: bool = True,
    desired_energy_var: float = 5e-4,
    initial_point: dict[str, np.ndarray] | np.ndarray | None = None,
    include_transformed: bool = False,
    compile_step: bool = True,
    random_seed: RandomSeed = None,
    warmup_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> DataTree:
    """
    Sample a model with unadjusted MCLMC on the Apple Silicon GPU.

    Microcanonical Langevin Monte Carlo evolves an isokinetic Hamiltonian: the momentum is held
    on the unit sphere and partially refreshed each step, so trajectories decorrelate without a
    Metropolis accept step. Dropping that accept step is what makes the sampler cheap, and also
    what makes it approximate. Draws carry a bias of order :math:`\\epsilon^4` in the step size,
    which ``desired_energy_var`` controls, so treat the result as an approximation to the
    posterior rather than an exact sample from it.

    The log-density is compiled to MLX and run on Metal, so the model graph must be float32. Set
    ``pytensor.config.floatX = "float32"`` before building the model.

    Parameters
    ----------
    draws : int
        Number of draws to keep per chain. Default is 1000.
    tune : int
        Number of integrator steps given to the adaptation, which runs on a single chain and
        settles the step size, the diagonal metric, and ``L``. Default is 1000.
    burn_in : int
        Number of sampling steps to run and discard before the kept draws. The chains start
        jittered around the tuned position by ``sqrt(inverse_mass_matrix)``, which is a full unit
        when ``diagonal_preconditioning`` is False, so without a burn-in the leading draws are
        badly over-dispersed for a concentrated posterior. Default is 500.
    chains : int
        Number of chains, run simultaneously as the leading array axis. Default is 4.
    model : pm.Model, optional
        Defaults to the model on the context stack.
    integrator : str
        Either ``"mclachlan"``, which takes 2 gradient evaluations per step, or
        ``"velocity_verlet"``, which takes 1. Default is ``"mclachlan"``.
    diagonal_preconditioning : bool
        Whether to adapt a diagonal inverse mass matrix. Default is True.
    desired_energy_var : float
        Target energy variance per dimension. Smaller values give a smaller step size, so less
        bias and more compute per effective draw. Default is 5e-4.
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
    warmup_kwargs : dict, optional
        Extra keyword arguments for :func:`~pymc_extras.inference.mlx_mclmc.kernel.warmup`.
    compile_kwargs : dict, optional
        Extra keyword arguments for the PyTensor function that maps draws back to model space.

    Returns
    -------
    idata : DataTree
        Posterior draws, per-step energy errors and divergence flags under ``sample_stats``, and
        the adapted sampler parameters in the posterior group's attributes.

    References
    ----------
    .. [1] Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023). Microcanonical
       Hamiltonian Monte Carlo. Journal of Machine Learning Research, 24(311), 1-34.
    .. [2] Robnik, J., & Seljak, U. (2024). Fluctuation without dissipation: Microcanonical
       Langevin Monte Carlo. arXiv:2303.18221.
    """
    model = pm.modelcontext(model)
    check_model_is_sampleable(model)

    seed = int(_get_seeds_per_chain(random_seed, 1)[0])
    logdensity_fn = MLXLogp(model)

    # Everything downstream reads the frozen model, so the draws are unpacked against exactly the
    # value variables the density was compiled from.
    model = logdensity_fn.model

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

    # Merged rather than passed alongside, so an explicit warmup_kwargs entry overrides the
    # named argument instead of raising a duplicate-keyword TypeError.
    warmup_kwargs = {
        "diagonal_preconditioning": diagonal_preconditioning,
        "desired_energy_var": desired_energy_var,
        **(warmup_kwargs or {}),
    }

    _log.info("Sampling %d chains of %d draws in %d dimensions", chains, draws, logdensity_fn.dim)
    sampler_kwargs = dict(
        num_tune=tune,
        draws=draws,
        discard=burn_in,
        chains=chains,
        integrator=integrator,
        seed=seed,
        **warmup_kwargs,
    )
    try:
        output, tuned = warmup_and_sample(
            logdensity_fn, start, compile_step=compile_step, **sampler_kwargs
        )
    except RuntimeError as exc:
        if not (compile_step and _METAL_FUSION_LIMIT in str(exc)):
            raise
        warnings.warn(
            "The fused sampler step exceeded Metal's argument-buffer limit, so MCLMC is falling "
            "back to an unfused step, which is slower. Pass compile_step=False to skip this "
            "attempt.",
            RuntimeWarning,
            stacklevel=2,
        )
        output, tuned = warmup_and_sample(
            logdensity_fn, start, compile_step=False, **sampler_kwargs
        )

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

    if not np.isfinite(tuned.L) or tuned.step_size <= _COLLAPSED_STEP_SIZE:
        warnings.warn(
            f"MCLMC adaptation collapsed to step_size={tuned.step_size:.3g}, L={tuned.L:.3g}. "
            "The draws are not usable; check that the posterior is proper.",
            RuntimeWarning,
            stacklevel=3,
        )
