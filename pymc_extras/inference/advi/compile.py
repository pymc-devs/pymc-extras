from typing import Protocol

import numpy as np
import pytensor

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable
from pytensor_ml.optim import Gradients, Steps, Transform, Updates
from pytensor_ml.optim.base import require_unique_state_names
from pytensor_ml.pytensorf import collect_clock_updates

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class TrainingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


class SamplingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def shared_guide_params(guide: AutoGuideModel) -> dict[str, SharedVariable]:
    """A shared variable for each guide parameter at its initial value, keyed by parameter name."""
    return {
        param.name: pytensor.shared(np.asarray(value), name=param.name)
        for param, value in guide.params_init_values.items()
    }


def build_svi_step(
    model: Model,
    guide: AutoGuideModel,
    optimizer: Transform,
    shared_params: dict[str, SharedVariable],
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
) -> tuple[TensorVariable, Updates]:
    """Build one SVI step, calling ``optimizer`` once so every compile of it shares that state.

    Parameters
    ----------
    shared_params : dict
        The guide parameters, keyed by name, from :func:`shared_guide_params`.

    Returns
    -------
    negative_elbo : TensorVariable
        The step's negative ELBO estimate.
    updates : Updates
        The optimizer's updates, plus the advance of every training clock a schedule reads, as
        :func:`pytensor_ml.optim.compile_train` adds them.
    """
    logp, logq = get_logp_logq(
        model,
        guide,
        path_derivative_gradient=path_derivative_gradient,
        logp_scalings=logp_scalings,
    )
    scalar_negative_elbo = advi_objective(logp, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    params_to_shared = {param: shared_params[param.name] for param in guide.params}
    [negative_elbo] = graph_replace([negative_elbo], replace=params_to_shared)
    shared_param_list = list(params_to_shared.values())

    result = optimizer(rewrite_pregrad(negative_elbo), shared_param_list)
    if isinstance(result, Gradients):
        raise ValueError(
            "The optimizer returned gradients rather than the steps to take, so the guide "
            "parameters would move uphill. Put an update rule such as `adam(1e-3)` in the chain."
        )
    updates = Steps(result)
    if unwritten := [param.name for param in shared_param_list if param not in updates]:
        raise ValueError(f"The optimizer writes no update for the guide parameters {unwritten}.")

    for clock, next_count in collect_clock_updates(
        [negative_elbo, *updates.values()], already_written=updates
    ).items():
        updates[clock] = next_count
    require_unique_state_names(updates)

    return negative_elbo, updates


def compile_svi_step_fn(
    negative_elbo: TensorVariable, updates: Updates, random_seed=None, **compile_kwargs
) -> TrainingFn:
    """Compile a step from :func:`build_svi_step`, applying its updates in place.

    The step takes no inputs and returns the negative ELBO estimate.

    Parameters
    ----------
    random_seed : optional
        Seeds the guide's RNGs before compilation, through :func:`pymc.pytensorf.compile`.
    """
    compile_kwargs.setdefault("trust_input", True)

    return compile(
        inputs=[],
        outputs=negative_elbo,
        updates=updates,
        random_seed=random_seed,
        **compile_kwargs,
    )


def compile_sampling_fn(
    model: Model, guide: AutoGuideModel, draws: int, random_seed=None, **compile_kwargs
) -> SamplingFn:
    params = guide.params

    free_rvs = model.free_RVs
    parameterized_value_vars = [guide.model[rv.name] for rv in free_rvs]
    transformed_vars = [
        transform.backward(parameterized_var, *rv.owner.inputs)
        if (transform := model.rvs_to_transforms[rv]) is not None
        else parameterized_var
        for rv, parameterized_var in zip(free_rvs, parameterized_value_vars)
    ]

    sampled_rvs_draws = vectorize_random_graph(transformed_vars, batch_draws=draws)

    compile_kwargs.setdefault("trust_input", True)

    f_sample = compile(
        inputs=list(params), outputs=sampled_rvs_draws, random_seed=random_seed, **compile_kwargs
    )

    return f_sample
