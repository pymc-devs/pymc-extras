from typing import Protocol

import numpy as np
import pytensor

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor import tensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import graph_replace

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.optimizers import GradientTransformation
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


def shared_optimizer_state(
    optimizer: GradientTransformation,
    guide: AutoGuideModel,
    shared_params: dict[str, SharedVariable],
) -> dict[str, SharedVariable]:
    """The optimizer's state buffers for ``guide``'s parameters, keyed by name."""
    return optimizer.pytensor_init([shared_params[param.name] for param in guide.params])


def compile_svi_step_fn(
    model: Model,
    guide: AutoGuideModel,
    optimizer: GradientTransformation,
    shared_params: dict[str, SharedVariable],
    optimizer_state: dict[str, SharedVariable],
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
    random_seed=None,
    **compile_kwargs,
) -> TrainingFn:
    """Compile one full SVI step, with optimizer updates applied in-graph.

    The step takes no inputs and returns the negative ELBO estimate. It reads and writes the
    guide parameters and the optimizer state in place through the shared variables it is given,
    which the caller owns, so several compiled steps can drive one training state.

    Parameters
    ----------
    shared_params : dict
        The guide parameters, keyed by name, from :func:`shared_guide_params`.
    optimizer_state : dict
        The optimizer's state buffers, keyed by name, from :func:`shared_optimizer_state`.
    random_seed : optional
        Seeds the guide's RNGs before compilation, through :func:`pymc.pytensorf.compile`.
    """
    if optimizer.pytensor is None:
        raise ValueError(
            f"The optimizer {optimizer} does not have a PyTensor implementation "
            "and cannot be compiled into the step function."
        )

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

    grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=shared_param_list)

    new_grads, updates = optimizer.pytensor(grads, shared_param_list, optimizer_state)

    for param, grad in zip(shared_param_list, new_grads):
        updates[param] = param + grad

    compile_kwargs.setdefault("trust_input", True)

    step_fn = compile(
        inputs=[],
        outputs=negative_elbo,
        updates=updates,
        random_seed=random_seed,
        **compile_kwargs,
    )

    return step_fn


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
