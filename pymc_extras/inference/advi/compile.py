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


def compile_svi_step_fn(
    model: Model,
    guide: AutoGuideModel,
    optimizer: GradientTransformation,
    draws: int = 1,
    path_derivative_gradient: bool = True,
    **compile_kwargs,
) -> tuple[TrainingFn, dict[str, SharedVariable], dict[str, SharedVariable]]:
    """Compile one full SVI step, with optimizer updates applied in-graph.

    The guide parameters and the optimizer state live in shared variables that the
    compiled function updates in place. It takes no inputs and returns only the
    negative ELBO estimate, so no parameters or gradients round-trip through Python
    during training.

    Together the two returned dicts hold the whole training state: reading their values
    snapshots a run, writing them resumes one exactly.

    Returns
    -------
    step_fn :
        Compiled function ``step_fn() -> negative_elbo``.
    shared_params : dict
        Maps each guide parameter name to the shared variable holding its value.
    shared_optimizer_state : dict
        Maps each optimizer state variable name to the shared variable holding its
        value. Empty for stateless optimizers such as ``sgd``.
    """
    if optimizer.pytensor is None:
        raise ValueError(
            f"The optimizer {optimizer} does not have a PyTensor implementation "
            "and cannot be compiled into the step function."
        )

    logp, logq = get_logp_logq(model, guide, path_derivative_gradient=path_derivative_gradient)
    scalar_negative_elbo = advi_objective(logp, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    params_to_shared = {
        param: pytensor.shared(np.asarray(value), name=param.name)
        for param, value in guide.params_init_values.items()
    }
    [negative_elbo] = graph_replace([negative_elbo], replace=params_to_shared)
    shared_params = list(params_to_shared.values())

    grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=shared_params)

    new_grads, updates = optimizer.pytensor(grads, shared_params)

    # The optimizer's own state variables are the update keys that are not the guide
    # parameters themselves.
    param_ids = {id(param) for param in shared_params}
    shared_optimizer_state = {var.name: var for var in updates if id(var) not in param_ids}

    for param, grad in zip(shared_params, new_grads):
        updates[param] = param + grad

    compile_kwargs.setdefault("trust_input", True)

    step_fn = compile(inputs=[], outputs=negative_elbo, updates=updates, **compile_kwargs)

    shared_params_by_name = {param.name: shared for param, shared in params_to_shared.items()}

    return step_fn, shared_params_by_name, shared_optimizer_state


def compile_sampling_fn(
    model: Model, guide: AutoGuideModel, draws: int, **compile_kwargs
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

    f_sample = compile(inputs=list(params), outputs=sampled_rvs_draws, **compile_kwargs)

    return f_sample
