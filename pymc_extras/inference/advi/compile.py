from collections import Counter
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


def compile_svi_step_fn(
    model: Model,
    guide: AutoGuideModel,
    optimizer: GradientTransformation,
    shared_params: dict[str, SharedVariable],
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
    random_seed=None,
    **compile_kwargs,
) -> tuple[TrainingFn, dict[str, SharedVariable]]:
    """Compile one full SVI step, with optimizer updates applied in-graph.

    The step takes no inputs and returns the negative ELBO estimate. It reads and writes the
    guide parameters through ``shared_params``, and the optimizer's own state lives in shared
    variables the step creates.

    Parameters
    ----------
    shared_params : dict
        The shared variables holding the guide parameters, keyed by name, from
        :func:`shared_guide_params`. The caller owns them, so several compiled steps can share
        one set.
    random_seed : optional
        Seeds the guide's RNGs before compilation, through :func:`pymc.pytensorf.compile`.

    Returns
    -------
    step_fn :
        Compiled function ``step_fn() -> negative_elbo``.
    shared_optimizer_state : dict
        Maps each optimizer state variable name to the shared variable holding its value.
        Empty for stateless optimizers such as ``sgd``.
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

    new_grads, updates = optimizer.pytensor(grads, shared_param_list)

    # The optimizer's own state variables are the update keys that are not the guide
    # parameters themselves. Snapshotting and restoring them keys on the name, so a
    # duplicate would quietly drop one buffer and resume it from whatever it held.
    guide_params = set(shared_param_list)
    state_variables = [var for var in updates if var not in guide_params]
    name_counts = Counter(var.name for var in state_variables)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"The optimizer has more than one state variable named {duplicate_names}, so its "
            "state cannot be snapshotted or restored unambiguously. Give each transform in the "
            "chain state variables with distinct names."
        )
    shared_optimizer_state = {var.name: var for var in state_variables}

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

    return step_fn, shared_optimizer_state


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
