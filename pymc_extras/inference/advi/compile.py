from collections import Counter
from typing import Protocol

import numpy as np
import pytensor

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor import tensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import graph_replace
from pytensor_ml.optim import Transform
from pytensor_ml.pytensorf import collect_clock_updates

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class TrainingFn(Protocol):
    def __call__(self) -> np.ndarray: ...


class SamplingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_step_fn(
    model: Model,
    guide: AutoGuideModel,
    optimizer: Transform,
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
    **compile_kwargs,
) -> tuple[TrainingFn, dict[str, SharedVariable], dict[str, SharedVariable]]:
    """Compile one full SVI step, with the optimizer updates applied in-graph.

    The guide parameters and the optimizer state live in shared variables that the
    compiled function updates in place. It takes no inputs and returns only the negative
    ELBO estimate, so no parameters, gradients or learning rates round-trip through
    Python during training.

    Together the returned dicts hold the whole training state: reading their values
    snapshots a run, writing them resumes one exactly.

    Parameters
    ----------
    model : Model
        The PyMC model to fit.
    guide : AutoGuideModel
        The variational guide whose parameters are optimized.
    optimizer : Transform
        A configured :mod:`pytensor_ml.optim` rule, ``(loss_or_gradients, parameters) ->
        updates``, carrying its own rate, schedule and clipping.
    draws : int, optional
        Number of guide draws per step used to estimate the ELBO gradient. Default 1.
    path_derivative_gradient : bool, optional
        Whether to use the path-derivative ("sticking the landing") gradient estimator.
        Default True.
    logp_scalings : dict, optional
        Per-variable likelihood scalings, mapping model variables to the factor their
        logp is multiplied by (for unbiased minibatch estimates). Default None.

    Returns
    -------
    step_fn :
        Compiled function ``step_fn() -> negative_elbo``.
    shared_params : dict
        Maps each guide parameter name to the shared variable holding its value.
    shared_optimizer_state : dict
        Maps the name of every other shared variable the optimizer writes -- its moment
        estimates and the training clocks -- to that variable.
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

    params_to_shared = {
        param: pytensor.shared(np.asarray(value), name=param.name)
        for param, value in guide.params_init_values.items()
    }
    [negative_elbo] = graph_replace([negative_elbo], replace=params_to_shared)
    shared_params = list(params_to_shared.values())

    grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=shared_params)

    updates = optimizer(grads, shared_params)
    updates |= collect_clock_updates(list(updates.values()), already_written=updates)

    # The optimizer's own state variables are the update keys that are not the guide
    # parameters themselves. Snapshotting and restoring them keys on the name, so a
    # duplicate would quietly drop one buffer and resume it from whatever it held.
    guide_params = set(shared_params)
    state_variables = [var for var in updates if var not in guide_params]
    name_counts = Counter(var.name for var in state_variables)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"The optimizer has more than one state variable named {duplicate_names}, so its "
            "state cannot be snapshotted or restored unambiguously. Give each transform in the "
            "chain state variables with distinct names (e.g. via their namespace argument)."
        )
    shared_optimizer_state = {var.name: var for var in state_variables}

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
