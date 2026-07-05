from typing import Protocol

import numpy as np
import pytensor

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor import config
from pytensor import tensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.replace import graph_replace

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class SamplingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


class TrainingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_training_fn(
    model: Model,
    guide: AutoGuideModel,
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
    **compile_kwargs,
) -> TrainingFn:
    # draws is a compile-time constant: backends like JAX cannot handle inputs that
    # determine the shapes of random variables
    params = guide.params
    inputs = list(params)

    logp, logq = get_logp_logq(
        model,
        guide,
        path_derivative_gradient=path_derivative_gradient,
        logp_scalings=logp_scalings,
    )

    scalar_negative_elbo = advi_objective(logp, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    negative_elbo_grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=params)

    compile_kwargs.setdefault("trust_input", True)

    f_loss_dloss = compile(
        inputs=inputs, outputs=[negative_elbo, *negative_elbo_grads], **compile_kwargs
    )

    return f_loss_dloss


def compile_svi_step_fn(
    model: Model,
    guide: AutoGuideModel,
    draws: int = 1,
    path_derivative_gradient: bool = True,
    logp_scalings: dict | None = None,
    clip_norm: float | None = 10.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    **compile_kwargs,
) -> tuple[TrainingFn, dict[str, SharedVariable]]:
    """Compile one full SVI step, with clipped-Adam updates applied in-graph.

    The guide parameters and the optimizer state live in shared variables that the
    compiled function updates in place. Its only input is the learning rate and its
    only output the negative ELBO estimate, so no parameters or gradients round-trip
    through Python during training.

    Returns
    -------
    step_fn :
        Compiled function ``step_fn(learning_rate) -> negative_elbo``.
    shared_params : dict
        Maps each guide parameter name to the shared variable holding its value.
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

    if clip_norm is not None:
        global_norm = pt.sqrt(pt.sum([pt.sum(pt.square(g)) for g in grads]))
        scale = pt.minimum(1.0, clip_norm / (global_norm + 1e-12))
        grads = [g * scale for g in grads]

    learning_rate = pt.scalar("learning_rate", dtype=config.floatX)
    t = pytensor.shared(np.zeros((), dtype="int64"), name="adam_t")
    t_new = t + 1
    # The bias-correction powers `beta**t_new` mix a (weak) python float base with an int64
    # exponent, which pytensor resolves to float64. Under floatX=float32 that upcasts param_new
    # to float64 and the shared-variable update fails the dtype check. Compute the powers in
    # floatX so the whole update stays in the parameter dtype.
    t_new_float = t_new.astype(config.floatX)
    updates = {t: t_new}
    for shared_param, grad in zip(shared_params, grads):
        value = shared_param.get_value(borrow=True)
        m = pytensor.shared(np.zeros_like(value), name=f"adam_m_{shared_param.name}")
        v = pytensor.shared(np.zeros_like(value), name=f"adam_v_{shared_param.name}")
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * pt.square(grad)
        m_hat = m_new / (1 - beta1**t_new_float)
        v_hat = v_new / (1 - beta2**t_new_float)
        param_new = shared_param - learning_rate * m_hat / (pt.sqrt(v_hat) + epsilon)
        updates.update({m: m_new, v: v_new, shared_param: param_new})

    compile_kwargs.setdefault("trust_input", True)

    step_fn = compile(
        inputs=[learning_rate], outputs=negative_elbo, updates=updates, **compile_kwargs
    )

    return step_fn, {param.name: shared for param, shared in params_to_shared.items()}


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
