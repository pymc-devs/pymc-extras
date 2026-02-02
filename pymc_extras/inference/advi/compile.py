from typing import Protocol

import numpy as np

from pymc import Model, compile
from pymc.pytensorf import rewrite_pregrad
from pytensor import tensor as pt

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.objective import advi_objective, get_logp_logq
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class TrainingFn(Protocol):
    def __call__(self, draws: int, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


def compile_svi_training_fn(
    model: Model,
    guide: AutoGuideModel,
    stick_the_landing: bool = True,
    minibatch: bool = False,
    **compile_kwargs,
) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params
    inputs = [draws, *params]

    logp_scale = 1

    if minibatch:
        data = model.data_vars
        inputs = [*inputs, *data]

    logp, logq = get_logp_logq(model, guide, stick_the_landing=stick_the_landing)

    scalar_negative_elbo = advi_objective(logp / logp_scale, logq)
    [negative_elbo_draws] = vectorize_random_graph([scalar_negative_elbo], batch_draws=draws)
    negative_elbo = negative_elbo_draws.mean(axis=0)

    negative_elbo_grads = pt.grad(rewrite_pregrad(negative_elbo), wrt=params)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_loss_dloss = compile(
        inputs=inputs, outputs=[negative_elbo, *negative_elbo_grads], **compile_kwargs
    )

    return f_loss_dloss


def compile_sampling_fn(model: Model, guide: AutoGuideModel, **compile_kwargs) -> TrainingFn:
    draws = pt.scalar("draws", dtype=int)
    params = guide.params

    parameterized_value_vars = [
        guide.model[rv.name] for rv in model.rvs_to_values.keys() if rv not in model.observed_RVs
    ]
    transformed_vars = [
        transform.backward(parameterized_var)
        if (transform := model.rvs_to_transforms[rv]) is not None
        else parameterized_var
        for rv, parameterized_var in zip(model.rvs_to_values.keys(), parameterized_value_vars)
    ]

    sampled_rvs_draws = vectorize_random_graph(transformed_vars, batch_draws=draws)

    if "trust_input" not in compile_kwargs:
        compile_kwargs["trust_input"] = True

    f_sample = compile(inputs=[draws, *params], outputs=sampled_rvs_draws, **compile_kwargs)

    return f_sample
