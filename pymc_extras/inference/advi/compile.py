from typing import Protocol

import numpy as np

from pymc import Model, compile

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.advi.pytensorf import vectorize_random_graph


class SamplingFn(Protocol):
    def __call__(self, *params: np.ndarray) -> tuple[np.ndarray, ...]: ...


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
