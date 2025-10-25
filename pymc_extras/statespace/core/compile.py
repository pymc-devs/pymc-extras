import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc_extras.statespace.core import PyMCStateSpace
from pymc_extras.statespace.filters.distributions import LinearGaussianStateSpace
from pymc_extras.statespace.utils.constants import SHORT_NAME_TO_LONG


def compile_statespace(
    statespace_model: PyMCStateSpace, steps: int | None = None, **compile_kwargs
):
    if steps is None:
        steps = pt.iscalar("steps")

    x0, _, c, d, T, Z, R, H, Q = statespace_model._unpack_statespace_with_placeholders()

    sequence_names = [x.name for x in [c, d] if x.ndim == 2]
    sequence_names += [x.name for x in [T, Z, R, H, Q] if x.ndim == 3]

    rename_dict = {v: k for k, v in SHORT_NAME_TO_LONG.items()}
    sequence_names = list(map(rename_dict.get, sequence_names))

    P0 = pt.zeros((x0.shape[0], x0.shape[0]))

    outputs = LinearGaussianStateSpace.dist(
        x0, P0, c, d, T, Z, R, H, Q, steps=steps, sequence_names=sequence_names
    )

    inputs = list(pytensor.graph.basic.explicit_graph_inputs(outputs))

    _f = pm.compile(inputs, outputs, on_unused_input="ignore", **compile_kwargs)

    def f(*, draws=1, **params):
        if isinstance(steps, pt.Variable):
            inner_steps = params.get("steps", 100)
        else:
            inner_steps = steps

        output = [np.empty((draws, inner_steps + 1, x.type.shape[-1])) for x in outputs]
        for i in range(draws):
            draw = _f(**params)
            for j, x in enumerate(draw):
                output[j][i] = x
        return [x.squeeze() for x in output]

    return f
