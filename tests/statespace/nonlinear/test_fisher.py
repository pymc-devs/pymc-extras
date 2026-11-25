import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import grad
from pytensor.tensor.random.basic import (
    GammaRV,
    bernoulli,
    binomial,
    exponential,
    lognormal,
    negative_binomial,
    normal,
    poisson,
    t,
)

from pymc_extras.statespace.nonlinear.fisher import fisher_information


@pytest.mark.filterwarnings("ignore:Numba will use object mode to run tri_gamma")
@pytest.mark.parametrize(
    "op, values, positions",
    [
        (poisson, [3.0], [0]),
        (normal, [0.5, 2.0], [0, 1]),
        (lognormal, [0.3, 0.7], [0, 1]),
        (bernoulli, [0.3], [0]),
        (binomial, [7, 0.4], [1]),
        (exponential, [1.5], [0]),
        (GammaRV(), [2.5, 0.8], [0, 1]),
        (t, [5.0, 0.2, 1.3], [0, 1, 2]),
    ],
    ids=["poisson", "normal", "lognormal", "bernoulli", "binomial", "exponential", "gamma", "t"],
)
def test_fisher_information_is_the_expected_negative_hessian(op, values, positions):
    n_draws = 400_000
    rng = pytensor.shared(np.random.default_rng(0))
    size = pt.as_tensor([n_draws])
    draws = op.make_node(rng, size, *map(pt.as_tensor, values)).outputs[1].eval()

    parameters = [pt.as_tensor(value) for value in values]
    for position in positions:
        parameters[position] = pt.scalar(f"parameter_{position}", dtype="float64")
    rv = op.make_node(rng, size, *parameters).outputs[1]
    logp = pm.logp(rv, draws).sum()
    gradient = grad(logp, [parameters[position] for position in positions])
    negative_hessian = -pt.stack(
        [
            pt.stack(grad(entry, [parameters[position] for position in positions]))
            for entry in gradient
        ]
    )
    point = {parameters[position]: values[position] for position in positions}
    average = negative_hessian.eval(point) / n_draws

    information = fisher_information(
        op=op,
        distribution_parameters=[pt.as_tensor(np.full(1, value, dtype=float)) for value in values],
        positions=positions,
    )

    expected = information.eval()[0]
    assert_allclose(average, expected, rtol=0.03, atol=0.03 * np.abs(expected).max())


def test_fisher_information_is_none_without_a_closed_form():
    parameters = [pt.ones(3), pt.full(3, 0.5)]

    assert (
        fisher_information(op=negative_binomial, distribution_parameters=parameters, positions=[1])
        is None
    )
    assert (
        fisher_information(op=binomial, distribution_parameters=parameters, positions=[0, 1])
        is None
    )
