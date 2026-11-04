import pymc as pm
import pytest

from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable

from pymc_extras.inference.advi import default_optimizer
from pymc_extras.inference.advi.autoguide import AutoMultivariateNormal
from pymc_extras.inference.advi.compile import build_svi_step, shared_guide_params


@pytest.mark.parametrize("draws, expected_shape", [(1, (3,)), (2, (2, 3))])
def test_build_svi_step_only_batches_multiple_draws(draws, expected_shape):
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1, shape=3)
        pm.Normal("y", x.sum(), 1, observed=1.0)

    guide = AutoMultivariateNormal(model, random_seed=1)
    negative_elbo, _ = build_svi_step(
        model, guide, default_optimizer(), shared_guide_params(guide), draws=draws
    )

    guide_draws = [
        v
        for v in ancestors([negative_elbo])
        if v.owner and isinstance(v.owner.op, RandomVariable) and v is v.owner.outputs[1]
    ]
    assert [draw.type.shape for draw in guide_draws] == [expected_shape]
