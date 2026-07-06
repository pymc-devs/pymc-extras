import numpy as np
import pymc as pm

from pymc.logprob.abstract import _logprob
from pymc.pytensorf import collect_default_updates
from pytensor import tensor as pt

from pymc_extras.model.marginal.distributions import MarginalFiniteDiscreteRV


def test_marginalized_bernoulli_logp():
    """Test logp of IR TestFiniteMarginalDiscreteRV directly"""
    mu = pt.vector("mu")

    idx = pm.Bernoulli.dist(0.7, name="idx")
    y = pm.Normal.dist(mu=mu[idx], sigma=1.0, name="y")
    # The inner RVs draw from shared RNGs, which the OpFromGraph requires as
    # explicit inputs (with their updates as extra outputs).
    updates = collect_default_updates([idx, y])
    rngs, rng_updates = list(updates.keys()), list(updates.values())
    marginal_rv_node = MarginalFiniteDiscreteRV(
        [mu, *rngs],
        [idx, y, *rng_updates],
        n_dependent_rvs=1,
        dims_connections=(((),),),
        marginalized_name="idx",
        marginalized_dims=(),
    )(mu, *rngs)[0].owner

    y_vv = y.clone()
    (logp,) = _logprob(
        marginal_rv_node.op,
        (y_vv,),
        *marginal_rv_node.inputs,
    )

    ref_logp = pm.logp(pm.NormalMixture.dist(w=[0.3, 0.7], mu=mu, sigma=1.0), y_vv)
    np.testing.assert_almost_equal(
        logp.eval({mu: [-1, 1], y_vv: 2}),
        ref_logp.eval({mu: [-1, 1], y_vv: 2}),
    )
