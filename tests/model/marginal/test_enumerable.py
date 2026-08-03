import numpy as np
import pymc as pm

from pymc.logprob.abstract import _logprob
from pymc.pytensorf import collect_default_updates
from pytensor import tensor as pt
from scipy.special import logsumexp

from pymc_extras.marginal import marginalize
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


def test_categorical_domain_from_dims():
    """The number of categories may only be known at runtime, when it comes from a model dim."""
    n_obs, n_groups = 4, 3
    y_data = np.array([-0.9, 0.3, 1.7, 0.1])

    with pm.Model(coords={"obs_idx": range(n_obs), "group": range(n_groups)}) as m:
        logit_p = pm.Normal("logit_p", dims=("obs_idx", "group"))
        mu = pm.Normal("mu", dims=("group",))
        # The last axis of the Categorical parameter is only known once the dim length is resolved
        assert logit_p.type.shape == (None, None)
        idx = pm.Categorical("idx", logit_p=logit_p, dims=("obs_idx",))
        pm.Normal("y", mu=mu[idx], sigma=1.0, observed=y_data, dims=("obs_idx",))

    ip = m.initial_point()
    ip.pop("idx")
    # Observations are independent given p, so enumerate each one over the shared domain
    ref_logp_fn = m.compile_logp([m["idx"], m["y"]], sum=False)
    ref_logp = logsumexp(
        [sum(ref_logp_fn({**ip, "idx": np.full(n_obs, g)})) for g in range(n_groups)],
        axis=0,
    ).sum()

    marginal_m = marginalize(m, ["idx"])
    np.testing.assert_allclose(marginal_m.compile_logp([marginal_m["y"]])(ip), ref_logp)
