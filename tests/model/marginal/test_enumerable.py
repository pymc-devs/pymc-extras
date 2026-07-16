import numpy as np
import pymc as pm
import pytest
import scipy.special
import scipy.stats

from pymc.logprob.abstract import _logprob
from pymc.pytensorf import collect_default_updates
from pytensor import tensor as pt

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


def test_multivariate_dependent_with_extra_batch_dim():
    """A dependent whose own logp collapses its core dims, plus a dim to marginalize over.

    ``dims_connections`` is indexed against the dependent RV, but MvNormal's logp has already
    dropped ``c``, so the axes left to reduce have to be renumbered against the logp.
    """
    with pm.Model() as m:
        idx = pm.Categorical("idx", p=[0.3, 0.7], shape=(3,))
        mu = np.zeros((5, 3, 4)) + idx[None, :, None] * 2.0
        pm.MvNormal("y", mu=mu, chol=np.eye(4), shape=(5, 3, 4))

    logp = marginalize(m, ["idx"]).compile_logp()({"y": np.zeros((5, 3, 4))})

    # idx is shared across the 5 obs, so each trial marginalizes over one idx draw
    expected = 3 * scipy.special.logsumexp(
        [
            np.log(w)
            + 5
            * scipy.stats.multivariate_normal.logpdf(np.zeros(4), np.full(4, 2.0 * k), np.eye(4))
            for k, w in [(0, 0.3), (1, 0.7)]
        ]
    )
    np.testing.assert_allclose(logp, expected)


def test_multiple_dependents_logp_term_shapes():
    """The joint logp goes to the first value; the rest get zeros of the right shape.

    A bare scalar placeholder would still sum to the correct total, but leaves
    ``compile_logp(sum=False)`` returning a term that doesn't match its variable's shape.
    """
    with pm.Model() as m:
        idx = pm.Categorical("idx", p=[0.3, 0.7], shape=(3,))
        pm.Normal("y1", mu=idx * 2.0, sigma=1.0, shape=(5, 3))
        pm.Normal("y2", mu=idx * 3.0, sigma=1.0, shape=(3,))

    marginal_m = marginalize(m, ["idx"])
    point = {"y1": np.zeros((5, 3)), "y2": np.zeros(3)}
    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        logp_fn = marginal_m.compile_logp(sum=False)
    y1_term, y2_term = logp_fn(point)

    # Each term keeps its variable's batch shape, placeholder included
    assert y1_term.shape == (3,)
    assert y2_term.shape == (3,)
    np.testing.assert_allclose(y2_term, 0.0)

    # The whole joint sits on the first term, so the total is still exact
    expected = 3 * scipy.special.logsumexp(
        [
            np.log(w)
            + scipy.stats.norm.logpdf(np.zeros(5), 2.0 * k, 1).sum()
            + scipy.stats.norm.logpdf(0.0, 3.0 * k, 1)
            for k, w in [(0, 0.3), (1, 0.7)]
        ]
    )
    np.testing.assert_allclose(y1_term.sum() + y2_term.sum(), expected)
