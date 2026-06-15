import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy

from arviz_base import from_dict
from scipy.special import logsumexp
from scipy.stats import norm

from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.marginal import conditional, marginalize, recover


@pytest.mark.parametrize("batch_chain", (False, True), ids=lambda x: f"batch_chain={x}")
@pytest.mark.parametrize("batch_emission", (False, True), ids=lambda x: f"batch_emission={x}")
def test_marginalized_discrete_markov_chain_normal_emission(batch_chain, batch_emission):
    if batch_chain and not batch_emission:
        pytest.skip("Redundant implicit combination")

    with pm.Model() as m:
        P = [[0, 1], [1, 0]]
        init_dist = pm.Categorical.dist(p=[1, 0])
        chain = DiscreteMarkovChain(
            "chain", P=P, init_dist=init_dist, steps=3, shape=(3, 4) if batch_chain else None
        )
        emission = pm.Normal(
            "emission", mu=chain * 2 - 1, sigma=1e-1, shape=(3, 4) if batch_emission else None
        )

    marginal_m = marginalize(m, [chain])
    logp_fn = marginal_m.compile_logp()

    test_value = np.array([-1, 1, -1, 1])
    expected_logp = pm.logp(pm.Normal.dist(0, 1e-1), np.zeros_like(test_value)).sum().eval()
    if batch_emission:
        test_value = np.broadcast_to(test_value, (3, 4))
        expected_logp *= 3
    np.testing.assert_allclose(logp_fn({"emission": test_value}), expected_logp)


@pytest.mark.parametrize(
    "categorical_emission",
    [False, True],
)
def test_marginalized_discrete_markov_chain_categorical_emission(categorical_emission):
    """Example adapted from https://www.youtube.com/watch?v=9-sPm4CfcD0"""
    with pm.Model() as m:
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        init_dist = pm.Categorical.dist(p=[0.375, 0.625])
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=2)
        if categorical_emission:
            emission = pm.Categorical("emission", p=pt.constant([[0.8, 0.2], [0.4, 0.6]])[chain])
        else:
            emission = pm.Bernoulli("emission", p=pt.where(pt.eq(chain, 0), 0.2, 0.6))
    marginal_m = marginalize(m, [chain])

    test_value = np.array([0, 0, 1])
    expected_logp = np.log(0.1344)  # Shown at the 10m22s mark in the video
    logp_fn = marginal_m.compile_logp()
    np.testing.assert_allclose(logp_fn({"emission": test_value}), expected_logp)


@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply:RuntimeWarning")
@pytest.mark.parametrize("batch_chain", (False, True))
@pytest.mark.parametrize("batch_emission1", (False, True))
@pytest.mark.parametrize("batch_emission2", (False, True))
def test_marginalized_discrete_markov_chain_multiple_emissions(
    batch_chain, batch_emission1, batch_emission2
):
    chain_shape = (3, 1, 4) if batch_chain else (4,)
    emission1_shape = (
        (2, *reversed(chain_shape)) if batch_emission1 else tuple(reversed(chain_shape))
    )
    emission2_shape = (*chain_shape, 2) if batch_emission2 else chain_shape
    with pm.Model() as m:
        P = [[0, 1], [1, 0]]
        init_dist = pm.Categorical.dist(p=[1, 0])
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, shape=chain_shape)
        emission_1 = pm.Normal(
            "emission_1", mu=(chain * 2 - 1).T, sigma=1e-1, shape=emission1_shape
        )

        emission2_mu = (1 - chain) * 2 - 1
        if batch_emission2:
            emission2_mu = emission2_mu[..., None]
        emission_2 = pm.Normal("emission_2", mu=emission2_mu, sigma=1e-1, shape=emission2_shape)

    marginal_m = marginalize(m, [chain])

    with pytest.warns(UserWarning, match="multiple dependent variables"):
        logp_fn = marginal_m.compile_logp(sum=False)

    test_value = np.array([-1, 1, -1, 1])
    multiplier = 2 + batch_emission1 + batch_emission2
    if batch_chain:
        multiplier *= 3
    expected_logp = norm.logpdf(np.zeros_like(test_value), 0, 1e-1).sum() * multiplier

    test_value = np.broadcast_to(test_value, chain_shape)
    test_value_emission1 = np.broadcast_to(test_value.T, emission1_shape)
    if batch_emission2:
        test_value_emission2 = np.broadcast_to(-test_value[..., None], emission2_shape)
    else:
        test_value_emission2 = np.broadcast_to(-test_value, emission2_shape)
    test_point = {"emission_1": test_value_emission1, "emission_2": test_value_emission2}
    res_logp, dummy_logp = logp_fn(test_point)
    assert res_logp.shape == ((3, 1) if batch_chain else ())
    np.testing.assert_allclose(res_logp.sum(), expected_logp)


def test_recover_discrete_markov_chain_propagates_noise():
    """recover() propagates the emission noise from the posterior.

    The emission sigma is a free RV; we feed a posterior alternating between a
    low and a high value. Under low noise the emissions pin the states, so the
    recovered path is deterministic and matches the observations; under high
    noise the emissions are uninformative and the recovered path is not.
    """
    P = np.array([[0.7, 0.3], [0.3, 0.7]])
    low, high = 0.05, 5.0
    obs = np.array([0, 0, 1, 1, 0])
    with pm.Model() as m:
        sigma_rv = pm.HalfNormal("sigma", 1.0, default_transform=None)
        init_dist = pm.Categorical.dist(p=[0.5, 0.5])
        states = DiscreteMarkovChain("states", P=P, init_dist=init_dist, steps=len(obs) - 1)
        pm.Normal("emission", mu=states, sigma=sigma_rv, observed=obs)

    marginal_m = marginalize(m, [states])

    sigmas = np.tile([low, high], 50)
    idata = from_dict({"posterior": {"sigma": sigmas[None, :]}})
    out = recover(idata, model=marginal_m, random_seed=42)

    post = out.posterior
    assert "states" in post
    assert post["states"].shape == (1, 100, len(obs))

    rec = post["states"].isel(chain=0).values
    # Low-noise draws: deterministic, every draw is exactly the observations.
    np.testing.assert_array_equal(rec[sigmas == low], np.broadcast_to(obs, (50, len(obs))))
    # High-noise draws: not all equal to the observations.
    assert not np.array_equal(rec[sigmas == high], np.broadcast_to(obs, (50, len(obs))))


def test_conditional_discrete_markov_chain():
    """conditional() logp and recover() marginals for an HMM both match brute force."""
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    pi0 = np.array([0.6, 0.4])
    sigma = 0.9
    obs = np.array([0.2, 1.1, 0.9, -0.3])
    n_steps = len(obs)

    with pm.Model() as m:
        sigma_rv = pm.HalfNormal("sigma", 1.0, default_transform=None)
        init_dist = pm.Categorical.dist(p=pi0)
        states = DiscreteMarkovChain("states", P=P, init_dist=init_dist, steps=n_steps - 1)
        pm.Normal("emission", mu=states, sigma=sigma_rv, observed=obs)
    marginal_m = marginalize(m, [states])
    cond_m = conditional(marginal_m)

    # Brute-force log joint over all 2^T paths -> normalizer -> marginals.
    log_joint = {}
    for path in itertools.product([0, 1], repeat=n_steps):
        lp = np.log(pi0[path[0]])
        for t in range(1, n_steps):
            lp += np.log(P[path[t - 1], path[t]])
        lp += scipy.stats.norm.logpdf(obs, loc=np.array(path), scale=sigma).sum()
        log_joint[path] = lp
    log_z = logsumexp(list(log_joint.values()))
    brute_marginal = np.array(
        [sum(np.exp(log_joint[s] - log_z) for s in log_joint if s[t] == 1) for t in range(n_steps)]
    )

    # Exact conditional logp p(s | y, sigma)
    logp_fn = cond_m.compile_logp(vars=[cond_m["states"]])
    for path in [(0, 0, 1, 0), (1, 1, 0, 1), (0, 1, 1, 0)]:
        got = logp_fn({"sigma": sigma, "states": np.array(path)})
        np.testing.assert_allclose(got, log_joint[path] - log_z, atol=1e-5)

    # Recovered marginals match brute force; sigma is propagated from the
    # posterior (states are 0/1, so mean == P(s=1)).
    idata = from_dict({"posterior": {"sigma": np.full((2, 2000), sigma)}})
    recovered = (
        recover(idata, model=marginal_m, random_seed=0).posterior["states"].mean(("chain", "draw"))
    )
    np.testing.assert_allclose(recovered, brute_marginal, atol=0.02)


def test_marginalized_discrete_markov_chain_time_varying_P():
    """Marginalizing a non-homogeneous (time-varying P) chain matches brute force."""
    pi0 = np.array([0.6, 0.4])
    sigma = 0.8
    obs = np.array([0.1, 1.2, 0.8, -0.2, 0.9])
    T, k = len(obs), 2
    A_t = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.1, 0.9], [0.5, 0.5]],
            [[0.7, 0.3], [0.6, 0.4]],
        ]
    )

    with pm.Model() as m:
        init = pm.Categorical.dist(p=pi0)
        # steps inferred from A_t's time axis
        states = DiscreteMarkovChain("states", P=A_t, init_dist=init, time_varying_P=True)
        pm.Normal("emission", mu=states, sigma=sigma, observed=obs)

    marginal_m = marginalize(m, [states])

    # Brute-force marginal likelihood log p(y) over all k^T paths.
    log_joint = []
    for s in itertools.product(range(k), repeat=T):
        lp = np.log(pi0[s[0]]) + sum(np.log(A_t[t - 1, s[t - 1], s[t]]) for t in range(1, T))
        lp += norm.logpdf(obs, loc=np.array(s), scale=sigma).sum()
        log_joint.append(lp)
    expected = logsumexp(log_joint)

    np.testing.assert_allclose(marginal_m.compile_logp()({}), expected)


def test_conditional_discrete_markov_chain_time_varying_P():
    """conditional()/recover() for a non-homogeneous (time-varying P) HMM."""
    pi0 = np.array([0.6, 0.4])
    sigma = 0.8
    obs = np.array([0.1, 1.2, 0.8, -0.2, 0.9])
    n_steps, k = len(obs), 2
    A_t = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.4, 0.6], [0.3, 0.7]],
            [[0.1, 0.9], [0.5, 0.5]],
            [[0.7, 0.3], [0.6, 0.4]],
        ]
    )

    with pm.Model() as m:
        sigma_rv = pm.HalfNormal("sigma", 1.0, default_transform=None)
        init_dist = pm.Categorical.dist(p=pi0)
        # steps inferred from A_t's time axis
        states = DiscreteMarkovChain("states", P=A_t, init_dist=init_dist, time_varying_P=True)
        pm.Normal("emission", mu=states, sigma=sigma_rv, observed=obs)

    marginal_m = marginalize(m, [states])
    cond_m = conditional(marginal_m)

    log_joint = {}
    for s in itertools.product(range(k), repeat=n_steps):
        lp = np.log(pi0[s[0]]) + sum(np.log(A_t[t - 1, s[t - 1], s[t]]) for t in range(1, n_steps))
        lp += scipy.stats.norm.logpdf(obs, loc=np.array(s), scale=sigma).sum()
        log_joint[s] = lp
    log_z = logsumexp(list(log_joint.values()))
    brute_marginal = np.array(
        [sum(np.exp(log_joint[s] - log_z) for s in log_joint if s[t] == 1) for t in range(n_steps)]
    )

    logp_fn = cond_m.compile_logp(vars=[cond_m["states"]])
    for path in [(0, 0, 1, 0, 1), (1, 1, 0, 1, 0), (0, 1, 1, 0, 1)]:
        got = logp_fn({"sigma": sigma, "states": np.array(path)})
        np.testing.assert_allclose(got, log_joint[path] - log_z, atol=1e-5)

    # Recovered marginals match brute force; sigma is propagated from the
    # posterior (states are 0/1, so mean == P(s=1)).
    idata = from_dict({"posterior": {"sigma": np.full((1, 5000), sigma)}})
    recovered = (
        recover(idata, model=marginal_m, random_seed=0).posterior["states"].mean(("chain", "draw"))
    )
    np.testing.assert_allclose(recovered, brute_marginal, atol=0.02)
