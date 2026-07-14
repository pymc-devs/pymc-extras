import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy

from arviz_base import from_dict
from scipy.special import logsumexp
from scipy.stats import norm

from pymc_extras.distributions import DiscreteMarkovChain, JointCategorical
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


@pytest.mark.parametrize("batched", [False, True], ids=lambda b: f"batched={b}")
def test_marginalized_discrete_markov_chain_time_varying_P(batched):
    """Marginalizing a non-homogeneous (time-varying P) chain matches brute force. When batched,
    independent chains share one per-step transition sequence, so the chain's batch axes must not
    collide with P's core (time, from, to) axes in the forward filter."""
    rng = np.random.default_rng(8)
    pi0 = np.array([0.6, 0.4])
    sigma = 0.8
    T, k = 5, 2
    A_t = rng.dirichlet(np.ones(k), size=(T - 1, k))  # (T - 1, k, k): one matrix per transition
    if batched:
        n_chains = 3
        obs = rng.normal(size=(n_chains, T))
    else:
        obs = rng.normal(size=T)

    with pm.Model() as m:
        init = pm.Categorical.dist(p=pi0, shape=(n_chains,) if batched else None)
        # steps inferred from A_t's time axis
        states = DiscreteMarkovChain(
            "states",
            P=A_t,
            init_dist=init,
            time_varying_P=True,
            shape=(n_chains, T) if batched else None,
        )
        pm.Normal("emission", mu=states, sigma=sigma, observed=obs)

    marginal_m = marginalize(m, [states])

    def brute(obs_row):
        # Marginal likelihood log p(y) over all k**T paths of a single chain.
        log_joint = []
        for s in itertools.product(range(k), repeat=T):
            lp = np.log(pi0[s[0]]) + sum(np.log(A_t[t - 1, s[t - 1], s[t]]) for t in range(1, T))
            lp += norm.logpdf(obs_row, loc=np.array(s), scale=sigma).sum()
            log_joint.append(lp)
        return logsumexp(log_joint)

    expected = sum(brute(o) for o in obs) if batched else brute(obs)
    np.testing.assert_allclose(marginal_m.compile_logp()({}), expected)


@pytest.mark.parametrize("batched", [False, True], ids=lambda b: f"batched={b}")
def test_marginalized_discrete_markov_chain_higher_order(batched):
    """Marginalizing a second-order (n_lags=2) chain matches brute force over all k**T paths. When
    batched, a leading batch dim on P gives independent chains each with their own transition
    tensor, exercising P's batch axis in the forward filter."""
    rng = np.random.default_rng(4)
    pi0 = np.array([0.4, 0.6])
    sigma = 0.7
    T, k, n_lags = 5, 2, 2
    n_chains = 3
    # (k, k, k) == P[s_{t-2}, s_{t-1}, s_t]; a leading batch axis when batched.
    P = rng.dirichlet(np.ones(k), size=((n_chains, k, k) if batched else (k, k)))
    obs = rng.normal(size=(n_chains, T)) if batched else rng.normal(size=T)

    with pm.Model() as m:
        init = pm.Categorical.dist(p=pi0)
        chain = DiscreteMarkovChain(
            "chain",
            P=P,
            init_dist=init,
            steps=T - n_lags,
            n_lags=n_lags,
            shape=obs.shape if batched else None,
        )
        pm.Normal("emission", mu=chain, sigma=sigma, observed=obs)
    marginal_m = marginalize(m, [chain])

    def brute(P_row, obs_row):
        log_joint = []
        for s in itertools.product(range(k), repeat=T):
            lp = np.log(pi0[s[0]]) + np.log(pi0[s[1]])
            lp += sum(np.log(P_row[s[t - 2], s[t - 1], s[t]]) for t in range(n_lags, T))
            lp += norm.logpdf(obs_row, loc=np.array(s), scale=sigma).sum()
            log_joint.append(lp)
        return logsumexp(log_joint)

    expected = sum(brute(P[b], obs[b]) for b in range(n_chains)) if batched else brute(P, obs)
    np.testing.assert_allclose(marginal_m.compile_logp()({}), expected)


@pytest.mark.parametrize("batched", [False, True], ids=lambda b: f"batched={b}")
def test_marginalized_discrete_markov_chain_higher_order_time_varying(batched):
    """Second-order (n_lags=2) chain with a per-step transition tensor, vs brute force. When
    batched, independent chains share one per-step transition tensor."""
    rng = np.random.default_rng(5)
    pi0 = np.array([0.4, 0.6])
    sigma = 0.7
    T, k, n_lags = 5, 2, 2
    P_t = rng.dirichlet(np.ones(k), size=(T - n_lags, k, k))  # (steps, k, k, k)
    obs = rng.normal(size=(3, T)) if batched else rng.normal(size=T)

    with pm.Model() as m:
        init = pm.Categorical.dist(p=pi0)
        chain = DiscreteMarkovChain(
            "chain",
            P=P_t,
            init_dist=init,
            n_lags=n_lags,
            time_varying_P=True,
            shape=obs.shape if batched else None,
        )
        pm.Normal("emission", mu=chain, sigma=sigma, observed=obs)
    marginal_m = marginalize(m, [chain])

    def brute(obs_row):
        log_joint = []
        for s in itertools.product(range(k), repeat=T):
            lp = np.log(pi0[s[0]]) + np.log(pi0[s[1]])
            lp += sum(np.log(P_t[t - n_lags, s[t - 2], s[t - 1], s[t]]) for t in range(n_lags, T))
            lp += norm.logpdf(obs_row, loc=np.array(s), scale=sigma).sum()
            log_joint.append(lp)
        return logsumexp(log_joint)

    expected = sum(brute(o) for o in obs) if batched else brute(obs)
    np.testing.assert_allclose(marginal_m.compile_logp()({}), expected)


@pytest.mark.parametrize("batched", [False, True], ids=lambda b: f"batched={b}")
def test_conditional_discrete_markov_chain_higher_order(batched):
    """conditional() logp and recover() marginals for a second-order (n_lags=2) HMM, vs brute force.

    The posterior over the first two states is correlated, so the recovered chain carries a joint
    (JointCategorical) init; when batched, a leading batch dim on P gives independent chains each
    with their own transition tensor, and that init and the smoothed time-varying transitions are
    themselves batched."""
    rng = np.random.default_rng(7 if batched else 3)
    k, n_lags = 2, 2
    pi0 = np.array([0.5, 0.5])
    sigma = 0.8
    obs = np.array([[0.2, 1.1, -0.3, 0.9], [-0.7, 0.5, 1.3, 0.1]])
    if not batched:
        obs = obs[:1]
    n_chains, T = obs.shape
    # (k, k, k) == P[s_{t-2}, s_{t-1}, s_t]; a leading batch axis when batched (each chain its own).
    P = rng.dirichlet(np.ones(k), size=((n_chains, k, k) if batched else (k, k)))

    with pm.Model() as m:
        init = pm.Categorical.dist(p=pi0)
        states = DiscreteMarkovChain(
            "states",
            P=P,
            init_dist=init,
            steps=T - n_lags,
            n_lags=n_lags,
            shape=obs.shape if batched else None,
        )
        pm.Normal("emission", mu=states, sigma=sigma, observed=obs if batched else obs[0])
    marginal_m = marginalize(m, [states])
    cond_m = conditional(marginal_m)

    # Per-row brute force over all k**T paths -> normalizer -> marginals.
    log_joint = [{} for _ in range(n_chains)]
    for b in range(n_chains):
        P_row = P[b] if batched else P
        for s in itertools.product(range(k), repeat=T):
            lp = np.log(pi0[s[0]]) + np.log(pi0[s[1]])
            lp += sum(np.log(P_row[s[t - 2], s[t - 1], s[t]]) for t in range(n_lags, T))
            lp += norm.logpdf(obs[b], loc=np.array(s), scale=sigma).sum()
            log_joint[b][s] = lp
    log_z = [logsumexp(list(d.values())) for d in log_joint]
    brute_marginal = np.array(
        [
            [sum(np.exp(d[s] - z) for s in d if s[t] == 1) for t in range(T)]
            for d, z in zip(log_joint, log_z)
        ]
    )

    # Exact conditional logp p(s | y) (summed over the batch rows when batched).
    logp_fn = cond_m.compile_logp(vars=[cond_m["states"]])
    if batched:
        paths = np.array([[0, 0, 1, 0], [1, 1, 0, 1]])
        expected_logp = sum(log_joint[b][tuple(paths[b])] - log_z[b] for b in range(n_chains))
        np.testing.assert_allclose(logp_fn({"states": paths}), expected_logp, atol=1e-5)
    else:
        for path in [(0, 0, 1, 0), (1, 1, 0, 1), (0, 1, 1, 0)]:
            np.testing.assert_allclose(
                logp_fn({"states": np.array(path)}), log_joint[0][path] - log_z[0], atol=1e-5
            )

    # Recovered marginals match each row's brute force.
    idata = from_dict({"posterior": {"sigma": np.full((2, 4000), sigma)}})
    recovered = (
        recover(idata, model=marginal_m, random_seed=0).posterior["states"].mean(("chain", "draw"))
    )
    expected_marginal = brute_marginal if batched else brute_marginal[0]
    np.testing.assert_allclose(recovered, expected_marginal, atol=0.02)


def test_marginalized_discrete_markov_chain_joint_init_dist():
    """Marginalizing a second-order chain whose initial states have a correlated joint
    (JointCategorical) init_dist matches brute force. This closes the loop over recovery: the
    init of a recovered higher-order chain is a JointCategorical, so its model can itself be
    marginalized again."""
    rng = np.random.default_rng(6)
    k, n_lags = 2, 2
    P = rng.dirichlet(np.ones(k), size=(k, k))  # (k, k, k): P[s_{t-2}, s_{t-1}, s_t]
    pi0_joint = np.array([[0.1, 0.4], [0.3, 0.2]])  # pi0_joint[s_0, s_1]
    sigma = 0.7
    obs = np.array([0.2, 1.1, -0.3, 0.9])
    T = len(obs)

    with pm.Model() as m:
        init = JointCategorical.dist(p=pi0_joint, n_lags=n_lags)
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init, steps=T - n_lags, n_lags=n_lags)
        pm.Normal("emission", mu=chain, sigma=sigma, observed=obs)
    marginal_m = marginalize(m, [chain])

    log_joint = []
    for s in itertools.product(range(k), repeat=T):
        lp = np.log(pi0_joint[s[0], s[1]])
        lp += sum(np.log(P[s[t - 2], s[t - 1], s[t]]) for t in range(n_lags, T))
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
