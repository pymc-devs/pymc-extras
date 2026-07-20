import numpy as np
import pymc as pm
import pytest
import scipy

from arviz_base import from_dict

from pymc_extras.marginal import conditional, marginalize, recover


def test_normal_normal():
    with pm.Model() as m:
        x = pm.Normal("x", mu=0, sigma=1)
        y = pm.Normal("y", mu=x + np.pi - 1, sigma=1.0)
        z = pm.Normal("z", mu=y + 2 * np.pi, sigma=np.sqrt(np.e))

    marginal_m = marginalize(m, m["y"])

    test_point = {"x": 1, "z": -1}

    np.testing.assert_allclose(
        marginal_m.compile_logp([marginal_m["z"]])(test_point),
        scipy.stats.norm.logpdf(test_point["z"], np.pi * 3, np.sqrt(1 + np.e)),
    )


def test_normal_normal_batched_integer_mu():
    with pm.Model() as m:
        x = pm.Normal("x", mu=0, sigma=1, shape=(3,))
        y = pm.Normal("y", mu=x + 1, sigma=1.0, shape=(3,))

    np.testing.assert_allclose(
        marginalize(m, "x").compile_logp()({"y": [0.5, 1.0, 2.0]}),
        scipy.stats.norm.logpdf([0.5, 1.0, 2.0], 1, np.sqrt(2)).sum(),
    )


@pytest.mark.parametrize(
    "mu_fn, a, b",
    [
        (lambda x: x + x, 0, 2),
        (lambda x: 2 * x, 0, 2),
        (lambda x: 3 * x + 1, 1, 3),
        (lambda x: 1 + x * 3, 1, 3),
        (lambda x: x * 2 + x, 0, 3),
        (lambda x: x + x + x, 0, 3),
        (lambda x: 1 + 2 + 3 * x, 3, 3),
    ],
    ids=["x+x", "2x", "3x+1", "1+x3", "2x+x", "x+x+x", "1+2+3x"],
)
def test_normal_normal_affine(mu_fn, a, b):
    """Dependent mean affine in the marginalized rv, mu = a + b*x, over the
    flattened variadic Add/Mul forms (operand order and repetition vary)."""
    with pm.Model() as m:
        x = pm.Normal("x", mu=1, sigma=2)
        y = pm.Normal("y", mu=mu_fn(x), sigma=3)

    marginal_m = marginalize(m, m["x"])

    expected_mu = a + b * 1
    expected_sigma = np.sqrt(3**2 + (b * 2) ** 2)
    np.testing.assert_allclose(
        marginal_m.compile_logp()({"y": 5.0}),
        scipy.stats.norm.logpdf(5.0, expected_mu, expected_sigma),
    )


def test_normal_normal_affine_conditional():
    """The conjugate posterior of an affine Normal-Normal accounts for the slope."""
    sigma_prior = 3.0
    a, b = 1.5, 2.0
    sigma_lik = 4.0
    y_obs = 10.0

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        x = pm.Normal("x", mu=mu, sigma=sigma_prior)
        y = pm.Normal("y", mu=b * x + a, sigma=sigma_lik, observed=y_obs)

    marginal_m = marginalize(m, "x")
    cond_m = conditional(marginal_m)

    mu_val = 1.0
    x_test = 3.0
    prec_p = 1 / sigma_prior**2
    post_prec = prec_p + b**2 / sigma_lik**2
    post_sigma = np.sqrt(1 / post_prec)
    post_mu = (mu_val * prec_p + b * (y_obs - a) / sigma_lik**2) / post_prec

    expected = scipy.stats.norm.logpdf(x_test, post_mu, post_sigma)
    actual = cond_m.compile_logp(vars=[cond_m["x"]])({"mu": mu_val, "x": x_test})
    np.testing.assert_allclose(actual, expected)


def test_normal_normal_nonlinear_in_sigma():
    """Marginalized rv in sigma — not valid for closed-form Normal-Normal."""
    with pm.Model() as m:
        x = pm.Normal("x", mu=0, sigma=1)
        y = pm.Normal("y", mu=0, sigma=x**2 + 1)

    with pytest.raises(NotImplementedError):
        marginalize(m, m["x"])


def test_normal_normal_nonlinear_in_mu():
    """Marginalized rv entering mu nonlinearly (x**2) has no closed-form Normal marginal."""
    with pm.Model() as m:
        x = pm.Normal("x", mu=0, sigma=1)
        y = pm.Normal("y", mu=x**2, sigma=1)

    with pytest.raises(NotImplementedError):
        marginalize(m, m["x"])


@pytest.mark.parametrize("x_shape", [(), (1,)], ids=["scalar", "size-1"])
def test_normal_normal_shared_scalar_latent(x_shape):
    """A scalar (or size-1) latent broadcast into a wider dependent makes those
    dependents jointly MvNormal, correlated through the single shared latent."""
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=x_shape)
        y = pm.Normal("y", mu=x, sigma=1.0, shape=(3,))

    yv = np.array([0.5, 1.0, 2.0])
    cov = np.eye(3) + np.ones((3, 3))  # diag(sigma_d**2) + sigma_m**2 shared everywhere
    np.testing.assert_allclose(
        marginalize(m, "x").compile_logp()({"y": yv}),
        scipy.stats.multivariate_normal.logpdf(yv, np.zeros(3), cov),
    )


@pytest.mark.parametrize(
    "mu_fn, y_shape, along",
    [(lambda x: x, (5, 3), "col"), (lambda x: x[:, None], (5, 3), "row")],
    ids=["shared-leading", "shared-trailing"],
)
def test_normal_normal_batched_shared_latent(mu_fn, y_shape, along):
    """A vector latent broadcast along an extra dependent dim: MvNormal over the
    shared dim, independent (batched) along the dim it matches one-to-one."""
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=(3,) if along == "col" else (5,))
        y = pm.Normal("y", mu=mu_fn(x), sigma=1.0, shape=y_shape)

    yv = np.arange(15.0).reshape(y_shape) / 5
    if along == "col":  # 5 rows share x[j] -> 5-dim event, 3 independent columns
        cov, blocks = np.eye(5) + np.ones((5, 5)), (yv[:, j] for j in range(3))
    else:  # 3 cols share x[i] -> 3-dim event, 5 independent rows
        cov, blocks = np.eye(3) + np.ones((3, 3)), (yv[i] for i in range(5))
    expected = sum(scipy.stats.multivariate_normal.logpdf(b, np.zeros(len(b)), cov) for b in blocks)
    np.testing.assert_allclose(marginalize(m, "x").compile_logp()({"y": yv}), expected)


def test_normal_normal_cross_broadcast_not_supported():
    """A dependent element coupling two different latent entries (x[None,:] + x[:,None])
    is not a closed-form conjugate marginal."""
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=(3,))
        y = pm.Normal("y", mu=x[None, :] + x[:, None], sigma=1.0, shape=(3, 3))

    with pytest.raises(NotImplementedError):
        marginalize(m, "x")


def test_normal_normal_conditional_logp():
    """Test that conditional gives correct conjugate posterior logp for Normal-Normal."""
    sigma_prior = 3.0
    offset = 1.5
    sigma_lik = 4.0
    y_obs = 10.0

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        x = pm.Normal("x", mu=mu, sigma=sigma_prior)
        y = pm.Normal("y", mu=x + offset, sigma=sigma_lik, observed=y_obs)

    marginal_m = marginalize(m, "x")
    cond_m = conditional(marginal_m)

    assert "x" in [rv.name for rv in cond_m.free_RVs]

    logp_fn = cond_m.compile_logp(vars=[cond_m["x"]])
    mu_val = 1.0
    x_test = 3.0

    prec_p = 1 / sigma_prior**2
    prec_l = 1 / sigma_lik**2
    post_prec = prec_p + prec_l
    post_sigma = np.sqrt(1 / post_prec)
    post_mu = (mu_val * prec_p + (y_obs - offset) * prec_l) / post_prec

    expected = scipy.stats.norm.logpdf(x_test, post_mu, post_sigma)
    actual = logp_fn({"mu": mu_val, "x": x_test})
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "x_shape, y_shape, sum_axis",
    [((), (3,), None), ((5,), (5, 3), 1), ((3,), (5, 3), 0)],
    ids=["scalar", "batched-trailing", "batched-leading"],
)
def test_normal_normal_broadcast_conditional_logp(x_shape, y_shape, sum_axis):
    """A latent broadcast into several dependents accumulates all their evidence in
    its conditional: the posterior sums over the shared axes back onto the latent."""
    y_obs = np.arange(np.prod(y_shape), dtype=float).reshape(y_shape) / 5

    with pm.Model() as m:
        x = pm.Normal("x", 0, 1, shape=x_shape)
        mu = x if x_shape != (5,) else x[:, None]
        y = pm.Normal("y", mu=mu, sigma=1.0, observed=y_obs, shape=y_shape)

    cond_m = conditional(marginalize(m, "x"))
    assert cond_m["x"].type.shape == x_shape

    # n shared observations of unit precision: post precision 1 + n, mean = sum(y)/(1+n)
    n_shared = y_obs.size if sum_axis is None else y_shape[sum_axis]
    post_prec = 1 + n_shared
    evidence = y_obs.sum() if sum_axis is None else y_obs.sum(axis=sum_axis)
    post_mu, post_sigma = evidence / post_prec, np.sqrt(1 / post_prec)

    x_test = np.full(x_shape, 0.3)
    expected = scipy.stats.norm.logpdf(x_test, post_mu, post_sigma).sum()
    np.testing.assert_allclose(cond_m.compile_logp(vars=[cond_m["x"]])({"x": x_test}), expected)


def test_recover_normal_normal_marginal():
    """Test that recover produces correct conjugate posterior samples."""
    sigma_prior = 3.0
    offset = 1.5
    sigma_lik = 4.0
    y_obs = 10.0

    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        x = pm.Normal("x", mu=mu, sigma=sigma_prior)
        y = pm.Normal("y", mu=x + offset, sigma=sigma_lik, observed=y_obs)

    marginal_m = marginalize(m, "x")

    prec_prior = 1 / sigma_prior**2
    prec_lik = 1 / sigma_lik**2
    post_prec = prec_prior + prec_lik
    expected_sigma = np.sqrt(1 / post_prec)

    # Use constant mu across many draws for statistical precision
    mu_val = 1.0
    expected_mu = (mu_val * prec_prior + (y_obs - offset) * prec_lik) / post_prec

    n_draws = 500
    idata = from_dict({"posterior": {"mu": np.full((4, n_draws), mu_val)}})

    post = recover(idata, model=marginal_m, random_seed=42)
    assert "x" in post.posterior

    x_samples = post.posterior.x.values.flatten()
    np.testing.assert_allclose(x_samples.mean(), expected_mu, atol=0.1)
    np.testing.assert_allclose(x_samples.std(), expected_sigma, atol=0.1)
