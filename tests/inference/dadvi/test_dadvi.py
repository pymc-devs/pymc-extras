import numpy as np
import pymc as pm
import pytest

import pymc_extras as pmx

from pymc_extras.inference.dadvi.dadvi import fit_dadvi


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_dadvi"))
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_fit_dadvi_basic(mode, gradient_backend):
    # Example from BDA3, section 4.1 (same as Laplace test)
    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size
    draws = 10000

    with pm.Model() as m:
        mu = pm.Flat("mu")
        logsigma = pm.Flat("logsigma")

        pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)

        idata = pmx.fit(
            method="dadvi",
            optimizer_method="trust-ncg",
            n_fixed_draws=draws,
            n_draws=draws,
            compile_kwargs={"mode": mode},
            gradient_backend=gradient_backend,
        )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (n,)

    bda_map = [y.mean(), np.log(y.std())]

    np.testing.assert_allclose(idata.posterior["mu"].mean(), bda_map[0], atol=1, rtol=1e-1)
    np.testing.assert_allclose(idata.posterior["logsigma"].mean(), bda_map[1], atol=1, rtol=1e-1)


def test_fit_dadvi_outside_model_context():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.Exponential("sigma", 1)
        y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=np.random.normal(size=10))

    idata = fit_dadvi(
        model=m,
        optimizer_method="L-BFGS-B",
        use_grad=True,
        progressbar=False,
        n_draws=100,
    )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "optimizer_result")
    assert idata.posterior["mu"].shape == (1, 100)


@pytest.mark.parametrize(
    "include_transformed", [True, False], ids=["include_transformed", "no_transformed"]
)
def test_fit_dadvi_coords(include_transformed, rng):
    coords = {"city": ["A", "B", "C"], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=3, sigma=0.5, dims=["city"])
        sigma = pm.Exponential("sigma", 1, dims=["city"])
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        idata = pmx.fit(
            method="dadvi",
            optimizer_method="trust-ncg",
            n_draws=1000,
            include_transformed=include_transformed,
        )

    np.testing.assert_allclose(
        idata.posterior.mu.mean(dim=["chain", "draw"]).values, np.full((3,), 3), atol=0.5
    )
    np.testing.assert_allclose(
        idata.posterior.sigma.mean(dim=["chain", "draw"]).values, np.full((3,), 1.5), atol=0.3
    )

    if include_transformed:
        assert "unconstrained_posterior" in idata
        assert "sigma_log__" in idata.unconstrained_posterior

        # FIXME: The automatic coordinate inference used in MAP/Laplace doesn't work in DADVI yet, so city is not
        #  propagated to the unconstrained_posterior group.
        with pytest.raises(AssertionError):
            assert "city" in idata.unconstrained_posterior.coords


def test_fit_dadvi_ragged_coords(rng):
    coords = {"city": ["A", "B", "C"], "feature": [0, 1], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as ragged_dim_model:
        X = pm.Data("X", np.ones((100, 2)), dims=["obs_idx", "feature"])
        beta = pm.Normal(
            "beta", mu=[[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]], dims=["city", "feature"]
        )
        mu = pm.Deterministic(
            "mu", (X[:, None, :] * beta[None]).sum(axis=-1), dims=["obs_idx", "city"]
        )
        sigma = pm.HalfNormal("sigma", sigma=3.0, dims=["city"])

        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        idata = fit_dadvi(
            optimizer_method="trust-ncg",
            use_grad=True,
            use_hessp=True,
        )

    assert idata["posterior"].beta.shape[-2:] == (3, 2)
    assert idata["posterior"].sigma.shape[-1:] == (3,)

    # Check that everything got unraveled correctly -- feature 0 should be strictly negative, feature 1
    # strictly positive
    assert (idata["posterior"].beta.sel(feature=0).to_numpy() < 0).all()
    assert (idata["posterior"].beta.sel(feature=1).to_numpy() > 0).all()


@pytest.mark.parametrize(
    "method, use_grad, use_hess, use_hessp",
    [
        ("Newton-CG", True, True, False),
        ("Newton-CG", True, False, True),
    ],
)
def test_dadvi_basinhopping(method, use_grad, use_hess, use_hessp, rng):
    pytest.importorskip("jax")

    with pm.Model() as m:
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y_hat", mu=mu, sigma=sigma, observed=rng.normal(loc=3, scale=1.5, size=10))

        idata = fit_dadvi(
            optimizer_method="basinhopping",
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            progressbar=False,
            include_transformed=True,
            minimizer_kwargs=dict(method=method),
            niter=1,
            n_draws=100,
        )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "unconstrained_posterior")

    posterior = idata.posterior
    unconstrained_posterior = idata.unconstrained_posterior
    assert "mu" in posterior
    assert posterior["mu"].shape == (1, 100)

    assert "sigma_log__" in unconstrained_posterior
    assert unconstrained_posterior["sigma_log__"].shape == (1, 100)
