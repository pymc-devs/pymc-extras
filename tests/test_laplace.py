#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import pymc as pm
import pytest

import pymc_extras as pmx

from pymc_extras.inference.find_map import GradientBackend, find_MAP
from pymc_extras.inference.laplace import (
    fit_laplace,
    fit_mvn_at_MAP,
    get_conditional_gaussian_approximation,
    sample_laplace_posterior,
)


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_laplace"))
    return np.random.default_rng(seed)


@pytest.mark.filterwarnings(
    "ignore:hessian will stop negating the output in a future version of PyMC.\n"
    + "To suppress this warning set `negate_output=False`:FutureWarning",
)
@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_laplace(mode, gradient_backend: GradientBackend):
    # Example originates from Bayesian Data Analyses, 3rd Edition
    # By Andrew Gelman, John Carlin, Hal Stern, David Dunson,
    # Aki Vehtari, and Donald Rubin.
    # See section. 4.1

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size
    draws = 100000

    with pm.Model() as m:
        mu = pm.Uniform("mu", -10000, 10000)
        logsigma = pm.Uniform("logsigma", 1, 100)

        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        vars = [mu, logsigma]

        idata = pmx.fit(
            method="laplace",
            optimize_method="trust-ncg",
            draws=draws,
            random_seed=173300,
            chains=1,
            compile_kwargs={"mode": mode},
            gradient_backend=gradient_backend,
        )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (n,)
    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [y.mean(), np.log(y.std())]
    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])

    np.testing.assert_allclose(idata.fit["mean_vector"].values, bda_map)
    np.testing.assert_allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)


@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_laplace_only_fit(mode, gradient_backend: GradientBackend):
    # Example originates from Bayesian Data Analyses, 3rd Edition
    # By Andrew Gelman, John Carlin, Hal Stern, David Dunson,
    # Aki Vehtari, and Donald Rubin.
    # See section. 4.1

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size

    with pm.Model() as m:
        logsigma = pm.Uniform("logsigma", 1, 100)
        mu = pm.Uniform("mu", -10000, 10000)
        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        vars = [mu, logsigma]

        idata = pmx.fit(
            method="laplace",
            optimize_method="BFGS",
            progressbar=True,
            gradient_backend=gradient_backend,
            compile_kwargs={"mode": mode},
            optimizer_kwargs=dict(maxiter=100_000, gtol=1e-100),
            random_seed=173300,
        )

    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [np.log(y.std()), y.mean()]
    bda_cov = np.array([[1 / (2 * n), 0], [0, y.var() / n]])

    np.testing.assert_allclose(idata.fit["mean_vector"].values, bda_map)
    np.testing.assert_allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)


@pytest.mark.parametrize(
    "transform_samples",
    [True, False],
    ids=["transformed", "untransformed"],
)
@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_fit_laplace_coords(rng, transform_samples, mode, gradient_backend: GradientBackend):
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

        optimized_point = find_MAP(
            method="trust-ncg",
            use_grad=True,
            use_hessp=True,
            progressbar=False,
            compile_kwargs=dict(mode=mode),
            gradient_backend=gradient_backend,
        )

        for value in optimized_point.values():
            assert value.shape == (3,)

        mu, H_inv = fit_mvn_at_MAP(
            optimized_point=optimized_point,
            model=model,
            transform_samples=transform_samples,
        )

        idata = sample_laplace_posterior(
            mu=mu, H_inv=H_inv, model=model, transform_samples=transform_samples
        )

    np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2, 3), 3), atol=0.5)
    np.testing.assert_allclose(
        np.mean(idata.posterior.sigma, axis=1), np.full((2, 3), 1.5), atol=0.3
    )

    suffix = "_log__" if transform_samples else ""
    assert idata.fit.rows.values.tolist() == [
        "mu[A]",
        "mu[B]",
        "mu[C]",
        f"sigma{suffix}[A]",
        f"sigma{suffix}[B]",
        f"sigma{suffix}[C]",
    ]


@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_fit_laplace_ragged_coords(mode, gradient_backend: GradientBackend, rng):
    coords = {"city": ["A", "B", "C"], "feature": [0, 1], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as ragged_dim_model:
        X = pm.Data("X", np.ones((100, 2)), dims=["obs_idx", "feature"])
        beta = pm.Normal(
            "beta", mu=[[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]], dims=["city", "feature"]
        )
        mu = pm.Deterministic(
            "mu", (X[:, None, :] * beta[None]).sum(axis=-1), dims=["obs_idx", "city"]
        )
        sigma = pm.Normal("sigma", mu=1.5, sigma=0.5, dims=["city"])

        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        idata = fit_laplace(
            optimize_method="Newton-CG",
            progressbar=False,
            use_grad=True,
            use_hessp=True,
            gradient_backend=gradient_backend,
            compile_kwargs={"mode": mode},
        )

    assert idata["posterior"].beta.shape[-2:] == (3, 2)
    assert idata["posterior"].sigma.shape[-1:] == (3,)

    # Check that everything got unraveled correctly -- feature 0 should be strictly negative, feature 1
    # strictly positive
    assert (idata["posterior"].beta.sel(feature=0).to_numpy() < 0).all()
    assert (idata["posterior"].beta.sel(feature=1).to_numpy() > 0).all()


@pytest.mark.parametrize(
    "fit_in_unconstrained_space",
    [True, False],
    ids=["transformed", "untransformed"],
)
@pytest.mark.parametrize(
    "mode, gradient_backend",
    [(None, "pytensor"), ("NUMBA", "pytensor"), ("JAX", "jax"), ("JAX", "pytensor")],
)
def test_fit_laplace(fit_in_unconstrained_space, mode, gradient_backend: GradientBackend):
    with pm.Model() as simp_model:
        mu = pm.Normal("mu", mu=3, sigma=0.5)
        sigma = pm.Exponential("sigma", 1)
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=np.random.default_rng().normal(loc=3, scale=1.5, size=(10000,)),
        )

        idata = fit_laplace(
            optimize_method="trust-ncg",
            use_grad=True,
            use_hessp=True,
            fit_in_unconstrained_space=fit_in_unconstrained_space,
            optimizer_kwargs=dict(maxiter=100_000, tol=1e-100),
            compile_kwargs={"mode": mode},
            gradient_backend=gradient_backend,
        )

        np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2,), 3), atol=0.1)
        np.testing.assert_allclose(
            np.mean(idata.posterior.sigma, axis=1), np.full((2,), 1.5), atol=0.1
        )

        if fit_in_unconstrained_space:
            assert idata.fit.rows.values.tolist() == ["mu", "sigma_log__"]
            np.testing.assert_allclose(idata.fit.mean_vector.values, np.array([3.0, 0.4]), atol=0.1)
        else:
            assert idata.fit.rows.values.tolist() == ["mu", "sigma"]
            np.testing.assert_allclose(idata.fit.mean_vector.values, np.array([3.0, 1.5]), atol=0.1)


def test_laplace_scalar():
    # Example model from Statistical Rethinking
    data = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

    with pm.Model():
        p = pm.Uniform("p", 0, 1)
        w = pm.Binomial("w", n=len(data), p=p, observed=data.sum())

        idata_laplace = pmx.fit_laplace(progressbar=False)

    assert idata_laplace.fit.mean_vector.shape == (1,)
    assert idata_laplace.fit.covariance_matrix.shape == (1, 1)

    np.testing.assert_allclose(idata_laplace.fit.mean_vector.values.item(), data.mean(), atol=0.1)


def test_get_conditional_gaussian_approximation():
    rng = np.random.default_rng(42)
    n = 100
    d = 3

    mu_true = rng.random(d)
    cov_true = np.diag(rng.random(d))
    Q_val = np.diag(rng.random(d))
    cov_param_val = rng.random(d**2).reshape((d, d))

    x_val = rng.random(d)
    mu_val = rng.random(d)

    mu_mu = rng.random(d)
    mu_cov = np.diag(rng.random(d))
    cov_mu = rng.random(d**2)
    cov_cov = np.diag(rng.random(d**2))
    Q_mu = rng.random(d**2)
    Q_cov = np.diag(rng.random(d**2))

    with pm.Model() as model:
        y_obs = rng.multivariate_normal(mean=mu_true, cov=cov_true, size=n)

        mu_param = pm.MvNormal("mu_param", mu=mu_mu, cov=mu_cov)
        cov_param = pm.MvNormal("cov_param", mu=cov_mu, cov=cov_cov)
        Q = pm.MvNormal("Q", mu=Q_mu, cov=Q_cov)

        x = pm.MvNormal("x", mu=mu_param, cov=np.linalg.inv(Q_val))

        y = pm.MvNormal(
            "y",
            mu=x,
            cov=cov_param.reshape((d, d)),
            observed=y_obs,
        )

        # logp(x | y, params)
        cga = get_conditional_gaussian_approximation(
            x=model.rvs_to_values[x],
            Q=Q.reshape((d, d)),
            mu=mu_param,
            optimizer_kwargs={"tol": 1e-25},
        )

    x0, log_x_posterior = cga(
        x=x_val, mu_param=mu_val, cov_param=cov_param_val.flatten(), Q=Q_val.flatten()
    )

    cov_param_inv = np.linalg.inv(cov_param_val)

    x0_true = np.linalg.inv(n * cov_param_inv - 2 * Q_val) @ (
        cov_param_inv @ y_obs.sum(axis=0) - 2 * Q_val @ mu_val
    )

    log_x_posterior_true = (
        -0.5 * x_val.T @ (-n * cov_param_inv + Q_val) @ x_val
        + x_val.T
        @ (
            Q_val @ mu_val
            - cov_param_inv @ (y_obs - x0_true).sum(axis=0)
            - n * cov_param_inv @ x0_true
        )
        + 0.5 * np.log(np.linalg.det(Q_val))
        + -0.5 * cov_param_val.flatten().T @ np.linalg.inv(cov_cov) @ cov_param_val.flatten()
        - ((d**2) / 2) * np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(cov_cov))
        + -0.5 * mu_val.T @ np.linalg.inv(mu_cov) @ mu_val
        - (d / 2) * np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(mu_cov))
        + -0.5 * (x_val - mu_val).T @ np.linalg.inv(Q_val) @ (x_val - mu_val)
        - (d / 2) * np.log(2 * np.pi)
        - 0.5 * np.log(np.linalg.det(Q_val))
    )
    np.testing.assert_allclose(x0, x0_true, atol=0.1, rtol=0.1)
    np.testing.assert_allclose(log_x_posterior, log_x_posterior_true, atol=0.1, rtol=0.1)
