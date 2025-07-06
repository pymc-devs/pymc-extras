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
