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

from pymc_extras.inference.laplace_approx.find_map import GradientBackend
from pymc_extras.inference.laplace_approx.laplace import (
    fit_laplace,
    get_conditional_gaussian_approximation,
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
def test_fit_laplace_basic(mode, gradient_backend: GradientBackend):
    # Example originates from Bayesian Data Analyses, 3rd Edition
    # By Andrew Gelman, John Carlin, Hal Stern, David Dunson,
    # Aki Vehtari, and Donald Rubin.
    # See section. 4.1

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size
    draws = 100000

    with pm.Model() as m:
        mu = pm.Flat("mu")
        logsigma = pm.Flat("logsigma")

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
            optimizer_kwargs=dict(tol=1e-20),
        )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (n,)
    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [y.mean(), np.log(y.std())]
    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])

    np.testing.assert_allclose(idata.posterior["mu"].mean(), bda_map[0], atol=1)
    np.testing.assert_allclose(idata.posterior["logsigma"].mean(), bda_map[1], rtol=1e-3)

    np.testing.assert_allclose(idata.fit["covariance_matrix"].values, bda_cov, rtol=1e-3, atol=1e-3)


def test_fit_laplace_outside_model_context():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.Exponential("sigma", 1)
        y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=np.random.normal(size=10))

    idata = fit_laplace(
        model=m,
        optimize_method="L-BFGS-B",
        use_grad=True,
        progressbar=False,
        chains=1,
        draws=100,
    )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")

    assert idata.posterior["mu"].shape == (1, 100)


@pytest.mark.parametrize(
    "include_transformed", [True, False], ids=["include_transformed", "no_transformed"]
)
def test_fit_laplace_coords(include_transformed, rng):
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
            method="laplace",
            optimize_method="trust-ncg",
            chains=1,
            draws=1000,
            optimizer_kwargs=dict(tol=1e-20),
            include_transformed=include_transformed,
        )

    np.testing.assert_allclose(
        idata.posterior.mu.mean(dim=["chain", "draw"]).values, np.full((3,), 3), atol=0.5
    )
    np.testing.assert_allclose(
        idata.posterior.sigma.mean(dim=["chain", "draw"]).values, np.full((3,), 1.5), atol=0.3
    )

    assert idata.fit.rows.values.tolist() == [
        "mu[A]",
        "mu[B]",
        "mu[C]",
        "sigma_log__[A]",
        "sigma_log__[B]",
        "sigma_log__[C]",
    ]

    assert hasattr(idata, "unconstrained_posterior") == include_transformed
    if include_transformed:
        assert "sigma_log__" in idata.unconstrained_posterior
        assert "city" in idata.unconstrained_posterior.coords


def test_fit_laplace_ragged_coords(rng):
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
        )

    # These should have been dropped when the laplace idata was created
    assert "laplace_approximation" not in list(idata.posterior.data_vars.keys())
    assert "unpacked_var_names" not in list(idata.posterior.coords.keys())

    assert idata["posterior"].beta.shape[-2:] == (3, 2)
    assert idata["posterior"].sigma.shape[-1:] == (3,)

    # Check that everything got unraveled correctly -- feature 0 should be strictly negative, feature 1
    # strictly positive
    assert (idata["posterior"].beta.sel(feature=0).to_numpy() < 0).all()
    assert (idata["posterior"].beta.sel(feature=1).to_numpy() > 0).all()


def test_model_with_nonstandard_dimensionality(rng):
    y_obs = np.concatenate(
        [rng.normal(-1, 2, size=150), rng.normal(3, 1, size=350), rng.normal(5, 4, size=50)]
    )

    with pm.Model(coords={"obs_idx": range(y_obs.size), "class": ["A", "B", "C"]}) as model:
        y = pm.Data("y", y_obs, dims=["obs_idx"])

        mu = pm.Normal("mu", mu=1, sigma=3, dims=["class"])
        sigma = pm.HalfNormal("sigma", sigma=3, dims=["class"])

        w = pm.Dirichlet(
            "w",
            a=np.ones(
                3,
            ),
            dims=["class"],
        )
        class_idx = pm.Categorical("class_idx", p=w, dims=["obs_idx"])
        y_hat = pm.Normal(
            "obs", mu=mu[class_idx], sigma=sigma[class_idx], observed=y, dims=["obs_idx"]
        )

    with pmx.marginalize(model, [class_idx]):
        idata = pmx.fit_laplace(progressbar=False)

    # The dirichlet value variable has a funky shape; check that it got a default
    assert "w_simplex___dim_0" in list(idata.unconstrained_posterior.w_simplex__.coords.keys())
    assert "class" not in list(idata.unconstrained_posterior.w_simplex__.coords.keys())
    assert len(idata.unconstrained_posterior.coords["w_simplex___dim_0"]) == 2

    # On the other hand, check that the actual w has the correct dims
    assert "class" in list(idata.posterior.w.coords.keys())

    # The log transform is 1-to-1, so it should have the same dims as the original rv
    assert "class" in list(idata.unconstrained_posterior.sigma_log__.coords.keys())


def test_laplace_nonstandard_dims_2d():
    true_P = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.4, 0.4]])
    y_obs = pm.draw(
        pmx.DiscreteMarkovChain.dist(
            P=true_P,
            init_dist=pm.Categorical.dist(
                logit_p=np.ones(
                    3,
                )
            ),
            shape=(100, 5),
        )
    )

    with pm.Model(
        coords={
            "time": range(y_obs.shape[0]),
            "state": list("ABC"),
            "next_state": list("ABC"),
            "unit": [1, 2, 3, 4, 5],
        }
    ) as model:
        y = pm.Data("y", y_obs, dims=["time", "unit"])
        init_dist = pm.Categorical.dist(
            logit_p=np.ones(
                3,
            )
        )
        P = pm.Dirichlet("P", a=np.eye(3) * 2 + 1, dims=["state", "next_state"])
        y_hat = pmx.DiscreteMarkovChain(
            "y_hat", P=P, init_dist=init_dist, dims=["time", "unit"], observed=y_obs
        )

        idata = pmx.fit_laplace(progressbar=True)

        # The simplex transform should drop from the right-most dimension, so the left dimension should be unmodified
        assert "state" in list(idata.unconstrained_posterior.P_simplex__.coords.keys())

        # The mutated dimension should be unknown coords
        assert "P_simplex___dim_1" in list(idata.unconstrained_posterior.P_simplex__.coords.keys())

        assert idata.unconstrained_posterior.P_simplex__.shape[-2:] == (3, 2)


def test_laplace_nonscalar_rv_without_dims():
    with pm.Model(coords={"test": ["A", "B", "C"]}) as model:
        x_loc = pm.Normal("x_loc", mu=0, sigma=1, dims=["test"])
        x = pm.Normal("x", mu=x_loc, sigma=1, shape=(2, 3))
        y = pm.Normal("y", mu=x, sigma=1, observed=np.random.randn(10, 2, 3))

        idata = pmx.fit_laplace(progressbar=False)

    assert idata.posterior["x"].shape == (2, 500, 2, 3)
    assert all(f"x_dim_{i}" in idata.posterior.coords for i in range(2))
    assert idata.fit.rows.values.tolist() == [
        "x_loc[A]",
        "x_loc[B]",
        "x_loc[C]",
        "x[0,0]",
        "x[0,1]",
        "x[0,2]",
        "x[1,0]",
        "x[1,1]",
        "x[1,2]",
    ]


# Test these three optimizers because they are either special cases for H_inv (BFGS, L-BFGS-B) or are
# gradient free and require re-compilation of hessp (powell).
@pytest.mark.parametrize("optimizer_method", ["BFGS", "L-BFGS-B", "powell"])
def test_laplace_scalar_basinhopping(optimizer_method):
    # Example model from Statistical Rethinking
    data = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

    with pm.Model():
        p = pm.Uniform("p", 0, 1)
        w = pm.Binomial("w", n=len(data), p=p, observed=data.sum())

        idata_laplace = pmx.fit_laplace(
            optimize_method="basinhopping",
            optimizer_kwargs={"minimizer_kwargs": {"method": optimizer_method}, "niter": 1},
            progressbar=False,
        )

    assert idata_laplace.fit.mean_vector.shape == (1,)
    assert idata_laplace.fit.covariance_matrix.shape == (1, 1)

    np.testing.assert_allclose(
        idata_laplace.posterior.p.mean(dim=["chain", "draw"]), data.mean(), atol=0.1
    )


def test_get_conditional_gaussian_approximation():
    """
    Consider the trivial case of:

    y | x ~ N(x, cov_param)
    x | param ~ N(mu_param, Q^-1)

    cov_param ~ N(cov_mu, cov_cov)
    mu_param ~ N(mu_mu, mu_cov)
    Q ~ N(Q_mu, Q_cov)

    This has an analytic solution at the mode which we can compare against.
    """
    rng = np.random.default_rng(12345)
    n = 10000
    d = 10

    # Initialise arrays
    mu_true = rng.random(d)
    cov_true = np.diag(rng.random(d))
    Q_val = np.diag(rng.random(d))
    cov_param_val = np.diag(rng.random(d))

    x_val = rng.random(d)
    mu_val = rng.random(d)

    mu_mu = rng.random(d)
    mu_cov = np.diag(np.ones(d))
    cov_mu = rng.random(d**2)
    cov_cov = np.diag(np.ones(d**2))
    Q_mu = rng.random(d**2)
    Q_cov = np.diag(np.ones(d**2))

    with pm.Model() as model:
        y_obs = rng.multivariate_normal(mean=mu_true, cov=cov_true, size=n)

        mu_param = pm.MvNormal("mu_param", mu=mu_mu, cov=mu_cov)
        cov_param = pm.MvNormal("cov_param", mu=cov_mu, cov=cov_cov)
        Q = pm.MvNormal("Q", mu=Q_mu, cov=Q_cov)

        # Pytensor currently doesn't support autograd for pt inverses, so we use a numeric Q instead
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

    # Get analytic values of the mode and Laplace-approximated log posterior
    cov_param_inv = np.linalg.inv(cov_param_val)

    x0_true = np.linalg.inv(n * cov_param_inv + 2 * Q_val) @ (
        cov_param_inv @ y_obs.sum(axis=0) + 2 * Q_val @ mu_val
    )

    jac_true = cov_param_inv @ (y_obs - x0_true).sum(axis=0) - Q_val @ (x0_true - mu_val)
    hess_true = -n * cov_param_inv - Q_val

    log_x_posterior_laplace_true = (
        -0.5 * x_val.T @ (-hess_true + Q_val) @ x_val
        + x_val.T @ (Q_val @ mu_val + jac_true - hess_true @ x0_true)
        + 0.5 * np.log(np.linalg.det(Q_val))
    )

    np.testing.assert_allclose(x0, x0_true, atol=0.1, rtol=0.1)
    np.testing.assert_allclose(log_x_posterior, log_x_posterior_laplace_true, atol=0.1, rtol=0.1)
