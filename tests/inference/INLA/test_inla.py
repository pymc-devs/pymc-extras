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


@pytest.fixture(scope="session")
def rng():
    seed = 123
    return np.random.default_rng(seed)


@pytest.mark.filterwarnings(
    r"ignore:INLA is currently experimental",
    r"ignore:Numba will use object mode to run",
)
def test_AR1(rng):
    T = 10
    N = 100
    x = np.zeros((T,))

    # true stationarity:
    true_theta = 0.95
    # true standard deviation of the innovation:
    true_sigma = 2.0
    # true process mean:
    true_center = 0.0

    for t in range(1, T):
        x[t] = true_theta * x[t - 1] + rng.normal(loc=true_center, scale=true_sigma)

    y_obs = rng.poisson(np.exp(x))

    with pm.Model() as ar1_inla:
        theta = pm.Normal("theta", 0, 1.0)
        tau = pm.Exponential("tau", 0.5)

        x = pm.AR("x", rho=theta, tau=tau, steps=T - 1, init_dist=pm.Normal.dist(0, N, shape=(T,)))

        y = pm.Poisson("y", mu=pm.math.exp(x), observed=y_obs)

        # Use INLA
        idata_inla = pmx.fit(method="INLA", x=x, Q=tau, return_latent_posteriors=False)

        # Use direct sampling to get ground truth for variances
        idata_true = pm.sample()

    theta_inla = idata_inla.posterior.theta.mean(axis=(0, 1))

    tau_inla = idata_inla.posterior.tau.mean(axis=(0, 1))
    tau_true = idata_true.posterior.tau.mean(axis=(0, 1))

    theta_var_inla = idata_inla.posterior.theta.var(axis=(0, 1))
    theta_var_true = idata_true.posterior.theta.var(axis=(0, 1))

    tau_var_inla = idata_inla.posterior.tau.var(axis=(0, 1))
    tau_var_true = idata_true.posterior.tau.var(axis=(0, 1))

    np.testing.assert_allclose(theta_inla, np.array([true_theta]), atol=0.2)
    np.testing.assert_allclose(tau_inla, tau_true, atol=0.3)
    np.testing.assert_allclose(theta_var_inla, theta_var_true, atol=0.1)
    np.testing.assert_allclose(tau_var_inla, tau_var_true, atol=0.2)


@pytest.mark.filterwarnings(
    r"ignore:INLA is currently experimental",
    r"ignore:Numba will use object mode to run",
)
def test_3_layer_normal(rng):
    """
    Test INLA against a simple toy problem:

    mu ~ N(mu_mu, tau^-1)
    x ~ N(mu, tau^-1)
    y ~ N(x, tau^-1)
    """
    n = 10000
    d = 3

    mu_mu = 10 * rng.random(d)
    mu_true = rng.random(d)

    tau = np.diag(rng.random(d))
    cov = np.linalg.inv(tau)

    y_obs = rng.multivariate_normal(mean=mu_true, cov=cov, size=n)

    with pm.Model() as model:
        mu = pm.MvNormal("mu", mu=mu_mu, tau=tau)
        x = pm.MvNormal("x", mu=mu, tau=tau)
        y = pm.MvNormal("y", mu=x, tau=tau, observed=y_obs)

        # Use INLA
        idata_inla = pmx.fit(
            method="INLA",
            x=x,
            Q=tau,
            return_latent_posteriors=False,
        )

        # Compare against direct sampling
        idata_true = pm.sample()

    posterior_mean_true = idata_true.posterior.mu.mean(axis=(0, 1))
    posterior_mean_inla = idata_inla.posterior.mu.mean(axis=(0, 1))

    posterior_var_true = idata_true.posterior.mu.var(axis=(0, 1))
    posterior_var_inla = idata_inla.posterior.mu.var(axis=(0, 1))

    np.testing.assert_allclose(posterior_mean_inla, posterior_mean_true, atol=0.1)
    np.testing.assert_allclose(posterior_var_inla, posterior_var_true, atol=0.2)
