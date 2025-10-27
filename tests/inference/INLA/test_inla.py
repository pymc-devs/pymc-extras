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


@pytest.mark.filterwarnings(r"ignore:INLA is currently experimental")
def test_AR1(rng):
    T = 10
    x = np.zeros((T,))

    # true stationarity:
    true_theta = 0.95
    # true standard deviation of the innovation:
    true_sigma = 2.0
    # true process mean: #
    true_center = 0.0

    for t in range(1, T):
        x[t] = true_theta * x[t - 1] + rng.normal(loc=true_center, scale=true_sigma)

    y_obs = rng.poisson(np.exp(x))

    with pm.Model() as ar1_inla:
        theta = pm.Normal("theta", 0, 1.0)
        tau = pm.Exponential("tau", 0.5)

        x = pm.AR(
            "x", rho=theta, tau=tau, steps=T - 1, init_dist=pm.Normal.dist(0, 100, shape=(T,))
        )

        y = pm.Poisson("y", mu=pm.math.exp(x), observed=y_obs)

        # Use INLA
        idata = pmx.fit(method="INLA", x=x, Q=tau, return_latent_posteriors=False)

    theta_inla = idata.posterior.theta.mean(axis=(0, 1))
    np.testing.assert_allclose(np.array([true_theta]), theta_inla, atol=0.2)


@pytest.mark.filterwarnings(r"ignore:INLA is currently experimental")
def test_3_layer_normal(rng):
    """
    Test INLA against a simple toy problem:

    mu ~ N(mu_mu, I)
    x ~ N(mu, I)
    y ~ N(x, I)

    The mean of the posterior should be the midpoint between mu_mu and mu_true
    """
    n = 10000
    d = 3

    mu_mu = 10 * rng.random(d)
    mu_true = rng.random(d)
    tau = np.identity(d)
    cov = np.linalg.inv(tau)
    y_obs = rng.multivariate_normal(mean=mu_true, cov=cov, size=n)

    with pm.Model() as model:
        mu = pm.MvNormal("mu", mu=mu_mu, tau=tau)
        x = pm.MvNormal("x", mu=mu, tau=tau)
        y = pm.MvNormal("y", mu=x, tau=tau, observed=y_obs)

        idata = pmx.fit(
            method="INLA",
            x=x,
            Q=tau,
            return_latent_posteriors=False,
        )

        posterior_mean_true = (mu_mu + mu_true) / 2
        posterior_mean_inla = idata.posterior.mu.mean(axis=(0, 1))
        np.testing.assert_allclose(posterior_mean_true, posterior_mean_inla, atol=0.1)
