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
    seed = 12345
    return np.random.default_rng(seed)


def test_non_gaussian_latent(rng):
    """
    INLA should raise an error if trying to use a non-Gaussian latent
    """
    n = 10000

    mu_mu = 0
    mu_true = rng.random()
    tau = 1
    y_obs = rng.normal(loc=mu_true, scale=1 / tau, size=n)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu_mu, tau=tau)
        x = pm.Normal("x", mu=mu, tau=tau)
        y = pm.Normal("y", mu=x, tau=tau, observed=y_obs)

        with pytest.raises(ValueError):
            pmx.fit(method="INLA", x=x)


def test_non_precision_MvNormal(rng):
    """
    INLA should raise an error if trying to use a latent not in precision form
    """
    n = 10000
    d = 3

    mu_mu = np.zeros((d,))
    mu_true = rng.random(d)
    tau = np.identity(d)
    cov = np.linalg.inv(tau)
    y_obs = rng.multivariate_normal(mean=mu_true, cov=cov, size=n)

    with pm.Model() as model:
        mu = pm.MvNormal("mu", mu=mu_mu, tau=tau)
        x = pm.MvNormal("x", mu=mu, cov=cov)
        y = pm.MvNormal("y", mu=x, tau=tau, observed=y_obs)

        with pytest.raises(ValueError):
            pmx.fit(method="INLA", x=x)


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
            temp_kwargs=[n, y_obs],  # TODO REMOVE LATER - DEBUGGING TOOL
            return_latent_posteriors=False,
        )

        posterior_mean_true = (mu_mu + mu_true) / 2
        posterior_mean_inla = idata.posterior.mu.mean(axis=(0, 1))
        np.testing.assert_allclose(posterior_mean_true, posterior_mean_inla, atol=0.1)
