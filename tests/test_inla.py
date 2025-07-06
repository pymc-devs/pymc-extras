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
import pytensor

from pymc_extras.inference.inla import get_conditional_gaussian_approximation


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

        x = pm.MvNormal("x", mu=mu_param, tau=Q_val)

        y = pm.MvNormal(
            "y",
            mu=x,
            cov=cov_param.reshape((d, d)),
            observed=y_obs,
        )

        args = model.continuous_value_vars + model.discrete_value_vars

        # logp(x | y, params)
        x0, x_g = get_conditional_gaussian_approximation(
            x=model.rvs_to_values[x],
            Q=Q.reshape((d, d)),
            mu=mu_param,
            optimizer_kwargs={"tol": 1e-8},
        )

    cga = pytensor.function(args, [x0, pm.logp(x_g, model.rvs_to_values[x])])

    x0, log_x_posterior = cga(
        x=x_val, mu_param=mu_val, cov_param=cov_param_val.flatten(), Q=Q_val.flatten()
    )

    # Get analytic values of the mode and Laplace-approximated log posterior
    cov_param_inv = np.linalg.inv(cov_param_val)

    x0_true = np.linalg.inv(n * cov_param_inv + 2 * Q_val) @ (
        cov_param_inv @ y_obs.sum(axis=0) + 2 * Q_val @ mu_val
    )

    hess_true = -n * cov_param_inv - Q_val
    tau_true = Q_val - hess_true

    log_x_taylor = (
        -0.5 * (x_val - x0_true).T @ tau_true @ (x_val - x0_true)
        + 0.5 * np.log(np.linalg.det(tau_true))
        - 0.5 * d * np.log(2 * np.pi)
    )

    np.testing.assert_allclose(x0, x0_true, atol=0.1, rtol=0.1)
    np.testing.assert_allclose(log_x_posterior, log_x_taylor, atol=0.1, rtol=0.1)
