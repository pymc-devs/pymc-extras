#   Copyright 2022 The PyMC Developers
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

import sys

import numpy as np
import pymc as pm
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:compile_pymc was renamed to compile:FutureWarning")

import pymc_extras as pmx


def eight_schools_model() -> pm.Model:
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        obs = pm.Normal("obs", mu=mu + tau * theta, sigma=sigma, shape=J, observed=y)

    return model


@pytest.fixture
def reference_idata():
    model = eight_schools_model()
    with model:
        idata = pmx.fit(
            method="pathfinder",
            num_paths=10,
            jitter=12.0,
            random_seed=41,
            inference_backend="pymc",
        )
    return idata


@pytest.mark.parametrize("inference_backend", ["pymc", "blackjax"])
def test_pathfinder(inference_backend, reference_idata):
    if inference_backend == "blackjax" and sys.platform == "win32":
        pytest.skip("JAX not supported on windows")

    if inference_backend == "blackjax":
        model = eight_schools_model()
        with model:
            idata = pmx.fit(
                method="pathfinder",
                num_paths=10,
                jitter=12.0,
                random_seed=41,
                inference_backend=inference_backend,
            )
    else:
        idata = reference_idata
        np.testing.assert_allclose(idata.posterior["mu"].mean(), 5.0, atol=0.95)
        np.testing.assert_allclose(idata.posterior["tau"].mean(), 4.15, atol=1.35)

    assert idata.posterior["mu"].shape == (1, 1000)
    assert idata.posterior["tau"].shape == (1, 1000)
    assert idata.posterior["theta"].shape == (1, 1000, 8)


@pytest.mark.parametrize("concurrent", ["thread", "process"])
def test_concurrent_results(reference_idata, concurrent):
    model = eight_schools_model()
    with model:
        idata_conc = pmx.fit(
            method="pathfinder",
            num_paths=10,
            jitter=12.0,
            random_seed=41,
            inference_backend="pymc",
            concurrent=concurrent,
        )

    np.testing.assert_allclose(
        reference_idata.posterior.mu.data.mean(),
        idata_conc.posterior.mu.data.mean(),
        atol=0.4,
    )

    np.testing.assert_allclose(
        reference_idata.posterior.tau.data.mean(),
        idata_conc.posterior.tau.data.mean(),
        atol=0.4,
    )


def test_seed(reference_idata):
    model = eight_schools_model()
    with model:
        idata_41 = pmx.fit(
            method="pathfinder",
            num_paths=4,
            jitter=10.0,
            random_seed=41,
            inference_backend="pymc",
        )

        idata_123 = pmx.fit(
            method="pathfinder",
            num_paths=4,
            jitter=10.0,
            random_seed=123,
            inference_backend="pymc",
        )

    assert not np.allclose(idata_41.posterior.mu.data.mean(), idata_123.posterior.mu.data.mean())

    assert np.allclose(idata_41.posterior.mu.data.mean(), idata_41.posterior.mu.data.mean())


def test_bfgs_sample():
    import pytensor.tensor as pt

    from pymc_extras.inference.pathfinder.pathfinder import (
        alpha_recover,
        bfgs_sample,
        inverse_hessian_factors,
    )

    """test BFGS sampling"""
    Lp1, N = 8, 10
    L = Lp1 - 1
    J = 6
    num_samples = 1000

    # mock data
    x_data = np.random.randn(Lp1, N)
    g_data = np.random.randn(Lp1, N)

    # get factors
    x_full = pt.as_tensor(x_data, dtype="float64")
    g_full = pt.as_tensor(g_data, dtype="float64")
    epsilon = 1e-11

    x = x_full[1:]
    g = g_full[1:]
    alpha, S, Z, update_mask = alpha_recover(x_full, g_full, epsilon)
    beta, gamma = inverse_hessian_factors(alpha, S, Z, update_mask, J)

    # sample
    phi, logq = bfgs_sample(
        num_samples=num_samples,
        x=x,
        g=g,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # check shapes
    assert beta.eval().shape == (L, N, 2 * J)
    assert gamma.eval().shape == (L, 2 * J, 2 * J)
    assert phi.eval().shape == (L, num_samples, N)
    assert logq.eval().shape == (L, num_samples)


@pytest.mark.parametrize("importance_sampling", ["psis", "psir", "identity", None])
def test_pathfinder_importance_sampling(importance_sampling):
    model = eight_schools_model()

    num_paths = 4
    num_draws_per_path = 300
    num_draws = 750

    with model:
        idata = pmx.fit(
            method="pathfinder",
            num_paths=num_paths,
            num_draws_per_path=num_draws_per_path,
            num_draws=num_draws,
            maxiter=5,
            random_seed=41,
            inference_backend="pymc",
            importance_sampling=importance_sampling,
        )

    if importance_sampling is None:
        assert idata.posterior["mu"].shape == (num_paths, num_draws_per_path)
        assert idata.posterior["tau"].shape == (num_paths, num_draws_per_path)
        assert idata.posterior["theta"].shape == (num_paths, num_draws_per_path, 8)
    else:
        assert idata.posterior["mu"].shape == (1, num_draws)
        assert idata.posterior["tau"].shape == (1, num_draws)
        assert idata.posterior["theta"].shape == (1, num_draws, 8)
