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

import re
import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

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


def unstable_lbfgs_update_mask_model() -> pm.Model:
    # data and model from: https://github.com/pymc-devs/pymc-extras/issues/445
    # this scenario made LBFGS struggle leading to a lot of rejected iterations, (result.nit being moderate, but only history.count <= 1).
    # this scenario is used to test that the LBFGS history manager is rejecting iterations as expected and PF can run to completion.

    # fmt: off
    inp = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0])

    res = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,1,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,0,1,0]])
    # fmt: on

    n_ordered = res.shape[1]
    coords = {
        "obs": np.arange(len(inp)),
        "inp": np.arange(max(inp) + 1),
        "outp": np.arange(res.shape[1]),
    }
    with pm.Model(coords=coords) as mdl:
        mu = pm.Normal("intercept", sigma=3.5)[None]

        offset = pm.Normal(
            "offset", dims=("inp"), transform=pm.distributions.transforms.ZeroSumTransform([0])
        )

        scale = 3.5 * pm.HalfStudentT("scale", nu=5)
        mu += (scale * offset)[inp]

        phi_delta = pm.Dirichlet("phi_diffs", [1.0] * (n_ordered - 1))
        phi = pt.concatenate([[0], pt.cumsum(phi_delta)])
        s_mu = pm.Normal(
            "stereotype_intercept",
            size=n_ordered,
            transform=pm.distributions.transforms.ZeroSumTransform([-1]),
        )
        fprobs = pm.math.softmax(s_mu[None, :] + phi[None, :] * mu[:, None], axis=-1)

        pm.Multinomial("y_res", p=fprobs, n=np.ones(len(inp)), observed=res, dims=("obs", "outp"))

    return mdl


@pytest.mark.parametrize("jitter", [12.0, 500.0, 1000.0])
def test_unstable_lbfgs_update_mask(capsys, jitter):
    model = unstable_lbfgs_update_mask_model()

    if jitter < 1000:
        with model:
            idata = pmx.fit(
                method="pathfinder",
                jitter=jitter,
                random_seed=4,
            )
        out, err = capsys.readouterr()
        status_pattern = [
            r"INIT_FAILED_LOW_UPDATE_PCT\s+\d+",
            r"LOW_UPDATE_PCT\s+\d+",
            r"LBFGS_FAILED\s+\d+",
            r"SUCCESS\s+\d+",
        ]
        for pattern in status_pattern:
            assert re.search(pattern, out) is not None

    else:
        with pytest.raises(ValueError, match="All paths failed"):
            with model:
                idata = pmx.fit(
                    method="pathfinder",
                    jitter=1000,
                    random_seed=2,
                    num_paths=4,
                )
            out, err = capsys.readouterr()

            status_pattern = [
                r"INIT_FAILED_LOW_UPDATE_PCT\s+2",
                r"LOW_UPDATE_PCT\s+2",
                r"LBFGS_FAILED\s+4",
            ]
            for pattern in status_pattern:
                assert re.search(pattern, out) is not None


@pytest.mark.parametrize("inference_backend", ["pymc", "blackjax"])
@pytest.mark.filterwarnings("ignore:JAXopt is no longer maintained.:DeprecationWarning")
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

    x = x_full[1:]
    g = g_full[1:]
    alpha, s, z = alpha_recover(x_full, g_full)
    beta, gamma = inverse_hessian_factors(alpha, s, z, J)

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
    assert all(phi.shape.eval() == (L, num_samples, N))
    assert all(logq.shape.eval() == (L, num_samples))


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


def test_pathfinder_initvals():
    # Run a model with an ordered transform that will fail unless initvals are in place
    with pm.Model() as mdl:
        pm.Normal("ordered", size=10, transform=pm.distributions.transforms.ordered)
        idata = pmx.fit_pathfinder(initvals={"ordered": np.linspace(0, 1, 10)})

    # Check that the samples are ordered to make sure transform was applied
    assert np.all(
        idata.posterior["ordered"][..., 1:].values > idata.posterior["ordered"][..., :-1].values
    )
