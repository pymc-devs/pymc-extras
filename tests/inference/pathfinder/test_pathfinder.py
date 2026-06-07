import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import pymc_extras as pmx

from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig


def unstable_lbfgs_update_mask_model() -> pm.Model:
    # data and model from: https://github.com/pymc-devs/pymc-extras/issues/445
    # this scenario made LBFGS struggle leading to a lot of rejected iterations, (result.nit being moderate, but only
    # history.count <= 1). this scenario is used to test that the LBFGS history manager is rejecting iterations as
    # expected and PF can run to completion.

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


@pytest.mark.parametrize("jitter", [12.0, 750.0])
def test_unstable_lbfgs_update_mask(jitter):
    model = unstable_lbfgs_update_mask_model()

    if jitter < 750.0:
        # Low jitter values should succeed
        with model:
            idata = pmx.fit(
                method="pathfinder",
                jitter=jitter,
                random_seed=4,
                max_init_retries=0,
                parallel=True,
            )
        # With epsilon=1e-12 the curvature condition is permissive, so we expect at least
        # one path to flag an update-quality issue and at least one to succeed.
        lbfgs_counts = idata.lbfgs.status_counts
        path_counts = idata.pathfinder.path_status_counts
        assert lbfgs_counts.sel(status="LOW_UPDATE_PCT").item() > 0
        assert path_counts.sel(status="SUCCESS").item() > 0

    else:
        # High jitter values (>=750) cause numerical overflow and all paths fail
        with pytest.raises(ValueError, match="(All paths failed|BUG: Failed to iterate)"):
            with model:
                idata = pmx.fit(
                    method="pathfinder",
                    jitter=jitter,
                    random_seed=4,
                    num_paths=4,
                    max_init_retries=0,
                    parallel=True,
                )


def test_pathfinder_pymc(reference_idata):
    idata = reference_idata
    np.testing.assert_allclose(idata.posterior["mu"].mean(), 5.0, atol=0.95)
    np.testing.assert_allclose(idata.posterior["tau"].mean(), 4.15, atol=1.35)

    assert idata.posterior["mu"].shape == (1, 1000)
    assert idata.posterior["tau"].shape == (1, 1000)
    assert idata.posterior["theta"].shape == (1, 1000, 8)


@pytest.mark.parametrize("importance_sampling", ["psis", "psir", "identity", None])
def test_pathfinder_importance_sampling(eight_schools_model, importance_sampling):
    num_paths = 4
    num_draws_per_path = 300
    num_draws = 750

    with eight_schools_model:
        idata = pmx.fit(
            method="pathfinder",
            num_paths=num_paths,
            num_draws_per_path=num_draws_per_path,
            num_draws=num_draws,
            lbfgs_config=LBFGSConfig(maxiter=5),
            random_seed=41,
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


def test_fit_pathfinder_invalid_importance_sampling():
    with pm.Model():
        pm.Normal("x")
        with pytest.raises(ValueError, match="Invalid importance sampling method"):
            pmx.fit(method="pathfinder", importance_sampling="not_a_method")


def test_fit_pathfinder_importance_sampling_case_insensitive(eight_schools_model):
    # "PSIS" is normalised to "psis" rather than rejected
    with eight_schools_model:
        idata = pmx.fit(
            method="pathfinder", num_paths=2, random_seed=41, importance_sampling="PSIS"
        )
    assert idata.pathfinder["importance_sampling_method"].item() == "psis"


def test_pathfinder_initvals():
    # Run a model with an ordered transform that will fail unless initvals are in place
    with pm.Model() as mdl:
        pm.Normal("ordered", size=10, transform=pm.distributions.transforms.ordered)
        idata = pmx.fit_pathfinder(initvals={"ordered": np.linspace(0, 1, 10)})

    # Check that the samples are ordered to make sure transform was applied
    assert np.all(
        idata.posterior["ordered"][..., 1:].values > idata.posterior["ordered"][..., :-1].values
    )


@pytest.mark.filterwarnings("ignore:JAXopt is no longer maintained.:DeprecationWarning")
def test_pathfinder_blackjax(eight_schools_model):
    if sys.platform == "win32":
        pytest.skip("JAX not supported on windows")
    pytest.importorskip("blackjax")

    from pymc_extras.inference import fit_blackjax_pathfinder

    with eight_schools_model:
        idata = fit_blackjax_pathfinder(random_seed=41)

    assert idata.posterior["mu"].shape == (1, 1000)
    assert idata.posterior["tau"].shape == (1, 1000)
    assert idata.posterior["theta"].shape == (1, 1000, 8)
