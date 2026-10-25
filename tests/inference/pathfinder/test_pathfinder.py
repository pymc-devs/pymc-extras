import sys

import numpy as np
import pymc as pm
import pytest

import pymc_extras as pmx

from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig


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
