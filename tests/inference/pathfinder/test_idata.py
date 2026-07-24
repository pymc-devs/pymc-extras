"""Tests for pathfinder InferenceData integration."""

from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from pymc_extras.inference.pathfinder import pathfinder_report
from pymc_extras.inference.pathfinder.idata import (
    add_pathfinder_to_inference_data,
    convert_flat_trace_to_idata,
)
from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus
from pymc_extras.inference.pathfinder.results import PathStatus
from tests.inference.pathfinder.conftest import (
    MockMultiPathfinderResult,
    make_config,
    make_result,
)


def test_get_param_coords_fallback():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import get_param_coords

    assert get_param_coords(None, 3) == ["0", "1", "2"]


def test_get_param_coords_from_model():
    pytest.importorskip("pymc")
    import pymc as pm

    from pymc_extras.inference.pathfinder.idata import get_param_coords

    with pm.Model() as model:
        pm.Normal("x", 0, 1)
        pm.Normal("y", 0, 1, shape=2)

    assert get_param_coords(model, 3) == ["x", "y[0]", "y[1]"]


def test_get_param_coords_uses_model_dims():
    pytest.importorskip("pymc")
    import pymc as pm

    from pymc_extras.inference.pathfinder.idata import get_param_coords

    with pm.Model(coords={"feature": ["a", "b"]}) as model:
        pm.Normal("beta", 0, 1, dims="feature")

    assert get_param_coords(model, 2) == ["beta[a]", "beta[b]"]


def test_status_counter_to_dataarray():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _status_counter_to_dataarray

    counter = Counter({LBFGSStatus.CONVERGED: 2, LBFGSStatus.MAX_ITER_REACHED: 1})
    da = _status_counter_to_dataarray(counter, LBFGSStatus)

    assert isinstance(da, xr.DataArray)
    assert da.sel(status="CONVERGED").item() == 2
    assert da.sel(status="MAX_ITER_REACHED").item() == 1


def test_determine_num_paths():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _determine_num_paths

    # From lbfgs_niter
    assert _determine_num_paths(make_result(lbfgs_niter=np.array([10, 15, 12]))) == 3

    # From logP when lbfgs_niter is None
    r = make_result(lbfgs_niter=None, elbo_argmax=None)
    assert _determine_num_paths(r) == 3

    # Status counter fallback
    r = make_result(
        samples=None,
        lbfgs_niter=None,
        elbo_argmax=None,
        logP=None,
        logQ=None,
        lbfgs_status=Counter({LBFGSStatus.CONVERGED: 2}),
    )
    assert _determine_num_paths(r) == 2


def test_pathfinder_dataset_basic():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _pathfinder_dataset

    ds = _pathfinder_dataset(make_result(), model=None)

    # Top-level summary
    assert "num_paths" in ds.data_vars
    assert "num_draws" in ds.data_vars
    assert "compile_time" in ds.data_vars
    assert "compute_time" in ds.data_vars
    assert "total_time" in ds.data_vars
    assert "pareto_k" in ds.data_vars
    assert "path_status_counts" in ds.data_vars
    assert "importance_sampling_method" in ds.data_vars
    assert "all_paths_failed" in ds.data_vars

    # logP/logQ stats
    for key in ("logP_mean", "logP_std", "logP_max", "logQ_mean", "logQ_std", "logQ_max"):
        assert key in ds.data_vars

    # Per-path arrays
    assert "path" in ds.dims
    assert ds.sizes["path"] == 3
    assert ds["elbo_argmax"].shape == (3,)
    assert ds["inv_hessian_diag"].shape == (3, 2)
    assert ds["final_sample"].shape == (3, 2)

    # Pathfinder-side config (pathfinder-specific bits, not lbfgs)
    assert "num_draws_per_path" in ds.data_vars
    assert "num_elbo_draws" in ds.data_vars
    assert "jitter" in ds.data_vars

    # No lbfgs fields here — they live in the lbfgs group
    for key in ("maxcor", "maxiter", "ftol", "gtol", "maxls", "epsilon", "status_counts"):
        assert key not in ds.data_vars


def test_num_successful_paths_counts_paths_not_draws():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _pathfinder_dataset

    # After importance sampling, samples collapse to (num_draws, N), so samples.shape[0] is the
    # draw count. num_successful_paths must report the path count instead.
    num_paths, num_draws, n_params = 4, 1000, 2
    result = make_result(
        samples=np.zeros((num_draws, n_params)),
        logP=np.zeros((num_paths, 50)),
        logQ=np.zeros((num_paths, 50)),
        lbfgs_niter=np.arange(num_paths),
        elbo_argmax=np.arange(num_paths),
        inv_hessian_diag=np.ones((num_paths, n_params)),
    )

    ds = _pathfinder_dataset(result, model=None)

    assert int(ds["num_successful_paths"]) == num_paths
    assert ds.sizes["path"] == num_paths


def test_idata_reports_requested_counts(reference_idata):
    # num_paths/num_draws are populated on the result, so a real fit's idata group exposes them.
    pf = reference_idata.pathfinder
    assert int(pf["num_paths"]) == 10
    assert int(pf["num_draws"]) > 0


def test_lbfgs_dataset():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _lbfgs_dataset

    ds = _lbfgs_dataset(make_result())

    # Per-path L-BFGS niter
    assert ds["niter"].shape == (3,)
    assert "niter_mean" in ds.data_vars
    assert "niter_std" in ds.data_vars

    # Status counts
    assert "status_counts" in ds.data_vars
    assert ds["status_counts"].sel(status="CONVERGED").item() == 3

    # L-BFGS config fields
    for key in ("maxcor", "maxiter", "ftol", "gtol", "maxls", "epsilon"):
        assert key in ds.data_vars
    assert ds["maxcor"].item() == 5
    assert ds["maxiter"].item() == 1000


def test_add_pathfinder_to_inference_data():
    from pymc_extras.inference.pathfinder.idata import add_pathfinder_to_inference_data

    posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
    idata = xr.DataTree.from_dict({"posterior": posterior})

    add_pathfinder_to_inference_data(idata, make_result(), model=None)

    groups = list(idata.groups)
    assert "/posterior" in groups
    assert "/pathfinder" in groups
    assert "/lbfgs" in groups

    assert idata["pathfinder"]["num_paths"].item() == 3
    assert idata["lbfgs"]["niter"].shape == (3,)
    assert idata["lbfgs"]["maxcor"].item() == 5


def test_determine_num_paths_raises_when_undeterminable():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _determine_num_paths

    # No per-path arrays and empty status counters: nothing records how many paths ran.
    with pytest.raises(ValueError, match="Cannot determine number of paths"):
        _determine_num_paths(MockMultiPathfinderResult())


def test_pathfinder_dataset_all_paths_failed():
    pytest.importorskip("arviz")
    from pymc_extras.inference.pathfinder.idata import _pathfinder_dataset

    result = MockMultiPathfinderResult(
        all_paths_failed=True,
        samples=None,
        logP=None,
        logQ=None,
        lbfgs_niter=None,
        elbo_argmax=None,
        inv_hessian_diag=None,
        num_paths=4,
        num_draws=0,
        lbfgs_status=Counter({LBFGSStatus.INIT_FAILED: 4}),
        path_status=Counter({PathStatus.LBFGS_FAILED: 4}),
        pathfinder_config=make_config(),
    )

    ds = _pathfinder_dataset(result, model=None)

    assert ds["all_paths_failed"].item()
    # No per-path arrays are built when every path failed
    assert "path" not in ds.dims
    for key in ("final_sample", "inv_hessian_diag", "elbo_argmax", "num_successful_paths"):
        assert key not in ds.data_vars
    # The status summary is still recorded so the failure is diagnosable
    assert "path_status_counts" in ds.data_vars


def test_mock_result_is_subset_of_real_schema():
    """Guard against MockMultiPathfinderResult drifting from the real dataclass it stands in for."""
    from dataclasses import fields

    from pymc_extras.inference.pathfinder.results import MultiPathfinderResult

    mock_fields = {f.name for f in fields(MockMultiPathfinderResult)}
    real_fields = {f.name for f in fields(MultiPathfinderResult)}
    assert mock_fields <= real_fields, (
        f"mock has fields absent from MultiPathfinderResult: {mock_fields - real_fields}"
    )


def test_pathfinder_report_smoke():
    """pathfinder_report prints without raising on a fully populated idata."""
    posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
    idata = xr.DataTree.from_dict({"posterior": posterior})
    add_pathfinder_to_inference_data(idata, make_result(), model=None)

    pathfinder_report(idata)


def test_pathfinder_report_missing_group():
    idata = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"x": (["chain", "draw"], np.zeros((1, 1)))})}
    )
    with pytest.raises(ValueError, match="missing the 'pathfinder' group"):
        pathfinder_report(idata)


_POSTPROC_N = 3  # a (scalar) + b (shape 2)


@pytest.fixture
def postproc_model():
    with pm.Model() as m:
        pm.Normal("a")
        pm.Normal("b", shape=2)
    return m


@pytest.mark.parametrize("vectorize", [False, True])
def test_convert_flat_trace_collapsed(postproc_model, vectorize):
    """With importance sampling the path axis is gone: (num_draws, N) -> (1, num_draws, ...)."""
    num_draws = 20
    samples = np.random.default_rng(0).normal(size=(num_draws, _POSTPROC_N))

    idata = convert_flat_trace_to_idata(
        samples, model=postproc_model, importance_sampling="psis", vectorize=vectorize
    )

    assert idata.posterior["a"].shape == (1, num_draws)
    assert idata.posterior["b"].shape == (1, num_draws, 2)


@pytest.mark.parametrize("vectorize", [False, True])
def test_convert_flat_trace_keeps_paths(postproc_model, vectorize):
    """importance_sampling=None keeps the path axis: (num_paths, num_pdraws, N) round-trips."""
    num_paths, num_pdraws = 4, 10
    samples = np.random.default_rng(0).normal(size=(num_paths, num_pdraws, _POSTPROC_N))

    idata = convert_flat_trace_to_idata(
        samples, model=postproc_model, importance_sampling=None, vectorize=vectorize
    )

    assert idata.posterior["a"].shape == (num_paths, num_pdraws)
    assert idata.posterior["b"].shape == (num_paths, num_pdraws, 2)


@pytest.mark.parametrize("vectorize", [False, True])
def test_convert_flat_trace_jax_postprocessing_with_dynamic_shape(vectorize):
    """Postprocessing whose Deterministic has a data-dependent shape must lower under JAX.

    ``repeat`` gives the batched Deterministic an output dimension JAX cannot size until runtime.
    When the batch dims of the postprocessing graph are left symbolic, that dimension reaches
    JAX's reshape as a tracer and compilation fails; concrete batch sizes let it fold. This must
    hold for both postprocessing paths, and the result must match the default backend.
    """
    pytest.importorskip("jax")

    rows, cols, num_draws, repeats = 4, 3, 7, 2
    with pm.Model() as model:
        weight = pm.Normal("weight", 0, 1, shape=(rows, cols))
        pm.Deterministic("weight_repeated", pt.repeat(weight, repeats, axis=0))

    samples = np.random.default_rng(0).normal(size=(num_draws, rows * cols))

    default_idata = convert_flat_trace_to_idata(
        samples, model=model, importance_sampling="psis", vectorize=vectorize
    )
    jax_idata = convert_flat_trace_to_idata(
        samples,
        model=model,
        importance_sampling="psis",
        vectorize=vectorize,
        compile_kwargs={"mode": "JAX"},
    )

    assert jax_idata.posterior["weight_repeated"].shape == (1, num_draws, rows * repeats, cols)
    np.testing.assert_allclose(
        jax_idata.posterior["weight_repeated"].values,
        default_idata.posterior["weight_repeated"].values,
        rtol=1e-6,
    )
