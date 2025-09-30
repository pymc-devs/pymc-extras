"""Tests for pathfinder InferenceData integration."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

# Mock objects for testing without full dependencies
from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus
from pymc_extras.inference.pathfinder.pathfinder import PathfinderConfig, PathStatus


@dataclass
class MockPathfinderResult:
    """Mock PathfinderResult for testing."""

    samples: np.ndarray = None
    logP: np.ndarray = None
    logQ: np.ndarray = None
    lbfgs_niter: np.ndarray = None
    elbo_argmax: np.ndarray = None
    lbfgs_status: LBFGSStatus = LBFGSStatus.CONVERGED
    path_status: PathStatus = PathStatus.SUCCESS


@dataclass
class MockMultiPathfinderResult:
    """Mock MultiPathfinderResult for testing."""

    samples: np.ndarray = None
    logP: np.ndarray = None
    logQ: np.ndarray = None
    lbfgs_niter: np.ndarray = None
    elbo_argmax: np.ndarray = None
    lbfgs_status: Counter = None
    path_status: Counter = None
    importance_sampling: str = "psis"
    warnings: list = None
    pareto_k: float = None
    num_paths: int = None
    num_draws: int = None
    pathfinder_config: PathfinderConfig = None
    compile_time: float = None
    compute_time: float = None
    all_paths_failed: bool = False

    def __post_init__(self):
        if self.lbfgs_status is None:
            self.lbfgs_status = Counter()
        if self.path_status is None:
            self.path_status = Counter()
        if self.warnings is None:
            self.warnings = []


class TestPathfinderResultToXarray:
    """Tests for converting single PathfinderResult to xarray."""

    def test_single_result_basic_conversion(self):
        """Test basic conversion of PathfinderResult to xarray Dataset."""
        # Skip if dependencies not available
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import pathfinder_result_to_xarray

        # Create mock result
        result = MockPathfinderResult(
            samples=np.random.normal(0, 1, (1, 100, 2)),
            logP=np.random.normal(-10, 1, (1, 100)),
            logQ=np.random.normal(-11, 1, (1, 100)),
            lbfgs_niter=np.array([50]),
            elbo_argmax=np.array([25]),
        )

        ds = pathfinder_result_to_xarray(result, model=None)

        # Check basic structure
        assert isinstance(ds, xr.Dataset)
        assert "lbfgs_niter" in ds.data_vars
        assert "elbo_argmax" in ds.data_vars
        assert "lbfgs_status_code" in ds.data_vars
        assert "lbfgs_status_name" in ds.data_vars
        assert "path_status_code" in ds.data_vars
        assert "path_status_name" in ds.data_vars

        # Check attributes
        assert "lbfgs_status" in ds.attrs
        assert "path_status" in ds.attrs

    def test_parameter_coordinates(self):
        """Test parameter coordinate generation."""
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import get_param_coords

        # Test fallback to indices when no model
        coords = get_param_coords(None, 3)
        assert coords == ["0", "1", "2"]

    def test_get_param_coords_fail_fast(self):
        """Test that get_param_coords fails fast on model errors."""
        pytest.importorskip("arviz")
        pytest.importorskip("pymc")

        import pymc as pm

        from pymc_extras.inference.pathfinder.idata import get_param_coords

        # Test that it fails when model.initial_point() raises an exception
        with pm.Model() as broken_model:
            # Shape mismatch causes initial_point to fail
            x = pm.Normal("x", mu=[0, 1], sigma=1, shape=1)  # incompatible shapes

        with pytest.raises(ValueError, match=r".*incompatible.*"):
            get_param_coords(broken_model, 2)

        # Test that it works correctly with valid models
        with pm.Model() as valid_model:
            x = pm.Normal("x", 0, 1)  # scalar
            y = pm.Normal("y", 0, 1, shape=2)  # vector

        coords = get_param_coords(valid_model, 3)
        expected = ["x", "y[0]", "y[1]"]
        assert coords == expected

    def test_multipath_coordinate_dimensions_with_importance_sampling(self):
        """Test that path dimensions are calculated correctly when importance sampling collapses samples."""
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import multipathfinder_result_to_xarray

        # Mock a multi-path result where importance sampling has collapsed the samples
        # but per-path diagnostics are still available
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (1000, 2)),  # Collapsed: (total_draws, n_params)
            lbfgs_niter=np.array([50, 45, 55, 40]),  # Per-path: (4,)
            elbo_argmax=np.array([25, 30, 20, 35]),  # Per-path: (4,)
            logP=np.random.normal(-10, 1, (4, 250)),  # Per-path, per-draw: (4, 250)
            logQ=np.random.normal(-11, 1, (4, 250)),  # Per-path, per-draw: (4, 250)
            lbfgs_status=Counter({LBFGSStatus.CONVERGED: 4}),
            path_status=Counter({PathStatus.SUCCESS: 4}),
            num_paths=4,
            num_draws=1000,
        )

        summary_ds, paths_ds = multipathfinder_result_to_xarray(result, model=None)

        # Check that path dimension is correctly inferred as 4 (not 1000)
        assert paths_ds is not None
        assert "path" in paths_ds.dims
        assert paths_ds.sizes["path"] == 4  # Should be 4 paths, not 1000 samples

        # Check that per-path data has correct shape
        assert "lbfgs_niter" in paths_ds.data_vars
        assert paths_ds.lbfgs_niter.shape == (4,)
        assert "elbo_argmax" in paths_ds.data_vars
        assert paths_ds.elbo_argmax.shape == (4,)

    def test_determine_num_paths_helper(self):
        """Test the _determine_num_paths helper function."""
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import _determine_num_paths

        # Test with lbfgs_niter
        result1 = MockMultiPathfinderResult(
            lbfgs_niter=np.array([10, 15, 12]),
            elbo_argmax=None,
        )
        assert _determine_num_paths(result1) == 3

        # Test with logP when lbfgs_niter is None
        result2 = MockMultiPathfinderResult(
            lbfgs_niter=None,
            logP=np.random.normal(0, 1, (5, 100)),  # 5 paths, 100 samples each
        )
        assert _determine_num_paths(result2) == 5

        # Test fallback to status counters
        result3 = MockMultiPathfinderResult(
            lbfgs_niter=None,
            elbo_argmax=None,
            logP=None,
            logQ=None,
            lbfgs_status=Counter({LBFGSStatus.CONVERGED: 2}),
        )
        assert _determine_num_paths(result3) == 2

    def test_status_counter_conversion(self):
        """Test conversion of status counters to DataArray."""
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import _status_counter_to_dataarray

        counter = Counter({LBFGSStatus.CONVERGED: 2, LBFGSStatus.MAX_ITER_REACHED: 1})
        da = _status_counter_to_dataarray(counter, LBFGSStatus)

        assert isinstance(da, xr.DataArray)
        assert "status" in da.dims
        assert da.sel(status="CONVERGED").item() == 2
        assert da.sel(status="MAX_ITER_REACHED").item() == 1


class TestMultiPathfinderResultToXarray:
    """Tests for converting MultiPathfinderResult to xarray."""

    def test_multi_result_conversion(self):
        """Test conversion of MultiPathfinderResult to datasets."""
        pytest.importorskip("arviz")

        from pymc_extras.inference.pathfinder.idata import multipathfinder_result_to_xarray

        # Create mock multi-path result
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (3, 100, 2)),  # 3 paths, 100 draws, 2 params
            logP=np.random.normal(-10, 1, (3, 100)),
            logQ=np.random.normal(-11, 1, (3, 100)),
            lbfgs_niter=np.array([50, 45, 55]),
            elbo_argmax=np.array([25, 30, 20]),
            lbfgs_status=Counter({LBFGSStatus.CONVERGED: 3}),
            path_status=Counter({PathStatus.SUCCESS: 3}),
            num_paths=3,
            num_draws=300,
            compile_time=1.5,
            compute_time=10.2,
            pareto_k=0.5,
        )

        summary_ds, paths_ds = multipathfinder_result_to_xarray(result, model=None)

        # Check summary dataset
        assert isinstance(summary_ds, xr.Dataset)
        assert "num_paths" in summary_ds.data_vars
        assert "num_draws" in summary_ds.data_vars
        assert "compile_time" in summary_ds.data_vars
        assert "compute_time" in summary_ds.data_vars
        assert "total_time" in summary_ds.data_vars
        assert "pareto_k" in summary_ds.data_vars

        # Check per-path dataset
        assert isinstance(paths_ds, xr.Dataset)
        assert "path" in paths_ds.dims
        assert paths_ds.sizes["path"] == 3
        assert "lbfgs_niter" in paths_ds.data_vars
        assert "elbo_argmax" in paths_ds.data_vars


class TestAddPathfinderToInferenceData:
    """Tests for adding pathfinder results to InferenceData."""

    def test_add_to_inference_data(self):
        """Test adding pathfinder results to InferenceData object."""
        pytest.importorskip("arviz")

        import arviz as az

        from pymc_extras.inference.pathfinder.idata import add_pathfinder_to_inference_data

        # Create mock InferenceData
        posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
        idata = az.InferenceData(posterior=posterior)

        # Create mock result with proper single-path status values
        # (Note: MockMultiPathfinderResult isn't a real MultiPathfinderResult,
        #  so it will be treated as single-path by add_pathfinder_to_inference_data)
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (2, 50, 1)),
            num_paths=2,
            num_draws=100,
            lbfgs_status=LBFGSStatus.CONVERGED,  # Single enum value, not Counter
            path_status=PathStatus.SUCCESS,  # Single enum value, not Counter
        )

        # Add pathfinder groups
        idata_updated = add_pathfinder_to_inference_data(idata, result, model=None)

        # Check groups were added
        # Note: Since MockMultiPathfinderResult is not a real MultiPathfinderResult,
        # it gets treated as a single-path result, so only 'pathfinder' group is added
        groups = list(idata_updated.groups())
        assert "posterior" in groups
        assert "pathfinder" in groups
        # pathfinder_paths is only created for true MultiPathfinderResult instances


class TestDiagnosticsAndConfigGroups:
    """Tests for diagnostics and config group functionality."""

    def test_config_group_creation(self):
        """Test that config group is created when PathfinderConfig is available."""
        pytest.importorskip("arviz")

        import arviz as az

        from pymc_extras.inference.pathfinder.idata import (
            _build_config_dataset,
            add_pathfinder_to_inference_data,
        )
        from pymc_extras.inference.pathfinder.pathfinder import PathfinderConfig

        # Create mock InferenceData
        posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
        idata = az.InferenceData(posterior=posterior)

        # Create mock config
        config = PathfinderConfig(
            num_draws=1000,
            maxcor=5,
            maxiter=100,
            ftol=1e-5,
            gtol=1e-8,
            maxls=1000,
            jitter=2.0,
            epsilon=1e-8,
            num_elbo_draws=10,
        )

        # Test config dataset creation
        config_ds = _build_config_dataset(config)
        assert isinstance(config_ds, xr.Dataset)
        assert "num_draws" in config_ds.data_vars
        assert "maxcor" in config_ds.data_vars
        assert "maxiter" in config_ds.data_vars
        assert config_ds.num_draws.values == 1000
        assert config_ds.maxcor.values == 5

        # Test with MultiPathfinderResult that has config
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (2, 50, 1)),
            num_paths=2,
            pathfinder_config=config,
        )

        # Add pathfinder groups
        idata_updated = add_pathfinder_to_inference_data(
            idata, result, model=None, config_group="test_config"
        )

        # Check config group was added
        groups = list(idata_updated.groups())
        assert "test_config" in groups
        assert "num_draws" in idata_updated.test_config.data_vars

    def test_diagnostics_group_creation(self):
        """Test that diagnostics group is created when store_diagnostics=True."""
        pytest.importorskip("arviz")

        import arviz as az

        from pymc_extras.inference.pathfinder.idata import (
            _build_diagnostics_dataset,
            add_pathfinder_to_inference_data,
        )

        # Create mock InferenceData
        posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
        idata = az.InferenceData(posterior=posterior)

        # Create mock result with diagnostic data
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (2, 50, 3)),  # 2 paths, 50 draws, 3 params
            logP=np.random.normal(-10, 1, (2, 50)),  # Per-path, per-draw logP
            logQ=np.random.normal(-11, 1, (2, 50)),  # Per-path, per-draw logQ
            num_paths=2,
        )

        # Test diagnostics dataset creation
        diag_ds = _build_diagnostics_dataset(result, model=None)
        assert isinstance(diag_ds, xr.Dataset)
        assert "logP_full" in diag_ds.data_vars
        assert "logQ_full" in diag_ds.data_vars
        assert "samples_full" in diag_ds.data_vars
        assert diag_ds.logP_full.shape == (2, 50)
        assert diag_ds.samples_full.shape == (2, 50, 3)

        # Test with add_pathfinder_to_inference_data
        idata_updated = add_pathfinder_to_inference_data(
            idata, result, model=None, store_diagnostics=True, diagnostics_group="test_diag"
        )

        # Check diagnostics group was added
        groups = list(idata_updated.groups())
        assert "test_diag" in groups
        assert "logP_full" in idata_updated.test_diag.data_vars

    def test_no_diagnostics_when_store_false(self):
        """Test that diagnostics group is NOT created when store_diagnostics=False."""
        pytest.importorskip("arviz")

        import arviz as az

        from pymc_extras.inference.pathfinder.idata import add_pathfinder_to_inference_data

        # Create mock InferenceData
        posterior = xr.Dataset({"x": (["chain", "draw"], np.random.normal(0, 1, (1, 100)))})
        idata = az.InferenceData(posterior=posterior)

        # Create mock result with diagnostic data
        result = MockMultiPathfinderResult(
            samples=np.random.normal(0, 1, (2, 50, 3)),
            logP=np.random.normal(-10, 1, (2, 50)),
            logQ=np.random.normal(-11, 1, (2, 50)),
            num_paths=2,
        )

        # Test with store_diagnostics=False (default)
        idata_updated = add_pathfinder_to_inference_data(
            idata, result, model=None, store_diagnostics=False
        )

        # Check diagnostics group was NOT added
        groups = list(idata_updated.groups())
        assert "pathfinder_diagnostics" not in groups


def test_import_structure():
    """Test that all expected imports work."""
    # This test should pass even without full dependencies
    from pymc_extras.inference.pathfinder.idata import (
        _build_config_dataset,
        _build_diagnostics_dataset,
        add_pathfinder_to_inference_data,
        get_param_coords,
        multipathfinder_result_to_xarray,
        pathfinder_result_to_xarray,
    )

    # Check functions are callable
    assert callable(get_param_coords)
    assert callable(pathfinder_result_to_xarray)
    assert callable(multipathfinder_result_to_xarray)
    assert callable(add_pathfinder_to_inference_data)
    assert callable(_build_config_dataset)
    assert callable(_build_diagnostics_dataset)


if __name__ == "__main__":
    # Run basic import test
    test_import_structure()
    print("✓ Import structure test passed")
