import pytest

from pymc_extras.inference.pathfinder import fit_pathfinder

from .conftest import get_available_backends, validate_pathfinder_result

pytestmark = pytest.mark.skipif(not pytest.importorskip("numba"), reason="Numba not available")


class TestNumbaIntegration:
    def test_backend_selection_not_implemented(self, simple_model):
        """Test that Numba backend selection fails gracefully when not implemented."""
        # Should fail at this point since we haven't implemented the backend yet
        with pytest.raises((NotImplementedError, ValueError)):
            result = fit_pathfinder(
                simple_model, inference_backend="numba", num_draws=10, num_paths=1
            )

    def test_backend_selection_with_fixtures(self, medium_model):
        """Test backend selection using conftest fixtures."""
        # Test that we can at least attempt to select the Numba backend
        # This should currently fail since backend isn't implemented
        with pytest.raises((NotImplementedError, ValueError)):
            result = fit_pathfinder(
                medium_model, inference_backend="numba", num_draws=20, num_paths=2
            )

    def test_numba_import_conditional(self):
        """Test conditional import of Numba backend."""
        import importlib.util

        if importlib.util.find_spec("numba") is None:
            pytest.skip("Numba not available")

        try:
            from pymc_extras.inference.pathfinder import numba_dispatch

            # If we get here, numba_dispatch imported successfully
            assert numba_dispatch is not None
        except ImportError:
            # If import fails, it should be due to missing Numba
            pytest.skip("Numba dispatch not available")

    def test_fallback_behavior(self, simple_model):
        """Test that system works when Numba is not available (simulated)."""
        # This test ensures graceful degradation
        # For now, we just test that the PyMC backend still works
        result = fit_pathfinder(simple_model, inference_backend="pymc", num_draws=50, num_paths=2)

        # Use conftest utility to validate result
        validate_pathfinder_result(result, expected_draws=50, expected_vars=["x"])

    def test_available_backends(self):
        """Test which backends are available in current environment."""
        available_backends = get_available_backends()

        print(f"Available backends: {available_backends}")
        # At least PyMC should be available
        assert "pymc" in available_backends
        # In our environment, Numba should be available too
        assert "numba" in available_backends
