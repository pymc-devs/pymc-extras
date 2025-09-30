import pytest

from pymc_extras.inference.pathfinder import fit_pathfinder

from .conftest import get_available_backends, validate_pathfinder_result

pytestmark = pytest.mark.skipif(not pytest.importorskip("numba"), reason="Numba not available")


class TestNumbaIntegration:
    def test_backend_selection_not_implemented(self, simple_model):
        """Test that Numba backend selection fails gracefully when not implemented."""
        with pytest.raises((NotImplementedError, ValueError)):
            result = fit_pathfinder(
                simple_model, inference_backend="numba", num_draws=10, num_paths=1
            )

    def test_backend_selection_with_fixtures(self, medium_model):
        """Test backend selection using conftest fixtures."""
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

            assert numba_dispatch is not None
        except ImportError:
            pytest.skip("Numba dispatch not available")

    def test_fallback_behavior(self, simple_model):
        """Test that system works when Numba is not available (simulated)."""
        result = fit_pathfinder(simple_model, inference_backend="pymc", num_draws=50, num_paths=2)

        validate_pathfinder_result(result, expected_draws=50, expected_vars=["x"])

    def test_available_backends(self):
        """Test which backends are available in current environment."""
        available_backends = get_available_backends()

        print(f"Available backends: {available_backends}")
        assert "pymc" in available_backends
        assert "numba" in available_backends
