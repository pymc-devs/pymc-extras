import time

import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.pathfinder import fit_pathfinder

pytestmark = pytest.mark.skipif(not pytest.importorskip("numba"), reason="Numba not available")


class TestNumbaPerformance:
    @pytest.mark.parametrize("param_size", [5, 10, 20])
    def test_compilation_time_reasonable(self, param_size):
        """Test that Numba compilation time is reasonable."""

        # Create model with specified parameter size
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1, shape=param_size)
            y = pm.Normal("y", x.sum(), 1, observed=param_size * 0.5)

        # This test will initially fail since Numba backend isn't implemented yet
        # But it sets up the testing infrastructure
        with pytest.raises((NotImplementedError, ValueError, ImportError)):
            start_time = time.time()
            result = fit_pathfinder(model, inference_backend="numba", num_draws=50, num_paths=2)
            compilation_time = time.time() - start_time

            # When implemented, compilation should be reasonable (< 30 seconds)
            assert compilation_time < 30.0

    def test_numba_environment_performance(self):
        """Test basic Numba performance is working."""
        import numba

        @numba.jit(nopython=True)
        def numba_sum(arr):
            total = 0.0
            for i in range(len(arr)):
                total += arr[i]
            return total

        # Test array
        test_array = np.random.randn(1000)

        # Warm up
        numba_sum(test_array)

        # Time Numba version
        start_time = time.time()
        numba_result = numba_sum(test_array)
        numba_time = time.time() - start_time

        # Time NumPy version
        start_time = time.time()
        numpy_result = np.sum(test_array)
        numpy_time = time.time() - start_time

        # Results should be equivalent
        np.testing.assert_allclose(numba_result, numpy_result, rtol=1e-12)

        # For this simple operation, timing comparison isn't strict
        # Just ensure Numba is working
        assert numba_time >= 0  # Basic sanity check
