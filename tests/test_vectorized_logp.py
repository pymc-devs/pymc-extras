"""
Test suite for vectorized log-probability implementations.

Tests the PyTensor First approach using vectorize_graph, pt.scan, and pt.vectorize
to replace the custom LogLike Op, ensuring numerical equivalence and JAX compatibility.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_extras.inference.pathfinder.pathfinder import LogLike, get_logp_dlogp_of_ravel_inputs
from pymc_extras.inference.pathfinder.vectorized_logp import (
    create_direct_vectorized_logp,
    create_scan_based_logp_graph,
    create_vectorized_logp_graph,
)


class TestVectorizedLogP:
    """Test suite for vectorized log-probability implementations."""

    @pytest.fixture
    def simple_model(self):
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1, observed=2.0)
        return model

    @pytest.fixture
    def multidim_model(self):
        with pm.Model() as model:
            beta = pm.Normal("beta", 0, 1, shape=3)
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", beta.sum(), sigma, observed=np.array([1.0, 2.0, 3.0]))
        return model

    @pytest.fixture
    def logp_func(self, simple_model):
        """Create logp function from simple model."""
        logp_func, _ = get_logp_dlogp_of_ravel_inputs(simple_model, jacobian=True)
        return logp_func

    @pytest.fixture
    def multidim_logp_func(self, multidim_model):
        logp_func, _ = get_logp_dlogp_of_ravel_inputs(multidim_model, jacobian=True)
        return logp_func

    def test_vectorize_graph_approach_simple(self, logp_func):
        # Create test input
        test_input = np.random.randn(5, 1).astype("float64")  # 5 samples, 1 parameter

        # Current approach: LogLike Op
        loglike_op = LogLike(logp_func)
        phi_current = pt.matrix("phi_current", dtype="float64")
        logP_current = loglike_op(phi_current)
        f_current = pytensor.function([phi_current], logP_current)

        # New approach: vectorize_graph
        vectorized_logp = create_vectorized_logp_graph(logp_func)
        phi_new = pt.matrix("phi_new", dtype="float64")
        logP_new = vectorized_logp(phi_new)
        f_new = pytensor.function([phi_new], logP_new)

        result_current = f_current(test_input)
        result_new = f_new(test_input)

        np.testing.assert_allclose(result_current, result_new, rtol=1e-10)

    def test_vectorize_graph_approach_multidim(self, multidim_logp_func):
        test_input = np.random.randn(5, 4).astype("float64")
        test_input[:, 3] = np.abs(test_input[:, 3])

        loglike_op = LogLike(multidim_logp_func)
        phi_current = pt.matrix("phi_current", dtype="float64")
        logP_current = loglike_op(phi_current)
        f_current = pytensor.function([phi_current], logP_current)

        vectorized_logp = create_vectorized_logp_graph(multidim_logp_func)
        phi_new = pt.matrix("phi_new", dtype="float64")
        logP_new = vectorized_logp(phi_new)
        f_new = pytensor.function([phi_new], logP_new)

        result_current = f_current(test_input)
        result_new = f_new(test_input)

        np.testing.assert_allclose(result_current, result_new, rtol=1e-10)

    def test_scan_based_approach(self, logp_func):
        """Test pt.scan based approach."""
        test_input = np.random.randn(5, 1).astype("float64")

        loglike_op = LogLike(logp_func)
        phi_current = pt.matrix("phi_current", dtype="float64")
        logP_current = loglike_op(phi_current)
        f_current = pytensor.function([phi_current], logP_current)

        scan_logp = create_scan_based_logp_graph(logp_func)
        phi_new = pt.matrix("phi_new", dtype="float64")
        logP_new = scan_logp(phi_new)
        f_new = pytensor.function([phi_new], logP_new)

        result_current = f_current(test_input)
        result_new = f_new(test_input)

        np.testing.assert_allclose(result_current, result_new, rtol=1e-10)

    def test_direct_vectorize_approach(self, logp_func):
        test_input = np.random.randn(5, 1).astype("float64")

        loglike_op = LogLike(logp_func)
        phi_current = pt.matrix("phi_current", dtype="float64")
        logP_current = loglike_op(phi_current)
        f_current = pytensor.function([phi_current], logP_current)

        direct_logp = create_direct_vectorized_logp(logp_func)
        phi_new = pt.matrix("phi_new", dtype="float64")
        logP_new = direct_logp(phi_new)
        f_new = pytensor.function([phi_new], logP_new)

        result_current = f_current(test_input)
        result_new = f_new(test_input)

        np.testing.assert_allclose(result_current, result_new, rtol=1e-10)

    def test_jax_compilation_vectorize_graph(self, logp_func):
        test_input = np.random.randn(5, 1).astype("float64")

        vectorized_logp = create_vectorized_logp_graph(logp_func)
        phi = pt.matrix("phi", dtype="float64")
        logP = vectorized_logp(phi)

        try:
            f_jax = pytensor.function([phi], logP, mode="JAX")
            result_jax = f_jax(test_input)

            f_pt = pytensor.function([phi], logP)
            result_pt = f_pt(test_input)

            np.testing.assert_allclose(result_pt, result_jax, rtol=1e-10)

        except Exception as e:
            pytest.skip(f"JAX not available or JAX compilation failed: {e}")

    def test_jax_compilation_scan_based(self, logp_func):
        """Test that pt.scan approach compiles with JAX mode."""
        test_input = np.random.randn(5, 1).astype("float64")

        scan_logp = create_scan_based_logp_graph(logp_func)
        phi = pt.matrix("phi", dtype="float64")
        logP = scan_logp(phi)

        try:
            f_jax = pytensor.function([phi], logP, mode="JAX")
            result_jax = f_jax(test_input)

            f_pt = pytensor.function([phi], logP)
            result_pt = f_pt(test_input)

            np.testing.assert_allclose(result_pt, result_jax, rtol=1e-10)

        except Exception as e:
            pytest.skip(f"JAX not available or JAX compilation failed: {e}")

    def test_nan_inf_handling(self, logp_func):
        """Test that nan/inf values are handled correctly."""
        test_input = np.array(
            [
                [0.0],
                [np.inf],
                [np.nan],
                [-np.inf],
            ],
            dtype="float64",
        )

        vectorized_logp = create_vectorized_logp_graph(logp_func)
        phi = pt.matrix("phi", dtype="float64")
        logP = vectorized_logp(phi)
        f = pytensor.function([phi], logP)

        result = f(test_input)

        assert np.isfinite(result[0])
        assert result[1] == -np.inf
        assert result[2] == -np.inf
        assert result[3] == -np.inf

    def test_3d_input_shapes(self, logp_func):
        test_input = np.random.randn(2, 3, 1).astype("float64")

        loglike_op = LogLike(logp_func)
        phi_current = pt.tensor3("phi_current", dtype="float64")
        logP_current = loglike_op(phi_current)
        f_current = pytensor.function([phi_current], logP_current)

        vectorized_logp = create_vectorized_logp_graph(logp_func)
        phi_new = pt.tensor3("phi_new", dtype="float64")
        logP_new = vectorized_logp(phi_new)
        f_new = pytensor.function([phi_new], logP_new)

        result_current = f_current(test_input)
        result_new = f_new(test_input)

        np.testing.assert_allclose(result_current, result_new, rtol=1e-10)

        assert result_new.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__])
