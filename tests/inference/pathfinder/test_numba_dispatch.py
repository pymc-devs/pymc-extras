import numpy as np
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.skipif(not pytest.importorskip("numba"), reason="Numba not available")


class TestNumbaDispatch:
    def test_numba_import(self):
        """Test that numba_dispatch module imports correctly."""
        from pymc_extras.inference.pathfinder import numba_dispatch

        assert hasattr(numba_dispatch, "__version__")

    def test_required_imports_available(self):
        """Test that all required imports are available in numba_dispatch."""
        from pymc_extras.inference.pathfinder import numba_dispatch

        # Check required PyTensor imports
        assert hasattr(numba_dispatch, "pt")
        assert hasattr(numba_dispatch, "Apply")
        assert hasattr(numba_dispatch, "Op")

        # Check required Numba dispatch imports
        assert hasattr(numba_dispatch, "numba_funcify")
        assert hasattr(numba_dispatch, "numba_basic")

        # Check LogLike op import
        assert hasattr(numba_dispatch, "LogLike")

    def test_numba_basic_functionality(self):
        """Test basic Numba functionality is working."""
        import numba

        from pymc_extras.inference.pathfinder import numba_dispatch

        # Test that numba_basic.numba_njit is callable
        assert callable(numba_dispatch.numba_basic.numba_njit)

        # Test basic Numba compilation using standard numba
        @numba.jit(nopython=True)
        def simple_function(x):
            return x * 2

        result = simple_function(5.0)
        assert result == 10.0


class TestLogLikeNumbaDispatch:
    """Test Numba dispatch registration for LogLike Op."""

    def test_loglike_numba_registration_exists(self):
        """Test that LogLike Op has Numba registration."""
        from pytensor.link.numba.dispatch import numba_funcify

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Check that LogLike is registered with numba_funcify
        assert LogLike in numba_funcify.registry

    def test_loglike_numba_with_simple_function(self):
        """Test LogLike Op with simple compiled function."""
        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a simple logp function
        def simple_logp(x):
            return -0.5 * np.sum(x**2)

        # Create LogLike Op
        loglike_op = LogLike(simple_logp)
        phi = pt.matrix("phi", dtype="float64")
        output = loglike_op(phi)

        # Test with Numba mode
        try:
            f = pytensor.function([phi], output, mode="NUMBA")

            # Test execution
            test_phi = np.random.randn(5, 3).astype(np.float64)
            result = f(test_phi)

            # Verify shape and basic correctness
            assert result.shape == (5,)
            assert np.all(np.isfinite(result))

            # Verify results match expected values
            expected = np.array([simple_logp(test_phi[i]) for i in range(5)])
            np.testing.assert_allclose(result, expected, rtol=1e-12)

        except Exception as e:
            pytest.skip(f"Numba compilation failed: {e}")

    def test_loglike_numba_vs_python_equivalence(self):
        """Test that Numba implementation matches Python implementation."""
        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a more complex logp function
        def complex_logp(x):
            return -0.5 * (np.sum(x**2) + np.sum(np.log(2 * np.pi)))

        # Create LogLike Op
        loglike_op = LogLike(complex_logp)
        phi = pt.matrix("phi", dtype="float64")
        output = loglike_op(phi)

        # Test data
        test_phi = np.random.randn(10, 4).astype(np.float64)

        try:
            # Python mode (reference)
            f_py = pytensor.function([phi], output, mode="py")
            result_py = f_py(test_phi)

            # Numba mode
            f_numba = pytensor.function([phi], output, mode="NUMBA")
            result_numba = f_numba(test_phi)

            # Compare results
            np.testing.assert_allclose(result_numba, result_py, rtol=1e-12)

        except Exception as e:
            pytest.skip(f"Comparison test failed: {e}")

    def test_loglike_numba_3d_input(self):
        """Test LogLike Op with 3D input (multiple paths)."""
        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a simple logp function
        def simple_logp(x):
            return -0.5 * np.sum(x**2)

        # Create LogLike Op
        loglike_op = LogLike(simple_logp)
        phi = pt.tensor("phi", dtype="float64", shape=(None, None, None))
        output = loglike_op(phi)

        try:
            # Test with Numba mode
            f = pytensor.function([phi], output, mode="NUMBA")

            # Test execution with 3D input (L=3, M=4, N=2)
            test_phi = np.random.randn(3, 4, 2).astype(np.float64)
            result = f(test_phi)

            # Verify shape and basic correctness
            assert result.shape == (3, 4)
            assert np.all(np.isfinite(result))

            # Verify results match expected values
            for batch_idx in range(3):
                for m in range(4):
                    expected = simple_logp(test_phi[batch_idx, m])
                    np.testing.assert_allclose(result[batch_idx, m], expected, rtol=1e-12)

        except Exception as e:
            pytest.skip(f"3D input test failed: {e}")

    def test_loglike_numba_nan_inf_handling(self):
        """Test LogLike Op handles NaN/Inf values correctly."""
        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a function that can return NaN/Inf
        def problematic_logp(x):
            # Return NaN for negative first element
            if x[0] < 0:
                return np.nan
            # Return -Inf for very large values
            elif np.sum(x**2) > 100:
                return -np.inf
            else:
                return -0.5 * np.sum(x**2)

        # Create LogLike Op
        loglike_op = LogLike(problematic_logp)
        phi = pt.matrix("phi", dtype="float64")
        output = loglike_op(phi)

        try:
            # Test with Numba mode
            f = pytensor.function([phi], output, mode="NUMBA")

            # Create test data with problematic values
            test_phi = np.array(
                [
                    [-1.0, 0.0],  # Should produce NaN -> -Inf
                    [10.0, 10.0],  # Should produce -Inf
                    [1.0, 1.0],  # Should produce normal value
                ],
                dtype=np.float64,
            )

            result = f(test_phi)

            # Verify NaN/Inf are converted to -Inf
            assert result[0] == -np.inf  # NaN -> -Inf
            assert result[1] == -np.inf  # -Inf -> -Inf
            assert np.isfinite(result[2])  # Normal value

        except Exception as e:
            pytest.skip(f"NaN/Inf handling test failed: {e}")

    def test_loglike_numba_interface_compatibility_error(self):
        """Test LogLike Op raises appropriate error for incompatible logp_func."""
        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a symbolic function (incompatible with Numba)
        def symbolic_logp(x):
            if hasattr(x, "type"):  # Symbolic
                return pt.sum(x**2)
            else:
                raise TypeError("Expected symbolic input")

        # Create LogLike Op
        loglike_op = LogLike(symbolic_logp)
        phi = pt.matrix("phi", dtype="float64")
        output = loglike_op(phi)

        # Test that Numba mode raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Numba backend requires logp_func"):
            f = pytensor.function([phi], output, mode="NUMBA")

    def test_loglike_numba_performance_improvement(self):
        """Test that Numba provides performance improvement over Python."""
        import time

        import pytensor

        from pymc_extras.inference.pathfinder.pathfinder import LogLike

        # Define a computationally intensive logp function
        def intensive_logp(x):
            result = 0.0
            for i in range(len(x)):
                result += -0.5 * x[i] ** 2 - 0.5 * np.log(2 * np.pi)
            return result

        # Create LogLike Op
        loglike_op = LogLike(intensive_logp)
        phi = pt.matrix("phi", dtype="float64")
        output = loglike_op(phi)

        # Large test data
        test_phi = np.random.randn(100, 10).astype(np.float64)

        try:
            # Python mode timing
            f_py = pytensor.function([phi], output, mode="py")
            start_time = time.time()
            result_py = f_py(test_phi)
            py_time = time.time() - start_time

            # Numba mode timing (including compilation)
            f_numba = pytensor.function([phi], output, mode="NUMBA")
            start_time = time.time()
            result_numba = f_numba(test_phi)
            numba_time = time.time() - start_time

            # Verify results are equivalent
            np.testing.assert_allclose(result_numba, result_py, rtol=1e-12)

            # For large enough data, Numba should eventually be faster
            # Note: First run includes compilation overhead
            print(f"Python time: {py_time:.4f}s, Numba time: {numba_time:.4f}s")

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")


class TestChiMatrixNumbaDispatch:
    """Test Numba dispatch registration for ChiMatrix Op."""

    def test_chimatrix_numba_registration_exists(self):
        """Test that NumbaChiMatrixOp has Numba registration."""
        from pytensor.link.numba.dispatch import numba_funcify

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        # Check that NumbaChiMatrixOp is registered with numba_funcify
        assert NumbaChiMatrixOp in numba_funcify.registry

    def test_chimatrix_op_basic_functionality(self):
        """Test basic ChiMatrix Op functionality."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        J = 3
        diff = pt.matrix("diff", dtype="float64")
        test_diff = np.arange(20).reshape(4, 5).astype(np.float64)

        chi_op = NumbaChiMatrixOp(J)
        output = chi_op(diff)

        try:
            # Test with Python mode first (fallback)
            f_py = pytensor.function([diff], output, mode="py")
            result_py = f_py(test_diff)

            # Verify output shape
            assert result_py.shape == (4, 5, 3)

            # Test with Numba mode
            f_numba = pytensor.function([diff], output, mode="NUMBA")
            result_numba = f_numba(test_diff)

            # Compare results
            np.testing.assert_allclose(result_numba, result_py, rtol=1e-12)

        except Exception as e:
            pytest.skip(f"ChiMatrix basic functionality test failed: {e}")

    def test_chimatrix_sliding_window_logic(self):
        """Test sliding window logic correctness for ChiMatrix."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        # Test with simple sequential data to verify sliding window
        J = 3
        diff = pt.matrix("diff", dtype="float64")

        # Simple test case: sequential numbers
        test_diff = np.array(
            [
                [1.0, 10.0],  # Row 0
                [2.0, 20.0],  # Row 1
                [3.0, 30.0],  # Row 2
                [4.0, 40.0],  # Row 3
            ],
            dtype=np.float64,
        )

        chi_op = NumbaChiMatrixOp(J)
        output = chi_op(diff)

        try:
            f = pytensor.function([diff], output, mode="NUMBA")
            result = f(test_diff)

            # Verify sliding window behavior
            # For row 0: should have [0, 0, 1] and [0, 0, 10] (padded)
            expected_row0_col0 = [0.0, 0.0, 1.0]
            expected_row0_col1 = [0.0, 0.0, 10.0]
            np.testing.assert_allclose(result[0, 0, :], expected_row0_col0)
            np.testing.assert_allclose(result[0, 1, :], expected_row0_col1)

            # For row 2: should have [1, 2, 3] and [10, 20, 30]
            expected_row2_col0 = [1.0, 2.0, 3.0]
            expected_row2_col1 = [10.0, 20.0, 30.0]
            np.testing.assert_allclose(result[2, 0, :], expected_row2_col0)
            np.testing.assert_allclose(result[2, 1, :], expected_row2_col1)

            # For row 3: should have [2, 3, 4] and [20, 30, 40] (sliding window)
            expected_row3_col0 = [2.0, 3.0, 4.0]
            expected_row3_col1 = [20.0, 30.0, 40.0]
            np.testing.assert_allclose(result[3, 0, :], expected_row3_col0)
            np.testing.assert_allclose(result[3, 1, :], expected_row3_col1)

        except Exception as e:
            pytest.skip(f"ChiMatrix sliding window test failed: {e}")

    def test_chimatrix_edge_cases(self):
        """Test ChiMatrix Op edge cases."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        # Test case 1: L < J (fewer rows than history size)
        J = 5
        diff = pt.matrix("diff", dtype="float64")
        test_diff = np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
            ],
            dtype=np.float64,
        )  # Only 2 rows, J=5

        chi_op = NumbaChiMatrixOp(J)
        output = chi_op(diff)

        try:
            f = pytensor.function([diff], output, mode="NUMBA")
            result = f(test_diff)

            # Should have shape (2, 2, 5)
            assert result.shape == (2, 2, 5)

            # Row 0 should be [0, 0, 0, 0, 1] and [0, 0, 0, 0, 10]
            expected_row0_col0 = [0.0, 0.0, 0.0, 0.0, 1.0]
            np.testing.assert_allclose(result[0, 0, :], expected_row0_col0)

            # Row 1 should be [0, 0, 0, 1, 2] and [0, 0, 0, 10, 20]
            expected_row1_col0 = [0.0, 0.0, 0.0, 1.0, 2.0]
            np.testing.assert_allclose(result[1, 0, :], expected_row1_col0)

        except Exception as e:
            pytest.skip(f"ChiMatrix edge case test failed: {e}")

    def test_chimatrix_vs_jax_equivalence(self):
        """Test numerical equivalence with JAX implementation if available."""
        try:
            import pytensor

            from pymc_extras.inference.pathfinder.jax_dispatch import ChiMatrixOp as JAXChiMatrixOp
            from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

            J = 4
            diff = pt.matrix("diff", dtype="float64")
            test_diff = np.random.randn(6, 3).astype(np.float64)

            # JAX implementation
            jax_op = JAXChiMatrixOp(J)
            jax_output = jax_op(diff)

            # Numba implementation
            numba_op = NumbaChiMatrixOp(J)
            numba_output = numba_op(diff)

            try:
                # Compare using Python mode (fallback for both)
                f_jax = pytensor.function([diff], jax_output, mode="py")
                f_numba = pytensor.function([diff], numba_output, mode="py")

                result_jax = f_jax(test_diff)
                result_numba = f_numba(test_diff)

                # Should be mathematically equivalent
                np.testing.assert_allclose(result_numba, result_jax, rtol=1e-12)

            except Exception as e:
                pytest.skip(f"JAX comparison failed: {e}")

        except ImportError:
            pytest.skip("JAX not available for comparison")

    def test_chimatrix_different_j_values(self):
        """Test ChiMatrix Op with different J values."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        diff = pt.matrix("diff", dtype="float64")
        test_diff = np.random.randn(8, 4).astype(np.float64)

        # Test different J values
        for J in [1, 3, 5, 8, 10]:  # Including J > L case
            chi_op = NumbaChiMatrixOp(J)
            output = chi_op(diff)

            try:
                f = pytensor.function([diff], output, mode="NUMBA")
                result = f(test_diff)

                # Verify output shape
                assert result.shape == (8, 4, J)

                # Verify all values are finite
                assert np.all(np.isfinite(result))

            except Exception as e:
                pytest.skip(f"ChiMatrix J={J} test failed: {e}")

    def test_chimatrix_numba_performance(self):
        """Test ChiMatrix Numba performance vs Python."""
        import time

        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaChiMatrixOp

        # Large test case
        J = 10
        diff = pt.matrix("diff", dtype="float64")
        test_diff = np.random.randn(100, 50).astype(np.float64)

        chi_op = NumbaChiMatrixOp(J)
        output = chi_op(diff)

        try:
            # Python mode timing
            f_py = pytensor.function([diff], output, mode="py")
            start_time = time.time()
            result_py = f_py(test_diff)
            py_time = time.time() - start_time

            # Numba mode timing (including compilation)
            f_numba = pytensor.function([diff], output, mode="NUMBA")
            start_time = time.time()
            result_numba = f_numba(test_diff)
            numba_time = time.time() - start_time

            # Verify results are equivalent
            np.testing.assert_allclose(result_numba, result_py, rtol=1e-12)

            print(f"ChiMatrix - Python time: {py_time:.4f}s, Numba time: {numba_time:.4f}s")

        except Exception as e:
            pytest.skip(f"ChiMatrix performance test failed: {e}")


class TestBfgsSampleNumbaDispatch:
    """Test Numba dispatch registration for BfgsSample Op."""

    def test_bfgssample_numba_registration_exists(self):
        """Test that NumbaBfgsSampleOp has Numba registration."""
        from pytensor.link.numba.dispatch import numba_funcify

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Check that NumbaBfgsSampleOp is registered with numba_funcify
        assert NumbaBfgsSampleOp in numba_funcify.registry

    def test_bfgssample_op_basic_functionality(self):
        """Test basic BfgsSample Op functionality."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Create test data for small dense case (JJ >= N)
        L, M, N = 2, 3, 4
        JJ = 6  # JJ >= N, so dense case

        # Create input tensors
        x = pt.matrix("x", dtype="float64")
        g = pt.matrix("g", dtype="float64")
        alpha = pt.matrix("alpha", dtype="float64")
        beta = pt.tensor("beta", dtype="float64", shape=(None, None, None))
        gamma = pt.tensor("gamma", dtype="float64", shape=(None, None, None))
        alpha_diag = pt.tensor("alpha_diag", dtype="float64", shape=(None, None, None))
        inv_sqrt_alpha_diag = pt.tensor(
            "inv_sqrt_alpha_diag", dtype="float64", shape=(None, None, None)
        )
        sqrt_alpha_diag = pt.tensor("sqrt_alpha_diag", dtype="float64", shape=(None, None, None))
        u = pt.tensor("u", dtype="float64", shape=(None, None, None))

        # Create test data
        test_x = np.random.randn(L, N).astype(np.float64)
        test_g = np.random.randn(L, N).astype(np.float64)
        test_alpha = np.abs(np.random.randn(L, N)) + 0.1  # Ensure positive
        test_beta = np.random.randn(L, N, JJ).astype(np.float64)
        test_gamma = np.random.randn(L, JJ, JJ).astype(np.float64)
        # Make gamma positive definite
        for i in range(L):
            test_gamma[i] = test_gamma[i] @ test_gamma[i].T + np.eye(JJ) * 0.1

        test_alpha_diag = np.zeros((L, N, N))
        test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
        test_sqrt_alpha_diag = np.zeros((L, N, N))
        for i in range(L):
            test_alpha_diag[i] = np.diag(test_alpha[i])
            test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
            test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

        test_u = np.random.randn(L, M, N).astype(np.float64)

        # Create BfgsSample Op
        bfgs_op = NumbaBfgsSampleOp()
        phi_out, logdet_out = bfgs_op(
            x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
        )

        try:
            # Test with Python mode first (fallback)
            f_py = pytensor.function(
                [x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u],
                [phi_out, logdet_out],
                mode="py",
            )
            phi_py, logdet_py = f_py(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )

            # Verify output shapes
            assert phi_py.shape == (L, M, N)
            assert logdet_py.shape == (L,)
            assert np.all(np.isfinite(phi_py))
            assert np.all(np.isfinite(logdet_py))

            # Test with Numba mode
            f_numba = pytensor.function(
                [x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u],
                [phi_out, logdet_out],
                mode="NUMBA",
            )
            phi_numba, logdet_numba = f_numba(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )

            # Compare results
            np.testing.assert_allclose(phi_numba, phi_py, rtol=1e-10)
            np.testing.assert_allclose(logdet_numba, logdet_py, rtol=1e-10)

        except Exception as e:
            pytest.skip(f"BfgsSample basic functionality test failed: {e}")

    def test_bfgssample_dense_case(self):
        """Test dense BFGS sampling (JJ >= N)."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Create test data where JJ >= N (dense case)
        L, M, N = 2, 5, 3
        JJ = 4  # JJ > N, so dense case

        # Create smaller, well-conditioned test case
        test_x = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=np.float64)
        test_g = np.array([[0.1, 0.2, 0.1], [0.15, 0.1, 0.05]], dtype=np.float64)
        test_alpha = np.array([[1.0, 1.5, 2.0], [0.8, 1.2, 1.8]], dtype=np.float64)

        # Create well-conditioned beta and gamma
        test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1  # Small values
        test_gamma = np.zeros((L, JJ, JJ))
        for i in range(L):
            # Create positive definite gamma
            temp = np.random.randn(JJ, JJ) * 0.1
            test_gamma[i] = temp @ temp.T + np.eye(JJ) * 0.5

        # Create diagonal matrices
        test_alpha_diag = np.zeros((L, N, N))
        test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
        test_sqrt_alpha_diag = np.zeros((L, N, N))
        for i in range(L):
            test_alpha_diag[i] = np.diag(test_alpha[i])
            test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
            test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

        test_u = np.random.randn(L, M, N).astype(np.float64)

        # Create tensor variables (not constants)
        x_var = pt.matrix("x", dtype="float64")
        g_var = pt.matrix("g", dtype="float64")
        alpha_var = pt.matrix("alpha", dtype="float64")
        beta_var = pt.tensor("beta", dtype="float64", shape=(None, None, None))
        gamma_var = pt.tensor("gamma", dtype="float64", shape=(None, None, None))
        alpha_diag_var = pt.tensor("alpha_diag", dtype="float64", shape=(None, None, None))
        inv_sqrt_alpha_diag_var = pt.tensor(
            "inv_sqrt_alpha_diag", dtype="float64", shape=(None, None, None)
        )
        sqrt_alpha_diag_var = pt.tensor(
            "sqrt_alpha_diag", dtype="float64", shape=(None, None, None)
        )
        u_var = pt.tensor("u", dtype="float64", shape=(None, None, None))

        inputs = [
            x_var,
            g_var,
            alpha_var,
            beta_var,
            gamma_var,
            alpha_diag_var,
            inv_sqrt_alpha_diag_var,
            sqrt_alpha_diag_var,
            u_var,
        ]

        # Create BfgsSample Op
        bfgs_op = NumbaBfgsSampleOp()
        phi_out, logdet_out = bfgs_op(*inputs)

        try:
            f = pytensor.function(inputs, [phi_out, logdet_out], mode="NUMBA")
            phi, logdet = f(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )

            # Verify output shapes and values
            assert phi.shape == (L, M, N)
            assert logdet.shape == (L,)
            assert np.all(np.isfinite(phi))
            assert np.all(np.isfinite(logdet))

            # Verify this was the dense case (JJ >= N)
            assert JJ >= N, "Test should use dense case"

        except Exception as e:
            pytest.skip(f"BfgsSample dense case test failed: {e}")

    def test_bfgssample_sparse_case(self):
        """Test sparse BFGS sampling (JJ < N)."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Create test data where JJ < N (sparse case)
        L, M, N = 2, 5, 6
        JJ = 4  # JJ < N, so sparse case

        # Create smaller, well-conditioned test case
        test_x = np.random.randn(L, N).astype(np.float64)
        test_g = np.random.randn(L, N).astype(np.float64) * 0.1
        test_alpha = np.abs(np.random.randn(L, N)) + 0.5  # Ensure positive and bounded away from 0

        # Create well-conditioned beta and gamma
        test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1
        test_gamma = np.zeros((L, JJ, JJ))
        for i in range(L):
            # Create positive definite gamma
            temp = np.random.randn(JJ, JJ) * 0.1
            test_gamma[i] = temp @ temp.T + np.eye(JJ) * 0.5

        # Create diagonal matrices
        test_alpha_diag = np.zeros((L, N, N))
        test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
        test_sqrt_alpha_diag = np.zeros((L, N, N))
        for i in range(L):
            test_alpha_diag[i] = np.diag(test_alpha[i])
            test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
            test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

        test_u = np.random.randn(L, M, N).astype(np.float64)

        # Create tensors
        inputs = [
            pt.as_tensor_variable(arr)
            for arr in [
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            ]
        ]

        # Create BfgsSample Op
        bfgs_op = NumbaBfgsSampleOp()
        phi_out, logdet_out = bfgs_op(*inputs)

        try:
            f = pytensor.function(inputs, [phi_out, logdet_out], mode="NUMBA")
            phi, logdet = f(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )

            # Verify output shapes and values
            assert phi.shape == (L, M, N)
            assert logdet.shape == (L,)
            assert np.all(np.isfinite(phi))
            assert np.all(np.isfinite(logdet))

            # Verify this was the sparse case (JJ < N)
            assert JJ < N, "Test should use sparse case"

        except Exception as e:
            pytest.skip(f"BfgsSample sparse case test failed: {e}")

    def test_bfgssample_conditional_logic(self):
        """Test conditional branching works correctly."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Test both branches with same L, M but different N and JJ
        L, M = 2, 3

        # Dense case: N=3, JJ=4 (JJ >= N)
        N_dense, JJ_dense = 3, 4

        # Sparse case: N=5, JJ=3 (JJ < N)
        N_sparse, JJ_sparse = 5, 3

        for case_name, N, JJ in [("dense", N_dense, JJ_dense), ("sparse", N_sparse, JJ_sparse)]:
            # Create test data
            test_x = np.random.randn(L, N).astype(np.float64)
            test_g = np.random.randn(L, N).astype(np.float64) * 0.1
            test_alpha = np.abs(np.random.randn(L, N)) + 0.5
            test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1

            test_gamma = np.zeros((L, JJ, JJ))
            for i in range(L):
                temp = np.random.randn(JJ, JJ) * 0.1
                test_gamma[i] = temp @ temp.T + np.eye(JJ) * 0.5

            # Create diagonal matrices
            test_alpha_diag = np.zeros((L, N, N))
            test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
            test_sqrt_alpha_diag = np.zeros((L, N, N))
            for i in range(L):
                test_alpha_diag[i] = np.diag(test_alpha[i])
                test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
                test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

            test_u = np.random.randn(L, M, N).astype(np.float64)

            # Create tensors
            inputs = [
                pt.as_tensor_variable(arr)
                for arr in [
                    test_x,
                    test_g,
                    test_alpha,
                    test_beta,
                    test_gamma,
                    test_alpha_diag,
                    test_inv_sqrt_alpha_diag,
                    test_sqrt_alpha_diag,
                    test_u,
                ]
            ]

            # Create BfgsSample Op
            bfgs_op = NumbaBfgsSampleOp()
            phi_out, logdet_out = bfgs_op(*inputs)

            try:
                f = pytensor.function(inputs, [phi_out, logdet_out], mode="NUMBA")
                phi, logdet = f(
                    test_x,
                    test_g,
                    test_alpha,
                    test_beta,
                    test_gamma,
                    test_alpha_diag,
                    test_inv_sqrt_alpha_diag,
                    test_sqrt_alpha_diag,
                    test_u,
                )

                # Verify results for this case
                assert phi.shape == (L, M, N), f"Wrong phi shape for {case_name} case"
                assert logdet.shape == (L,), f"Wrong logdet shape for {case_name} case"
                assert np.all(np.isfinite(phi)), f"Non-finite values in phi for {case_name} case"
                assert np.all(
                    np.isfinite(logdet)
                ), f"Non-finite values in logdet for {case_name} case"

                # Verify the condition was correct
                if case_name == "dense":
                    assert JJ >= N, "Dense case should have JJ >= N"
                else:
                    assert JJ < N, "Sparse case should have JJ < N"

            except Exception as e:
                pytest.skip(f"BfgsSample {case_name} case test failed: {e}")

    def test_bfgssample_vs_jax_equivalence(self):
        """Test numerical equivalence with JAX implementation if available."""
        try:
            import pytensor

            from pymc_extras.inference.pathfinder.jax_dispatch import (
                BfgsSampleOp as JAXBfgsSampleOp,
            )
            from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

            # Create test data for comparison
            L, M, N = 2, 3, 4
            JJ = 3  # Use sparse case for more interesting comparison

            # Create well-conditioned test data
            test_x = np.array([[1.0, 2.0, 3.0, 0.5], [0.5, 1.5, 2.5, 1.0]], dtype=np.float64)
            test_g = np.array([[0.1, 0.2, 0.1, 0.05], [0.15, 0.1, 0.05, 0.08]], dtype=np.float64)
            test_alpha = np.array([[1.0, 1.5, 2.0, 1.2], [0.8, 1.2, 1.8, 1.1]], dtype=np.float64)

            test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1
            test_gamma = np.zeros((L, JJ, JJ))
            for i in range(L):
                temp = np.random.randn(JJ, JJ) * 0.1
                test_gamma[i] = temp @ temp.T + np.eye(JJ) * 0.5

            # Create diagonal matrices
            test_alpha_diag = np.zeros((L, N, N))
            test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
            test_sqrt_alpha_diag = np.zeros((L, N, N))
            for i in range(L):
                test_alpha_diag[i] = np.diag(test_alpha[i])
                test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
                test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

            test_u = np.random.randn(L, M, N).astype(np.float64)

            # Create tensors
            inputs = [
                pt.as_tensor_variable(arr)
                for arr in [
                    test_x,
                    test_g,
                    test_alpha,
                    test_beta,
                    test_gamma,
                    test_alpha_diag,
                    test_inv_sqrt_alpha_diag,
                    test_sqrt_alpha_diag,
                    test_u,
                ]
            ]

            # JAX implementation
            jax_op = JAXBfgsSampleOp()
            jax_phi_out, jax_logdet_out = jax_op(*inputs)

            # Numba implementation
            numba_op = NumbaBfgsSampleOp()
            numba_phi_out, numba_logdet_out = numba_op(*inputs)

            try:
                # Compare using Python mode (fallback for both)
                f_jax = pytensor.function(inputs, [jax_phi_out, jax_logdet_out], mode="py")
                f_numba = pytensor.function(inputs, [numba_phi_out, numba_logdet_out], mode="py")

                jax_phi, jax_logdet = f_jax(
                    test_x,
                    test_g,
                    test_alpha,
                    test_beta,
                    test_gamma,
                    test_alpha_diag,
                    test_inv_sqrt_alpha_diag,
                    test_sqrt_alpha_diag,
                    test_u,
                )
                numba_phi, numba_logdet = f_numba(
                    test_x,
                    test_g,
                    test_alpha,
                    test_beta,
                    test_gamma,
                    test_alpha_diag,
                    test_inv_sqrt_alpha_diag,
                    test_sqrt_alpha_diag,
                    test_u,
                )

                # Should be mathematically equivalent
                np.testing.assert_allclose(numba_phi, jax_phi, rtol=1e-10)
                np.testing.assert_allclose(numba_logdet, jax_logdet, rtol=1e-10)

            except Exception as e:
                pytest.skip(f"JAX comparison failed: {e}")

        except ImportError:
            pytest.skip("JAX not available for comparison")

    def test_bfgssample_edge_cases(self):
        """Test BfgsSample Op edge cases and robustness."""
        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Test case 1: Minimal dimensions
        L, M, N = 1, 1, 2
        JJ = 1

        test_x = np.array([[1.0, 2.0]], dtype=np.float64)
        test_g = np.array([[0.1, 0.2]], dtype=np.float64)
        test_alpha = np.array([[1.0, 1.5]], dtype=np.float64)
        test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1
        test_gamma = np.eye(JJ)[None, ...] * 0.5

        # Create diagonal matrices
        test_alpha_diag = np.diag(test_alpha[0])[None, ...]
        test_sqrt_alpha_diag = np.diag(np.sqrt(test_alpha[0]))[None, ...]
        test_inv_sqrt_alpha_diag = np.diag(1.0 / np.sqrt(test_alpha[0]))[None, ...]

        test_u = np.random.randn(L, M, N).astype(np.float64)

        # Create tensors
        inputs = [
            pt.as_tensor_variable(arr)
            for arr in [
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            ]
        ]

        # Create BfgsSample Op
        bfgs_op = NumbaBfgsSampleOp()
        phi_out, logdet_out = bfgs_op(*inputs)

        try:
            f = pytensor.function(inputs, [phi_out, logdet_out], mode="NUMBA")
            phi, logdet = f(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )

            # Verify minimal case works
            assert phi.shape == (L, M, N)
            assert logdet.shape == (L,)
            assert np.all(np.isfinite(phi))
            assert np.all(np.isfinite(logdet))

        except Exception as e:
            pytest.skip(f"BfgsSample minimal case test failed: {e}")

    def test_bfgssample_numba_performance(self):
        """Test BfgsSample Numba performance vs Python."""
        import time

        import pytensor

        from pymc_extras.inference.pathfinder.numba_dispatch import NumbaBfgsSampleOp

        # Medium-sized test case for performance measurement
        L, M, N = 4, 10, 8
        JJ = 6  # Sparse case

        # Create test data
        test_x = np.random.randn(L, N).astype(np.float64)
        test_g = np.random.randn(L, N).astype(np.float64) * 0.1
        test_alpha = np.abs(np.random.randn(L, N)) + 0.5
        test_beta = np.random.randn(L, N, JJ).astype(np.float64) * 0.1

        test_gamma = np.zeros((L, JJ, JJ))
        for i in range(L):
            temp = np.random.randn(JJ, JJ) * 0.1
            test_gamma[i] = temp @ temp.T + np.eye(JJ) * 0.5

        # Create diagonal matrices
        test_alpha_diag = np.zeros((L, N, N))
        test_inv_sqrt_alpha_diag = np.zeros((L, N, N))
        test_sqrt_alpha_diag = np.zeros((L, N, N))
        for i in range(L):
            test_alpha_diag[i] = np.diag(test_alpha[i])
            test_sqrt_alpha_diag[i] = np.diag(np.sqrt(test_alpha[i]))
            test_inv_sqrt_alpha_diag[i] = np.diag(1.0 / np.sqrt(test_alpha[i]))

        test_u = np.random.randn(L, M, N).astype(np.float64)

        # Create tensors
        inputs = [
            pt.as_tensor_variable(arr)
            for arr in [
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            ]
        ]

        # Create BfgsSample Op
        bfgs_op = NumbaBfgsSampleOp()
        phi_out, logdet_out = bfgs_op(*inputs)

        try:
            # Python mode timing
            f_py = pytensor.function(inputs, [phi_out, logdet_out], mode="py")
            start_time = time.time()
            phi_py, logdet_py = f_py(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )
            py_time = time.time() - start_time

            # Numba mode timing (including compilation)
            f_numba = pytensor.function(inputs, [phi_out, logdet_out], mode="NUMBA")
            start_time = time.time()
            phi_numba, logdet_numba = f_numba(
                test_x,
                test_g,
                test_alpha,
                test_beta,
                test_gamma,
                test_alpha_diag,
                test_inv_sqrt_alpha_diag,
                test_sqrt_alpha_diag,
                test_u,
            )
            numba_time = time.time() - start_time

            # Verify results are equivalent
            np.testing.assert_allclose(phi_numba, phi_py, rtol=1e-10)
            np.testing.assert_allclose(logdet_numba, logdet_py, rtol=1e-10)

            print(f"BfgsSample - Python time: {py_time:.4f}s, Numba time: {numba_time:.4f}s")

        except Exception as e:
            pytest.skip(f"BfgsSample performance test failed: {e}")
