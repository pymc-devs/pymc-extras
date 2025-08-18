#   Copyright 2024 The PyMC Developers
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

"""Basic tests for JAX dispatch conversions in Pathfinder.

This module tests the core JAX conversions for Pathfinder custom operations,
specifically the LogLike Op JAX conversion.
"""

import numpy as np
import pytensor.tensor as pt

from pytensor import function

from pymc_extras.inference.pathfinder.pathfinder import LogLike


class TestLogLikeJAXConversion:
    def test_loglike_simple_function(self):
        def simple_logp_func(x):
            return -0.5 * np.sum(x**2, axis=-1)

        loglike_op = LogLike(simple_logp_func)

        test_input_2d = np.random.randn(3, 2).astype(np.float64)
        inputs_2d = pt.tensor("inputs_2d", dtype="float64", shape=(None, None))
        output_2d = loglike_op(inputs_2d)

        f_pt_2d = function([inputs_2d], output_2d)
        result_pt_2d = f_pt_2d(test_input_2d)

        f_jax_2d = function([inputs_2d], output_2d, mode="JAX")
        result_jax_2d = f_jax_2d(test_input_2d)

        np.testing.assert_allclose(result_pt_2d, result_jax_2d, rtol=1e-10, atol=1e-12)

        test_input_3d = np.random.randn(2, 3, 2).astype(np.float64)
        inputs_3d = pt.tensor("inputs_3d", dtype="float64", shape=(None, None, None))
        output_3d = loglike_op(inputs_3d)

        f_pt_3d = function([inputs_3d], output_3d)
        result_pt_3d = f_pt_3d(test_input_3d)

        f_jax_3d = function([inputs_3d], output_3d, mode="JAX")
        result_jax_3d = f_jax_3d(test_input_3d)

        np.testing.assert_allclose(result_pt_3d, result_jax_3d, rtol=1e-10, atol=1e-12)

    def test_loglike_edge_cases(self):
        """Test LogLike Op handles edge cases like nan/inf."""

        def logp_func_with_inf(x):
            """Function that can produce inf values."""
            return np.where(np.abs(x) > 10, -np.inf, -0.5 * np.sum(x**2, axis=-1))

        loglike_op = LogLike(logp_func_with_inf)

        inputs = pt.tensor("inputs", dtype="float64", shape=(None, None))
        output = loglike_op(inputs)

        # Test with extreme values
        test_input = np.array([[1.0], [15.0], [-15.0], [0.0]]).astype(np.float64)

        f_jax = function([inputs], output, mode="JAX")
        result = f_jax(test_input)

        assert np.isfinite(result[0])
        assert result[1] == -np.inf
        assert result[2] == -np.inf
        assert np.isfinite(result[3])

    def test_loglike_2d_vs_3d_inputs(self):
        """Test LogLike Op handles both 2D and 3D inputs correctly."""

        def logp_func(x):
            return -0.5 * np.sum(x**2, axis=-1)

        loglike_op = LogLike(logp_func)

        inputs_2d = pt.tensor("inputs_2d", dtype="float64", shape=(None, None))
        output_2d = loglike_op(inputs_2d)
        f_2d = function([inputs_2d], output_2d, mode="JAX")

        test_2d = np.random.randn(4, 3).astype(np.float64)
        result_2d = f_2d(test_2d)
        assert result_2d.shape == (4,)

        inputs_3d = pt.tensor("inputs_3d", dtype="float64", shape=(None, None, None))
        output_3d = loglike_op(inputs_3d)
        f_3d = function([inputs_3d], output_3d, mode="JAX")

        test_3d = np.random.randn(2, 4, 3).astype(np.float64)
        result_3d = f_3d(test_3d)
        assert result_3d.shape == (2, 4)


if __name__ == "__main__":
    test_class = TestLogLikeJAXConversion()

    print("Running LogLike JAX conversion tests...")

    try:
        test_class.test_loglike_simple_function()
        print("✓ test_loglike_simple_function passed")
    except Exception as e:
        print(f"✗ test_loglike_simple_function failed: {e}")

    try:
        test_class.test_loglike_edge_cases()
        print("✓ test_loglike_edge_cases passed")
    except Exception as e:
        print(f"✗ test_loglike_edge_cases failed: {e}")

    try:
        test_class.test_loglike_2d_vs_3d_inputs()
        print("✓ test_loglike_2d_vs_3d_inputs passed")
    except Exception as e:
        print(f"✗ test_loglike_2d_vs_3d_inputs failed: {e}")

    print("All LogLike JAX tests completed!")
