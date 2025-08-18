#   Copyright 2022 The PyMC Developers
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

"""
Native PyTensor vectorized logp implementation.

This module provides a PyTensor First approach to vectorizing log-probability
computations, eliminating the need for custom LogLike Op and ensuring automatic
JAX compatibility through native PyTensor operations.

Expert Guidance Applied:
- Uses vectorize_graph instead of custom Ops (Jesse Grabowski's recommendation)
- Eliminates numpy.apply_along_axis dependency
- Leverages existing PyTensor functionality per "PyTensor First" principle
"""

from collections.abc import Callable as CallableType

import pytensor.tensor as pt

from pytensor.graph import vectorize_graph
from pytensor.tensor import TensorVariable


def create_vectorized_logp_graph(logp_func: CallableType) -> CallableType:
    """
    Create a vectorized log-probability computation graph using native PyTensor operations.

    IMPORTANT: This function now detects the interface type and handles both compiled
    functions and symbolic expressions properly to avoid the interface mismatch issue.

    Parameters
    ----------
    logp_func : Callable
        Log-probability function that takes a single parameter vector and returns scalar logp
        Can be either a compiled PyTensor function or a callable that works with symbolic inputs

    Returns
    -------
    Callable
        Function that takes a batch of parameter vectors and returns vectorized logp values

    Notes
    -----
    This implementation follows PyTensor expert recommendations:
    - "Can the perform method of that `Loglike` op be directly written in pytensor?" - Jesse Grabowski
    - "PyTensor vectorize / vectorize_graph directly" - Ricardo
    - Fixed interface mismatch between compiled functions and symbolic variables
    - Automatic JAX support through PyTensor's existing infrastructure
    """

    # Check if logp_func is a compiled function by testing its interface
    phi_test = pt.vector("phi_test", dtype="float64")

    try:
        # Try to call logp_func with symbolic input
        logP_scalar = logp_func(phi_test)
        if hasattr(logP_scalar, "type"):  # It's a symbolic variable
            use_symbolic_interface = True
        else:
            use_symbolic_interface = False
    except (TypeError, AttributeError):
        # logp_func is a compiled function that expects numeric input
        # Fall back to LogLike Op approach for now
        use_symbolic_interface = False

    if use_symbolic_interface:
        # Direct symbolic approach (ideal case)
        phi_scalar = pt.vector("phi_scalar", dtype="float64")
        logP_scalar = logp_func(phi_scalar)

        def vectorized_logp(phi: TensorVariable) -> TensorVariable:
            """Vectorized logp using symbolic interface."""
            # Use vectorize_graph to handle batch processing
            if phi.ndim == 2:
                result = vectorize_graph(logP_scalar, replace={phi_scalar: phi})
            else:
                # Multi-path case: (L, batch_size, num_params)
                phi_reshaped = phi.reshape((-1, phi.shape[-1]))
                result_flat = vectorize_graph(logP_scalar, replace={phi_scalar: phi_reshaped})
                result = result_flat.reshape(phi.shape[:-1])

            # Handle nan/inf values
            mask = pt.isnan(result) | pt.isinf(result)
            return pt.where(mask, -pt.inf, result)

        return vectorized_logp

    else:
        # Fallback to LogLike Op for compiled functions
        # This maintains compatibility while we transition to symbolic approach
        from .pathfinder import LogLike  # Import the existing LogLike Op

        def vectorized_logp(phi: TensorVariable) -> TensorVariable:
            """Vectorized logp using LogLike Op fallback."""
            loglike_op = LogLike(logp_func)
            result = loglike_op(phi)
            return result

        return vectorized_logp


def create_scan_based_logp_graph(logp_func: CallableType) -> CallableType:
    """
    Alternative implementation using pt.scan instead of vectorize_graph.

    This provides a direct replacement for numpy.apply_along_axis using native PyTensor scan.

    Parameters
    ----------
    logp_func : Callable
        Log-probability function that takes a single parameter vector and returns scalar logp

    Returns
    -------
    Callable
        Function that takes a batch of parameter vectors and returns vectorized logp values
    """

    def scan_logp(phi: TensorVariable) -> TensorVariable:
        """Compute log-probability using pt.scan for vectorization."""
        # Use pt.scan to apply logp_func along the batch dimension
        logP_result, _ = pt.scan(
            fn=lambda phi_row: logp_func(phi_row), sequences=[phi], outputs_info=None
        )

        # Handle nan/inf values
        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        return pt.where(mask, -pt.inf, logP_result)

    return scan_logp


def create_direct_vectorized_logp(logp_func: CallableType) -> CallableType:
    """
    Direct PyTensor implementation without custom Op using pt.vectorize.

    This is the simplest approach using PyTensor's built-in vectorize functionality.

    Parameters
    ----------
    logp_func : Callable
        Log-probability function that takes a single parameter vector and returns scalar logp

    Returns
    -------
    Callable
        Function that takes a batch of parameter vectors and returns vectorized logp values
    """
    # Use PyTensor's built-in vectorize
    vectorized_logp_func = pt.vectorize(logp_func, signature="(n)->()")

    def direct_logp(phi: TensorVariable) -> TensorVariable:
        """Compute log-probability using pt.vectorize."""
        logP_result = vectorized_logp_func(phi)

        # Handle nan/inf values
        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        return pt.where(mask, -pt.inf, logP_result)

    return direct_logp
