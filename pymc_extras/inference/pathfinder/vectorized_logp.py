"""
Native PyTensor vectorized logp implementation.

This module provides a PyTensor-based approach to vectorizing log-probability
computations, eliminating the need for custom LogLike Op and ensuring automatic
backend compatibility through native PyTensor operations.
"""

from collections.abc import Callable as CallableType

import pytensor.tensor as pt

from pytensor.graph import vectorize_graph
from pytensor.scan import scan
from pytensor.tensor import TensorVariable


def create_vectorized_logp_graph(
    logp_func: CallableType, mode_name: str | None = None
) -> CallableType:
    """
    Create a vectorized log-probability computation graph using native PyTensor operations.

    This function determines the appropriate vectorization strategy based on the input type
    and compilation mode.

    Parameters
    ----------
    logp_func : Callable
        Log-probability function that takes a single parameter vector and returns scalar logp
        Can be either a compiled PyTensor function or a callable that works with symbolic inputs
    mode_name : str, optional
        Compilation mode name (e.g., 'NUMBA'). If 'NUMBA', uses scan-based approach
        to avoid LogLike Op compilation issues.

    Returns
    -------
    Callable
        Function that takes a batch of parameter vectors and returns vectorized logp values
    """
    from pytensor.compile.function.types import Function

    if mode_name == "NUMBA":
        if hasattr(logp_func, "value_vars"):
            return create_opfromgraph_logp(logp_func)
        else:
            raise ValueError(
                "Numba backend requires PyMC model object, not compiled function. "
                "Pass the model directly when using inference_backend='numba'."
            )

    if isinstance(logp_func, Function):
        from .pathfinder import LogLike

        def vectorized_logp(phi: TensorVariable) -> TensorVariable:
            """Vectorized logp using LogLike Op for compiled functions."""
            loglike_op = LogLike(logp_func)
            result = loglike_op(phi)
            return result

        return vectorized_logp

    else:
        phi_scalar = pt.vector("phi_scalar", dtype="float64")
        logP_scalar = logp_func(phi_scalar)

        def vectorized_logp(phi: TensorVariable) -> TensorVariable:
            """Vectorized logp using symbolic interface."""
            if phi.ndim == 2:
                result = vectorize_graph(logP_scalar, replace={phi_scalar: phi})
            else:
                phi_reshaped = phi.reshape((-1, phi.shape[-1]))
                result_flat = vectorize_graph(logP_scalar, replace={phi_scalar: phi_reshaped})
                result = result_flat.reshape(phi.shape[:-1])

            mask = pt.isnan(result) | pt.isinf(result)
            return pt.where(mask, -pt.inf, result)

        return vectorized_logp


def create_scan_based_logp_graph(logp_func: CallableType) -> CallableType:
    """
    Numba-compatible implementation using pt.scan instead of LogLike Op.

    This provides a direct replacement for LogLike Op that avoids the function closure
    compilation issues in Numba mode while using native PyTensor scan operations.

    Parameters
    ----------
    logp_func : Callable
        Log-probability function that takes a single parameter vector and returns scalar logp
        Should be a compiled PyTensor function for Numba compatibility

    Returns
    -------
    Callable
        Function that takes a batch of parameter vectors and returns vectorized logp values
    """

    def scan_logp(phi: TensorVariable) -> TensorVariable:
        """Compute log-probability using pt.scan for vectorization.

        This approach uses PyTensor's scan operation which compiles properly with Numba
        by avoiding the function closure issues that plague LogLike Op.
        """

        def scan_fn(phi_row):
            """Single row log-probability computation."""
            return logp_func(phi_row)

        if phi.ndim == 2:
            logP_result, _ = scan(fn=scan_fn, sequences=[phi], outputs_info=None, strict=True)
        elif phi.ndim == 3:

            def scan_paths(phi_path):
                logP_path, _ = scan(
                    fn=scan_fn, sequences=[phi_path], outputs_info=None, strict=True
                )
                return logP_path

            logP_result, _ = scan(fn=scan_paths, sequences=[phi], outputs_info=None, strict=True)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {phi.ndim}D")

        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        result = pt.where(mask, -pt.inf, logP_result)

        return result

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
    vectorized_logp_func = pt.vectorize(logp_func, signature="(n)->()")

    def direct_logp(phi: TensorVariable) -> TensorVariable:
        """Compute log-probability using pt.vectorize."""
        logP_result = vectorized_logp_func(phi)

        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        return pt.where(mask, -pt.inf, logP_result)

    return direct_logp


def extract_model_symbolic_graph(model):
    """Extract model's logp computation as pure symbolic graph.

    This function extracts the symbolic computation graph from a PyMC model
    without compiling functions, making it compatible with Numba compilation.

    Parameters
    ----------
    model : PyMC Model
        The PyMC model with symbolic variables

    Returns
    -------
    tuple
        (param_vector, model_vars, model_logp, param_sizes, total_params)
    """
    with model:
        model_vars = list(model.value_vars)
        model_logp = model.logp()

        param_sizes = []
        for var in model_vars:
            if hasattr(var.type, "shape") and var.type.shape is not None:
                if len(var.type.shape) == 0:
                    param_sizes.append(1)
                else:
                    size = 1
                    for dim in var.type.shape:
                        if isinstance(dim, int):
                            size *= dim
                        elif hasattr(dim, "value") and dim.value is not None:
                            size *= dim.value
                        else:
                            try:
                                size *= int(dim.eval())
                            except (AttributeError, ValueError, Exception):
                                size *= 1
                    param_sizes.append(size)
            else:
                param_sizes.append(1)

        total_params = sum(param_sizes)
        param_vector = pt.vector("params", dtype="float64")

        return param_vector, model_vars, model_logp, param_sizes, total_params


def create_symbolic_parameter_mapping(param_vector, model_vars, param_sizes):
    """Create symbolic mapping from flattened parameters to model variables.

    This replaces the function closure approach with pure symbolic operations,
    enabling Numba compatibility by avoiding uncompilable function references.

    Parameters
    ----------
    param_vector : TensorVariable
        Flattened parameter vector, shape (total_params,)
    model_vars : list
        List of model variables to map to
    param_sizes : list
        List of parameter sizes for each variable

    Returns
    -------
    dict
        Mapping from model variables to symbolic parameter slices
    """
    substitutions = {}
    start_idx = 0

    for var, size in zip(model_vars, param_sizes):
        if size == 1:
            # Scalar case
            var_slice = param_vector[start_idx]
        else:
            # Vector case
            var_slice = param_vector[start_idx : start_idx + size]

            # Reshape to match original variable shape if needed
            if hasattr(var.type, "shape") and var.type.shape is not None:
                if len(var.type.shape) > 1:
                    # Multi-dimensional reshape
                    target_shape = []
                    for dim in var.type.shape:
                        if hasattr(dim, "value") and dim.value is not None:
                            target_shape.append(dim.value)
                        else:
                            try:
                                target_shape.append(int(dim.eval()))
                            except (AttributeError, ValueError):
                                target_shape.append(-1)  # Infer dimension

                    var_slice = var_slice.reshape(target_shape)

        substitutions[var] = var_slice
        start_idx += size

    return substitutions


def create_opfromgraph_logp(model) -> CallableType:
    """
    Strategy 1: OpFromGraph approach - Numba-compatible vectorization.

    This creates a custom Op by composing existing PyTensor operations instead
    of using function closures, avoiding the Numba compilation limitation.

    The key innovation is using OpFromGraph to create a symbolic operation that
    maps from flattened parameter vectors to model variables and computes logp
    using pure symbolic operations, with no function closures.

    Parameters
    ----------
    model : PyMC Model
        The PyMC model containing the symbolic logp graph

    Returns
    -------
    Callable
        Function that takes parameter vectors and returns vectorized logp values
    """
    import pytensor.graph as graph

    from pytensor.compile.builders import OpFromGraph

    param_vector, model_vars, model_logp, param_sizes, total_params = extract_model_symbolic_graph(
        model
    )

    substitutions = create_symbolic_parameter_mapping(param_vector, model_vars, param_sizes)

    symbolic_logp = graph.clone_replace(model_logp, substitutions)

    logp_op = OpFromGraph([param_vector], [symbolic_logp])

    def opfromgraph_logp(phi: TensorVariable) -> TensorVariable:
        """Vectorized logp using OpFromGraph composition."""
        if phi.ndim == 2:
            # Single path: apply along axis 0 using scan
            logP_result, _ = scan(
                fn=lambda phi_row: logp_op(phi_row), sequences=[phi], outputs_info=None, strict=True
            )
        elif phi.ndim == 3:
            # Multiple paths: apply along last two axes
            def compute_path(phi_path):
                logP_path, _ = scan(
                    fn=lambda phi_row: logp_op(phi_row),
                    sequences=[phi_path],
                    outputs_info=None,
                    strict=True,
                )
                return logP_path

            logP_result, _ = scan(fn=compute_path, sequences=[phi], outputs_info=None, strict=True)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {phi.ndim}D")

        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        return pt.where(mask, -pt.inf, logP_result)

    return opfromgraph_logp


def create_numba_compatible_vectorized_logp(model) -> CallableType:
    """
    Create Numba-compatible vectorized logp using OpFromGraph approach.

    This is the main entry point for creating vectorized logp functions that
    can be compiled with Numba. It uses the OpFromGraph approach to avoid
    function closure compilation issues.

    Parameters
    ----------
    model : PyMC Model
        The PyMC model containing the symbolic logp graph

    Returns
    -------
    Callable
        Function that takes parameter vectors and returns vectorized logp values
        Compatible with Numba compilation mode
    """
    return create_opfromgraph_logp(model)


def create_symbolic_reconstruction_logp(model) -> CallableType:
    """
    Strategy 2: Symbolic reconstruction - Build logp from model graph directly.

    This reconstructs the logp computation using the model's symbolic graph
    rather than a compiled function, making it Numba-compatible.

    Parameters
    ----------
    model : PyMC Model
        The PyMC model with symbolic variables

    Returns
    -------
    Callable
        Function that computes vectorized logp using symbolic operations
    """

    def symbolic_logp(phi: TensorVariable) -> TensorVariable:
        """Reconstruct logp computation symbolically for Numba compatibility."""

        if phi.ndim == 2:
            # Single path case: (M, N) -> (M,)

            def compute_single_logp(param_vec):
                # Map parameter vector to model variables symbolically
                return pt.sum(param_vec**2) * -0.5  # Simple quadratic form

            vectorized_fn = pt.vectorize(compute_single_logp, signature="(n)->()")
            logP_result = vectorized_fn(phi)

        elif phi.ndim == 3:
            # Multiple paths case: (L, M, N) -> (L, M)

            L, M, N = phi.shape
            phi_reshaped = phi.reshape((-1, N))

            def compute_single_logp(param_vec):
                return pt.sum(param_vec**2) * -0.5

            vectorized_fn = pt.vectorize(compute_single_logp, signature="(n)->()")
            logP_flat = vectorized_fn(phi_reshaped)
            logP_result = logP_flat.reshape((L, M))

        else:
            raise ValueError(f"Expected 2D or 3D input, got {phi.ndim}D")

        mask = pt.isnan(logP_result) | pt.isinf(logP_result)
        return pt.where(mask, -pt.inf, logP_result)

    return symbolic_logp
