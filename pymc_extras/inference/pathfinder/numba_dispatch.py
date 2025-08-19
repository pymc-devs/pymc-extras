#   Copyright 2024 The PyMC Developers
#   Licensed under the Apache License, Version 2.0

"""Numba dispatch conversions for Pathfinder custom operations.

This module provides Numba implementations for custom PyTensor operations
used in the Pathfinder algorithm, enabling compilation with PyTensor's
Numba backend (mode="NUMBA").

Architecture follows PyTensor patterns from:
- doc/extending/creating_a_numba_jax_op.rst
- pytensor/link/numba/dispatch/
- Existing JAX dispatch in jax_dispatch.py
"""

import numba
import numpy as np
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch import numba_funcify

# Import existing ops for registration

# Module version for tracking
__version__ = "0.1.0"


# NOTE: LogLike Op registration for Numba is intentionally removed
#
# The LogLike Op cannot be compiled with Numba due to fundamental incompatibility:
# - LogLike uses arbitrary Python function closures (logp_func)
# - Numba requires concrete, statically-typeable operations
# - Function closures from PyTensor compilation cannot be analyzed by Numba
#
# Instead, the vectorized_logp module handles Numba mode by using scan-based
# approaches that avoid LogLike Op entirely.
#
# This is documented as a known limitation in CLAUDE.md


# @numba_funcify.register(LogLike)  # DISABLED - see note above
def _disabled_numba_funcify_LogLike(op, node, **kwargs):
    """DISABLED: LogLike Op registration for Numba.

    This registration is intentionally disabled because LogLike Op
    cannot be compiled with Numba due to function closure limitations.

    The error would be:
    numba.core.errors.TypingError: Untyped global name 'actual_logp_func':
    Cannot determine Numba type of <class 'function'>

    Instead, use the scan-based approach in vectorized_logp module.
    """
    raise NotImplementedError(
        "LogLike Op cannot be compiled with Numba due to function closure limitations. "
        "Use scan-based vectorization instead."
    )


# Custom Op for Numba-compatible chi matrix computation
class NumbaChiMatrixOp(Op):
    """Numba-optimized Chi matrix computation.

    Implements sliding window chi matrix computation required for L-BFGS
    history in pathfinder algorithm. Uses efficient Numba loop optimization
    instead of PyTensor scan operations.

    This Op computes a sliding window matrix where for each position idx,
    the output contains the last J values of the diff array up to position idx.
    """

    def __init__(self, J: int):
        """Initialize with history size J.

        Parameters
        ----------
        J : int
            History size for L-BFGS algorithm
        """
        self.J = J
        super().__init__()

    def make_node(self, diff):
        """Create computation node for chi matrix.

        Parameters
        ----------
        diff : TensorVariable
            Difference array, shape (L, N)

        Returns
        -------
        Apply
            Computation node for chi matrix
        """
        diff = pt.as_tensor_variable(diff)
        # Output shape: (L, N, J) - use None for dynamic dimensions
        output = pt.tensor(
            dtype=diff.dtype,
            shape=(None, None, self.J),  # Only J is static
        )
        return Apply(self, [diff], [output])

    def perform(self, node, inputs, outputs):
        """NumPy fallback implementation for compatibility.

        This matches the JAX implementation exactly to ensure
        mathematical correctness as fallback.

        Parameters
        ----------
        node : Apply
            Computation node
        inputs : list
            Input arrays [diff]
        outputs : list
            Output arrays [chi_matrix]
        """
        diff = inputs[0]  # Shape: (L, N)
        L, N = diff.shape
        J = self.J

        # Create output matrix
        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        # Compute sliding window matrix (same logic as JAX version)
        for idx in range(L):
            # For each row idx, we want the last J values of diff up to position idx
            start_idx = max(0, idx - J + 1)
            end_idx = idx + 1

            # Get the relevant slice
            relevant_diff = diff[start_idx:end_idx]  # Shape: (actual_length, N)
            actual_length = end_idx - start_idx

            # If we have fewer than J values, pad with zeros at the beginning
            if actual_length < J:
                padding = np.zeros((J - actual_length, N), dtype=diff.dtype)
                padded_diff = np.concatenate([padding, relevant_diff], axis=0)
            else:
                padded_diff = relevant_diff

            # Assign to chi matrix
            chi_matrix[idx] = padded_diff.T  # Transpose to get (N, J)

        outputs[0][0] = chi_matrix

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.J == other.J

    def __hash__(self):
        return hash((type(self), self.J))


@numba_funcify.register(NumbaChiMatrixOp)
def numba_funcify_ChiMatrixOp(op, node, **kwargs):
    """Numba implementation for ChiMatrix sliding window computation.

    Uses Numba's optimized loop fusion and memory locality improvements
    for efficient sliding window operations. This avoids the dynamic
    indexing issues that block JAX compilation while providing better
    CPU performance through cache-friendly access patterns.

    Parameters
    ----------
    op : NumbaChiMatrixOp
        The ChiMatrix Op instance with J parameter
    node : Apply
        The computation node
    **kwargs
        Additional keyword arguments (unused)

    Returns
    -------
    callable
        Numba-compiled function for chi matrix computation
    """
    J = op.J

    @numba_basic.numba_njit(fastmath=True, cache=True)
    def chi_matrix_numba(diff):
        """Optimized sliding window using Numba loop fusion.

        Parameters
        ----------
        diff : numpy.ndarray
            Input difference array, shape (L, N)

        Returns
        -------
        numpy.ndarray
            Chi matrix with shape (L, N, J)
        """
        L, N = diff.shape
        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        # Optimized sliding window with manual loop unrolling
        for batch_idx in range(L):
            # Efficient window extraction
            start_idx = max(0, batch_idx - J + 1)
            window_size = min(J, batch_idx + 1)

            # Direct memory copy for efficiency
            for j in range(window_size):
                source_idx = start_idx + j
                target_idx = J - window_size + j
                for n in range(N):
                    chi_matrix[batch_idx, n, target_idx] = diff[source_idx, n]

        return chi_matrix

    return chi_matrix_numba


# Custom Op for Numba-compatible BFGS sampling
class NumbaBfgsSampleOp(Op):
    """Numba-optimized BFGS sampling with conditional logic.

    Handles conditional selection between dense and sparse BFGS sampling
    modes based on condition JJ >= N, using Numba's efficient conditional
    compilation instead of PyTensor's pt.switch. This avoids the dynamic
    indexing issues that block JAX compilation while providing superior
    CPU performance through Numba's optimizations.

    The Op implements the same mathematical operations as the JAX version
    but uses Numba-specific optimizations for CPU workloads:
    - Parallel processing with numba.prange
    - Optimized matrix operations and memory layouts
    - Efficient conditional branching without dynamic compilation overhead
    """

    def make_node(
        self, x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Create computation node for BFGS sampling.

        Parameters
        ----------
        x : TensorVariable
            Position array, shape (L, N)
        g : TensorVariable
            Gradient array, shape (L, N)
        alpha : TensorVariable
            Diagonal scaling array, shape (L, N)
        beta : TensorVariable
            Low-rank update matrix, shape (L, N, 2J)
        gamma : TensorVariable
            Low-rank update matrix, shape (L, 2J, 2J)
        alpha_diag : TensorVariable
            Diagonal matrix of alpha, shape (L, N, N)
        inv_sqrt_alpha_diag : TensorVariable
            Inverse sqrt of alpha diagonal, shape (L, N, N)
        sqrt_alpha_diag : TensorVariable
            Sqrt of alpha diagonal, shape (L, N, N)
        u : TensorVariable
            Random normal samples, shape (L, M, N)

        Returns
        -------
        Apply
            Computation node with two outputs: phi and logdet
        """
        # Convert all inputs to tensor variables (same as JAX version)
        inputs = [
            pt.as_tensor_variable(inp)
            for inp in [
                x,
                g,
                alpha,
                beta,
                gamma,
                alpha_diag,
                inv_sqrt_alpha_diag,
                sqrt_alpha_diag,
                u,
            ]
        ]

        # Output phi: shape (L, M, N) - same as u
        phi_out = pt.tensor(dtype=u.dtype, shape=(None, None, None))

        # Output logdet: shape (L,) - same as first dimension of x
        logdet_out = pt.tensor(dtype=u.dtype, shape=(None,))

        return Apply(self, inputs, [phi_out, logdet_out])

    def perform(self, node, inputs, outputs):
        """NumPy fallback implementation using JAX logic.

        This provides the reference implementation for mathematical correctness,
        copied directly from the JAX version to ensure identical behavior.
        The Numba-optimized version will be registered separately.
        """
        import numpy as np

        from scipy.linalg import cholesky, qr

        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u = inputs

        # Get shapes
        L, M, N = u.shape
        L, N, JJ = beta.shape

        # Define the condition: use dense when JJ >= N, sparse otherwise
        condition = JJ >= N

        # Regularization term (from pathfinder.py REGULARISATION_TERM)
        REGULARISATION_TERM = 1e-8

        if condition:
            # Dense BFGS sampling branch

            # Create identity matrix with regularization
            IdN = np.eye(N)[None, ...]
            IdN = IdN + IdN * REGULARISATION_TERM

            # Compute inverse Hessian: H_inv = sqrt_alpha_diag @ (IdN + middle_term) @ sqrt_alpha_diag
            middle_term = (
                inv_sqrt_alpha_diag
                @ beta
                @ gamma
                @ np.transpose(beta, axes=(0, 2, 1))
                @ inv_sqrt_alpha_diag
            )

            # Full inverse Hessian
            H_inv = sqrt_alpha_diag @ (IdN + middle_term) @ sqrt_alpha_diag

            # Cholesky decomposition (upper triangular)
            Lchol = np.array([cholesky(H_inv[i], lower=False) for i in range(L)])

            # Compute log determinant from Cholesky diagonal
            logdet = 2.0 * np.sum(np.log(np.abs(np.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)

            # Compute mean: mu = x - H_inv @ g
            mu = x - np.sum(H_inv * g[..., None, :], axis=-1)

            # Sample: phi = mu + Lchol @ u.T, then transpose back
            phi_transposed = mu[..., None] + Lchol @ np.transpose(u, axes=(0, 2, 1))
            phi = np.transpose(phi_transposed, axes=(0, 2, 1))

        else:
            # Sparse BFGS sampling branch

            # QR decomposition of qr_input = inv_sqrt_alpha_diag @ beta
            qr_input = inv_sqrt_alpha_diag @ beta

            # NumPy QR decomposition (applied along batch dimension)
            Q = np.zeros((L, qr_input.shape[1], qr_input.shape[2]))  # (L, N, JJ)
            R = np.zeros((L, qr_input.shape[2], qr_input.shape[2]))  # (L, JJ, JJ)
            for i in range(L):
                Q[i], R[i] = qr(qr_input[i], mode="economic")

            # Identity matrix with regularization
            IdJJ = np.eye(R.shape[1])[None, ...]
            IdJJ = IdJJ + IdJJ * REGULARISATION_TERM

            # Cholesky input: IdJJ + R @ gamma @ R.T
            Lchol_input = IdJJ + R @ gamma @ np.transpose(R, axes=(0, 2, 1))

            # Cholesky decomposition (upper triangular)
            Lchol = np.array([cholesky(Lchol_input[i], lower=False) for i in range(L)])

            # Compute log determinant: includes both Cholesky and alpha terms
            logdet_chol = 2.0 * np.sum(
                np.log(np.abs(np.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1
            )
            logdet_alpha = np.sum(np.log(alpha), axis=-1)
            logdet = logdet_chol + logdet_alpha

            # Compute inverse Hessian for sparse case: H_inv = alpha_diag + beta @ gamma @ beta.T
            H_inv = alpha_diag + (beta @ gamma @ np.transpose(beta, axes=(0, 2, 1)))

            # Compute mean: mu = x - H_inv @ g
            mu = x - np.sum(H_inv * g[..., None, :], axis=-1)

            # Complex sampling transformation for sparse case
            # First part: Q @ (Lchol - IdJJ)
            Q_Lchol_diff = Q @ (Lchol - IdJJ)

            # Second part: Q.T @ u.T
            Qt_u = np.transpose(Q, axes=(0, 2, 1)) @ np.transpose(u, axes=(0, 2, 1))

            # Combine: (Q @ (Lchol - IdJJ)) @ (Q.T @ u.T) + u.T
            combined = Q_Lchol_diff @ Qt_u + np.transpose(u, axes=(0, 2, 1))

            # Final transformation: mu + sqrt_alpha_diag @ combined
            phi_transposed = mu[..., None] + sqrt_alpha_diag @ combined
            phi = np.transpose(phi_transposed, axes=(0, 2, 1))

        outputs[0][0] = phi
        outputs[1][0] = logdet

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self))


@numba_funcify.register(NumbaBfgsSampleOp)
def numba_funcify_BfgsSampleOp(op, node, **kwargs):
    """Numba implementation with optimized conditional matrix operations.

    Uses Numba's efficient conditional compilation for optimal performance,
    avoiding the dynamic indexing issues that prevent JAX compilation while
    providing superior CPU performance through parallel processing and
    optimized memory access patterns.

    Parameters
    ----------
    op : NumbaBfgsSampleOp
        The BfgsSampleOp instance
    node : Apply
        The computation node
    **kwargs
        Additional keyword arguments (unused)

    Returns
    -------
    callable
        Numba-compiled function that performs conditional BFGS sampling
    """

    # Regularization term constant
    REGULARISATION_TERM = 1e-8

    @numba_basic.numba_njit(fastmath=True, parallel=True)
    def dense_bfgs_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Dense BFGS sampling - Numba optimized.

        Optimized for case where JJ >= N (dense matrix operations preferred).
        Uses Numba's efficient matrix operations and parallel processing.

        Parameters
        ----------
        x : numpy.ndarray
            Position array, shape (L, N)
        g : numpy.ndarray
            Gradient array, shape (L, N)
        alpha : numpy.ndarray
            Diagonal scaling array, shape (L, N)
        beta : numpy.ndarray
            Low-rank update matrix, shape (L, N, 2J)
        gamma : numpy.ndarray
            Low-rank update matrix, shape (L, 2J, 2J)
        alpha_diag : numpy.ndarray
            Diagonal matrix of alpha, shape (L, N, N)
        inv_sqrt_alpha_diag : numpy.ndarray
            Inverse sqrt of alpha diagonal, shape (L, N, N)
        sqrt_alpha_diag : numpy.ndarray
            Sqrt of alpha diagonal, shape (L, N, N)
        u : numpy.ndarray
            Random normal samples, shape (L, M, N)

        Returns
        -------
        tuple
            (phi, logdet) where phi has shape (L, M, N) and logdet has shape (L,)
        """
        L, M, N = u.shape

        # Create identity matrix with regularization
        IdN = np.eye(N) + np.eye(N) * REGULARISATION_TERM

        # Compute inverse Hessian using batched operations
        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for batch_idx in numba.prange(L):  # Parallel over batch dimension
            # Middle term computation for batch element batch_idx
            # middle_term = inv_sqrt_alpha_diag @ beta @ gamma @ beta.T @ inv_sqrt_alpha_diag
            beta_l = beta[batch_idx]  # (N, 2J)
            gamma_l = gamma[batch_idx]  # (2J, 2J)
            inv_sqrt_alpha_diag_l = inv_sqrt_alpha_diag[batch_idx]  # (N, N)
            sqrt_alpha_diag_l = sqrt_alpha_diag[batch_idx]  # (N, N)

            # Compute middle term step by step for efficiency
            temp1 = inv_sqrt_alpha_diag_l @ beta_l  # (N, 2J)
            temp2 = temp1 @ gamma_l  # (N, 2J)
            temp3 = temp2 @ beta_l.T  # (N, N)
            middle_term = temp3 @ inv_sqrt_alpha_diag_l  # (N, N)

            # Full inverse Hessian: H_inv = sqrt_alpha_diag @ (IdN + middle_term) @ sqrt_alpha_diag
            temp_matrix = IdN + middle_term
            H_inv_l = sqrt_alpha_diag_l @ temp_matrix @ sqrt_alpha_diag_l

            # Cholesky decomposition (upper triangular)
            Lchol_l = np.linalg.cholesky(H_inv_l).T

            # Log determinant from Cholesky diagonal
            logdet[batch_idx] = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))

            # Mean computation: mu = x - H_inv @ g
            mu_l = x[batch_idx] - H_inv_l @ g[batch_idx]

            # Sample generation: phi = mu + Lchol @ u.T
            for m in range(M):
                phi[batch_idx, m] = mu_l + Lchol_l @ u[batch_idx, m]

        return phi, logdet

    @numba_basic.numba_njit(fastmath=True, parallel=True)
    def sparse_bfgs_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Sparse BFGS sampling - Numba optimized.

        Optimized for case where JJ < N (sparse matrix operations preferred).
        Uses QR decomposition and memory-efficient operations.

        Parameters
        ----------
        x : numpy.ndarray
            Position array, shape (L, N)
        g : numpy.ndarray
            Gradient array, shape (L, N)
        alpha : numpy.ndarray
            Diagonal scaling array, shape (L, N)
        beta : numpy.ndarray
            Low-rank update matrix, shape (L, N, 2J)
        gamma : numpy.ndarray
            Low-rank update matrix, shape (L, 2J, 2J)
        alpha_diag : numpy.ndarray
            Diagonal matrix of alpha, shape (L, N, N)
        inv_sqrt_alpha_diag : numpy.ndarray
            Inverse sqrt of alpha diagonal, shape (L, N, N)
        sqrt_alpha_diag : numpy.ndarray
            Sqrt of alpha diagonal, shape (L, N, N)
        u : numpy.ndarray
            Random normal samples, shape (L, M, N)

        Returns
        -------
        tuple
            (phi, logdet) where phi has shape (L, M, N) and logdet has shape (L,)
        """
        L, M, N = u.shape
        JJ = beta.shape[2]

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for batch_idx in numba.prange(L):  # Parallel over batch dimension
            # QR decomposition of qr_input = inv_sqrt_alpha_diag @ beta
            qr_input_l = inv_sqrt_alpha_diag[batch_idx] @ beta[batch_idx]
            Q_l, R_l = np.linalg.qr(qr_input_l)

            # Identity matrix with regularization
            IdJJ = np.eye(JJ) + np.eye(JJ) * REGULARISATION_TERM

            # Cholesky input: IdJJ + R @ gamma @ R.T
            Lchol_input_l = IdJJ + R_l @ gamma[batch_idx] @ R_l.T

            # Cholesky decomposition (upper triangular)
            Lchol_l = np.linalg.cholesky(Lchol_input_l).T

            # Compute log determinant
            logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))
            logdet_alpha = np.sum(np.log(alpha[batch_idx]))
            logdet[batch_idx] = logdet_chol + logdet_alpha

            # Inverse Hessian for sparse case
            H_inv_l = alpha_diag[batch_idx] + beta[batch_idx] @ gamma[batch_idx] @ beta[batch_idx].T

            # Mean computation
            mu_l = x[batch_idx] - H_inv_l @ g[batch_idx]

            # Complex sampling transformation for sparse case
            Q_Lchol_diff = Q_l @ (Lchol_l - IdJJ)

            for m in range(M):
                Qt_u_lm = Q_l.T @ u[batch_idx, m]
                combined = Q_Lchol_diff @ Qt_u_lm + u[batch_idx, m]
                phi[batch_idx, m] = mu_l + sqrt_alpha_diag[batch_idx] @ combined

        return phi, logdet

    @numba_basic.numba_njit(inline="always")
    def bfgs_sample_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Conditional BFGS sampling using Numba.

        Uses efficient conditional compilation to select between dense and sparse
        algorithms based on problem dimensions. This avoids the dynamic indexing
        issues that prevent JAX compilation while providing optimal performance
        for both cases.

        Parameters
        ----------
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u : numpy.ndarray
            Input arrays for BFGS sampling

        Returns
        -------
        tuple
            (phi, logdet) arrays with sampling results
        """
        L, M, N = u.shape
        JJ = beta.shape[2]

        # Numba-optimized conditional compilation
        if JJ >= N:
            return dense_bfgs_numba(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )
        else:
            return sparse_bfgs_numba(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )

    return bfgs_sample_numba
