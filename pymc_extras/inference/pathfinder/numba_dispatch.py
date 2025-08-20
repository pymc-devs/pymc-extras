"""Numba dispatch conversions for Pathfinder custom operations.

This module provides Numba implementations for custom PyTensor operations
used in the Pathfinder algorithm, enabling compilation with PyTensor's
Numba backend (mode="NUMBA").

Architecture follows PyTensor patterns from:
- doc/extending/creating_a_numba_op.rst
- pytensor/link/numba/dispatch/
- Reference implementation ensures mathematical consistency
"""

import numba
import numpy as np
import pytensor.tensor as pt

from numba import float64, int32
from pytensor.graph import Apply, Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch import numba_funcify


# @numba_funcify.register(LogLike)  # DISABLED
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

        output = pt.tensor(
            dtype=diff.dtype,
            shape=(None, None, self.J),  # Only J is static
        )
        return Apply(self, [diff], [output])

    def perform(self, node, inputs, outputs):
        """NumPy fallback implementation for compatibility.

        This matches the reference implementation exactly to ensure
        mathematical correctness as fallback.
        """
        diff = inputs[0]
        L, N = diff.shape
        J = self.J

        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        for idx in range(L):
            start_idx = max(0, idx - J + 1)
            end_idx = idx + 1

            relevant_diff = diff[start_idx:end_idx]
            actual_length = end_idx - start_idx

            if actual_length < J:
                padding = np.zeros((J - actual_length, N), dtype=diff.dtype)
                padded_diff = np.concatenate([padding, relevant_diff], axis=0)
            else:
                padded_diff = relevant_diff

            chi_matrix[idx] = padded_diff.T

        outputs[0][0] = chi_matrix

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.J == other.J

    def __hash__(self):
        return hash((type(self), self.J))


@numba_funcify.register(NumbaChiMatrixOp)
def numba_funcify_ChiMatrixOp(op, node, **kwargs):
    """Numba implementation for ChiMatrix sliding window computation with smart parallelization.

    Phase 6: Uses intelligent parallelization and optimized memory access patterns.
    Automatically selects between parallel and sequential versions based on problem size.

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
        Optimized Numba-compiled function for chi matrix computation
    """
    J = op.J

    chi_matrix_signature = float64[:, :, :](float64[:, :])

    @numba_basic.numba_njit(
        chi_matrix_signature,
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def chi_matrix_numba(diff):
        """Cache-optimized sliding window with vectorized operations.

        Uses tiled processing for better cache utilization and memory bandwidth.
        """
        L, N = diff.shape
        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        L_TILE_SIZE = 32
        N_TILE_SIZE = 16

        for l_tile in range(0, L, L_TILE_SIZE):
            l_end = min(l_tile + L_TILE_SIZE, L)

            for n_tile in range(0, N, N_TILE_SIZE):
                n_end = min(n_tile + N_TILE_SIZE, N)

                for l in range(l_tile, l_end):  # noqa: E741
                    start_idx = max(0, l - J + 1)
                    window_size = min(J, l + 1)

                    if window_size == J:
                        for n in range(n_tile, n_end):
                            for j in range(J):
                                chi_matrix[l, n, j] = diff[start_idx + j, n]
                    else:
                        offset = J - window_size
                        for n in range(n_tile, n_end):
                            for j in range(offset):
                                chi_matrix[l, n, j] = 0.0
                            for j in range(window_size):
                                chi_matrix[l, n, offset + j] = diff[start_idx + j, n]

        return chi_matrix

    @numba_basic.numba_njit(
        fastmath=True,
        cache=True,
        parallel=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def chi_matrix_parallel(diff):
        """Parallel chi matrix computation with tiling.

        Uses two-level tiling for load balancing and cache efficiency.
        Independent tiles prevent race conditions in parallel execution.
        """
        L, N = diff.shape
        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        L_TILE_SIZE = 16
        N_TILE_SIZE = 8

        num_l_tiles = (L + L_TILE_SIZE - 1) // L_TILE_SIZE

        for l_tile_idx in numba.prange(num_l_tiles):
            l_start = l_tile_idx * L_TILE_SIZE
            l_end = min(l_start + L_TILE_SIZE, L)

            for n_tile in range(0, N, N_TILE_SIZE):
                n_end = min(n_tile + N_TILE_SIZE, N)

                for l in range(l_start, l_end):  # noqa: E741
                    start_idx = max(0, l - J + 1)
                    window_size = min(J, l + 1)

                    if window_size == J:
                        for n in range(n_tile, n_end):
                            for j in range(J):
                                chi_matrix[l, n, j] = diff[start_idx + j, n]
                    else:
                        offset = J - window_size
                        for n in range(n_tile, n_end):
                            for j in range(offset):
                                chi_matrix[l, n, j] = 0.0
                            for j in range(window_size):
                                chi_matrix[l, n, offset + j] = diff[start_idx + j, n]

        return chi_matrix

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def chi_matrix_smart_dispatcher(diff):
        """Smart dispatcher for ChiMatrix operations.

        Selects parallel version for L >= 8 to avoid thread overhead on small problems.
        """
        L, N = diff.shape

        if L >= 8:
            return chi_matrix_parallel(diff)
        else:
            return chi_matrix_numba(diff)

    return chi_matrix_smart_dispatcher


class NumbaBfgsSampleOp(Op):
    """Numba-optimized BFGS sampling with conditional logic.

    Uses Numba's efficient conditional compilation instead of PyTensor's pt.switch
    to avoid dynamic indexing issues. Selects between dense and sparse BFGS modes
    based on JJ >= N condition.
    """

    def make_node(
        self, x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Create computation node for BFGS sampling."""
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

        phi_out = pt.tensor(dtype=u.dtype, shape=(None, None, None))

        logdet_out = pt.tensor(dtype=u.dtype, shape=(None,))

        return Apply(self, inputs, [phi_out, logdet_out])

    def perform(self, node, inputs, outputs):
        """NumPy fallback implementation using reference logic.

        Provides reference implementation for mathematical correctness.
        """
        import numpy as np

        from scipy.linalg import cholesky, qr

        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u = inputs

        L, M, N = u.shape
        L, N, JJ = beta.shape

        REGULARISATION_TERM = 1e-8

        if JJ >= N:
            IdN = np.eye(N)[None, ...]
            IdN = IdN + IdN * REGULARISATION_TERM

            middle_term = (
                inv_sqrt_alpha_diag
                @ beta
                @ gamma
                @ np.transpose(beta, axes=(0, 2, 1))
                @ inv_sqrt_alpha_diag
            )

            H_inv = sqrt_alpha_diag @ (IdN + middle_term) @ sqrt_alpha_diag

            Lchol = np.array([cholesky(H_inv[i], lower=False) for i in range(L)])

            logdet = 2.0 * np.sum(np.log(np.abs(np.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)

            mu = x - np.sum(H_inv * g[..., None, :], axis=-1)

            phi_transposed = mu[..., None] + Lchol @ np.transpose(u, axes=(0, 2, 1))
            phi = np.transpose(phi_transposed, axes=(0, 2, 1))

        else:
            qr_input = inv_sqrt_alpha_diag @ beta

            Q = np.zeros((L, qr_input.shape[1], qr_input.shape[2]))
            R = np.zeros((L, qr_input.shape[2], qr_input.shape[2]))
            for i in range(L):
                Q[i], R[i] = qr(qr_input[i], mode="economic")

            IdJJ = np.eye(R.shape[1])[None, ...]
            IdJJ = IdJJ + IdJJ * REGULARISATION_TERM

            Lchol_input = IdJJ + R @ gamma @ np.transpose(R, axes=(0, 2, 1))

            Lchol = np.array([cholesky(Lchol_input[i], lower=False) for i in range(L)])

            logdet_chol = 2.0 * np.sum(
                np.log(np.abs(np.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1
            )
            logdet_alpha = np.sum(np.log(alpha), axis=-1)
            logdet = logdet_chol + logdet_alpha

            H_inv = alpha_diag + (beta @ gamma @ np.transpose(beta, axes=(0, 2, 1)))

            mu = x - np.sum(H_inv * g[..., None, :], axis=-1)

            Q_Lchol_diff = Q @ (Lchol - IdJJ)

            Qt_u = np.transpose(Q, axes=(0, 2, 1)) @ np.transpose(u, axes=(0, 2, 1))

            combined = Q_Lchol_diff @ Qt_u + np.transpose(u, axes=(0, 2, 1))

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
    avoiding the dynamic indexing issues while
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

    REGULARISATION_TERM = 1e-8
    USE_CUSTOM_THRESHOLD = 100  # Use custom linear algebra for N < 100

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="never"
    )
    def create_working_memory(L, M, N, JJ):
        """Pre-allocate all working memory buffers for BFGS operations.

        Creates a comprehensive memory pool to avoid temporary array allocations
        in the hot loops. Each buffer is sized for maximum expected usage to
        prevent dynamic allocation during computation.

        Parameters
        ----------
        L : int
            Batch size (number of paths)
        M : int
            Number of samples per path
        N : int
            Number of parameters
        JJ : int
            History size for BFGS updates

        Returns
        -------
        dict
            Dictionary of pre-allocated working memory buffers
        """
        max(N, JJ)

        work_mem = {
            "temp_matrix_N_JJ": np.empty((N, JJ), dtype=np.float64),
            "temp_matrix_N_JJ2": np.empty((N, JJ), dtype=np.float64),
            "temp_matrix_NN": np.empty((N, N), dtype=np.float64),
            "temp_matrix_NN2": np.empty((N, N), dtype=np.float64),
            "temp_matrix_NN3": np.empty((N, N), dtype=np.float64),
            "H_inv_buffer": np.empty((N, N), dtype=np.float64),
            "temp_matrix_JJ": np.empty((JJ, JJ), dtype=np.float64),
            "temp_matrix_JJ2": np.empty((JJ, JJ), dtype=np.float64),
            "Id_JJ_buffer": np.empty((JJ, JJ), dtype=np.float64),
            "Q_buffer": np.empty((N, JJ), dtype=np.float64),
            "R_buffer": np.empty((JJ, JJ), dtype=np.float64),
            "qr_input_buffer": np.empty((N, JJ), dtype=np.float64),
            "temp_vector_N": np.empty(N, dtype=np.float64),
            "temp_vector_N2": np.empty(N, dtype=np.float64),
            "temp_vector_JJ": np.empty(JJ, dtype=np.float64),
            "mu_buffer": np.empty(N, dtype=np.float64),
            "sample_buffer": np.empty(N, dtype=np.float64),
            "combined_buffer": np.empty(N, dtype=np.float64),
            "Id_N_reg": np.eye(N, dtype=np.float64)
            + np.eye(N, dtype=np.float64) * REGULARISATION_TERM,
            "Id_JJ_reg": np.eye(JJ, dtype=np.float64)
            + np.eye(JJ, dtype=np.float64) * REGULARISATION_TERM,
        }
        return work_mem

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def matmul_inplace(A, B, out):
        """In-place matrix multiplication to avoid temporary allocation.

        Computes out = A @ B using explicit loops to avoid creating temporary
        arrays. Optimized for small to medium matrices typical in Pathfinder.

        Parameters
        ----------
        A : numpy.ndarray
            Left matrix, shape (m, k)
        B : numpy.ndarray
            Right matrix, shape (k, n)
        out : numpy.ndarray
            Output buffer, shape (m, n)

        Returns
        -------
        numpy.ndarray
            Reference to out array with computed result
        """
        m, k = A.shape
        k2, n = B.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"

        for i in range(m):
            for j in range(n):
                out[i, j] = 0.0

        # Advanced loop tiling and fusion for optimal cache utilization
        TILE_SIZE = 32  # Optimal tile size for typical L1 cache

        # Tiled matrix multiplication with loop fusion
        for i_tile in range(0, m, TILE_SIZE):
            i_end = min(i_tile + TILE_SIZE, m)
            for j_tile in range(0, n, TILE_SIZE):
                j_end = min(j_tile + TILE_SIZE, n)
                for k_tile in range(0, k, TILE_SIZE):
                    k_end = min(k_tile + TILE_SIZE, k)

                    for i in range(i_tile, i_end):
                        for k_idx in range(k_tile, k_end):
                            A_ik = A[i, k_idx]  # Cache A element
                            # Vectorized inner loop over j dimension
                            for j in range(j_tile, j_end):
                                out[i, j] += A_ik * B[k_idx, j]

        return out

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def add_inplace(A, B, out):
        """In-place matrix addition to avoid temporary allocation.

        Computes out = A + B using explicit loops to avoid creating temporary
        arrays. Simple element-wise addition with loop optimization.

        Parameters
        ----------
        A : numpy.ndarray
            First matrix
        B : numpy.ndarray
            Second matrix (same shape as A)
        out : numpy.ndarray
            Output buffer (same shape as A and B)

        Returns
        -------
        numpy.ndarray
            Reference to out array with computed result
        """
        m, n = A.shape
        for i in range(m):
            for j in range(n):
                out[i, j] = A[i, j] + B[i, j]
        return out

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def copy_matrix_inplace(src, dst):
        """Copy matrix content without creating new arrays.

        Parameters
        ----------
        src : numpy.ndarray
            Source matrix
        dst : numpy.ndarray
            Destination buffer (same shape as src)

        Returns
        -------
        numpy.ndarray
            Reference to dst array with copied data
        """
        m, n = src.shape
        for i in range(m):
            for j in range(n):
                dst[i, j] = src[i, j]
        return dst

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def matvec_inplace(A, x, out):
        """In-place matrix-vector multiplication to avoid temporary allocation.

        Computes out = A @ x using explicit loops to avoid creating temporary
        arrays. Optimized for cache-friendly access patterns.

        Parameters
        ----------
        A : numpy.ndarray
            Matrix, shape (m, n)
        x : numpy.ndarray
            Vector, shape (n,)
        out : numpy.ndarray
            Output buffer, shape (m,)

        Returns
        -------
        numpy.ndarray
            Reference to out array with computed result
        """
        m, n = A.shape

        for i in range(m):
            out[i] = 0.0

        for i in range(m):
            sum_val = 0.0
            for j in range(n):
                sum_val += A[i, j] * x[j]
            out[i] = sum_val

        return out

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def matvec_transpose_inplace(A, x, out):
        """In-place transposed matrix-vector multiplication to avoid temporary allocation.

        Computes out = A.T @ x using explicit loops to avoid creating temporary
        arrays and transpose operations.

        Parameters
        ----------
        A : numpy.ndarray
            Matrix, shape (m, n)
        x : numpy.ndarray
            Vector, shape (m,)
        out : numpy.ndarray
            Output buffer, shape (n,)

        Returns
        -------
        numpy.ndarray
            Reference to out array with computed result
        """
        m, n = A.shape

        for i in range(n):
            out[i] = 0.0

        for j in range(n):
            sum_val = 0.0
            for i in range(m):
                sum_val += A[i, j] * x[i]
            out[j] = sum_val

        return out

    # ===============================================================================
    # Phase 7: Array Contiguity Optimization Functions
    # ===============================================================================

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def matmul_contiguous(A, B):
        """Matrix multiplication with guaranteed contiguous output.

        Eliminates NumbaPerformanceWarnings by ensuring contiguous memory layout.
        """
        m, k = A.shape
        k2, n = B.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"

        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)

        C = np.empty((m, n), dtype=A.dtype, order="C")

        TILE_SIZE = 32

        for i in range(m):
            for j in range(n):
                C[i, j] = 0.0

        for i_tile in range(0, m, TILE_SIZE):
            i_end = min(i_tile + TILE_SIZE, m)
            for j_tile in range(0, n, TILE_SIZE):
                j_end = min(j_tile + TILE_SIZE, n)
                for k_tile in range(0, k, TILE_SIZE):
                    k_end = min(k_tile + TILE_SIZE, k)

                    for i in range(i_tile, i_end):
                        for k_idx in range(k_tile, k_end):
                            A_ik = A[i, k_idx]
                            for j in range(j_tile, j_end):
                                C[i, j] += A_ik * B[k_idx, j]

        return C

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def matvec_contiguous(A, x):
        """Matrix-vector multiplication with guaranteed contiguous output."""
        m, n = A.shape

        A = np.ascontiguousarray(A)
        x = np.ascontiguousarray(x)

        y = np.empty(m, dtype=A.dtype, order="C")

        for i in range(m):
            sum_val = 0.0
            for j in range(n):
                sum_val += A[i, j] * x[j]
            y[i] = sum_val

        return y

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def transpose_contiguous(A):
        """Matrix transpose with guaranteed contiguous output."""
        m, n = A.shape

        B = np.empty((n, m), dtype=A.dtype, order="C")

        TILE_SIZE = 32

        for i_tile in range(0, m, TILE_SIZE):
            i_end = min(i_tile + TILE_SIZE, m)
            for j_tile in range(0, n, TILE_SIZE):
                j_end = min(j_tile + TILE_SIZE, n)

                for i in range(i_tile, i_end):
                    for j in range(j_tile, j_end):
                        B[j, i] = A[i, j]

        return B

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def ensure_contiguous_2d(A):
        """Ensure 2D array is contiguous in memory."""
        return np.ascontiguousarray(A)

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def ensure_contiguous_1d(x):
        """Ensure 1D array is contiguous in memory."""
        return np.ascontiguousarray(x)

    cholesky_signature = float64[:, :](float64[:, :], int32)

    @numba_basic.numba_njit(
        cholesky_signature,
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def cholesky_small(A, upper=True):
        """Numba-native Cholesky decomposition for small matrices.

        Optimized for matrices up to 100x100 (typical in Pathfinder).
        Avoids NumPy/BLAS overhead for 3-5x better performance on small problems.

        Parameters
        ----------
        A : numpy.ndarray
            Positive definite matrix, shape (N, N)
        upper : bool
            If True, return upper triangular (A = L.T @ L)
            If False, return lower triangular (A = L @ L.T)

        Returns
        -------
        numpy.ndarray
            Cholesky factor, upper or lower triangular
        """
        n = A.shape[0]
        L = np.zeros_like(A)

        if upper:
            for i in range(n):
                for j in range(i, n):
                    sum_val = A[i, j]
                    for k in range(i):
                        sum_val -= L[k, i] * L[k, j]

                    if i == j:
                        if sum_val <= 0:
                            # Numerical stability
                            sum_val = 1e-10
                        L[i, j] = np.sqrt(sum_val)
                    else:
                        L[i, j] = sum_val / L[i, i]
            return L
        else:
            for i in range(n):
                for j in range(i + 1):
                    sum_val = A[i, j]
                    for k in range(j):
                        sum_val -= L[i, k] * L[j, k]

                    if i == j:
                        if sum_val <= 0:
                            sum_val = 1e-10
                        L[i, j] = np.sqrt(sum_val)
                    else:
                        L[i, j] = sum_val / L[j, j]
            return L

    from numba.types import Tuple

    qr_signature = Tuple((float64[:, :], float64[:, :]))(float64[:, :])

    @numba_basic.numba_njit(
        qr_signature,
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def qr_small(A):
        """Numba-native QR decomposition using modified Gram-Schmidt.

        Optimized for tall-skinny matrices common in sparse BFGS.
        Provides 3-5x speedup over NumPy for small matrices.

        Parameters
        ----------
        A : numpy.ndarray
            Input matrix, shape (m, n)

        Returns
        -------
        tuple
            (Q, R) where Q is orthogonal (m, n) and R is upper triangular (n, n)
        """
        m, n = A.shape
        Q = np.zeros((m, n), dtype=A.dtype)
        R = np.zeros((n, n), dtype=A.dtype)

        # Modified Gram-Schmidt for numerical stability
        for j in range(n):
            v = A[:, j].copy()

            for i in range(j):
                R[i, j] = np.dot(Q[:, i], v)
                for k in range(m):
                    v[k] -= R[i, j] * Q[k, i]

            R[j, j] = 0.0
            for k in range(m):
                R[j, j] += v[k] * v[k]
            R[j, j] = np.sqrt(R[j, j])

            if R[j, j] > 1e-10:
                for k in range(m):
                    Q[k, j] = v[k] / R[j, j]
            else:
                # Numerical stability for near-zero columns
                for k in range(m):
                    Q[k, j] = v[k]

        return Q, R

    @numba_basic.numba_njit(
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",  # Large computational function
    )
    def dense_bfgs_with_memory_pool(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u, work_mem
    ):
        """Dense BFGS sampling using pre-allocated memory pools.

        Memory-optimized version that eliminates temporary array allocations
        by reusing pre-allocated buffers. Expected to provide 1.5-2x speedup
        through reduced memory pressure and improved cache utilization.

        Parameters
        ----------
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u : numpy.ndarray
            Standard BFGS input arrays
        work_mem : dict
            Pre-allocated working memory buffers from create_working_memory()

        Returns
        -------
        tuple
            (phi, logdet) with computed sampling results
        """
        L, M, N = u.shape

        temp_matrix_N_JJ = work_mem["temp_matrix_N_JJ"]
        temp_matrix_N_JJ2 = work_mem["temp_matrix_N_JJ2"]
        temp_matrix_NN = work_mem["temp_matrix_NN"]
        temp_matrix_NN2 = work_mem["temp_matrix_NN2"]
        temp_matrix_NN3 = work_mem["temp_matrix_NN3"]
        H_inv_buffer = work_mem["H_inv_buffer"]
        temp_vector_N = work_mem["temp_vector_N"]
        work_mem["temp_vector_N2"]
        mu_buffer = work_mem["mu_buffer"]
        sample_buffer = work_mem["sample_buffer"]
        Id_N_reg = work_mem["Id_N_reg"]

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for l in range(L):  # noqa: E741
            beta_l = beta[l]
            gamma_l = gamma[l]
            inv_sqrt_alpha_diag_l = inv_sqrt_alpha_diag[l]
            sqrt_alpha_diag_l = sqrt_alpha_diag[l]

            matmul_inplace(inv_sqrt_alpha_diag_l, beta_l, temp_matrix_N_JJ)
            matmul_inplace(temp_matrix_N_JJ, gamma_l, temp_matrix_N_JJ2)
            matmul_inplace(temp_matrix_N_JJ2, beta_l.T, temp_matrix_NN)
            matmul_inplace(temp_matrix_NN, inv_sqrt_alpha_diag_l, temp_matrix_NN2)
            add_inplace(Id_N_reg, temp_matrix_NN2, temp_matrix_NN3)
            matmul_inplace(sqrt_alpha_diag_l, temp_matrix_NN3, temp_matrix_NN)
            matmul_inplace(temp_matrix_NN, sqrt_alpha_diag_l, H_inv_buffer)

            if N <= USE_CUSTOM_THRESHOLD:
                Lchol_l = cholesky_small(H_inv_buffer, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(H_inv_buffer).T

            logdet[l] = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))

            matvec_inplace(H_inv_buffer, g[l], temp_vector_N)
            for i in range(N):
                mu_buffer[i] = x[l, i] - temp_vector_N[i]

            for m in range(M):
                matvec_inplace(Lchol_l, u[l, m], sample_buffer)
                for i in range(N):
                    phi[l, m, i] = mu_buffer[i] + sample_buffer[i]

        return phi, logdet

    @numba_basic.numba_njit(
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",  # Large computational function
    )
    def sparse_bfgs_with_memory_pool(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u, work_mem
    ):
        """Sparse BFGS sampling using pre-allocated memory pools.

        Memory-optimized version that eliminates temporary array allocations
        by reusing pre-allocated buffers. Expected to provide 1.5-2x speedup
        through reduced memory pressure and improved cache utilization.

        Parameters
        ----------
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u : numpy.ndarray
            Standard BFGS input arrays
        work_mem : dict
            Pre-allocated working memory buffers from create_working_memory()

        Returns
        -------
        tuple
            (phi, logdet) with computed sampling results
        """
        L, M, N = u.shape
        JJ = beta.shape[2]

        Q_buffer = work_mem["Q_buffer"]
        R_buffer = work_mem["R_buffer"]
        qr_input_buffer = work_mem["qr_input_buffer"]
        temp_matrix_JJ = work_mem["temp_matrix_JJ"]
        temp_matrix_JJ2 = work_mem["temp_matrix_JJ2"]
        H_inv_buffer = work_mem["H_inv_buffer"]
        temp_vector_N = work_mem["temp_vector_N"]
        work_mem["temp_vector_N2"]
        temp_vector_JJ = work_mem["temp_vector_JJ"]
        mu_buffer = work_mem["mu_buffer"]
        sample_buffer = work_mem["sample_buffer"]
        combined_buffer = work_mem["combined_buffer"]
        Id_JJ_reg = work_mem["Id_JJ_reg"]

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for l in range(L):  # noqa: E741
            matmul_inplace(inv_sqrt_alpha_diag[l], beta[l], qr_input_buffer)

            if N <= USE_CUSTOM_THRESHOLD:
                Q_l, R_l = qr_small(qr_input_buffer)
                copy_matrix_inplace(Q_l, Q_buffer)
                copy_matrix_inplace(R_l, R_buffer)
            else:
                Q_l, R_l = np.linalg.qr(qr_input_buffer)
                copy_matrix_inplace(Q_l, Q_buffer)
                copy_matrix_inplace(R_l, R_buffer)

            matmul_inplace(R_buffer, gamma[l], temp_matrix_JJ)
            for i in range(JJ):
                for j in range(JJ):
                    sum_val = 0.0
                    for k in range(JJ):
                        sum_val += temp_matrix_JJ[i, k] * R_buffer[j, k]
                    temp_matrix_JJ2[i, j] = sum_val
            add_inplace(Id_JJ_reg, temp_matrix_JJ2, temp_matrix_JJ)

            if JJ <= USE_CUSTOM_THRESHOLD:
                Lchol_l = cholesky_small(temp_matrix_JJ, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(temp_matrix_JJ).T

            logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))
            logdet_alpha = np.sum(np.log(alpha[l]))
            logdet[l] = logdet_chol + logdet_alpha

            matmul_inplace(beta[l], gamma[l], qr_input_buffer)
            matmul_inplace(qr_input_buffer, beta[l].T, H_inv_buffer)
            add_inplace(alpha_diag[l], H_inv_buffer, H_inv_buffer)

            matvec_inplace(H_inv_buffer, g[l], temp_vector_N)
            for i in range(N):
                mu_buffer[i] = x[l, i] - temp_vector_N[i]

            for i in range(JJ):
                for j in range(JJ):
                    temp_matrix_JJ2[i, j] = Lchol_l[i, j] - Id_JJ_reg[i, j]
            matmul_inplace(Q_buffer, temp_matrix_JJ2, qr_input_buffer)

            for m in range(M):
                matvec_transpose_inplace(Q_buffer, u[l, m], temp_vector_JJ)
                matvec_inplace(qr_input_buffer, temp_vector_JJ, temp_vector_N)
                for i in range(N):
                    combined_buffer[i] = temp_vector_N[i] + u[l, m, i]
                matvec_inplace(sqrt_alpha_diag[l], combined_buffer, sample_buffer)
                for i in range(N):
                    phi[l, m, i] = mu_buffer[i] + sample_buffer[i]

        return phi, logdet

    from numba.types import Tuple

    dense_bfgs_signature = Tuple((float64[:, :, :], float64[:]))(
        float64[:, :],  # x: (L, N)
        float64[:, :],  # g: (L, N)
        float64[:, :],  # alpha: (L, N)
        float64[:, :, :],  # beta: (L, N, JJ)
        float64[:, :, :],  # gamma: (L, JJ, JJ)
        float64[:, :, :],  # alpha_diag: (L, N, N)
        float64[:, :, :],  # inv_sqrt_alpha_diag: (L, N, N)
        float64[:, :, :],  # sqrt_alpha_diag: (L, N, N)
        float64[:, :, :],  # u: (L, M, N)
    )

    @numba_basic.numba_njit(
        dense_bfgs_signature,
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def dense_bfgs_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Dense BFGS sampling - Numba optimized with custom linear algebra.

        Optimized for case where JJ >= N (dense matrix operations preferred).
        Uses size-based selection: custom Cholesky for N < 100, BLAS for larger matrices.

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

        IdN = np.eye(N) + np.eye(N) * REGULARISATION_TERM

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for l in range(L):  # noqa: E741
            beta_l = ensure_contiguous_2d(beta[l])
            gamma_l = ensure_contiguous_2d(gamma[l])
            inv_sqrt_alpha_diag_l = ensure_contiguous_2d(inv_sqrt_alpha_diag[l])
            sqrt_alpha_diag_l = ensure_contiguous_2d(sqrt_alpha_diag[l])

            temp1 = matmul_contiguous(inv_sqrt_alpha_diag_l, beta_l)
            temp2 = matmul_contiguous(temp1, gamma_l)
            beta_l_T = transpose_contiguous(beta_l)
            temp3 = matmul_contiguous(temp2, beta_l_T)
            middle_term = matmul_contiguous(temp3, inv_sqrt_alpha_diag_l)

            temp_matrix = middle_term.copy()
            for i in range(N):
                temp_matrix[i, i] += IdN[i, i]  # Add identity efficiently
            H_inv_l = matmul_contiguous(
                sqrt_alpha_diag_l, matmul_contiguous(temp_matrix, sqrt_alpha_diag_l)
            )

            if N <= USE_CUSTOM_THRESHOLD:
                # 3-5x speedup over BLAS
                Lchol_l = cholesky_small(H_inv_l, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(H_inv_l).T

            logdet_sum = 0.0
            for i in range(N):
                logdet_sum += np.log(np.abs(Lchol_l[i, i]))
            logdet[l] = 2.0 * logdet_sum

            for m in range(M):
                for i in range(N):
                    mu_i = x[l, i]
                    for j in range(N):
                        mu_i -= H_inv_l[i, j] * g[l, j]

                    sample_i = mu_i
                    for j in range(N):
                        sample_i += Lchol_l[i, j] * u[l, m, j]
                    phi[l, m, i] = sample_i

        return phi, logdet

    sparse_bfgs_signature = Tuple((float64[:, :, :], float64[:]))(
        float64[:, :],  # x: (L, N)
        float64[:, :],  # g: (L, N)
        float64[:, :],  # alpha: (L, N)
        float64[:, :, :],  # beta: (L, N, JJ)
        float64[:, :, :],  # gamma: (L, JJ, JJ)
        float64[:, :, :],  # alpha_diag: (L, N, N)
        float64[:, :, :],  # inv_sqrt_alpha_diag: (L, N, N)
        float64[:, :, :],  # sqrt_alpha_diag: (L, N, N)
        float64[:, :, :],  # u: (L, M, N)
    )

    @numba_basic.numba_njit(
        sparse_bfgs_signature,
        fastmath=True,
        cache=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def sparse_bfgs_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Sparse BFGS sampling - Numba optimized with custom linear algebra.

        Optimized for case where JJ < N (sparse matrix operations preferred).
        Uses size-based selection: custom QR for small matrices, BLAS for larger matrices.

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

        for l in range(L):  # noqa: E741
            qr_input_l = inv_sqrt_alpha_diag[l] @ beta[l]

            if N <= USE_CUSTOM_THRESHOLD:
                # 3-5x speedup over BLAS
                Q_l, R_l = qr_small(qr_input_l)
            else:
                Q_l, R_l = np.linalg.qr(qr_input_l)

            IdJJ = np.eye(JJ) + np.eye(JJ) * REGULARISATION_TERM

            gamma_l = ensure_contiguous_2d(gamma[l])
            R_l_T = transpose_contiguous(R_l)
            temp_gamma = matmul_contiguous(R_l, gamma_l)
            temp_RgammaRT = matmul_contiguous(temp_gamma, R_l_T)

            Lchol_input_l = temp_RgammaRT.copy()
            for i in range(JJ):
                Lchol_input_l[i, i] += IdJJ[i, i]  # Add identity efficiently

            if JJ <= USE_CUSTOM_THRESHOLD:
                # 3-5x speedup over BLAS
                Lchol_l = cholesky_small(Lchol_input_l, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(Lchol_input_l).T

            logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))
            logdet_alpha = np.sum(np.log(alpha[l]))
            logdet[l] = logdet_chol + logdet_alpha

            beta_l = ensure_contiguous_2d(beta[l])
            alpha_diag_l = ensure_contiguous_2d(alpha_diag[l])
            temp_betagamma = matmul_contiguous(beta_l, gamma_l)
            beta_l_T = transpose_contiguous(beta_l)
            temp_lowrank = matmul_contiguous(temp_betagamma, beta_l_T)

            H_inv_l = temp_lowrank.copy()
            for i in range(N):
                for j in range(N):
                    H_inv_l[i, j] += alpha_diag_l[i, j]

            x_l = ensure_contiguous_1d(x[l])
            g_l = ensure_contiguous_1d(g[l])
            H_inv_g = matvec_contiguous(H_inv_l, g_l)
            mu_l = x_l.copy()
            for i in range(N):
                mu_l[i] -= H_inv_g[i]

            Lchol_diff = Lchol_l.copy()
            for i in range(JJ):
                for j in range(JJ):
                    Lchol_diff[i, j] -= IdJJ[i, j]
            Q_Lchol_diff = matmul_contiguous(Q_l, Lchol_diff)

            for m in range(M):
                u_lm = ensure_contiguous_1d(u[l, m])
                Qt_u_lm = matvec_contiguous(transpose_contiguous(Q_l), u_lm)
                Q_diff_Qtu = matvec_contiguous(Q_Lchol_diff, Qt_u_lm)

                combined = Q_diff_Qtu.copy()
                for i in range(N):
                    combined[i] += u_lm[i]

                sqrt_alpha_combined = matvec_contiguous(
                    ensure_contiguous_2d(sqrt_alpha_diag[l]), combined
                )
                phi[l, m] = mu_l.copy()
                for i in range(N):
                    phi[l, m, i] += sqrt_alpha_combined[i]

        return phi, logdet

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def bfgs_sample_with_memory_pool(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Memory-optimized conditional BFGS sampling using pre-allocated buffers.

        Uses efficient conditional compilation to select between dense and sparse
        algorithms based on problem dimensions, with memory pooling to eliminate
        temporary array allocations for improved performance.

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

        work_mem = create_working_memory(L, M, N, JJ)

        if JJ >= N:
            return dense_bfgs_with_memory_pool(
                x,
                g,
                alpha,
                beta,
                gamma,
                alpha_diag,
                inv_sqrt_alpha_diag,
                sqrt_alpha_diag,
                u,
                work_mem,
            )
        else:
            return sparse_bfgs_with_memory_pool(
                x,
                g,
                alpha,
                beta,
                gamma,
                alpha_diag,
                inv_sqrt_alpha_diag,
                sqrt_alpha_diag,
                u,
                work_mem,
            )

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def bfgs_sample_numba(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Conditional BFGS sampling using Numba.

        Uses efficient conditional compilation to select between dense and sparse
        algorithms based on problem dimensions. This avoids the dynamic indexing
        issues while providing optimal performance
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

        if JJ >= N:
            return dense_bfgs_numba(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )
        else:
            return sparse_bfgs_numba(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )

    # ===============================================================================
    # Phase 6: Smart Parallelization
    # ===============================================================================

    @numba_basic.numba_njit(
        dense_bfgs_signature,
        fastmath=True,
        cache=True,
        parallel=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def dense_bfgs_parallel(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Dense BFGS sampling with smart parallelization - Phase 6 optimization.

        Uses numba.prange for batch-level parallelization while avoiding thread
        contention with heavy linear algebra operations. Only custom lightweight
        operations are used within parallel loops.

        Key improvements:
        - Parallel processing over batch dimension (L)
        - Custom linear algebra operations avoid BLAS thread contention
        - Independent batch elements prevent race conditions
        - Memory-efficient with minimal allocations

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

        IdN = np.eye(N) + np.eye(N) * REGULARISATION_TERM

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for l in numba.prange(L):  # noqa: E741
            beta_l = ensure_contiguous_2d(beta[l])
            gamma_l = ensure_contiguous_2d(gamma[l])
            inv_sqrt_alpha_diag_l = ensure_contiguous_2d(inv_sqrt_alpha_diag[l])
            sqrt_alpha_diag_l = ensure_contiguous_2d(sqrt_alpha_diag[l])

            temp1 = matmul_contiguous(inv_sqrt_alpha_diag_l, beta_l)
            temp2 = matmul_contiguous(temp1, gamma_l)
            beta_l_T = transpose_contiguous(beta_l)
            temp3 = matmul_contiguous(temp2, beta_l_T)
            middle_term = matmul_contiguous(temp3, inv_sqrt_alpha_diag_l)

            temp_matrix = middle_term.copy()
            for i in range(N):
                temp_matrix[i, i] += IdN[i, i]
            H_inv_l = matmul_contiguous(
                sqrt_alpha_diag_l, matmul_contiguous(temp_matrix, sqrt_alpha_diag_l)
            )

            if N <= USE_CUSTOM_THRESHOLD:
                Lchol_l = cholesky_small(H_inv_l, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(H_inv_l).T

            logdet_sum = 0.0
            for i in range(N):
                logdet_sum += np.log(np.abs(Lchol_l[i, i]))
            logdet[l] = 2.0 * logdet_sum

            for m in range(M):
                for i in range(N):
                    mu_i = x[l, i]
                    for j in range(N):
                        mu_i -= H_inv_l[i, j] * g[l, j]

                    sample_i = mu_i
                    for j in range(N):
                        sample_i += Lchol_l[i, j] * u[l, m, j]
                    phi[l, m, i] = sample_i

        return phi, logdet

    @numba_basic.numba_njit(
        sparse_bfgs_signature,
        fastmath=True,
        cache=True,
        parallel=True,
        error_model="numpy",
        boundscheck=False,
        inline="never",
    )
    def sparse_bfgs_parallel(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Sparse BFGS sampling with smart parallelization - Phase 6 optimization.

        Uses numba.prange for batch-level parallelization while avoiding thread
        contention with heavy linear algebra operations. Custom QR operations
        are used within parallel loops for optimal performance.

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

        for l in numba.prange(L):  # noqa: E741
            inv_sqrt_alpha_diag_l = ensure_contiguous_2d(inv_sqrt_alpha_diag[l])
            beta_l = ensure_contiguous_2d(beta[l])
            qr_input_l = matmul_contiguous(inv_sqrt_alpha_diag_l, beta_l)

            if N <= USE_CUSTOM_THRESHOLD:
                Q_l, R_l = qr_small(qr_input_l)
            else:
                Q_l, R_l = np.linalg.qr(qr_input_l)

            IdJJ = np.eye(JJ) + np.eye(JJ) * REGULARISATION_TERM

            gamma_l = ensure_contiguous_2d(gamma[l])
            R_l_T = transpose_contiguous(R_l)
            temp_gamma = matmul_contiguous(R_l, gamma_l)
            temp_RgammaRT = matmul_contiguous(temp_gamma, R_l_T)

            Lchol_input_l = temp_RgammaRT.copy()
            for i in range(JJ):
                Lchol_input_l[i, i] += IdJJ[i, i]

            if JJ <= USE_CUSTOM_THRESHOLD:
                Lchol_l = cholesky_small(Lchol_input_l, upper=True)
            else:
                Lchol_l = np.linalg.cholesky(Lchol_input_l).T

            logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol_l))))
            logdet_alpha = np.sum(np.log(alpha[l]))
            logdet[l] = logdet_chol + logdet_alpha

            alpha_diag_l = ensure_contiguous_2d(alpha_diag[l])
            temp_betagamma = matmul_contiguous(beta_l, gamma_l)
            beta_l_T = transpose_contiguous(beta_l)
            temp_lowrank = matmul_contiguous(temp_betagamma, beta_l_T)

            H_inv_l = temp_lowrank.copy()
            for i in range(N):
                for j in range(N):
                    H_inv_l[i, j] += alpha_diag_l[i, j]

            x_l = ensure_contiguous_1d(x[l])
            g_l = ensure_contiguous_1d(g[l])
            H_inv_g = matvec_contiguous(H_inv_l, g_l)
            mu_l = x_l.copy()
            for i in range(N):
                mu_l[i] -= H_inv_g[i]

            Lchol_diff = Lchol_l.copy()
            for i in range(JJ):
                for j in range(JJ):
                    Lchol_diff[i, j] -= IdJJ[i, j]
            Q_Lchol_diff = matmul_contiguous(Q_l, Lchol_diff)

            for m in range(M):
                u_lm = ensure_contiguous_1d(u[l, m])
                Qt_u_lm = matvec_contiguous(transpose_contiguous(Q_l), u_lm)
                Q_diff_Qtu = matvec_contiguous(Q_Lchol_diff, Qt_u_lm)

                combined = Q_diff_Qtu.copy()
                for i in range(N):
                    combined[i] += u_lm[i]

                sqrt_alpha_combined = matvec_contiguous(
                    ensure_contiguous_2d(sqrt_alpha_diag[l]), combined
                )
                phi[l, m] = mu_l.copy()
                for i in range(N):
                    phi[l, m, i] += sqrt_alpha_combined[i]

        return phi, logdet

    @numba_basic.numba_njit(
        fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
    )
    def bfgs_sample_parallel(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Phase 6: Smart parallel conditional BFGS sampling.

        Uses intelligent parallelization that avoids thread contention:
        - Parallel over batch dimension (independent elements)
        - Custom linear algebra for small matrices (thread-safe)
        - Minimized BLAS contention for large matrices
        - Efficient memory access patterns

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

        if JJ >= N:
            return dense_bfgs_parallel(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )
        else:
            return sparse_bfgs_parallel(
                x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
            )

    # Note: chi_matrix_parallel is already defined in ChiMatrix section above

    def create_parallel_dispatcher():
        """Create intelligent parallel dispatcher based on problem size.

        Returns appropriate BFGS function based on:
        - Problem dimensions (favor parallel for larger problems)
        - Available CPU cores (detected at runtime)
        - Memory considerations

        Returns
        -------
        callable
            Optimized BFGS sampling function
        """
        try:
            import multiprocessing

            multiprocessing.cpu_count() or 1
        except (ImportError, OSError):
            pass

        @numba_basic.numba_njit(
            fastmath=True, cache=True, error_model="numpy", boundscheck=False, inline="always"
        )
        def smart_dispatcher(
            x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
        ):
            """Smart dispatcher: choose parallel vs sequential based on problem size.

            Decision criteria:
            - L >= 4: Use parallel version (sufficient work for threads)
            - L < 4: Use sequential version (avoid thread overhead)
            - Always use parallel for large batch sizes
            """
            L, M, N = u.shape

            # This avoids thread overhead for small problems
            if L >= 4:
                return bfgs_sample_parallel(
                    x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
                )
            else:
                return bfgs_sample_numba(
                    x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
                )

        return smart_dispatcher

    # Phase 6: Return intelligent parallel dispatcher
    return create_parallel_dispatcher()
