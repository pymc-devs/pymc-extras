import numba
import numpy as np
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch import numba_funcify

# Import LogLike Op for Numba dispatch registration
from .pathfinder import LogLike

# Ensure consistent regularization with main pathfinder module
REGULARISATION_TERM = 1e-8


class NumbaChiMatrixOp(Op):
    """Numba-optimized Chi matrix computation.

    Computes sliding window chi matrix for L-BFGS history in pathfinder algorithm.
    """

    def __init__(self, J: int):
        self.J = J
        super().__init__()

    def make_node(self, diff):
        """Create computation node for chi matrix."""
        diff = pt.as_tensor_variable(diff)
        output = pt.tensor(dtype=diff.dtype, shape=(None, None, self.J))
        return Apply(self, [diff], [output])

    def perform(self, node, inputs, outputs):
        """NumPy fallback implementation."""
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
    """Simplified Numba implementation for ChiMatrix computation."""
    J = op.J

    @numba_basic.numba_njit(parallel=True, fastmath=True, cache=True)
    def chi_matrix_simplified(diff):
        L, N = diff.shape
        chi_matrix = np.zeros((L, N, J), dtype=diff.dtype)

        for idx in numba.prange(L):
            start_idx = max(0, idx - J + 1)
            end_idx = idx + 1
            window_size = end_idx - start_idx

            if window_size < J:
                chi_matrix[idx, :, J - window_size :] = diff[start_idx:end_idx].T
            else:
                chi_matrix[idx] = diff[start_idx:end_idx].T

        return chi_matrix

    return chi_matrix_simplified


class NumbaBfgsSampleOp(Op):
    """Numba-optimized BFGS sampling.

    Uses simple conditional logic to select between dense and sparse algorithms
    based on problem dimensions.
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
        """NumPy fallback implementation using native operations."""
        from scipy.linalg import cholesky, qr

        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u = inputs
        L, M, N = u.shape
        JJ = beta.shape[2]
        REGULARISATION_TERM = 1e-8

        if JJ >= N:
            # Dense case
            IdN = np.eye(N)[None, ...] * (1.0 + REGULARISATION_TERM)
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
            # Sparse case
            qr_input = inv_sqrt_alpha_diag @ beta
            Q = np.zeros((L, qr_input.shape[1], qr_input.shape[2]))
            R = np.zeros((L, qr_input.shape[2], qr_input.shape[2]))
            for i in range(L):
                Q[i], R[i] = qr(qr_input[i], mode="economic")

            IdJJ = np.eye(JJ)[None, ...] * (1.0 + REGULARISATION_TERM)
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
    """Simplified Numba implementation for BFGS sampling."""

    REGULARISATION_TERM = 1e-8

    @numba_basic.numba_njit(parallel=True, fastmath=True, cache=True)
    def bfgs_sample_simplified(
        x, g, alpha, beta, gamma, alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag, u
    ):
        """Single unified BFGS sampling function with automatic optimization."""
        L, M, N = u.shape
        JJ = beta.shape[2]

        phi = np.empty((L, M, N), dtype=u.dtype)
        logdet = np.empty(L, dtype=u.dtype)

        for l in numba.prange(L):  # noqa: E741
            if JJ >= N:
                IdN = np.eye(N, dtype=u.dtype) * (1.0 + REGULARISATION_TERM)
                middle_term = (
                    inv_sqrt_alpha_diag[l] @ beta[l] @ gamma[l] @ beta[l].T @ inv_sqrt_alpha_diag[l]
                )
                H_inv = sqrt_alpha_diag[l] @ (IdN + middle_term) @ sqrt_alpha_diag[l]

                Lchol = np.linalg.cholesky(H_inv).T
                logdet[l] = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol))))

                mu = x[l] - H_inv @ g[l]
                phi[l] = (mu[:, None] + Lchol @ u[l].T).T

            else:
                Q, R = np.linalg.qr(inv_sqrt_alpha_diag[l] @ beta[l])
                IdJJ = np.eye(JJ, dtype=u.dtype) * (1.0 + REGULARISATION_TERM)
                Lchol_input = IdJJ + R @ gamma[l] @ R.T

                Lchol = np.linalg.cholesky(Lchol_input).T
                logdet_chol = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol))))
                logdet_alpha = np.sum(np.log(alpha[l]))
                logdet[l] = logdet_chol + logdet_alpha

                H_inv = alpha_diag[l] + beta[l] @ gamma[l] @ beta[l].T
                mu = x[l] - H_inv @ g[l]

                Q_Lchol_diff = Q @ (Lchol - IdJJ)
                Qt_u = Q.T @ u[l].T
                combined = Q_Lchol_diff @ Qt_u + u[l].T
                phi[l] = (mu[:, None] + sqrt_alpha_diag[l] @ combined).T

        return phi, logdet

    return bfgs_sample_simplified


@numba_funcify.register(LogLike)
def numba_funcify_LogLike(op, node=None, **kwargs):
    """Optimized Numba implementation for LogLike computation.

    Handles vectorized log-probability calculations with automatic parallelization
    and efficient NaN/Inf handling. Uses hybrid approach for maximum compatibility.
    """
    logp_func = op.logp_func

    @numba_basic.numba_njit(parallel=True, fastmath=True, cache=True)
    def loglike_vectorized_hybrid(phi):
        """Vectorized log-likelihood with hybrid Python/Numba approach.

        Uses objmode to call the Python logp_func while keeping array operations
        in nopython mode.
        """
        L, N = phi.shape
        logP = np.empty(L, dtype=phi.dtype)

        for i in numba.prange(L):
            row = phi[i].copy()
            with numba.objmode(val="float64"):
                val = logp_func(row)
            logP[i] = val

        mask = np.isnan(logP) | np.isinf(logP)

        if np.all(mask):
            logP[:] = -np.inf
        else:
            logP = np.where(mask, -np.inf, logP)

        return logP

    return loglike_vectorized_hybrid
