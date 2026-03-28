"""Cholesky-form Kalman filtering utilities.

This file holds the phase-1 reference implementation for the square-root
Kalman filter. The NumPy functions are the baseline implementation used by the
tests, and they are written to favor clarity and numerical safety over being as
factorized as possible.

There are two main pieces here:

1. A NumPy predict/update path used as the ground-truth implementation for the
    current milestone.
2. A small PyTensor Op that reuses the same forward logic and rebuilds the step
    symbolically when gradients are needed.

The filter keeps covariance matrices in Cholesky form, uses a QR-based predict
step, and computes the update with triangular solves instead of explicit matrix
inversion. That keeps the implementation stable enough for long filtering runs
while still being easy to check against a standard covariance-form filter.

Missing data is handled conservatively for now: if any entry of `y` is NaN, the
whole observation is treated as missing and the step becomes predict-only.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from pytensor.gradient import DisconnectedType
from pytensor.gradient import Rop as _Rop
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from scipy.linalg import solve_triangular


def _shape_tuple(a: np.ndarray) -> tuple[int, ...]:
    """Return a plain tuple so shape errors read cleanly."""
    return tuple(int(d) for d in a.shape)


def _validate_numeric_shapes(
    x_prev: np.ndarray,
    L_prev: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    c: np.ndarray | None = None,
) -> None:
    """Fail early on shape mismatches in the NumPy path."""
    if x_prev.ndim != 1:
        raise ValueError(f"x_prev must be 1-D, got shape {_shape_tuple(x_prev)}")
    if L_prev.ndim != 2:
        raise ValueError(f"L_prev must be 2-D, got shape {_shape_tuple(L_prev)}")
    if T.ndim != 2:
        raise ValueError(f"T must be 2-D, got shape {_shape_tuple(T)}")
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D, got shape {_shape_tuple(R)}")
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2-D, got shape {_shape_tuple(Q)}")
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {_shape_tuple(Z)}")
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D, got shape {_shape_tuple(H)}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {_shape_tuple(y)}")
    if c is not None and c.ndim != 1:
        raise ValueError(f"c must be 1-D, got shape {_shape_tuple(c)}")

    n = x_prev.shape[0]
    if L_prev.shape != (n, n):
        raise ValueError(f"L_prev must be square with shape ({n}, {n}), got {_shape_tuple(L_prev)}")
    if T.shape != (n, n):
        raise ValueError(f"T must have shape ({n}, {n}), got {_shape_tuple(T)}")
    if R.shape[0] != n:
        raise ValueError(f"R.shape[0] must be {n}, got {R.shape[0]}")

    q = R.shape[1]
    if Q.shape != (q, q):
        raise ValueError(f"Q must have shape ({q}, {q}), got {_shape_tuple(Q)}")

    m = y.shape[0]
    if Z.shape != (m, n):
        raise ValueError(f"Z must have shape ({m}, {n}), got {_shape_tuple(Z)}")
    if H.shape != (m, m):
        raise ValueError(f"H must have shape ({m}, {m}), got {_shape_tuple(H)}")
    if c is not None and c.shape != (n,):
        raise ValueError(f"c must have shape ({n},), got {_shape_tuple(c)}")


def _validate_static_shapes_make_node(
    x_prev,
    L_prev,
    T,
    R,
    Q,
    Z,
    H,
    y,
) -> None:
    """Check static PyTensor shapes when enough shape information is available."""
    sx = x_prev.type.shape
    sL = L_prev.type.shape
    sT = T.type.shape
    sR = R.type.shape
    sQ = Q.type.shape
    sZ = Z.type.shape
    sH = H.type.shape
    sy = y.type.shape

    n = sx[0]
    m = sy[0]

    if sL[0] is not None and sL[1] is not None and sL[0] != sL[1]:
        raise TypeError(f"L_prev must be square, got static shape {sL}")
    if n is not None and sL[0] is not None and sL[0] != n:
        raise TypeError(f"L_prev.shape[0] must match x_prev length ({n}), got {sL[0]}")
    if n is not None and sL[1] is not None and sL[1] != n:
        raise TypeError(f"L_prev.shape[1] must match x_prev length ({n}), got {sL[1]}")

    if sT[0] is not None and sT[1] is not None and sT[0] != sT[1]:
        raise TypeError(f"T must be square, got static shape {sT}")
    if n is not None and sT[0] is not None and sT[0] != n:
        raise TypeError(f"T.shape[0] must match x_prev length ({n}), got {sT[0]}")
    if n is not None and sT[1] is not None and sT[1] != n:
        raise TypeError(f"T.shape[1] must match x_prev length ({n}), got {sT[1]}")

    if n is not None and sR[0] is not None and sR[0] != n:
        raise TypeError(f"R.shape[0] must match x_prev length ({n}), got {sR[0]}")
    q = sR[1]
    if q is not None and sQ[0] is not None and sQ[0] != q:
        raise TypeError(f"Q.shape[0] must match R.shape[1] ({q}), got {sQ[0]}")
    if q is not None and sQ[1] is not None and sQ[1] != q:
        raise TypeError(f"Q.shape[1] must match R.shape[1] ({q}), got {sQ[1]}")

    if m is not None and sZ[0] is not None and sZ[0] != m:
        raise TypeError(f"Z.shape[0] must match y length ({m}), got {sZ[0]}")
    if n is not None and sZ[1] is not None and sZ[1] != n:
        raise TypeError(f"Z.shape[1] must match x_prev length ({n}), got {sZ[1]}")

    if sH[0] is not None and sH[1] is not None and sH[0] != sH[1]:
        raise TypeError(f"H must be square, got static shape {sH}")
    if m is not None and sH[0] is not None and sH[0] != m:
        raise TypeError(f"H.shape[0] must match y length ({m}), got {sH[0]}")
    if m is not None and sH[1] is not None and sH[1] != m:
        raise TypeError(f"H.shape[1] must match y length ({m}), got {sH[1]}")


def _qr_chol_predict_numpy(
    L_prev: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    L_Q: np.ndarray,
) -> np.ndarray:
    """Return the predicted covariance factor using a QR step.

    The QR form is a little more robust than propagating the full covariance
    directly and then taking a Cholesky factor afterwards.
    """
    A = np.concatenate([T @ L_prev, R @ L_Q], axis=1)  # (n, n+q)

    _, R_thin = np.linalg.qr(A.T, mode="reduced")  # R_thin: (n, n)

    # Keep the usual Cholesky sign convention.
    signs = np.sign(np.diag(R_thin))
    signs[signs == 0.0] = 1.0
    R_thin = R_thin * signs[:, np.newaxis]  # flip rows

    return R_thin.T  # lower-triangular, positive diagonal


def _chol_kalman_update_numpy(
    x_pred: np.ndarray,
    L_pred: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the NumPy update step and return the new mean, factor, and logp.

    This uses triangular solves for both the gain and the likelihood term, so
    there is no explicit matrix inverse in the update.
    """
    n = x_pred.shape[0]
    m = y.shape[0]

    # In this version, a partially missing observation is treated as missing.
    if np.isnan(y).any():
        return x_pred.copy(), L_pred.copy(), 0.0

    P_pred = L_pred @ L_pred.T
    v = y - Z @ x_pred  # (m,)

    S = Z @ P_pred @ Z.T + H
    S = 0.5 * (S + S.T)
    S += jitter * np.eye(m)

    L_S = np.linalg.cholesky(S)  # (m, m) lower-triangular

    ZP = Z @ P_pred  # (m, n)
    tmp1 = solve_triangular(L_S, ZP, lower=True)  # L_S  tmp1  = ZP
    tmp2 = solve_triangular(L_S.T, tmp1, lower=False)  # L_S^T K^T = tmp1
    K = tmp2.T  # (n, m)

    x_new = x_pred + K @ v  # (n,)

    IKZ = np.eye(n) - K @ Z  # (n, n)
    P_new = IKZ @ P_pred @ IKZ.T + K @ H @ K.T  # (n, n)
    P_new = 0.5 * (P_new + P_new.T)
    P_new += jitter * np.eye(n)

    L_new = np.linalg.cholesky(P_new)  # (n, n)

    log_det_S = 2.0 * np.sum(np.log(np.diag(L_S)))
    Sinv_v = solve_triangular(L_S, v, lower=True)  # (m,)
    quad = float(np.dot(Sinv_v, Sinv_v))
    log_lik = -0.5 * (log_det_S + quad + m * np.log(2.0 * np.pi))

    return x_new, L_new, float(log_lik)


def chol_kalman_step_numpy(
    x_prev: np.ndarray,
    L_prev: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    y: np.ndarray,
    c: np.ndarray | None = None,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one predict/update step with the NumPy reference code."""
    if jitter <= 0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    _validate_numeric_shapes(x_prev, L_prev, T, R, Q, Z, H, y, c=c)
    q = Q.shape[0]

    L_Q = np.linalg.cholesky(Q + jitter * np.eye(q))

    if c is None:
        x_pred = T @ x_prev
    else:
        x_pred = T @ x_prev + c
    L_pred = _qr_chol_predict_numpy(L_prev, T, R, L_Q)

    x_new, L_new, log_lik = _chol_kalman_update_numpy(x_pred, L_pred, Z, H, y, jitter=jitter)
    return x_new, L_new, log_lik


def run_chol_kalman_filter_numpy(
    x0: np.ndarray,
    P0: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    observations: np.ndarray,
    c: np.ndarray | None = None,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter a full observation sequence with the NumPy reference code."""
    n = x0.shape[0]
    T_steps = observations.shape[0]

    L0 = np.linalg.cholesky(P0 + jitter * np.eye(n))

    means = np.zeros((T_steps, n))
    chols = np.zeros((T_steps, n, n))
    lls = np.zeros(T_steps)

    x, L = x0.copy(), L0.copy()
    for t in range(T_steps):
        x, L, ll = chol_kalman_step_numpy(x, L, T, R, Q, Z, H, observations[t], c=c, jitter=jitter)
        means[t] = x
        chols[t] = L
        lls[t] = ll

    return means, chols, lls


def _chol_kalman_step_symbolic(
    x_prev,
    L_prev,
    T,
    R,
    Q,
    Z,
    H,
    y,
    jitter: float = 1e-8,
    c=None,
):
    """Symbolic version of the same step, used for gradients.

    The forward `perform` method uses NumPy, but `grad` rebuilds the step with
    PyTensor ops so autodiff can see through it.

    Note: The predict step here uses the standard covariance form
    (T P T' + R Q R') rather than the QR decomposition used in the NumPy path.
    PyTensor's QR gradient support is incomplete, so the covariance form is used
    to keep the symbolic graph differentiable. The two forms are mathematically
    equivalent.
    """
    n_state = T.shape[0]
    n_obs = H.shape[0]

    # Match the NumPy missing-data rule.
    any_missing = pt.any(pt.isnan(y))
    y_safe = pt.switch(any_missing, pt.zeros_like(y), y)

    if c is None:
        x_pred = T @ x_prev
    else:
        x_pred = T @ x_prev + c

    # PyTensor's QR path is awkward here, so use the equivalent covariance form.
    P_prev = L_prev @ L_prev.T
    P_pred_temp = T @ P_prev @ T.T + R @ Q @ R.T

    P_pred_temp = 0.5 * (P_pred_temp + P_pred_temp.T) + jitter * pt.eye(n_state)
    L_pred = pt.linalg.cholesky(P_pred_temp, lower=True)

    P_pred = L_pred @ L_pred.T  # (n, n)

    v = y_safe - Z @ x_pred  # (m,)

    S = Z @ P_pred @ Z.T + H
    S = 0.5 * (S + S.T)
    S = S + jitter * pt.eye(n_obs)

    L_S = pt.linalg.cholesky(S, lower=True)  # (m, m)

    ZP = Z @ P_pred  # (m, n)
    tmp1 = pt.linalg.solve_triangular(L_S, ZP, lower=True)  # (m, n)
    tmp2 = pt.linalg.solve_triangular(L_S.T, tmp1, lower=False)  # (m, n)
    K = tmp2.T  # (n, m)

    x_new = x_pred + K @ v  # (n,)

    I_n = pt.eye(n_state)
    IKZ = I_n - K @ Z  # (n, n)
    P_new = IKZ @ P_pred @ IKZ.T + K @ H @ K.T  # (n, n)
    P_new = 0.5 * (P_new + P_new.T) + jitter * pt.eye(n_state)

    L_new = pt.linalg.cholesky(P_new, lower=True)  # (n, n)

    log_det_S = 2.0 * pt.sum(pt.log(pt.diag(L_S)))
    Sinv_v = pt.linalg.solve_triangular(L_S, v, lower=True)  # (m,)
    quad = pt.dot(Sinv_v, Sinv_v)
    log_lik = -0.5 * (log_det_S + quad + n_obs * pt.log(2.0 * np.pi))

    x_out = pt.switch(any_missing, x_pred, x_new)
    L_out = pt.switch(any_missing, L_pred, L_new)
    ll_out = pt.switch(any_missing, pt.as_tensor(0.0, dtype="float64"), log_lik)

    return x_out, L_out, ll_out


class CholKalmanUpdateOp(Op):
    """PyTensor wrapper around one Kalman predict/update step.

    The Op keeps the execution path simple: NumPy for forward evaluation,
    symbolic reconstruction for gradients.

    The state offset vector ``c`` is not supported at the Op level in Phase 1.
    Use the NumPy path (``chol_kalman_step_numpy``) directly if ``c`` is needed.
    """

    __props__ = ("jitter",)

    def __init__(self, jitter: float = 1e-8):
        if jitter <= 0:
            raise ValueError(f"jitter must be positive, got {jitter}")
        self.jitter = float(jitter)
        super().__init__()

    def make_node(self, x_prev, L_prev, T, R, Q, Z, H, y):
        x_prev = pt.as_tensor_variable(x_prev).astype("float64")
        L_prev = pt.as_tensor_variable(L_prev).astype("float64")
        T = pt.as_tensor_variable(T).astype("float64")
        R = pt.as_tensor_variable(R).astype("float64")
        Q = pt.as_tensor_variable(Q).astype("float64")
        Z = pt.as_tensor_variable(Z).astype("float64")
        H = pt.as_tensor_variable(H).astype("float64")
        y = pt.as_tensor_variable(y).astype("float64")

        inputs = [x_prev, L_prev, T, R, Q, Z, H, y]

        if x_prev.ndim != 1:
            raise TypeError(f"x_prev must be 1-D, got ndim={x_prev.ndim}")
        if L_prev.ndim != 2:
            raise TypeError(f"L_prev must be 2-D, got ndim={L_prev.ndim}")
        if y.ndim != 1:
            raise TypeError(f"y must be 1-D, got ndim={y.ndim}")

        _validate_static_shapes_make_node(x_prev, L_prev, T, R, Q, Z, H, y)

        x_new_type = pt.dvector()
        L_new_type = pt.dmatrix()
        log_lik_type = pt.dscalar()

        return Apply(self, inputs, [x_new_type, L_new_type, log_lik_type])

    def perform(self, node, inputs, outputs):
        """Run the Op through the NumPy reference implementation."""
        x_prev, L_prev, T, R, Q, Z, H, y = (np.asarray(a, dtype=np.float64) for a in inputs)
        _validate_numeric_shapes(x_prev, L_prev, T, R, Q, Z, H, y)

        x_new, L_new, log_lik = chol_kalman_step_numpy(
            x_prev, L_prev, T, R, Q, Z, H, y, jitter=self.jitter
        )

        outputs[0][0] = x_new
        outputs[1][0] = L_new
        outputs[2][0] = np.array(log_lik, dtype=np.float64)

    def grad(self, inputs, output_grads):
        """Build the symbolic step and let PyTensor differentiate it."""
        x_prev, L_prev, T, R, Q, Z, H, y = inputs
        g_x_new, g_L_new, g_ell = output_grads

        x_new_sym, L_new_sym, log_lik_sym = _chol_kalman_step_symbolic(
            x_prev, L_prev, T, R, Q, Z, H, y, jitter=self.jitter
        )

        if isinstance(g_x_new.type, DisconnectedType):
            g_x_new = pt.zeros_like(x_new_sym)

        if isinstance(g_L_new.type, DisconnectedType):
            g_L_new = pt.zeros_like(L_new_sym)

        if isinstance(g_ell.type, DisconnectedType):
            g_ell = pt.zeros_like(log_lik_sym)

        obj = pt.dot(g_x_new, x_new_sym) + pt.sum(g_L_new * L_new_sym) + g_ell * log_lik_sym

        grads = pt.grad(
            obj,
            wrt=[x_prev, L_prev, T, R, Q, Z, H, y],
            disconnected_inputs="zero",
        )
        return grads

    def R_op(self, inputs, eval_points):
        """Forward-mode derivative helper."""
        x_prev, L_prev, T, R, Q, Z, H, y = inputs
        ev_x, ev_L, ev_T, ev_R, ev_Q, ev_Z, ev_H, ev_y = eval_points

        def _safe_ev(ev, var):
            if ev is None:
                return pt.zeros_like(var)
            return ev

        ev_x = _safe_ev(ev_x, x_prev)
        ev_L = _safe_ev(ev_L, L_prev)
        ev_T = _safe_ev(ev_T, T)
        ev_R = _safe_ev(ev_R, R)
        ev_Q = _safe_ev(ev_Q, Q)
        ev_Z = _safe_ev(ev_Z, Z)
        ev_H = _safe_ev(ev_H, H)
        ev_y = _safe_ev(ev_y, y)
        eval_points_safe = [ev_x, ev_L, ev_T, ev_R, ev_Q, ev_Z, ev_H, ev_y]

        x_new_sym, L_new_sym, log_lik_sym = _chol_kalman_step_symbolic(
            x_prev, L_prev, T, R, Q, Z, H, y, jitter=self.jitter
        )

        rop_x = _Rop(x_new_sym, inputs, eval_points_safe, disconnected_outputs="ignore")
        rop_L = _Rop(L_new_sym, inputs, eval_points_safe, disconnected_outputs="ignore")
        rop_ll = _Rop(log_lik_sym, inputs, eval_points_safe, disconnected_outputs="ignore")

        return [rop_x, rop_L, rop_ll]

    def infer_shape(self, fgraph, node, input_shapes):
        n = input_shapes[0][0]
        return [(n,), (n, n), ()]

    def connection_pattern(self, node):
        return [[True, True, True]] * 8

    def __repr__(self):
        return f"CholKalmanUpdateOp(jitter={self.jitter})"


def make_chol_kalman_op(jitter: float = 1e-8) -> CholKalmanUpdateOp:
    """Create a `CholKalmanUpdateOp` with the jitter."""
    return CholKalmanUpdateOp(jitter=jitter)
