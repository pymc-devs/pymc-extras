from collections.abc import Iterable

import pytensor
import pytensor.tensor as pt

from pytensor.assumptions import assume

from pymc_extras.statespace.filters.utilities import (
    mask_missing_values,
    quad_form_sym,
    stabilize,
)
from pymc_extras.statespace.utils.constants import (
    JITTER_DEFAULT,
    LONG_NAME_TO_SHORT,
    MISSING_FILL,
)

# The smoother's backward pass only ever touches these three matrices.
SMOOTHER_PARAM_NAMES = ("T", "R", "Q")

# The disturbance smoother's backward pass reads the observation block as well as the transition.
DISTURBANCE_PARAM_NAMES = ("d", "T", "Z", "H")


class RTSSmoother:
    """
    Rauch-Tung-Striebel fixed-interval smoother.

    Runs the backward recursion on the smoothing gain
    :math:`G_t = P_{t|t} T^T P_{t+1|t}^{-1}`, which requires a ``k_states`` square inverse at every
    step. :class:`DisturbanceSmoother` computes the same quantities without that inverse and is the
    default; this class is kept for cross-checking against it.
    """

    def __init__(
        self,
        time_varying_names: Iterable[str] = (),
        cov_jitter: float | None = None,
    ):
        """
        Kalman smoother.

        :meth:`build_graph` only reads these settings, so one smoother builds any number of graphs.

        Parameters
        ----------
        time_varying_names : iterable of str, optional
            Long names of the matrices the model declared time-varying. Only the transition, selection
            and state covariance matrices reach the smoother. Default is no time-varying matrices.
        cov_jitter : float, optional
            Jitter added to the diagonal of every covariance matrix at each step. Default 1e-8, or
            1e-6 if ``pytensor.config.floatX`` is float32.
        """
        self.cov_jitter = JITTER_DEFAULT if cov_jitter is None else cov_jitter

        time_varying_short = {LONG_NAME_TO_SHORT[name] for name in time_varying_names}
        self.seq_names = [name for name in SMOOTHER_PARAM_NAMES if name in time_varying_short]
        self.non_seq_names = [
            name for name in SMOOTHER_PARAM_NAMES if name not in time_varying_short
        ]

    def build_graph(self, data, matrices, filter_outputs):
        """
        Build the backward smoothing recursion.

        Parameters
        ----------
        data : TensorVariable
            Observed series, shape ``(T, k_endog)``. Unused by this smoother.
        matrices : sequence of TensorVariable
            The nine state-space matrices, in the order ``x0, P0, c, d, T, Z, R, H, Q``.
        filter_outputs : sequence of TensorVariable
            Output of :meth:`BaseFilter.build_graph`, whose first six entries are the filtered,
            predicted and observed means followed by their covariances.

        Returns
        -------
        smoothed_states : TensorVariable
            Smoothed state means, shape ``(T, k_states)``.
        smoothed_covariances : TensorVariable
            Smoothed state covariances, shape ``(T, k_states, k_states)``.
        """
        *_, T, _, R, _, Q = matrices
        filtered_states, filtered_covariances = filter_outputs[0], filter_outputs[3]
        k = filtered_states.type.shape[1]

        a_last = pt.specify_shape(filtered_states[-1], (k,))
        P_last = pt.specify_shape(filtered_covariances[-1], (k, k))

        params = dict(zip(SMOOTHER_PARAM_NAMES, [T, R, Q], strict=True))
        sequences = [params[name] for name in self.seq_names]
        non_sequences = [params[name] for name in self.non_seq_names]

        # After the filtered moments, scan passes the time-varying matrices, then the carried
        # smoothed moments, then the static matrices.
        def step(a, P, *args):
            names = [*self.seq_names, "a_smooth", "P_smooth", *self.non_seq_names]
            return self.smoother_step(a, P, **dict(zip(names, args, strict=True)))

        smoothed_states, smoothed_covariances = pytensor.scan(
            step,
            sequences=[filtered_states[:-1], filtered_covariances[:-1], *sequences],
            outputs_info=[a_last, P_last],
            non_sequences=non_sequences,
            go_backwards=True,
            name="kalman_smoother",
            return_updates=False,
        )

        smoothed_states = pt.concatenate(
            [smoothed_states[::-1], pt.expand_dims(a_last, axis=(0,))], axis=0
        )
        smoothed_covariances = pt.concatenate(
            [smoothed_covariances[::-1], pt.expand_dims(P_last, axis=(0,))], axis=0
        )

        smoothed_covariances = assume(smoothed_covariances, symmetric=True)
        smoothed_states.name = "smoothed_states"
        smoothed_covariances.name = "smoothed_covariances"

        return smoothed_states, smoothed_covariances

    def smoother_step(self, a, P, a_smooth, P_smooth, T, R, Q):
        a_hat, P_hat = self.predict(a, P, T, R, Q)

        # Use pinv, otherwise P_hat is singular when there is missing data
        smoother_gain = (pt.linalg.pinv(P_hat, hermitian=True) @ T @ P).mT
        a_smooth_next = a + smoother_gain @ (a_smooth - a_hat)

        P_smooth_next = P + quad_form_sym(smoother_gain, P_smooth - P_hat)
        P_smooth_next = stabilize(P_smooth_next, self.cov_jitter)
        P_smooth_next = pt.specify_shape(P_smooth_next, P_smooth.type.shape)

        return a_smooth_next, P_smooth_next

    def predict(self, a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = quad_form_sym(T, P) + quad_form_sym(R, Q)
        P_hat = stabilize(P_hat, self.cov_jitter)

        return a_hat, P_hat


class DisturbanceSmoother:
    r"""
    Durbin-Koopman disturbance smoother.

    Runs the backward recursion on the scaled smoothing errors :math:`r_t` and :math:`N_t` rather
    than on a state-space gain, so the only matrix inverted is the innovation covariance
    :math:`F = Z P Z^T + H`, of size ``k_endog``. The classical fixed-interval form inverts
    :math:`P` instead, which is ``k_states`` square and singular whenever a state carries no
    process noise.

    Parameters
    ----------
    time_varying_names : iterable of str, optional
        Long names of the matrices the model declared time-varying. Default is none.
    cov_jitter : float, optional
        Jitter added to the innovation covariance before factoring it. Default 1e-8, or 1e-6 when
        ``pytensor.config.floatX`` is float32.
    missing_fill_value : float, optional
        Sentinel standing in for a missing observation, alongside ``nan``.
    """

    def __init__(
        self,
        time_varying_names: Iterable[str] = (),
        cov_jitter: float | None = None,
        missing_fill_value: float | None = None,
    ):
        self.cov_jitter = JITTER_DEFAULT if cov_jitter is None else cov_jitter
        self.missing_fill_value = MISSING_FILL if missing_fill_value is None else missing_fill_value

        time_varying_short = {LONG_NAME_TO_SHORT[name] for name in time_varying_names}
        self.seq_names = [name for name in DISTURBANCE_PARAM_NAMES if name in time_varying_short]
        self.non_seq_names = [
            name for name in DISTURBANCE_PARAM_NAMES if name not in time_varying_short
        ]

    def smoother_step(self, y, a, P, r, N, d, T, Z, H):
        y, Z, H, d, nan_mask = mask_missing_values(y, Z, H, d, self.missing_fill_value)

        # A masked row of F is all zeros. A one on its diagonal keeps the factorization defined,
        # and contributes nothing because the innovation and Z are zero on that row.
        F = Z @ P @ Z.mT + stabilize(H, self.cov_jitter)
        observed = pt.bitwise_not(nan_mask).astype(F.dtype)
        F = F * pt.outer(observed, observed) + pt.diag(1 - observed)
        F_chol = pt.linalg.cholesky(F, lower=True)
        v = y - Z @ a - d

        # The one inverse in the recursion, and it is k_endog square rather than k_states.
        Z_F_inv = pt.linalg.cho_solve((F_chol, True), Z, b_ndim=2)
        gain = T @ pt.linalg.cho_solve((F_chol, True), Z @ P.mT, b_ndim=2).mT
        L = T - gain @ Z

        r_next = Z_F_inv.mT @ v + L.mT @ r
        N_next = Z_F_inv.mT @ Z + quad_form_sym(L.mT, N)

        return r_next, N_next, a + P @ r_next, P - quad_form_sym(P, N_next)

    def build_graph(self, data, matrices, filter_outputs):
        """
        Build the backward smoothing recursion over the filter's one-step-ahead moments.

        Parameters
        ----------
        data : TensorVariable
            Observed series, shape ``(T, k_endog)``.
        matrices : sequence of TensorVariable
            The nine state-space matrices, in the order ``x0, P0, c, d, T, Z, R, H, Q``. Any matrix
            named time-varying at construction carries a leading time axis.
        filter_outputs : sequence of TensorVariable
            Output of :meth:`BaseFilter.build_graph`, whose first six entries are the filtered,
            predicted and observed means followed by their covariances.

        Returns
        -------
        smoothed_states : TensorVariable
            Smoothed state means, shape ``(T, k_states)``.
        smoothed_covariances : TensorVariable
            Smoothed state covariances, shape ``(T, k_states, k_states)``.
        """
        _, _, _, d, T, Z, _, H, _ = matrices
        predicted_states, predicted_covariances = filter_outputs[1], filter_outputs[4]
        k_states = predicted_states.type.shape[1] or predicted_states.shape[1]

        params = dict(zip(DISTURBANCE_PARAM_NAMES, [d, T, Z, H], strict=True))
        sequences = [params[name] for name in self.seq_names]
        non_sequences = [params[name] for name in self.non_seq_names]

        # After the data and predicted moments, scan passes the time-varying matrices, then the
        # carried r and N, then the static matrices.
        def step(y, a, P, *args):
            names = [*self.seq_names, "r", "N", *self.non_seq_names]
            return self.smoother_step(y, a, P, **dict(zip(names, args, strict=True)))

        _, _, smoothed_states, smoothed_covariances = pytensor.scan(
            step,
            sequences=[data, predicted_states, predicted_covariances, *sequences],
            outputs_info=[pt.zeros(k_states), pt.zeros((k_states, k_states)), None, None],
            non_sequences=non_sequences,
            go_backwards=True,
            name="disturbance_smoother",
            strict=True,
            return_updates=False,
        )

        smoothed_states = smoothed_states[::-1]
        smoothed_covariances = smoothed_covariances[::-1]

        smoothed_covariances = assume(smoothed_covariances, symmetric=True)
        smoothed_states.name = "smoothed_states"
        smoothed_covariances.name = "smoothed_covariances"

        return smoothed_states, smoothed_covariances
