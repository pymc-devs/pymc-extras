import pytensor
import pytensor.tensor as pt

from pymc_extras.statespace.filters.utilities import (
    quad_form_sym,
    split_by_time_axis,
    stabilize,
    unpack_scan_step,
)
from pymc_extras.statespace.utils.constants import JITTER_DEFAULT

# The smoother's backward pass only ever touches these three matrices.
SMOOTHER_PARAM_NAMES = ("T", "R", "Q")


class KalmanSmoother:
    """
    Kalman Smoother

    """

    def __init__(self, cov_jitter: float | None = None):
        """
        Kalman smoother.

        :meth:`build_graph` only reads this setting, so one smoother builds any number of graphs.
        Which matrices ``scan`` receives as sequences is read off the matrices themselves, which
        callers mark with
        :func:`~pymc_extras.statespace.core.assumptions.declare_time_varying`.

        Parameters
        ----------
        cov_jitter : float, optional
            Jitter added to the diagonal of every covariance matrix at each step. Default 1e-8, or
            1e-6 if ``pytensor.config.floatX`` is float32.
        """
        self.cov_jitter = JITTER_DEFAULT if cov_jitter is None else cov_jitter

    def build_graph(
        self,
        T,
        R,
        Q,
        filtered_states,
        filtered_covariances,
    ):
        k = filtered_states.type.shape[1]

        a_last = pt.specify_shape(filtered_states[-1], (k,))
        P_last = pt.specify_shape(filtered_covariances[-1], (k, k))

        sequences, non_sequences, seq_names, non_seq_names = split_by_time_axis(
            dict(zip(SMOOTHER_PARAM_NAMES, [T, R, Q], strict=True))
        )

        def step_fn(a, P, *rest):
            (a_smooth, P_smooth), matrices = unpack_scan_step(
                rest, seq_names, non_seq_names, SMOOTHER_PARAM_NAMES
            )
            return self.smoother_step(a, P, a_smooth, P_smooth, *matrices)

        smoothed_states, smoothed_covariances = pytensor.scan(
            step_fn,
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
        P_smooth_next = pt.specify_shape(stabilize(P_smooth_next), P_smooth.type.shape)

        return a_smooth_next, P_smooth_next

    def predict(self, a, P, T, R, Q):
        a_hat = T.dot(a)
        P_hat = quad_form_sym(T, P) + quad_form_sym(R, Q)
        P_hat = stabilize(P_hat, self.cov_jitter)

        return a_hat, P_hat
