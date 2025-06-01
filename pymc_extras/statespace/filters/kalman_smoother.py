import pytensor
import pytensor.tensor as pt

from pytensor.tensor.nlinalg import matrix_dot

from pymc_extras.statespace.filters.utilities import (
    quad_form_sym,
    split_vars_into_seq_and_nonseq,
    stabilize,
)
from pymc_extras.statespace.utils.constants import JITTER_DEFAULT


class KalmanSmoother:
    """
    Kalman Smoother

    """

    def __init__(self):
        self.cov_jitter = JITTER_DEFAULT
        self.seq_names = []
        self.non_seq_names = []

    def unpack_args(self, args):
        """
        The order of inputs to the inner scan function is not known, since some, all, or none of the input matrices
        can be time varying. The order arguments are fed to the inner function is sequences, outputs_info,
        non-sequences. This function works out which matrices are where, and returns a standardized order expected
        by the kalman_step function.

        The standard order is: a, P, a_smooth, P_smooth, T, R, Q
        """
        # If there are no sequence parameters (all params are static),
        # no changes are needed, params will be in order.
        args = list(args)
        n_seq = len(self.seq_names)
        if n_seq == 0:
            return args

        # The first two args are always a and P
        a = args.pop(0)
        P = args.pop(0)

        # There are always two outputs_info wedged between the seqs and non_seqs
        seqs, (a_smooth, P_smooth), non_seqs = (
            args[:n_seq],
            args[n_seq : n_seq + 2],
            args[n_seq + 2 :],
        )
        return_ordered = []
        for name in ["T", "R", "Q"]:
            if name in self.seq_names:
                idx = self.seq_names.index(name)
                return_ordered.append(seqs[idx])
            else:
                idx = self.non_seq_names.index(name)
                return_ordered.append(non_seqs[idx])

        T, R, Q = return_ordered

        return a, P, a_smooth, P_smooth, T, R, Q

    def build_graph(
        self, T, R, Q, filtered_states, filtered_covariances, cov_jitter=JITTER_DEFAULT
    ):
        self.cov_jitter = cov_jitter

        n, k = filtered_states.type.shape

        a_last = pt.specify_shape(filtered_states[-1], (k,))
        P_last = pt.specify_shape(filtered_covariances[-1], (k, k))

        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            [T, R, Q], ["T", "R", "Q"]
        )

        self.seq_names = seq_names
        self.non_seq_names = non_seq_names

        smoother_result, updates = pytensor.scan(
            self.smoother_step,
            sequences=[filtered_states[:-1], filtered_covariances[:-1], *sequences],
            outputs_info=[a_last, P_last],
            non_sequences=non_sequences,
            go_backwards=True,
            name="kalman_smoother",
        )

        smoothed_states, smoothed_covariances = smoother_result
        smoothed_states = pt.concatenate(
            [smoothed_states[::-1], pt.expand_dims(a_last, axis=(0,))], axis=0
        )
        smoothed_covariances = pt.concatenate(
            [smoothed_covariances[::-1], pt.expand_dims(P_last, axis=(0,))], axis=0
        )

        smoothed_states.name = "smoothed_states"
        smoothed_covariances.name = "smoothed_covariances"

        return smoothed_states, smoothed_covariances

    def smoother_step(self, *args):
        a, P, a_smooth, P_smooth, T, R, Q = self.unpack_args(args)
        a_hat, P_hat = self.predict(a, P, T, R, Q)

        # Use pinv, otherwise P_hat is singular when there is missing data
        smoother_gain = matrix_dot(pt.linalg.pinv(P_hat), T, P).T
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
