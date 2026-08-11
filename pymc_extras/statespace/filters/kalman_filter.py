import copy

from abc import ABC

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.pytensorf import constant_fold
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import disconnected_grad
from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert
from pytensor.scan.utils import until
from pytensor.tensor import TensorConstant, TensorVariable
from pytensor.tensor.linalg import solve_triangular

from pymc_extras.statespace.filters.utilities import (
    quad_form_sym,
    split_vars_into_seq_and_nonseq,
    stabilize,
)
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_NAMES,
    JITTER_DEFAULT,
    LONG_NAME_TO_SHORT,
    MATRIX_NAMES,
    MISSING_FILL,
    static_matrix_shapes,
)

MVN_CONST = pt.log(2 * pt.constant(np.pi, dtype="float64"))
PARAM_NAMES = MATRIX_NAMES[2:]

assert_time_varying_dim_correct = Assert(
    "The first dimension of a time varying matrix (the time dimension) must be "
    "equal to the first dimension of the data (the time dimension)."
)


class BaseFilter(ABC):
    def __init__(self):
        """
        Kalman Filter.

        Notes
        -----
        The BaseFilter class is an abstract base class (ABC) for implementing kalman filters.
        It defines common attributes and methods used by kalman filter implementations.

        Attributes
        ----------
        seq_names : list[str]
            A list of name representing time-varying statespace matrices. That is, inputs that will need to be
            provided to the `sequences` argument of `pytensor.scan`

        non_seq_names : list[str]
            A list of names representing static statespace matrices. That is, inputs that will need to be provided
            to the `non_sequences` argument of `pytensor.scan`
        """

        self.seq_names: list[str] = []
        self.non_seq_names: list[str] = []

        self.n_states = None
        self.n_posdef = None
        self.n_endog = None

        self.missing_fill_value: float | None = None
        self.cov_jitter = None

    def copy(self) -> "BaseFilter":
        """Return a shallow copy whose ``seq_names``/``non_seq_names`` are independent."""
        return copy.copy(self)

    def check_params(self, data, a0, P0, c, d, T, Z, R, H, Q):
        """
        Apply any checks on validity of inputs. For most filters this is just the identity function.
        """
        return data, a0, P0, c, d, T, Z, R, H, Q

    @staticmethod
    def add_check_on_time_varying_shapes(
        data: TensorVariable, sequence_params: list[TensorVariable]
    ) -> list[Variable]:
        """
        Insert a check that time-varying matrices match the data shape to the computational graph.

        If any matrices are time-varying, they need to have the same length as the data. This function wraps each
        element of `sequence_params` in an assert `Op` that makes sure all inputs have the correct shape.

        Parameters
        ----------
        data : TensorVariable
            The tensor representing the data.

        sequence_params : list[TensorVariable]
            A list of tensors to be provided to `pytensor.scan` as `sequences`.

        Returns
        -------
        list[TensorVariable]
            A list of tensors wrapped in an `Assert` `Op` that checks the shape of the 0th dimension on each is equal
             to the shape of the 0th dimension on the data.
        """
        # TODO: The PytensorRepresentation object puts the time dimension last, should the reshaping happen here in
        #    the Kalman filter, or in the StateSpaceModel, before passing into the KF?

        params_with_assert = [
            assert_time_varying_dim_correct(param, pt.eq(param.shape[0], data.shape[0]))
            for param in sequence_params
        ]

        return params_with_assert

    def unpack_args(self, args) -> tuple:
        """
        The order of inputs to the inner scan function is not known, since some, all, or none of the input matrices
        can be time varying. The order arguments are fed to the inner function is sequences, outputs_info,
        non-sequences. This function works out which matrices are where, and returns a standardized order expected
        by the kalman_step function.

        The standard order is: y, a0, P0, c, d, T, Z, R, H, Q
        """
        # If there are no sequence parameters (all params are static),
        # no changes are needed, params will be in order.
        args = list(args)
        n_seq = len(self.seq_names)
        if n_seq == 0:
            return tuple(args)

        # The first arg is always y
        y = args.pop(0)

        # There are always two outputs_info wedged between the seqs and non_seqs
        seqs, (a0, P0), non_seqs = args[:n_seq], args[n_seq : n_seq + 2], args[n_seq + 2 :]
        return_ordered = []
        for name in PARAM_NAMES:
            if name in self.seq_names:
                idx = self.seq_names.index(name)
                return_ordered.append(seqs[idx])
            else:
                idx = self.non_seq_names.index(name)
                return_ordered.append(non_seqs[idx])

        c, d, T, Z, R, H, Q = return_ordered

        return y, a0, P0, c, d, T, Z, R, H, Q

    def build_graph(
        self,
        data,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        missing_fill_value=None,
        cov_jitter=None,
        time_varying_names=(),
    ) -> list[TensorVariable]:
        """
        Construct the computation graph for the Kalman filter. See [1] for details.

        Parameters
        ----------
        data : TensorVariable
            Data to be filtered

        return_updates: bool, default False
            Whether to return updates associated with the pytensor scan. Should only be requried to debug pruposes.

        missing_fill_value: float, default -9999
            Fill value used to mark missing values. Used to avoid PyMC's automatic interpolation, which conflict's with
            the Kalman filter's hidden state inference. Change if your data happens to have legitimate values of -9999

        cov_jitter: float, default 1e-8 or 1e-6 if pytensor.config.floatX is float32
            The Kalman filter is known to be numerically unstable, especially at half precision. This value is added to
            the diagonal of every covariance matrix -- predicted, filtered, and smoothed -- at every step, to ensure
            all matrices are strictly positive semi-definite.

            Obviously, if this can be zero, that's best. In general:
                - Having measurement error makes Kalman Filters more robust. A large source of numerical errors come
                  from the Filtered and Smoothed matrices having a zero in the (0, 0) position, which always occurs
                  when there is no measurement error.

                - The Univariate Filter is more robust than other filters, and can tolerate a lower jitter value

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        if missing_fill_value is None:
            missing_fill_value = MISSING_FILL
        if cov_jitter is None:
            cov_jitter = JITTER_DEFAULT

        self.missing_fill_value = missing_fill_value
        self.cov_jitter = cov_jitter

        [R_shape] = constant_fold([R.shape], raise_not_constant=False)
        [Z_shape] = constant_fold([Z.shape], raise_not_constant=False)

        self.n_states, self.n_shocks = R_shape[-2:]
        self.n_endog = Z_shape[-2]

        data, a0, P0, *params = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)
        data = pt.specify_shape(data, (data.type.shape[0], self.n_endog))
        # ``time_varying_names`` is keyed on long names ("transition", ...) but the filter
        # tracks ordering with the short single-letter names. Translate.
        time_varying_short = {LONG_NAME_TO_SHORT[n] for n in time_varying_names}
        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES, time_varying_short
        )

        self.seq_names = seq_names
        self.non_seq_names = non_seq_names

        if len(sequences) > 0:
            sequences = self.add_check_on_time_varying_shapes(data, sequences)

        results = pytensor.scan(
            self.kalman_step,
            sequences=[data, *sequences],
            outputs_info=[None, a0, None, None, P0, None, None],
            non_sequences=non_sequences,
            name="forward_kalman_pass",
            strict=False,
            return_updates=False,
        )

        return self._postprocess_scan_results(results, a0, P0, n=data.type.shape[0])

    def _postprocess_scan_results(self, results, a0, P0, n) -> list[TensorVariable]:
        """
        Transform the values returned by the Kalman Filter scan into a form expected by users. In particular:
        1. Append the initial state and covariance matrix to their respective Kalman predictions. This matches the
        output returned by Statsmodels state space models.

        2. Discard the last state and covariance matrix from the Kalman predictions. This is beacuse the kalman filter
        starts with the (random variable) initial state x0, and treats it as a predicted state. The first step (t=0)
        will filter x0 to make filtered_states[0], then do a predict step to make predicted_states[1]. This means
        the last step (t=T) predicted state will be a *forecast* for T+1. If the user wants this forecast, he should
        use the forecast method.

        3. Squeeze away extra dimensions from the filtered and predicted states, as well as the likelihoods.
        """
        (
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances,
            predicted_covariances,
            observed_covariances,
            loglike_obs,
        ) = results

        predicted_states = pt.concatenate(
            [pt.expand_dims(a0, axis=(0,)), predicted_states[:-1]], axis=0
        )
        predicted_covariances = pt.concatenate(
            [pt.expand_dims(P0, axis=(0,)), predicted_covariances[:-1]], axis=0
        )

        filtered_states = pt.specify_shape(filtered_states, (n, self.n_states))
        filtered_states.name = FILTER_OUTPUT_NAMES[0]

        predicted_states = pt.specify_shape(predicted_states, (n, self.n_states))
        predicted_states.name = FILTER_OUTPUT_NAMES[1]

        filtered_covariances = pt.specify_shape(
            filtered_covariances, (n, self.n_states, self.n_states)
        )
        filtered_covariances.name = FILTER_OUTPUT_NAMES[2]

        predicted_covariances = pt.specify_shape(
            predicted_covariances, (n, self.n_states, self.n_states)
        )
        predicted_covariances.name = FILTER_OUTPUT_NAMES[3]

        observed_states = pt.specify_shape(observed_states, (n, self.n_endog))
        observed_states.name = FILTER_OUTPUT_NAMES[4]

        observed_covariances = pt.specify_shape(
            observed_covariances, (n, self.n_endog, self.n_endog)
        )
        observed_covariances.name = FILTER_OUTPUT_NAMES[5]

        loglike_obs = pt.specify_shape(loglike_obs.squeeze(), (n,))
        loglike_obs.name = "loglike_obs"

        filter_results = [
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances,
            predicted_covariances,
            observed_covariances,
            loglike_obs,
        ]

        return filter_results

    def handle_missing_values(
        self, y, Z, H, d
    ) -> tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, float]:
        """
        Handle missing values in the observation data ``y``.

        Adjust the design matrix ``Z``, the observation noise covariance matrix ``H``, and the observation
        intercept ``d`` by zeroing the rows associated with observations that are missing at this iteration.
        The missing entries of ``y`` are replaced with zeros to prevent propagating NaNs through the
        computation. With ``y``, ``Z @ a``, and ``d`` all zero on the missing rows, the innovation
        :math:`v = y - (Z a + d)` is exactly zero there, so missing observations contribute nothing to the
        state update.

        Return a binary flag tensor ``all_nan_flag`` indicating whether every component of the observation
        is missing. This flag is used for numerical adjustments in the update method.

        Parameters
        ----------
        y : TensorVariable
            The observation data at time t.
        Z : TensorVariable
            The design matrix.
        H : TensorVariable
            The observation noise covariance matrix.
        d : TensorVariable
            The observation intercept.

        Returns
        -------
        y_masked : TensorVariable
            Observation vector with missing values replaced by zeros.

        Z_masked : TensorVariable
            Design matrix with the rows corresponding to missing observations zeroed out.

        H_masked : TensorVariable
            Observation noise covariance matrix with the rows *and columns* corresponding to missing
            observations zeroed out, so the result remains symmetric.

        d_masked : TensorVariable
            Observation intercept with the entries corresponding to missing observations zeroed out. Without
            this masking, missing rows of the innovation become :math:`-d`, injecting a fake observation
            into the state update and inflating the log-likelihood by :math:`d^2 / \\text{jitter}`.

        all_nan_flag : float
            1 if every component of the observation is missing.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        nan_mask = pt.or_(pt.isnan(y), pt.eq(y, self.missing_fill_value))
        all_nan_flag = pt.all(nan_mask).astype(pytensor.config.floatX)
        W = pt.diag(pt.bitwise_not(nan_mask).astype(pytensor.config.floatX))

        Z_masked = W.dot(Z)
        H_masked = W.dot(H).dot(W.mT)
        d_masked = W.dot(d)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        return y_masked, Z_masked, H_masked, d_masked, all_nan_flag

    @staticmethod
    def predict(a, P, c, T, R, Q) -> tuple[TensorVariable, TensorVariable]:
        """
        Perform the prediction step of the Kalman filter.

        This function computes the one-step forecast of the hidden states and the covariance matrix of the forecasted
        states, based on the current state estimates and model parameters. For computational stability, the estimated
        covariance matrix is forced to by symmetric by averaging it with its own transpose. The prediction equations
        are:

        .. math::

            \begin{align}
            a_{t+1 | t} &= T_t a_{t | t} \\
            P_{t+1 | t} &= T_t P_{t | t} T_t^T + R_t Q_t R_t^T
            \\end{align}


        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate computed by the update step, a[t | t].
        P : TensorVariable
            The current covariance matrix estimate computed by the update step, P[t | t].
        c : TensorVariable
            The hidden state intercept/bias vector.
        T : TensorVariable
            The state transition matrix.
        R : TensorVariable
            The selection matrix.
        Q : TensorVariable
            The state innovation covariance matrix.

        Returns
        -------
        a_hat : TensorVariable
            One-step forecast of the hidden states
        P_hat : TensorVariable
            Covariance matrix of the forecasted hidden states

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        a_hat = T @ a + c
        P_hat = quad_form_sym(T, P) + quad_form_sym(R, Q)

        return a_hat, P_hat

    @staticmethod
    def update(
        a, P, y, d, Z, H, all_nan_flag
    ) -> tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """
        Perform the update step of the Kalman filter.

        This function updates the state vector and covariance matrix estimates based on the current observation data,
        previous predictions, and model parameters. The filtering equations are:

        .. math::

            \begin{align}
            \\hat{y}_t &= Z_t a_{t | t-1} + d_t \\
            v_t &= y_t - \\hat{y}_t \\
            F_t &= Z_t P_{t | t-1} Z_t^T + H_t \\
            a_{t|t} &= a_{t | t-1} + P_{t | t-1} Z_t^T F_t^{-1} v_t \\
            P_{t|t} &= P_{t | t-1} - P_{t | t-1} Z_t^T F_t^{-1} Z_t P_{t | t-1}
            \\end{align}


        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate, conditioned on information up to time t-1.
        P : TensorVariable
            The current covariance matrix estimate, conditioned on information up to time t-1.
        y : TensorVariable
            The observation data at time t.
        d : TensorVariable
            The matrix d.
        Z : TensorVariable
            The matrix Z.
        H : TensorVariable
            The matrix H.
        all_nan_flag : TensorVariable
            A binary flag tensor indicating whether there are any missing values in the observation data.

        Returns
        -------
        tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable]
            A tuple containing the updated state vector `a_filtered`, the updated covariance matrix `P_filtered`, the
            predicted observation `obs_mu`, the predicted observation covariance matrix `obs_cov`, and the log-likelihood `ll`.
        """
        raise NotImplementedError

    def kalman_step(self, *args) -> tuple:
        """
        Performs a single iteration of the Kalman filter, which is composed of two steps : an update step and a
        prediction step. The timing convention follows [1], in which initial state and covariance estimates a0 and P0
        are taken to be predictions. As a result, the update step is applied first. The update step computes:

        .. math::

            \begin{align}
            \\hat{y}_t &= Z_t a_{t | t-1} \\
            v_t &= y_t - \\hat{y}_t \\
            F_t &= Z_t P_{t | t-1} Z_t^T + H_t \\
            a_{t|t} &= a_{t | t-1} + P_{t | t-1} Z_t^T F_t^{-1} v_t \\
            P_{t|t} &= P_{t | t-1} - P_{t | t-1} Z_t^T F_t^{-1} Z_t P_{t | t-1}
            \\end{align}

        Where the quantities :math:`a_{t|t}` and :math:`P_{t|t}` are the best linear estimates of the hidden states
        at time t, incorporating all information up to and including the observation :math:`y_t`. After the update step,
        new one-step forecasts of the hidden states can be obtained by applying the model transition dynamics in
        the prediction step:

        .. math::

            \begin{align}
            a_{t+1 | t} &= T_t a_{t | t} \\
            P_{t+1 | t} &= T_t P_{t | t} T_t^T + R_t Q_t R_t^T
            \\end{align}

        Recursive application of these two steps results in the best linear estimate of the hidden states, including
        missing values and observations subject to measurement error.

        Parameters
        ----------
        Kalman filter inputs:
            y, a, P, c, d, T, Z, R, H, Q. See the docstring for the kalman filter class for details.

        Returns
        -------
        a_filtered : TensorVariable
            Best linear estimate of hidden states given all information up to and including the present
             observation, a[t | t].

        a_hat: TensorVariable
            One-step forecast of next-period hidden states given all information up to and including the present
            observation, a[t+1 | t]

        obs_mu: TensorVariable
            Estimates of the current observation given all information available prior to the current state,
             d + Z @ a[t | t-1]

        P_filtered: TensorVariable
            Best linear estimate of the covariance between hidden states, given all information up to and including
            the present observation, P[t | t]

        P_hat: TensorVariable
            Covariance between the one-step forecasted hidden states given all information up to and including the
            present observation, P[t+1 | t]

        obs_cov: TensorVariable
            Covariance between estimated present observations, given all information available prior to the current
            state, Z @ P[t | t-1] @ Z.T + H

        ll: float
            Likelihood of the time t observation vector under the multivariate normal distribution parameterized by
            `obs_mu` and `obs_cov`

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        y, a, P, c, d, T, Z, R, H, Q = self.unpack_args(args)
        y_masked, Z_masked, H_masked, d_masked, all_nan_flag = self.handle_missing_values(
            y, Z, H, d
        )

        a_filtered, P_filtered, obs_mu, obs_cov, ll = self.update(
            y=y_masked, a=a, d=d_masked, P=P, Z=Z_masked, H=H_masked, all_nan_flag=all_nan_flag
        )

        P_filtered = stabilize(P_filtered, self.cov_jitter)

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)
        outputs = (a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll)

        return outputs


class StandardFilter(BaseFilter):
    """
    Basic Kalman Filter
    """

    def update(self, a, P, y, d, Z, H, all_nan_flag):
        """
        Compute one-step forecasts for observed states conditioned on information up to, but not including, the current
        timestep, `y_hat`, along with the forcast covariance matrix, `F`. Marginalize over observed states to obtain
        the best linear estimate of the unobserved states, `a_filtered`, as well as the associated covariance matrix,
        `P_filtered`,  conditioned on all information, up to and including the present.

        Derivation of the Kalman filter, along with a deeper discussion of the computational elements, can be found in
        [1].

        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate, conditioned on information up to time t-1.

        P : TensorVariable
            The current covariance matrix estimate, conditioned on information up to time t-1.

        y : TensorVariable
            Observations at time t.

        d : TensorVariable
            Observed state bias term.

        Z : TensorVariable
            Linear map between unobserved and observed states.

        H : TensorVariable
            Observation noise covariance matrix

        all_nan_flag : TensorVariable
            A flag indicating whether all elements in the data `y` are NaNs.

        Returns
        -------
        tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, float]
            A tuple containing the updated state vector `a_filtered`, the updated covariance matrix `P_filtered`,
            the one-step forecast mean `y_hat`, one-step forcast covariance matrix  `F`, and the log-likelihood of
            the data, given the one-step forecasts, `ll`.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        y_hat = d + Z @ a
        v = y - y_hat

        PZT = P.dot(Z.mT)
        F = Z.dot(PZT) + stabilize(H, self.cov_jitter)

        F_chol = pt.linalg.cholesky(F, lower=True)

        K = pt.linalg.cho_solve((F_chol, True), PZT.mT).mT
        I_KZ = pt.eye(self.n_states) - K.dot(Z)

        a_filtered = a + K @ v
        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        F_inv_v = pt.linalg.cho_solve((F_chol, True), v)
        inner_term = v.T @ F_inv_v

        F_logdet = 2 * pt.log(pt.diag(F_chol)).sum()

        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * (MVN_CONST + F_logdet + inner_term).ravel()[0],
        )

        return a_filtered, P_filtered, y_hat, F, ll


class SquareRootFilter(BaseFilter):
    """
    Kalman filter with Cholesky factorization

    Kalman filter implementation using a Cholesky factorization plus pt.solve_triangular to (attempt) to speed up
    inversion of the observation covariance matrix `F`.

    """

    def predict(self, a, P, c, T, R, Q):
        """
        Compute one-step forecasts for the hidden states conditioned on information up to, but not including, the current
        timestep, `a_hat`, along with the forcast covariance matrix, `P_hat`.

        .. warning::
            Very important -- In this function, $P$ is the **cholesky factor** of the covariance matrix, not the
            covariance matrix itself. The name `P` is kept for consistency with the superclass.
        """
        # Rename P to P_chol for clarity
        P_chol = P

        a_hat = T.dot(a) + c
        Q_chol = pt.linalg.cholesky(Q, lower=True)

        M = pt.horizontal_stack(T @ P_chol, R @ Q_chol).mT
        R_decomp = pt.linalg.qr(M, mode="r")
        P_chol_hat = R_decomp[..., : self.n_states, : self.n_states].mT

        return a_hat, P_chol_hat

    def update(self, a, P, y, d, Z, H, all_nan_flag):
        """
        Compute posterior estimates of the hidden state distributions conditioned on the observed data, up to and
        including the present timestep. Also compute the log-likelihood of the data given the one-step forecasts.

        .. warning::
            Very important -- In this function, $P$ is the **cholesky factor** of the covariance matrix, not the
            covariance matrix itself. The name `P` is kept for consistency with the superclass.
        """

        # Rename P to P_chol for clarity
        P_chol = P

        y_hat = Z.dot(a) + d
        v = y - y_hat
        H_chol = pt.linalg.cholesky(stabilize(H, self.cov_jitter), lower=True)

        # The following notation comes from https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A.pdf
        # Construct upper-triangular block matrix A = [[chol(H), Z @ L_pred],
        #                                              [0,           L_pred]]
        # The Schur decomposition of this matrix will be B (upper triangular). We are
        # more interested in B^T:
        # Structure of B^T = [[chol(F),     0              ],
        #                    [K @ chol(F), chol(P_filtered)]
        zeros = pt.zeros((self.n_states, self.n_endog))
        upper = pt.horizontal_stack(H_chol, Z @ P_chol)
        lower = pt.horizontal_stack(zeros, P_chol)
        A_T = pt.vertical_stack(upper, lower)
        B = pt.linalg.qr(A_T.mT, mode="r").mT

        F_chol = B[: self.n_endog, : self.n_endog]
        K_F_chol = B[self.n_endog :, : self.n_endog]
        P_chol_filtered = B[self.n_endog :, self.n_endog :]

        def compute_non_degenerate(P_chol_filtered, F_chol, K_F_chol, v):
            a_filtered = a + K_F_chol @ solve_triangular(F_chol, v, lower=True)

            inner_term = solve_triangular(
                F_chol, solve_triangular(F_chol, v, lower=True), lower=True
            )

            loss = (v.T @ inner_term).ravel()

            # abs necessary because we're not guaranteed a positive diagonal from the schur decomposition
            logdet = 2 * pt.log(pt.abs(pt.diag(F_chol))).sum()

            ll = -0.5 * (self.n_endog * (MVN_CONST + logdet) + loss)[0]

            return [a_filtered, P_chol_filtered, ll]

        def compute_degenerate(P_chol_filtered, F_chol, K_F_chol, v):
            """
            If F is zero (usually because there were no observations this period), then we want:
            K = 0, a = a, P = P, ll = 0
            """
            return [a, P_chol, pt.zeros(())]

        degenerate = pt.eq(all_nan_flag, 1.0)
        F_chol = pytensor.ifelse(degenerate, pt.eye(*F_chol.shape), F_chol)
        [a_filtered, P_chol_filtered, ll] = pytensor.ifelse(
            degenerate,
            compute_degenerate(P_chol_filtered, F_chol, K_F_chol, v),
            compute_non_degenerate(P_chol_filtered, F_chol, K_F_chol, v),
        )

        a_filtered = pt.specify_shape(a_filtered, (self.n_states,))
        P_chol_filtered = pt.specify_shape(P_chol_filtered, (self.n_states, self.n_states))

        return a_filtered, P_chol_filtered, y_hat, F_chol, ll

    def _postprocess_scan_results(self, results, a0, P0, n) -> list[TensorVariable]:
        """
        Convert the Cholesky factor of the covariance matrix back to the covariance matrix itself.
        """
        results = super()._postprocess_scan_results(results, a0, P0, n)
        (
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances_cholesky,
            predicted_covariances_cholesky,
            observed_covariances_cholesky,
            loglike_obs,
        ) = results

        def square_sequnece(L, k):
            X = pt.einsum("...ij,...kj->...ik", L, L.copy())
            X = pt.specify_shape(X, (n, k, k))
            return X

        filtered_covariances = square_sequnece(filtered_covariances_cholesky, k=self.n_states)
        predicted_covariances = square_sequnece(predicted_covariances_cholesky, k=self.n_states)
        observed_covariances = square_sequnece(observed_covariances_cholesky, k=self.n_endog)

        return [
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances,
            predicted_covariances,
            observed_covariances,
            loglike_obs,
        ]


class UnivariateFilter(BaseFilter):
    """
    The univariate kalman filter, described in [1], section 6.4.2, avoids inversion of the F matrix, as well as two
    matrix multiplications, at the cost of an additional loop. Note that the name doesn't mean there's only one
    observed time series. This is called univariate because it updates the state mean and covariance matrices one
    variable at a time, using an inner-inner loop.

    This is useful when states are perfectly observed, because the F matrix can easily become degenerate in these cases.

    References
    ----------
    .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
            2nd ed, Oxford University Press, 2012.

    """

    def _univariate_inner_filter_step(self, y, Z_row, d_row, sigma_H, nan_flag, a, P):
        y_hat = Z_row.dot(a) + d_row
        v = y - y_hat

        PZT = P.dot(Z_row.T)
        F = Z_row.dot(PZT) + sigma_H

        # Set the zero flag for F first, then jitter it to avoid a divide-by-zero NaN later
        F_zero_flag = pt.or_(pt.eq(F, 0), nan_flag)
        F = F + self.cov_jitter

        # If F is zero (implies y is NAN or another degenerate case), then we want:
        # K = 0, a = a, P = P, ll = 0
        K = PZT / F * (1 - F_zero_flag)

        a_filtered = a + K * v
        P_filtered = P - pt.outer(K, K) * F

        ll_inner = pt.switch(F_zero_flag, 0.0, pt.log(F) + v**2 / F)

        return a_filtered, P_filtered, pt.atleast_1d(y_hat), pt.atleast_2d(F), ll_inner

    def kalman_step(self, *args):
        y, a, P, c, d, T, Z, R, H, Q = self.unpack_args(args)

        nan_mask = pt.or_(pt.isnan(y), pt.eq(y, self.missing_fill_value))
        y_masked, Z_masked, H_masked, d_masked, all_nan_flag = self.handle_missing_values(
            y, Z, H, d
        )

        result = pytensor.scan(
            self._univariate_inner_filter_step,
            sequences=[y_masked, Z_masked, d_masked, pt.diag(H_masked), nan_mask],
            outputs_info=[a, P, None, None, None],
            name="univariate_inner_scan",
            return_updates=False,
        )

        a_filtered, P_filtered, obs_mu, obs_cov, ll_inner = result
        a_filtered, P_filtered, obs_mu, obs_cov = (
            a_filtered[-1],
            P_filtered[-1],
            obs_mu[-1],
            obs_cov[-1],
        )

        P_filtered = stabilize(0.5 * (P_filtered + P_filtered.mT), self.cov_jitter)
        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * ((pt.neq(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum()),
        )

        return a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll


class ConvergentFilter(StandardFilter):
    """Kalman filter that exploits Riccati convergence to a fixed-gain steady state.

    For a time-invariant system whose Riccati recursion reaches a steady state -- detectability and
    stabilizability are sufficient, which is weaker than stationarity and admits unit-root models such
    as the local level -- the predicted-covariance recursion
    ``P_{t+1|t} = T (P_{t|t-1} - K_t F_t K_t^T) T^T + R Q R^T`` is data-independent and converges to the
    discrete algebraic Riccati equation (DARE) fixed point ``P_star``. Once converged, the Kalman gain
    ``K_t = P_{t|t-1} Z^T F_t^{-1}`` and innovation covariance ``F_t = Z P_{t|t-1} Z^T + H`` are constant
    (``K_star``, ``F_star``) and the filter becomes a linear time-invariant recursion on the predicted
    state ``a_{t|t-1}`` alone. ``ConvergentFilter`` splits the forward pass at the convergence step ``k``:

    - **pre-convergence steps** (``t = 0..k-1``, the "transient"): full Kalman with update + Riccati predict.
    - **post-convergence steps** (``t = k..n-1``, the "tail"): linear recursion on ``a`` with cached
      ``K_star``, ``F_star``, ``log det F_star``. No per-step Cholesky; loss is ``log det F_star + v_t^T F_star^{-1} v_t``
      evaluated via a pre-computed Cholesky of ``F_star`` and a per-step triangular solve.

    The gradient is delivered via a custom pullback on an :class:`~pytensor.compile.builders.OpFromGraph`
    rather than via autodiff. Two reasons:

    1. **Correctness.** Autodiff through the two-scan forward gives *incorrect* gradients when the ``until``
       clause on the pre-convergence scan actually fires. The forward value is fine, and one scan in
       isolation autodiffs correctly -- but the combination of an ``until``-scan followed by a second scan
       that depends on quantities derived from the first scan's *last* outputs (our case: ``K_star``,
       ``F_star`` derived from ``P_hat_tr[-1]``) produces parameter gradients that are wrong by 5--100%
       relative error even though the likelihood value matches to machine precision. With ``tol`` set so
       the until never fires, autodiff is exact; with a real post-convergence tail it is not.

    2. **Speed.** In the post-convergence tail, per-step Cholesky and the full Joseph-form backward are
       unnecessary because ``K``, ``F``, and ``P`` are constant. The custom pullback hoists
       ``F^{-1}, A = T (I - K_star Z), I - K_star Z`` out of the scan body and the per-step
       backward reduces to matvec and outer products. On typical state-space cells
       (m=10..50, p=3..10, n=2000..10000) this gives a ~1.8--2.9x speedup on ``logp + grad``
       versus :class:`StandardFilter` autodiff, on top of a ~7--12x forward speedup.

    Constraints (validated at graph build time):

    - All model matrices (``c, d, T, Z, R, H, Q``) must be time-invariant. Time-varying inputs raise
      :class:`ValueError`: a constant steady-state gain only exists when the system matrices are
      constant. This is a statement about time-invariance, not stationarity -- a time-invariant model
      whose Riccati recursion never settles within the series simply never triggers the ``until``
      clause, so ``k = n``, the post-convergence tail is empty, and the result reduces to
      :class:`StandardFilter`.
    - ``data`` must contain no missing values. A masked observation changes the effective ``Z_t`` (and
      so ``K_t`` and ``F_t``), breaking the cached-``K_star`` assumption. The statespace core replaces
      ``NaN`` with ``missing_fill_value`` before the filter runs, so both ``NaN`` and that sentinel are
      rejected. For :class:`SharedVariable` or :class:`TensorConstant` data the check runs at build
      time; fully symbolic data gets an :class:`~pytensor.raise_op.Assert` that fires at runtime.

    Measurement-error-free models (singular or zero ``H``) are fully supported, gradients included;
    the tail backward avoids forming ``H^{-1}``.

    Only ``d loglike_obs`` is supported as an upstream gradient; other output adjoints are treated as
    disconnected. For gradients of other filter outputs, use :class:`StandardFilter`.

    Parameters
    ----------
    tol : float, optional
        Convergence tolerance on the Frobenius norm of ``P_{t+1|t} - P_{t|t-1}`` in the pre-convergence
        scan. The scan stops the first time this drops below ``tol``. Default ``1e-10``.
    """

    _MISSING_DATA_MSG = (
        "ConvergentFilter does not support missing data in the observation series. "
        "Use StandardFilter for data with missing values."
    )

    def __init__(self, tol: float = 1e-10):
        super().__init__()
        self.tol = tol

    def _check_data(self, data):
        """Reject data containing NaN or the missing-value sentinel.

        The statespace core replaces NaN with ``missing_fill_value`` before the filter runs, so the
        sentinel must be rejected alongside NaN. Concrete data (shared/constant) is checked at build
        time; fully symbolic data gets a runtime :class:`~pytensor.raise_op.Assert`.
        """
        sentinel = self.missing_fill_value
        if isinstance(data, SharedVariable | TensorConstant):
            arr = data.get_value() if isinstance(data, SharedVariable) else data.data
            if np.isnan(arr).any() or (arr == sentinel).any():
                raise ValueError(self._MISSING_DATA_MSG)
            return data
        missing = pt.or_(pt.isnan(data), pt.eq(data, sentinel))
        return Assert(self._MISSING_DATA_MSG)(data, pt.all(pt.bitwise_not(missing)))

    def _kalman_step(self, y, a, P, c, d, T, Z, R, H, Q):
        a_filt, P_filt, y_hat, F, ll = self.update(
            a=a, P=P, y=y, d=d, Z=Z, H=H, all_nan_flag=pt.constant(0.0)
        )
        P_filt = stabilize(P_filt, self.cov_jitter)
        a_hat, P_hat = self.predict(a=a_filt, P=P_filt, c=c, T=T, R=R, Q=Q)
        return a_filt, a_hat, y_hat, P_filt, P_hat, F, ll

    def _full_kalman_scan(self, data, a0, P0, c, d, T, Z, R, H, Q, stop_early):
        def body(y, a_prev, P_prev, *params):
            out = self._kalman_step(y, a_prev, P_prev, *params)
            if stop_early:
                return out, until(pt.sqrt(((out[4] - P_prev) ** 2).sum()) < self.tol)
            return out

        return pytensor.scan(
            body,
            sequences=[data],
            outputs_info=[None, a0, None, None, P0, None, None],
            non_sequences=[c, d, T, Z, R, H, Q],
            strict=False,
            return_updates=False,
        )

    def _convergent_forward(self, data, a0, P0, c, d, T, Z, R, H, Q):
        """Two-phase forward pass.

        Phase 1 (pre-convergence): a standard Kalman scan with an ``until`` clause that stops once the Riccati
        recursion has converged (``||P_{t+1|t} - P_{t|t-1}||_F < tol``). At the stopping step ``k`` we have the DARE
        fixed point estimate ``P_star = P_hat_tr[-1]``.

        Phase 2 (post-convergence): ``K_star, F_star, F_chol_star, log det F_star, P_filt_star`` are computed once from
         ``P_star`` and plugged into a cheaper scan whose step only does ``v_t = y_t - Z a_prev - d``, a triangular
         solve against ``F_chol_star`` for the loss, a matvec ``a_filt = a_prev + K_star v_t``, and one matvec for
         the state predict ``a_hat = T a_filt + c``.

        Outputs are the 7 per-step sequences produced by :class:`StandardFilter` plus the convergence step ``k``
        (a symbolic integer). The pullback uses ``k`` to split the adjoint computation. In the post-convergence segment
        the sequences ``P_filt``, ``P_hat``, and ``F`` are broadcasts of constants.
        """
        n = data.shape[0]
        (a_filt_tr, a_hat_tr, y_hat_tr, P_filt_tr, P_hat_tr, F_tr, ll_tr) = self._full_kalman_scan(
            data, a0, P0, c, d, T, Z, R, H, Q, stop_early=True
        )

        # Cap the split one step short of the series so the post-convergence tail always has at least
        # one step.
        k = pt.minimum(a_filt_tr.shape[0], n - 1)
        a_star, P_star = a_hat_tr[k - 1], P_hat_tr[k - 1]

        # Tail constants from P*. Computed once here to make the tail function prettier.
        F_star = Z @ P_star @ Z.mT + stabilize(H, self.cov_jitter)
        F_chol_star = pt.linalg.cholesky(F_star, lower=True)
        logdet_F_star = 2 * pt.log(pt.diag(F_chol_star)).sum()
        K_star = pt.linalg.solve(
            F_star.mT, (P_star @ Z.mT).mT, assume_a="pos", check_finite=False
        ).mT
        I_KZ_star = pt.eye(self.n_states) - K_star @ Z
        P_filt_star = stabilize(
            quad_form_sym(I_KZ_star, P_star) + quad_form_sym(K_star, H), self.cov_jitter
        )

        # Tail scan: fixed K*, F* (no per-step Cholesky). Recurrent slot is a_hat.
        def tail_body(y, a_prev):
            y_hat = d + Z @ a_prev
            v = y - y_hat
            solved = solve_triangular(F_chol_star, v, lower=True)
            ll = -0.5 * (MVN_CONST + logdet_F_star + (solved * solved).sum())
            a_filt = a_prev + K_star @ v
            return a_filt, T @ a_filt + c, y_hat, ll

        a_filt_fx, a_hat_fx, y_hat_fx, ll_fx = pytensor.scan(
            tail_body,
            sequences=[data[k:]],
            outputs_info=[None, a_star, None, None],
            strict=False,
            return_updates=False,
        )

        # Stitch: in the tail, P_filt, P_hat, F are constants (broadcast).
        n_fx = n - k
        m_shape = (n_fx, self.n_states, self.n_states)
        P_filt_fx = pt.broadcast_to(P_filt_star, m_shape)
        P_hat_fx = pt.broadcast_to(P_star, m_shape)
        F_fx = pt.broadcast_to(F_star, (n_fx, self.n_endog, self.n_endog))

        def cat(a, b):
            return pt.concatenate([a, b], axis=0)

        return (
            cat(a_filt_tr[:k], a_filt_fx),
            cat(a_hat_tr[:k], a_hat_fx),
            cat(y_hat_tr[:k], y_hat_fx),
            cat(P_filt_tr[:k], P_filt_fx),
            cat(P_hat_tr[:k], P_hat_fx),
            cat(F_tr[:k], F_fx),
            cat(ll_tr[:k], ll_fx),
            k,
        )

    def _fixed_k_tail_backward(
        self,
        data,
        a_star,
        c,
        d,
        T,
        Z,
        K_star,
        F_star,
        F_chol_star,
        a_hat_fx,
        dlogp,
    ):
        """Analytic backward pass over the post-convergence segment.

        Runs a backward scan on the post-convergence data with ``F_inv, A = T (I - K_star Z),
        I - K_star Z`` hoisted out as non-sequences. Each step is closed-form (no Cholesky or solve in
        the body; only matvecs and outer products). Returns:

        - ``d_data, d_a_star``: ordinary adjoints flowing backward out of the segment.
        - ``d_P_star``: Riccati-style adjoint at the segment's entry. Used as the terminal condition for
          the pre-convergence backward (the transient's ``P_hat_grad`` at step ``k-1``).
        - ``d_T_alpha, d_c, d_Z_direct, d_d``: per-step contributions summed across post-convergence.
          These are the alpha-path (mean of the latent states) contributions; P-path contributions to d_T / d_Z
          come from the ``S`` aggregate below.
        - ``d_K_star, d_F_star``: adjoints treating ``K_star, F_star`` as if they were independent
          inputs to the post-convergence forward. The caller chains these to ``(P_star, Z, H)`` via the
          analytic definitions ``K_star = P_star Z^T F_star^{-1}`` and ``F_star = Z P_star Z^T + H``.
        - ``S = sum_t d_P_hat_t``: aggregate P-adjoint across the post-convergence segment. In the
          post-convergence segment ``P_t`` is the constant ``P_star``, so P-path contributions to
          d_T, d_Z, d_H, d_R, d_Q are linear in ``d_P_hat_t``. These aggregate into formulas that use
          ``S`` (rather than needing a separate per-step scan).
        """
        m = a_star.shape[0]
        p_dim = F_chol_star.shape[0]
        n_tail = data.shape[0]
        a_prev_seq = pt.concatenate([a_star[None, :], a_hat_fx[:-1]], axis=0)

        # Hoisted constants.
        I_KZ = pt.eye(m) - K_star @ Z
        A = T @ I_KZ
        F_inv = pt.linalg.cho_solve((F_chol_star, True), pt.eye(p_dim))
        ZtFinvZ = Z.T @ F_inv @ Z  # the data-independent half of direct_C, lifted out of the scan

        def bwd(y, a_prev, r_next, P_hat_next, c_, d_, T_, Z_, K_, F_inv_, A_, ZtFinvZ_):
            v = y - Z_ @ a_prev - d_
            Finv_v = F_inv_ @ v
            a_filt = a_prev + K_ @ v
            T_r_next = T_.T @ r_next
            A_r_next = A_.T @ r_next
            Zt_Finv_v = Z_.T @ Finv_v

            dL_dv = K_.T @ T_r_next + 2 * Finv_v
            r_t = A_r_next - 2 * Zt_Finv_v

            # P-adjoint recursion: d_P_prev = A^T @ d_P_hat_next @ A + cross_a + direct_C,
            # where A = T (I - K_star Z) is the closed-loop transition. The homogeneous part
            # d_P_prev = A^T @ d_P_hat_next @ A is the backward Lyapunov recursion: as the segment
            # length grows it converges to the solution of the Lyapunov equation X = A^T X A + C, which
            # is the steady-state P-adjoint. Because A is constant in the post-convergence segment,
            # we can hoist it; the per-step work is the two correction terms (cross_a and direct_C)
            # which still depend on y_t and a_prev. cross_a uses the identity Z(I - K Z) = H F^{-1} Z
            # to express what is naturally an H^{-1} term through F^{-1} instead, which stays finite for
            # singular H (a measurement-error-free model).
            cross_a = 0.5 * (pt.outer(A_r_next, Zt_Finv_v) + pt.outer(Zt_Finv_v, A_r_next))
            direct_C = ZtFinvZ_ - pt.outer(Zt_Finv_v, Zt_Finv_v)
            d_P_prev = A_.T @ P_hat_next @ A_ + cross_a + direct_C

            dL_dT = pt.outer(r_next, a_filt)
            dL_dc = r_next
            dL_dZ = -pt.outer(dL_dv, a_prev)
            dL_dd = -dL_dv
            dL_dK = pt.outer(T_r_next, v)
            vvT = pt.outer(v, v)

            return r_t, d_P_prev, dL_dv, dL_dT, dL_dc, dL_dZ, dL_dd, dL_dK, vvT

        (r_seq, dPp_seq, dy_seq, dT_seq, dc_seq, dZ_seq, dd_seq, dK_seq, vvT_seq) = pytensor.scan(
            bwd,
            sequences=[data, a_prev_seq],
            outputs_info=[
                pt.zeros_like(a_star),
                pt.zeros((m, m), dtype="float64"),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            non_sequences=[c, d, T, Z, K_star, F_inv, A, ZtFinvZ],
            go_backwards=True,
            strict=False,
            return_updates=False,
        )

        # d_P_prev_seq in go_backwards order is [d_P_hat_{n-2}, ..., d_P_hat_{k-1}]; the last
        # element is d_P_star (the handoff fed to the pre-convergence backward), so the aggregate
        # S = sum_t d_P_hat_t over the post-convergence segment excludes it.
        S = dlogp * dPp_seq[:-1].sum(axis=0)
        sum_vvT = vvT_seq.sum(axis=0)
        return (
            dlogp * dy_seq[::-1],  # d_data
            dlogp * r_seq[-1],  # d_a_star
            dlogp * dPp_seq[-1],  # d_P_star (Riccati)
            dlogp * dT_seq.sum(axis=0),  # d_T alpha-path
            dlogp * dc_seq.sum(axis=0),  # d_c
            dlogp * dZ_seq.sum(axis=0),  # d_Z direct
            dlogp * dd_seq.sum(axis=0),  # d_d
            dlogp * dK_seq.sum(axis=0),  # d_K_star
            dlogp * (n_tail * F_inv - F_inv @ sum_vvT @ F_inv),  # d_F_star
            S,
        )

    def build_graph(
        self,
        data,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        missing_fill_value=None,
        cov_jitter=None,
        time_varying_names=(),
    ):
        if time_varying_names:
            raise ValueError(
                "ConvergentFilter requires time-invariant matrices; got time-varying: "
                f"{sorted(time_varying_names)}. Use StandardFilter instead."
            )
        self.missing_fill_value = MISSING_FILL if missing_fill_value is None else missing_fill_value
        self.cov_jitter = JITTER_DEFAULT if cov_jitter is None else cov_jitter

        [R_shape] = constant_fold([R.shape], raise_not_constant=False)
        [Z_shape] = constant_fold([Z.shape], raise_not_constant=False)
        self.n_states, self.n_shocks = R_shape[-2:]
        self.n_endog = Z_shape[-2]

        data = self._check_data(data)
        data, a0, P0, c, d, T, Z, R, H, Q = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)
        data = pt.specify_shape(data, (data.type.shape[0], self.n_endog))
        self.seq_names = []
        self.non_seq_names = list(PARAM_NAMES)

        # The forward is two scans chained through derived quantities (K_star, F_star, log det F_star
        # and P_filt_star all depend on the first scan's last output). Autodiff gives incorrect
        # gradients on this structure once the `until` clause fires with a non-trivial post-convergence
        # segment. We wrap in an OFG and supply a custom pullback that splits the backward
        # into: an analytic backward on the post-convergence segment (cheap, closed form), a chain
        # rule step for (K_star, F_star) -> (P_star, Z, H) using the analytic definitions, and an
        # autodiff backward on the pre-convergence segment with the handoff adjoints as its terminal
        # condition (the plain Kalman iterations).
        op = self._make_ofg()
        a_filt, a_hat, y_hat, P_filt, P_hat, F, ll, _k = op(data, a0, P0, c, d, T, Z, R, H, Q)
        n = data.type.shape[0]
        return self._postprocess_scan_results(
            (a_filt, a_hat, y_hat, P_filt, P_hat, F, ll[..., None]), a0, P0, n=n
        )

    def _make_ofg(self):
        m, p, r_ = self.n_states, self.n_endog, self.n_shocks
        cov_jitter = self.cov_jitter

        sym = {
            "data": pt.tensor("data", dtype="float64", shape=(None, p)),
            **{
                name: pt.tensor(name, dtype="float64", shape=shape)
                for name, shape in static_matrix_shapes(m, p, r_).items()
            },
        }
        in_order = ["data", "a0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
        inputs_s = [sym[k] for k in in_order]
        outputs = list(self._convergent_forward(*inputs_s))
        filt_self = self

        def pullback(inputs, outputs, out_grads):
            # `outputs` are the 8 forward outputs of `_convergent_forward`:
            # (a_filt, a_hat, y_hat, P_filt, P_hat, F, ll, k). Only the sequences we need for the
            # backward are bound; the rest are `_`. Similarly, `out_grads[6]` is the adjoint on `ll`;
            # all other output adjoints are assumed disconnected (see class docstring).
            data_, a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_ = inputs
            _, a_hat_sym, _, _, P_hat_sym, _, _, k_sym = outputs
            dll = out_grads[6]

            # Split the recorded per-step sequences at the convergence step `k_sym`. The
            # pre-convergence slice is `[:k_sym]` and the post-convergence slice is `[k_sym:]`;
            # a_star and P_star are the handoff values flowing between them (a_{k|k-1}, P_{k|k-1}).
            a_hat_tr, P_hat_tr = a_hat_sym[:k_sym], P_hat_sym[:k_sym]
            data_tr, data_fx = data_[:k_sym], data_[k_sym:]
            a_star, P_star = a_hat_tr[-1], P_hat_tr[-1]
            a_hat_fx = a_hat_sym[k_sym:]
            # Reduce the per-step ll adjoint to a scalar by averaging. This is correct for the
            # standard use case loss = ll.sum() (dll is a broadcast of ones, average is 1.0) and
            # punts on the more general case of per-step weights.
            dlogp_up = dll.sum() / dll.shape[0]

            # Pre-compute tail constants for readability.
            F_star = Z_ @ P_star @ Z_.mT + stabilize(H_, cov_jitter)
            F_chol = pt.linalg.cholesky(F_star, lower=True)
            K_star = pt.linalg.solve(
                F_star.mT, (P_star @ Z_.mT).mT, assume_a="pos", check_finite=False
            ).mT
            F_inv = pt.linalg.cho_solve((F_chol, True), pt.eye(F_chol.shape[0]))

            # Post-convergence backward. This is the source of the speedup: K, F, and P are constant
            # across the segment, so the per-step backward drops to matvec/outer-product work.
            # The factor of -0.5 on dlogp_up is a convention adjustment: the analytic formulas in
            # `_fixed_k_tail_backward` are derived for ll = log |F| + v^T F^{-1} v (the "2 * negative
            # log-likelihood" form), while our per-step ll is -0.5 * (MVN_CONST + log |F| + v^T F^{-1} v).
            # So the caller's dlogp_up needs a -0.5 factor before being fed into the analytic formulas.
            (
                dd_fx,
                da_star,
                dP_star_ric,
                dT_a,
                dc_t,
                dZ_dir,
                dd_t,
                dK_s,
                dF_s,
                S,
            ) = filt_self._fixed_k_tail_backward(
                data_fx,
                a_star,
                c_,
                d_,
                T_,
                Z_,
                K_star,
                F_star,
                F_chol,
                a_hat_fx,
                -0.5 * dlogp_up,
            )

            # Use chain rule to recover dZ and dH from dK*, dF*. Fully analytic.
            dK, dF = disconnected_grad(dK_s), disconnected_grad(dF_s)
            P_filt_s = (pt.eye(m) - K_star @ Z_) @ P_star
            d_Z_chain = (
                F_inv @ dK.mT @ P_filt_s - K_star.mT @ dK @ K_star.mT + (dF + dF.mT) @ Z_ @ P_star
            )
            X = K_star.mT @ dK @ F_inv
            d_H_chain = dF - 0.5 * (X + X.mT)

            # P-path contributions to d_T, d_Z, d_H. Because P_t is the constant P_star across the
            # post-convergence segment, each step's contribution is linear in d_P_hat_t, so the sum
            # over the segment is obtained by substituting S = sum_t d_P_hat_t into the formulas.
            TtST = T_.mT @ S @ T_
            KtPg = K_star.mT @ TtST
            FinvZP = F_inv @ Z_ @ P_star
            KZP = (K_star @ Z_) @ P_star
            d_T_P = (S @ T_) @ P_filt_s.mT + (S.mT @ T_) @ P_filt_s
            d_Z_P = (
                -FinvZP @ TtST.mT @ P_star
                + KtPg @ P_star.mT @ Z_.mT @ K_star.mT
                + FinvZP @ TtST.mT @ KZP
                - KtPg @ P_star.mT
            )
            d_H_P = KtPg @ K_star
            # Chain S through W = R Q R^T for (d_R, d_Q).
            d_R_P = (S + S.mT) @ R_ @ Q_
            d_Q_P = R_.mT @ S @ R_

            # Pre-convergence backward. We rebuild the Kalman scan on the sliced pre-convergence data
            # (data[:k_sym]) and autodiff through it. The surrogate loss has three parts: the
            # pre-convergence log-likelihood itself, plus two inner products that encode the handoff
            # adjoints (da_star for the last predicted state, dP_star_ric for the last predicted
            # covariance). disconnected_grad on the adjoints keeps pt.grad from trying to chase their
            # own parameter dependence (they are outputs of the post-convergence backward and are
            # treated as constants here. If we connect their gradients we would be double-counting).
            _, a_hat_rb, _, _, P_hat_rb, _, ll_rb = filt_self._full_kalman_scan(
                data_tr,
                a0_,
                P0_,
                c_,
                d_,
                T_,
                Z_,
                R_,
                H_,
                Q_,
                stop_early=False,
            )
            loss_tr = (
                dlogp_up * ll_rb.sum()
                + (a_hat_rb[-1] * disconnected_grad(da_star)).sum()
                + (P_hat_rb[-1] * disconnected_grad(dP_star_ric)).sum()
            )
            (
                dd_full,
                da0,
                dP0,
                dc_tr,
                dd_tr,
                dT_tr,
                dZ_tr,
                dR_tr,
                dH_tr,
                dQ_tr,
            ) = pt.grad(
                loss_tr,
                [data_, a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_],
                disconnected_inputs="ignore",
            )

            return [
                pt.concatenate([dd_full[:k_sym], dd_fx], axis=0),
                da0,
                dP0,
                dc_tr + dc_t,
                dd_tr + dd_t,
                dT_tr + dT_a + d_T_P,
                dZ_tr + dZ_dir + d_Z_chain + d_Z_P,
                dR_tr + d_R_P,
                dH_tr + d_H_chain + d_H_P,
                dQ_tr + d_Q_P,
            ]

        # inline=True folds the forward into the surrounding graph, but the custom pullback still
        # wins: pt.grad builds the gradient graph before the inline rewrite runs, so the analytic
        # backward is what gets differentiated.
        return OpFromGraph(
            inputs=inputs_s,
            outputs=outputs,
            pullback=pullback,
            inline=True,
            name="convergent_kf",
        )
