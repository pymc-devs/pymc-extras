from abc import ABC

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.pytensorf import constant_fold
from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert
from pytensor.tensor import TensorVariable
from pytensor.tensor.slinalg import solve_triangular

from pymc_extras.statespace.filters.utilities import (
    quad_form_sym,
    split_vars_into_seq_and_nonseq,
    stabilize,
)
from pymc_extras.statespace.utils.constants import JITTER_DEFAULT, MISSING_FILL

MVN_CONST = pt.log(2 * pt.constant(np.pi, dtype="float64"))
PARAM_NAMES = ["c", "d", "T", "Z", "R", "H", "Q"]

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
        for name in ["c", "d", "T", "Z", "R", "H", "Q"]:
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
        return_updates=False,
        missing_fill_value=None,
        cov_jitter=None,
    ) -> list[TensorVariable] | tuple[list[TensorVariable], dict]:
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

        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES
        )

        self.seq_names = seq_names
        self.non_seq_names = non_seq_names

        if len(sequences) > 0:
            sequences = self.add_check_on_time_varying_shapes(data, sequences)

        results, updates = pytensor.scan(
            self.kalman_step,
            sequences=[data, *sequences],
            outputs_info=[None, a0, None, None, P0, None, None],
            non_sequences=non_sequences,
            name="forward_kalman_pass",
            strict=False,
        )

        filter_results = self._postprocess_scan_results(results, a0, P0, n=data.type.shape[0])

        if return_updates:
            return filter_results, updates
        return filter_results

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
        filtered_states.name = "filtered_states"

        predicted_states = pt.specify_shape(predicted_states, (n, self.n_states))
        predicted_states.name = "predicted_states"

        observed_states = pt.specify_shape(observed_states, (n, self.n_endog))
        observed_states.name = "observed_states"

        filtered_covariances = pt.specify_shape(
            filtered_covariances, (n, self.n_states, self.n_states)
        )
        filtered_covariances.name = "filtered_covariances"

        predicted_covariances = pt.specify_shape(
            predicted_covariances, (n, self.n_states, self.n_states)
        )
        predicted_covariances.name = "predicted_covariances"

        observed_covariances = pt.specify_shape(
            observed_covariances, (n, self.n_endog, self.n_endog)
        )
        observed_covariances.name = "observed_covariances"

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
        self, y, Z, H
    ) -> tuple[TensorVariable, TensorVariable, TensorVariable, float]:
        """
        Handle missing values in the observation data `y`

        Adjusts the design matrix `Z` and the observation noise covariance matrix `H` by removing rows and/or columns
        associated with the data that is not observed at this iteration. Missing values are replaced with zeros to prevent
        propagating NaNs through the computation.

        Return a binary flag tensor `all_nan_flag`,indicating if all values in the observation data are missing. This
        flag is used for numerical adjustments in the update method.

        Parameters
        ----------
        y : TensorVariable
            The observation data at time t.
        Z : TensorVariable
            The design matrix.
        H : TensorVariable
            The observation noise covariance matrix.

        Returns
        -------
        y_masked : TensorVariable
            Observation vector with missing values replaced by zeros.

        Z_masked: TensorVariable
            Design matrix adjusted to exclude the missing states from the information set of observed variables in the
            update step

        H_masked: TensorVariable
            Noise covariance matrix, adjusted to exclude the missing states

        all_nan_flag: float
            1 if the entire state vector is missing

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        nan_mask = pt.or_(pt.isnan(y), pt.eq(y, self.missing_fill_value))
        all_nan_flag = pt.all(nan_mask).astype(pytensor.config.floatX)
        W = pt.diag(pt.bitwise_not(nan_mask).astype(pytensor.config.floatX))

        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        return y_masked, Z_masked, H_masked, all_nan_flag

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
        a_hat = T.dot(a) + c
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
        y_masked, Z_masked, H_masked, all_nan_flag = self.handle_missing_values(y, Z, H)

        a_filtered, P_filtered, obs_mu, obs_cov, ll = self.update(
            y=y_masked, a=a, d=d, P=P, Z=Z_masked, H=H_masked, all_nan_flag=all_nan_flag
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
        y_hat = d + Z.dot(a)
        v = y - y_hat

        PZT = P.dot(Z.T)
        F = Z.dot(PZT) + stabilize(H, self.cov_jitter)

        K = pt.linalg.solve(F.T, PZT.T, assume_a="pos", check_finite=False).T
        I_KZ = pt.eye(self.n_states) - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        F_inv_v = pt.linalg.solve(F, v, assume_a="pos", check_finite=False)
        inner_term = v.T @ F_inv_v

        F_logdet = pt.log(pt.linalg.det(F))

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

        M = pt.horizontal_stack(T @ P_chol, R @ Q_chol).T
        R_decomp = pt.linalg.qr(M, mode="r")
        P_chol_hat = R_decomp[: self.n_states, : self.n_states].T

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

        H_chol = pytensor.ifelse(pt.all(pt.eq(H, 0.0)), H, pt.linalg.cholesky(H, lower=True))

        # The following notation comes from https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A.pdf
        # Construct upper-triangular block matrix A = [[chol(H), Z @ L_pred],
        #                                              [0,           L_pred]]
        # The Schur decomposition of this matrix will be B (upper triangular). We are
        # more insterested in B^T:
        # Structure of B^T = [[chol(F),     0              ],
        #                    [K @ chol(F), chol(P_filtered)]
        zeros = pt.zeros((self.n_states, self.n_endog))
        upper = pt.horizontal_stack(H_chol, Z @ P_chol)
        lower = pt.horizontal_stack(zeros, P_chol)
        A_T = pt.vertical_stack(upper, lower)
        B = pt.linalg.qr(A_T.T, mode="r").T

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

        [a_filtered, P_chol_filtered, ll] = pytensor.ifelse(
            pt.eq(all_nan_flag, 1.0),
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

    def kalman_step(self, y, a, P, c, d, T, Z, R, H, Q):
        nan_mask = pt.isnan(y)

        W = pt.set_subtensor(pt.eye(y.shape[0])[nan_mask, nan_mask], 0.0)
        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        result, updates = pytensor.scan(
            self._univariate_inner_filter_step,
            sequences=[y_masked, Z_masked, d, pt.diag(H_masked), nan_mask],
            outputs_info=[a, P, None, None, None],
            name="univariate_inner_scan",
        )

        a_filtered, P_filtered, obs_mu, obs_cov, ll_inner = result
        a_filtered, P_filtered, obs_mu, obs_cov = (
            a_filtered[-1],
            P_filtered[-1],
            obs_mu[-1],
            obs_cov[-1],
        )

        P_filtered = stabilize(0.5 * (P_filtered + P_filtered.T), self.cov_jitter)
        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        ll = -0.5 * ((pt.neq(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum())

        return a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll
