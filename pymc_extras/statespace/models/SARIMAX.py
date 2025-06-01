from collections.abc import Sequence
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pytensor.compile.mode import Mode
from pytensor.tensor.slinalg import solve_discrete_lyapunov

from pymc_extras.statespace.core.statespace import PyMCStateSpace, floatX
from pymc_extras.statespace.models.utilities import (
    make_default_coords,
    make_harvey_state_names,
    make_SARIMA_transition_matrix,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    MA_PARAM_DIM,
    OBS_STATE_DIM,
    SARIMAX_STATE_STRUCTURES,
    SEASONAL_AR_PARAM_DIM,
    SEASONAL_MA_PARAM_DIM,
)


def _verify_order(p, d, q, P, D, Q, S):
    for name, terms in zip(["AR", "MA"], [(p, P), (q, Q)]):
        a, A = terms
        seasonal_lags = [(1 + i) * S for i in range(A)]
        lags = [(1 + i) for i in range(a)]
        overlapping_terms = set(seasonal_lags).intersection(set(lags))
        if any(overlapping_terms):
            raise ValueError(
                f"The following {name} and seasonal {name} terms overlap, check model "
                f"definition: {overlapping_terms}"
            )


class BayesianSARIMA(PyMCStateSpace):
    r"""
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors

    Parameters
    ----------
    order: tuple(int, int, int)
        Order of the ARIMA process. The order has the notation (p, d, q), where p is the number of autoregressive
        lags, q is the number of moving average components, and d is order of integration -- the number of
        differences needed to render the data stationary.

        If d > 0, the differences are modeled as components of the hidden state, and all available data can be used.
        This is only possible if state_structure = 'fast'. For interpretable states, the user must manually
        difference the data prior to calling the `build_statespace_graph` method.

    seasonal_order: tuple(int, int, int, int), optional
        Seasonal order of the SARIMA process. The order has the notation (P, D, Q, S), where P is the number of seasonal
        lags to include, Q is the number of seasonal innovation lags to include, and D is the number of seasonal
        differences to perform. S is the length of the season.

        Seasonal terms are similar to ARIMA terms, in that they are merely lags of the data or innovations. It is thus
        possible for the seasonal lags and the ARIMA lags to overlap, for example if P <= p. In this case, an error
        will be raised.

    stationary_initialization: bool, default False
        If true, the initial state and initial state covariance will not be assigned priors. Instead, their steady
        state values will be used.

        .. warning:: This option is very sensitive to the priors placed on the AR and MA parameters. If the model dynamics
                  for a given sample are not stationary, sampling will fail with a "covariance is not positive semi-definite"
                  error.

    filter_type: str, default "standard"
        The type of Kalman Filter to use. Options are "standard", "single", "univariate", "steady_state",
        and "cholesky". See the docs for kalman filters for more details.

    state_structure: str, default "fast"
        How to represent the state-space system. Currently, there are two choices: "fast" or "interpretable"

        - "fast" corresponds to the state space used by [2], and is called the "Harvey" representation in statsmodels.
           This is also the default representation used by statsmodels.tsa.statespace.SARIMAX. The states combine lags
           and innovations at different lags to compress the dimension of the state vector to max(p, 1+q). As a result,
           it is very preformat, but only the first state has a clear interpretation.

        - "interpretable" maximally expands the state vector, doing zero state compression. As a result, the state has
          dimension max(1, p) + max(1, q). What is gained by doing this is that every state has an obvious meaning, as
          either the data, an innovation, or a lag thereof.

    measurement_error: bool, default True
        If true, a measurement error term is added to the model.

    verbose: bool, default True
        If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.

    mode: str or Mode, optional
        Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
        ``forecast``. The mode does **not** effect calls to ``pm.sample``.

        Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
        to all sampling methods.

    Notes
    -----
        The ARIMAX model is a univariate time series model that posits the future evolution of a stationary time series will
    be a function of its past values, together with exogenous "innovations" and their past history. The model is
    described by its "order", a 3-tuple (p, d, q), that are:

        - p: The number of past time steps that directly influence the present value of the time series, called the
            "autoregressive", or AR, component
        - d: The "integration" order of the time series
        - q: The number of past exogenous innovations that directly influence the present value of the time series,
            called the "moving average", or MA, component

    Given this 3-tuple, the model can be written:

    .. math::
        (1- \phi_1 B - \cdots - \phi_p B^p) (1-B)^d y_{t} = c + (1 + \theta_1 B + \cdots + \theta_q B^q) \varepsilon_t

    Where B is the backshift operator, :math:`By_{t} = y_{t-1}`.

    The model assumes that the data are stationary; that is, that they can be described by a time-invariant Gaussian
    distribution with fixed mean and finite variance. Non-stationary data, those that grow over time, are not suitable
    for ARIMA modeling without preprocessing. Stationary can be induced in any time series by the sequential application
    of differences. Given a hypothetical non-stationary process:

    .. math::
        y_{t} = c + \rho y_{t-1} + \varepsilon_{t}

    The process:

    .. math::
        \Delta y_{t} = y_{t} - y_{t-1} = \rho \Delta y_{t-1} + \Delta \varepsilon_t

    is stationary, as the non-stationary component :math:`c` was eliminated by the operation of differencing. This
    process is said to be "integrated of order 1", as it requires 1 difference to render stationary. This is the
    function of the `d` parameter in the ARIMA order.

    Alternatively, the non-stationary components can be directly estimated. In this case, the errors of a preliminary
    regression are assumed to be ARIMA distributed, so that:

    .. math::
        \begin{align}
        y_{t} &= X\beta + \eta_t \\
        (1- \phi_1 B - \cdots - \phi_p B^p) (1-B)^d \eta_{t} &= (1 + \theta_1 B + \cdots + \theta_q B^q) \varepsilon_t
        \end{align}

    Where the design matrix `X` can include a constant, trends, or exogenous regressors.

    ARIMA models can be represented in statespace form, as described in [1]. For more details, see chapters 3.4, 3.6,
    and 8.4.

    Examples
    --------
    The following example shows how to build an ARMA(1, 1) model -- ARIMA(1, 0, 1) -- using the BayesianSARIMA class:

    .. code:: python

        import pymc_extras.statespace as pmss
        import pymc as pm

        ss_mod = pmss.BayesianSARIMA(order=(1, 0, 1), verbose=True)

        with pm.Model(coords=ss_mod.coords) as arma_model:
            state_sigmas = pm.HalfNormal("sigma_state", sigma=1.0, dims=ss_mod.param_dims["sigma_state"])

            rho = pm.Beta("ar_params", alpha=5, beta=1, dims=ss_mod.param_dims["ar_params"])
            theta = pm.Normal("ma_params", mu=0.0, sigma=0.5, dims=ss_mod.param_dims["ma_params"])

            ss_mod.build_statespace_graph(df)
            idata = pm.sample(nuts_sampler='numpyro')

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.

    .. [2] Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the
           Kalman Filter. Cambridge: Cambridge University Press.
    """

    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] | None = None,
        stationary_initialization: bool = True,
        filter_type: str = "standard",
        state_structure: str = "fast",
        measurement_error: bool = False,
        verbose=True,
        mode: str | Mode | None = None,
    ):
        """

        Parameters
        ----------
        order
        seasonal_order
        stationary_initialization
        filter_type
        state_structure
        measurement_error
        verbose
        mode
        """
        # Model order
        self.p, self.d, self.q = order
        if seasonal_order is None:
            seasonal_order = (0, 0, 0, 0)

        self.P, self.D, self.Q, self.S = seasonal_order
        _verify_order(self.p, self.d, self.q, self.P, self.D, self.Q, self.S)

        self.stationary_initialization = stationary_initialization

        self.state_structure = state_structure

        self._p_max = max(1, self.p + self.P * self.S)
        self._q_max = max(1, self.q + self.Q * self.S)

        k_states = None
        self._k_diffs = self.d + self.S * self.D

        if state_structure not in SARIMAX_STATE_STRUCTURES:
            raise ValueError(
                f"Got invalid argument {state_structure} for state structure, expected one of "
                f'{", ".join(SARIMAX_STATE_STRUCTURES)}'
            )

        if state_structure == "interpretable" and (self.d + self.D) > 0:
            raise ValueError(
                "Cannot use interpretable state structure with statespace differencing. Difference the "
                'data by hand (leaving NaN values to be interpolated), or use state_structure="fast"'
            )

        if self.state_structure == "fast":
            k_states = max(self.p + self.P * self.S, self.q + self.Q * self.S + 1) + (
                self.S * self.D + self.d
            )
        elif self.state_structure == "interpretable":
            k_states = self._p_max + self._q_max

        k_posdef = 1
        k_endog = 1

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type,
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
        )

    @property
    def param_names(self):
        names = [
            "x0",
            "P0",
            "ar_params",
            "ma_params",
            "seasonal_ar_params",
            "seasonal_ma_params",
            "sigma_state",
            "sigma_obs",
        ]
        if self.stationary_initialization:
            names.remove("P0")
            names.remove("x0")
        if self.p == 0:
            names.remove("ar_params")
        if self.P == 0:
            names.remove("seasonal_ar_params")
        if self.q == 0:
            names.remove("ma_params")
        if self.Q == 0:
            names.remove("seasonal_ma_params")
        if not self.measurement_error:
            names.remove("sigma_obs")

        return names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        info = {
            "x0": {
                "shape": (self.k_states,),
                "constraints": None,
            },
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
            },
            "sigma_obs": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "Positive",
            },
            "sigma_state": {
                "shape": None if self.k_posdef == 1 else (self.k_posdef,),
                "constraints": "Positive",
            },
            "ar_params": {
                "shape": (self.p,),
                "constraints": "None",
            },
            "ma_params": {
                "shape": (self.q,),
                "constraints": "None",
            },
            "seasonal_ar_params": {"shape": (self.P,), "constraints": "None"},
            "seasonal_ma_params": {"shape": (self.Q,), "constraints": "None"},
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        if self.state_structure == "fast":
            p, d, q = self.p, self.d, self.q
            P, D, Q, S = self.P, self.D, self.Q, self.S
            states = make_harvey_state_names(p, d, q, P, D, Q, S)

        elif self.state_structure == "interpretable":
            states = ["data"]
            if self.p > 0:
                states += [f"L{i + 1}.data" for i in range(self._p_max - 1)]
            states += ["innovations"]
            if self.q > 0:
                states += [f"L{i + 1}.innovations" for i in range(self._q_max - 1)]
        else:
            raise NotImplementedError()

        return states

    @property
    def observed_states(self):
        return [self.state_names[0]]

    @property
    def shock_names(self):
        return ["innovation"]

    @property
    def param_dims(self):
        coord_map = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (OBS_STATE_DIM,),
            "ar_params": (AR_PARAM_DIM,),
            "ma_params": (MA_PARAM_DIM,),
            "seasonal_ar_params": (SEASONAL_AR_PARAM_DIM,),
            "seasonal_ma_params": (SEASONAL_MA_PARAM_DIM,),
        }
        if self.k_endog == 1:
            coord_map["sigma_state"] = None
            coord_map["sigma_obs"] = None
        if not self.measurement_error:
            del coord_map["sigma_obs"]
        if self.p == 0:
            del coord_map["ar_params"]
        if self.q == 0:
            del coord_map["ma_params"]
        if self.P == 0:
            del coord_map["seasonal_ar_params"]
        if self.Q == 0:
            del coord_map["seasonal_ma_params"]
        if self.stationary_initialization:
            del coord_map["P0"]
            del coord_map["x0"]

        return coord_map

    @property
    def coords(self) -> dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.p > 0:
            coords.update({AR_PARAM_DIM: list(range(1, self.p + 1))})
        if self.q > 0:
            coords.update({MA_PARAM_DIM: list(range(1, self.q + 1))})
        if self.P > 0:
            coords.update({SEASONAL_AR_PARAM_DIM: list(range(1, self.P + 1))})
        if self.Q > 0:
            coords.update({SEASONAL_MA_PARAM_DIM: list(range(1, self.Q + 1))})

        return coords

    def _stationary_initialization(self):
        # Solve for matrix quadratic for P0
        T = self.ssm["transition"]
        R = self.ssm["selection"]
        Q = self.ssm["state_cov"]
        c = self.ssm["state_intercept"]

        x0 = pt.linalg.solve(pt.identity_like(T) - T, c, assume_a="gen", check_finite=True)
        P0 = solve_discrete_lyapunov(T, pt.linalg.matrix_dot(R, Q, R.T), method="bilinear")

        return x0, P0

    def make_symbolic_graph(self) -> None:
        p, d, q = self.p, self.d, self.q
        P, D, Q, S = self.P, self.D, self.Q, self.S

        # Initial state and covariance can be handled first if we're not doing a stationary initialization
        if not self.stationary_initialization:
            x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)
            P0 = self.make_and_register_variable(
                "P0", shape=(self.k_states, self.k_states), dtype=floatX
            )

            self.ssm["initial_state", :] = x0
            self.ssm["initial_state_cov"] = P0

        # Design matrix has no RVs
        k_lags = self.k_states - self._k_diffs
        self.ssm["design"] = np.r_[[1] * d, ([0] * (S - 1) + [1]) * D, [1], [0] * (k_lags - 1)][
            None
        ]

        # Set up the transition and selection matrices, depending on the requested representation
        if self.state_structure == "fast":
            transition = make_SARIMA_transition_matrix(p, d, q, P, D, Q, S)
            selection = np.r_[
                [0] * self._k_diffs, [1.0], np.zeros(self.k_states - self._k_diffs - 1)
            ][:, None]

            ar_param_idx = np.s_[
                "transition", self._k_diffs : self._k_diffs + self.p, self._k_diffs
            ]
            ma_param_idx = np.s_["selection", 1 + self._k_diffs : 1 + self._k_diffs + self.q, 0]

            self.ssm["transition"] = transition
            self.ssm["selection"] = selection

            if p > 0:
                ar_params = self.make_and_register_variable("ar_params", shape=(p,), dtype=floatX)
                self.ssm[ar_param_idx] = ar_params

            if P > 0:
                seasonal_ar_params = self.make_and_register_variable(
                    "seasonal_ar_params", shape=(P,), dtype=floatX
                )
                idx_rows = self._k_diffs + (np.arange(1, P + 1) * S) - 1
                S_ar_param_idx = np.s_["transition", idx_rows, self._k_diffs]
                self.ssm[S_ar_param_idx] = seasonal_ar_params

                if p > 0:
                    cross_term_idx = np.s_[
                        "transition",
                        idx_rows.repeat(p) + np.tile(np.arange(p), P) + 1,
                        self._k_diffs,
                    ]
                    self.ssm[cross_term_idx] = -pt.repeat(seasonal_ar_params, p) * pt.tile(
                        ar_params, P
                    )

            if q > 0:
                ma_params = self.make_and_register_variable("ma_params", shape=(q,), dtype=floatX)
                self.ssm[ma_param_idx] = ma_params

            if Q > 0:
                seasonal_ma_params = self.make_and_register_variable(
                    "seasonal_ma_params", shape=(Q,), dtype=floatX
                )
                idx_rows = self._k_diffs + np.arange(1, Q + 1) * S
                S_ma_param_idx = np.s_["selection", idx_rows, 0]
                self.ssm[S_ma_param_idx] = seasonal_ma_params

                if q > 0:
                    cross_term_idx = np.s_[
                        "selection", idx_rows.repeat(q) + np.tile(np.arange(q), Q) + 1, 0
                    ]
                    self.ssm[cross_term_idx] = pt.repeat(seasonal_ma_params, q) * pt.tile(
                        ma_params, Q
                    )

        elif self.state_structure == "interpretable":
            ar_param_idx = np.s_["transition", 0, : max(1, p)]
            ma_param_idx = np.s_["transition", 0, self._p_max : self._p_max + max(1, q)]

            transition = np.eye(self.k_states, k=-1)
            transition[-self._q_max, self._p_max - 1] = 0

            selection = np.r_[[1.0], np.zeros(self.k_states - 1)][:, None]
            selection[-self._q_max, 0] = 1

            self.ssm["transition"] = transition
            self.ssm["selection"] = selection

            if self.p > 0:
                ar_params = self.make_and_register_variable(
                    "ar_params", shape=(self.p,), dtype=floatX
                )
                self.ssm[ar_param_idx] = ar_params

            if self.P > 0:
                seasonal_ar_params = self.make_and_register_variable(
                    "seasonal_ar_params", shape=(P,), dtype=floatX
                )
                idx_cols = np.arange(1, P + 1) * S - 1
                S_ar_param_idx = np.s_["transition", 0, idx_cols]
                self.ssm[S_ar_param_idx] = seasonal_ar_params

                if p > 0:
                    cross_term_idx = np.s_[
                        "transition", 0, idx_cols.repeat(p) + np.tile(np.arange(p), P) + 1
                    ]
                    self.ssm[cross_term_idx] = -pt.repeat(seasonal_ar_params, p) * pt.tile(
                        ar_params, P
                    )

            if self.q > 0:
                ma_params = self.make_and_register_variable(
                    "ma_params", shape=(self.q,), dtype=floatX
                )
                self.ssm[ma_param_idx] = ma_params

            if Q > 0:
                seasonal_ma_params = self.make_and_register_variable(
                    "seasonal_ma_params", shape=(Q,), dtype=floatX
                )
                idx_cols = self._p_max + np.arange(1, Q + 1) * S - 1
                S_ma_param_idx = np.s_["transition", 0, idx_cols]
                self.ssm[S_ma_param_idx] = seasonal_ma_params

                if q > 0:
                    cross_term_idx = np.s_[
                        "transition", 0, idx_cols.repeat(q) + np.tile(np.arange(q), Q) + 1
                    ]
                    self.ssm[cross_term_idx] = pt.repeat(seasonal_ma_params, q) * pt.tile(
                        ma_params, Q
                    )

        # Set up the state covariance matrix
        state_cov_idx = ("state_cov", *np.diag_indices(self.k_posdef))
        state_cov = self.make_and_register_variable(
            "sigma_state", shape=() if self.k_posdef == 1 else (self.k_posdef,), dtype=floatX
        )
        self.ssm[state_cov_idx] = state_cov**2

        if self.measurement_error:
            obs_cov_idx = ("obs_cov", *np.diag_indices(self.k_endog))
            obs_cov = self.make_and_register_variable(
                "sigma_obs", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )
            self.ssm[obs_cov_idx] = obs_cov**2

        # The initial conditions have to be done last in the case of stationary initialization, because it will depend
        # on c, T, R and Q
        if self.stationary_initialization:
            x0, P0 = self._stationary_initialization()
            self.ssm["initial_state", :] = x0
            self.ssm["initial_state_cov", :, :] = P0
