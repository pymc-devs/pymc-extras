import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.model import modelcontext
from pytensor.compile.mode import Mode
from pytensor.tensor.linalg import solve_discrete_lyapunov

from pymc_extras.statespace.core.properties import (
    Coord,
    Data,
    Parameter,
    Shock,
    State,
)
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.filters.distributions import StationaryVAR
from pymc_extras.statespace.models.utilities import validate_names
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    EXOG_STATE_DIM,
    JITTER_DEFAULT,
    MA_PARAM_DIM,
    MISSING_FILL,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    OBSERVED_LIKELIHOOD_NAME,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    TIME_DIM,
    TREND_DIM,
)
from pymc_extras.statespace.utils.trend import (
    TrendSpec,
    constant_as_regressor,
    parse_trend,
    trend_design,
    trend_names,
)

floatX = pytensor.config.floatX


class BayesianVARMAX(PyMCStateSpace):
    r"""
    Vector AutoRegressive Moving Average with eXogenous Regressors

    The VARMA model is a multivariate extension of the SARIMAX model. Given a set of timeseries :math:`\{x_t\}_{t=0}^T`,
    with :math:`x_t = \begin{bmatrix} x_{1,t} & x_{2,t} & \cdots & x_{k,t} \end{bmatrix}^T`, a VARMA models each series
    as a function of the histories of all series. Specifically, denoting the AR-MA order as (p, q),  a VARMA can be
    written:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + \cdots + A_p x_{t-p} + B_1 \varepsilon_{t-1} + \cdots
            + B_q \varepsilon_{t-q} + \beta z_t + \varepsilon_t

    Where :math:`\varepsilon_t = \begin{bmatrix} \varepsilon_{1,t} & \varepsilon_{2,t} & \cdots &
    \varepsilon_{k,t}\end{bmatrix}^T \sim N(0, \Sigma)` is a vector of i.i.d stochastic innovations or shocks that drive
    intertemporal variation in the data, and :math:`z_t` holds any exogenous regressors. Their coefficients
    :math:`\beta` measure the impact effect of a regressor, which the autoregressive dynamics then propagate, as in
    Lutkepohl's VARX and in statsmodels' ``VARMAX``. Matrices :math:`A_i, B_i` are :math:`k \times k` coefficient
    matrices:

    .. math::
        A_i = \begin{bmatrix} \rho_{1,i,1} & \rho_{1,i,2} & \cdots & \rho_{1,i,k} \\
                              \rho_{2,i,1} & \rho_{2,i,2} & \cdots & \rho_{2,i,k} \\
                              \vdots     &  \vdots    & \cdots & \vdots     \\
                              \rho{k,i,1}  & \rho_{k,i,2} & \cdots & rho_{k,i,k}  \end{bmatrix}

    Internally, this representation is not used. Instead, the vectors :math:`x_t, x_{t-1}, \cdots, x_{t-p},
    \varepsilon_{t-1}, \cdots, \varepsilon_{t-q}` are concatenated into a single column vector of length ``k * (p+q)``,
    while the coefficients matrices are likewise concatenated into a single coefficient matrix, :math:`T`.

    As the dimensionality of the VARMA system increases -- either because there are a large number of timeseries
    included in the analysis, or because the order is large -- the probability of sampling a stationary matrix :math:`T`
    goes to zero. This has two implications for applied work. First, a non-stationary system will exhibit explosive
    behavior, potentially rending impulse response functions and long-term forecasts useless. Secondly, it is not
    possible to do stationary initialization. Stationary initialization significantly speeds up sampling, and should be
    preferred when possible.

    Examples
    --------
    The following code snippet estimates a VARMA(1, 1):

    .. code:: python

        import pymc_extras.statespace as pmss
        import pymc as pm

        # Create VAR Statespace Model
        bvar_mod = pmss.BayesianVARMAX(endog_names=data.columns, order=(2, 0),
                                       stationary_initialization=False, measurement_error=False,
                                       filter_type="standard", verbose=True)

        # Unpack dims and coords
        x0_dims, P0_dims, state_cov_dims, ar_dims = bvar_mod.param_dims.values()
        coords = bvar_mod.coords

        # Estimate PyMC model
        with pm.Model(coords=coords) as var_mod:
            x0 = pm.Normal("x0", dims=x0_dims)
            P0_diag = pm.Gamma("P0_diag", alpha=2, beta=1, size=data.shape[1] * 2, dims=P0_dims[0])
            P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=P0_dims)

            state_chol, _, _ = pm.LKJCholeskyCov(
                "state_chol", eta=1, n=bvar_mod.k_posdef, sd_dist=pm.Exponential.dist(lam=1)
            )

            ar_params = pm.Normal("ar_params", mu=0, sigma=1, dims=ar_dims)
            state_cov = pm.Deterministic("state_cov", state_chol @ state_chol.T, dims=state_cov_dims)

            bvar_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler="numpyro")
    """

    def __init__(
        self,
        order: tuple[int, int],
        endog_names: list[str] | None = None,
        exog_state_names: list[str] | dict[str, list[str]] | None = None,
        trend: TrendSpec = None,
        trend_offset: int = 1,
        exog_in_observation: bool = False,
        stationary_initialization: bool = False,
        filter_type: str = "standard",
        smoother_type: str = "disturbance",
        measurement_error: bool = False,
        verbose: bool = True,
        mode: str | Mode | None = None,
        cov_jitter: float = JITTER_DEFAULT,
        missing_fill_value: float = MISSING_FILL,
    ):
        """
        Create a Bayesian VARMAX model.

        Parameters
        ----------
        order: tuple of (int, int)
            Number of autoregressive (AR) and moving average (MA) terms to include in the model. All terms up to the
            specified order are included. For restricted models, set zeros directly on the priors.

        endog_names: list of str, optional
            Names of the endogenous variables being modeled. Used to generate names for the state and shock coords.

        exog_state_names : list[str] or dict[str, list[str]], optional
            Names of the exogenous state variables. If a list, all endogenous variables will share the same exogenous
            variables. If a dict, keys should be the names of the endogenous variables, and values should be lists of the
            exogenous variable names for that endogenous variable. Endogenous variables not included in the dict will
            be assumed to have no exogenous variables. If None, no exogenous variables will be included.

        trend : {None, "n", "c", "ct", "ctt"} or sequence of int, optional
            Deterministic polynomial trend :math:`A(t)` added to the VAR equation, as in statsmodels. ``"c"`` is a
            constant, ``"ct"`` adds a linear term, ``"ctt"`` a quadratic. A sequence of ints gives presence flags per
            power, so ``[1, 1, 0, 1]`` includes :math:`1, t, t^3`. Each endogenous series gets its own coefficients,
            registered as ``trend_params``. Default None, no trend.

        trend_offset : int, default 1
            Value of :math:`t` at the first observation, so the linear term runs ``1, 2, ..., n``.
        exog_in_observation: bool, default False
            Where the exogenous regressors enter. By default, they enter the VAR equation,
            :math:`x_t = A_1 x_{t-1} + \\cdots + \\beta z_t + \\varepsilon_t`, so :math:`\\beta` is an impact
            effect that the dynamics propagate, as in statsmodels' ``VARMAX``. If True, they shift the level of the
            observations instead, :math:`y_t = \\beta z_t + x_t` with :math:`x_t` the VAR, so :math:`\\beta` is a
            long-run effect, as in regression with ARMA errors.

        stationary_initialization: bool, default False
            If true, the initial state and initial state covariance will not be assigned priors. Instead, their steady
            state values will be used. If False, the user is responsible for setting priors on the initial state and
            initial covariance.

            ..warning :: This option is very sensitive to the priors placed on the AR and MA parameters. If the model dynamics
                      for a given sample are not stationary, sampling will fail with a "covariance is not positive semi-definite"
                      error.

        filter_type: str, default "standard"
            The type of Kalman Filter to use. Options are "standard", "single", "univariate", "steady_state",
            and "cholesky". See the docs for kalman filters for more details.

        measurement_error: bool, default True
            If true, a measurement error term is added to the model.

        verbose: bool, default True
            If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.

        mode: str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
            ``forecast``. The mode does **not** effect calls to ``pm.sample``.

            Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
            to all sampling methods.


        cov_jitter: float, optional
            Jitter added to the diagonal of every covariance matrix at each filtering step, for numerical
            stability. Post-estimation graphs are built with this same value. Default 1e-8, or 1e-6 if
            ``pytensor.config.floatX`` is float32.

        missing_fill_value: float, optional
            Sentinel used to mask missing observations. Set this only if your data legitimately contains the
            default sentinel. Post-estimation graphs are built with this same value. Default -9999.0.
        """

        validate_names(endog_names, var_name="endog_names", optional=False)
        k_endog = len(endog_names)

        needs_exog_data = False

        if exog_state_names is not None and not isinstance(exog_state_names, list | dict):
            raise ValueError("If not None, exog_state_names must be either a list or a dict")

        if exog_state_names is not None:
            if isinstance(exog_state_names, list):
                k_exog = len(exog_state_names)
            elif isinstance(exog_state_names, dict):
                k_exog = {name: len(names) for name, names in exog_state_names.items()}
            needs_exog_data = True
        else:
            k_exog = None

        # If exog_state_names is a dict but 1) all endog variables are among the keys, and 2) all values are the same
        # then we can drop back to the list case.
        if (
            isinstance(exog_state_names, dict)
            and set(exog_state_names.keys()) == set(endog_names)
            and len({frozenset(val) for val in exog_state_names.values()}) == 1
        ):
            exog_state_names = exog_state_names[endog_names[0]]
            k_exog = len(exog_state_names)

        self.endog_names = list(endog_names)
        self.exog_state_names = exog_state_names
        self.trend_powers = parse_trend(trend)
        self.trend_offset = trend_offset
        self.exog_in_observation = exog_in_observation

        self.k_exog = k_exog
        self.p, self.q = order
        self.stationary_initialization = stationary_initialization

        k_order = max(self.p, 1) + self.q
        k_states = int(k_endog * k_order)
        k_posdef = k_endog

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type,
            smoother_type=smoother_type,
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
            cov_jitter=cov_jitter,
            missing_fill_value=missing_fill_value,
        )

        self._needs_exog_data = needs_exog_data

        # Save counts of the number of parameters in each category
        self.param_counts = {
            "x0": k_states * (1 - self.stationary_initialization),
            "P0": k_states**2 * (1 - self.stationary_initialization),
            "trend": k_endog * len(self.trend_powers),
            "AR": k_endog**2 * self.p,
            "MA": k_endog**2 * self.q,
            "state_cov": k_posdef**2,
            "sigma_obs": k_endog * self.measurement_error,
        }

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        k_endog = self.k_endog
        k_states = self.k_states
        k_posdef = self.k_posdef

        parameters = []

        if not self.stationary_initialization:
            parameters.append(
                Parameter(
                    name="x0",
                    shape=(k_states,),
                    dims=(ALL_STATE_DIM,),
                    constraints=None,
                )
            )
            parameters.append(
                Parameter(
                    name="P0",
                    shape=(k_states, k_states),
                    dims=(ALL_STATE_DIM, ALL_STATE_AUX_DIM),
                    constraints="Positive Semi-definite",
                )
            )

        if self.trend_powers:
            parameters.append(
                Parameter(
                    name="trend_params",
                    shape=(k_endog, len(self.trend_powers)),
                    dims=(OBS_STATE_DIM, TREND_DIM),
                    constraints=None,
                )
            )

        if self.p > 0:
            parameters.append(
                Parameter(
                    name="ar_params",
                    shape=(k_endog, self.p, k_endog),
                    dims=(OBS_STATE_DIM, AR_PARAM_DIM, OBS_STATE_AUX_DIM),
                    constraints=None,
                )
            )

        if self.q > 0:
            parameters.append(
                Parameter(
                    name="ma_params",
                    shape=(k_endog, self.q, k_endog),
                    dims=(OBS_STATE_DIM, MA_PARAM_DIM, OBS_STATE_AUX_DIM),
                    constraints=None,
                )
            )

        parameters.append(
            Parameter(
                name="state_cov",
                shape=(k_posdef, k_posdef),
                dims=(SHOCK_DIM, SHOCK_AUX_DIM),
                constraints="Positive Semi-definite",
            )
        )

        if self.measurement_error:
            parameters.append(
                Parameter(
                    name="sigma_obs",
                    shape=(k_endog,),
                    dims=(OBS_STATE_DIM,),
                    constraints="Positive",
                )
            )

        # Handle exogenous parameters
        if isinstance(self.exog_state_names, list):
            k_exog = len(self.exog_state_names)
            parameters.append(
                Parameter(
                    name="beta_exog",
                    shape=(k_endog, k_exog),
                    dims=(OBS_STATE_DIM, EXOG_STATE_DIM),
                    constraints=None,
                )
            )
        elif isinstance(self.exog_state_names, dict):
            for name, exog_names in self.exog_state_names.items():
                k_exog = len(exog_names)
                parameters.append(
                    Parameter(
                        name=f"beta_{name}",
                        shape=(k_exog,),
                        dims=(f"{EXOG_STATE_DIM}_{name}",),
                        constraints=None,
                    )
                )

        return tuple(parameters)

    def set_states(self) -> State | tuple[State, ...] | None:
        state_names = self.endog_names.copy()
        state_names += [
            f"L{i + 1}_{state}" for i in range(self.p - 1) for state in self.endog_names
        ]
        state_names += [
            f"L{i + 1}_{state}_innov" for i in range(self.q) for state in self.endog_names
        ]

        hidden_states = [State(name=name, observed=False) for name in state_names]

        # The first k_endog states are observed
        observed_states = [State(name=name, observed=True) for name in self.endog_names]

        return *hidden_states, *observed_states

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        return tuple(Shock(name=name) for name in self.endog_names)

    def set_data_info(self) -> tuple[Data, ...] | None:
        data = []

        if isinstance(self.exog_state_names, list):
            k_exog = len(self.exog_state_names)
            data.append(
                Data(
                    name="exogenous_data",
                    shape=(None, k_exog),
                    dims=(TIME_DIM, EXOG_STATE_DIM),
                    is_exogenous=True,
                )
            )
        elif isinstance(self.exog_state_names, dict):
            for endog_state, exog_names in self.exog_state_names.items():
                k_exog = len(exog_names)
                data.append(
                    Data(
                        name=f"{endog_state}_exogenous_data",
                        shape=(None, k_exog),
                        dims=(TIME_DIM, f"{EXOG_STATE_DIM}_{endog_state}"),
                        is_exogenous=True,
                    )
                )

        return tuple(data)

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        coords = list(self.default_coords())

        if self.trend_powers:
            coords.append(Coord(dimension=TREND_DIM, labels=trend_names(self.trend_powers)))

        # AR/MA param coords
        if self.p > 0:
            coords.append(Coord(dimension=AR_PARAM_DIM, labels=tuple(range(1, self.p + 1))))

        if self.q > 0:
            coords.append(Coord(dimension=MA_PARAM_DIM, labels=tuple(range(1, self.q + 1))))

        # Exogenous coords
        if isinstance(self.exog_state_names, list):
            coords.append(Coord(dimension=EXOG_STATE_DIM, labels=tuple(self.exog_state_names)))
        elif isinstance(self.exog_state_names, dict):
            for name, exog_names in self.exog_state_names.items():
                coords.append(Coord(dimension=f"{EXOG_STATE_DIM}_{name}", labels=tuple(exog_names)))

        return tuple(coords)

    @property
    def default_priors(self):
        raise NotImplementedError

    def add_default_priors(self):
        raise NotImplementedError

    def make_likelihood(self, data, matrices, dims, missing):
        """
        Register a :class:`~pymc_extras.statespace.filters.distributions.StationaryVAR`.

        The closed form covers an autoregression with no moving average terms and no measurement
        error, initialized at its stationary distribution, with at most a constant trend. Anything
        else is filtered, including data with missing values, which only the filter marginalizes.
        """
        if self.q > 0 or self.p == 0 or self.measurement_error:
            return super().make_likelihood(data, matrices, dims, missing)
        if not self.stationary_initialization or self.trend_powers not in ((), (0,)):
            return super().make_likelihood(data, matrices, dims, missing)
        if missing.any():
            return super().make_likelihood(data, matrices, dims, missing)

        k_endog = self.k_endog
        *_, transition, _, _, _, state_cov = matrices
        coefficients = transition[:k_endog, : k_endog * self.p]
        exog, exog_coefficients = self._exogenous_regression()

        if self.trend_powers == (0,):
            constant = modelcontext(None)["trend_params"][:, 0]
            level = (
                self._constant_mean(coefficients, constant)
                if self.exog_in_observation
                else constant
            )
            exog, exog_coefficients = constant_as_regressor(
                level, exog, exog_coefficients, data.shape[0]
            )

        return StationaryVAR(
            OBSERVED_LIKELIHOOD_NAME,
            coefficients,
            state_cov,
            data,
            exog=exog,
            exog_coefficients=exog_coefficients,
            exog_in_observation=self.exog_in_observation,
            observed=data,
            dims=dims,
        )

    def _constant_mean(self, coefficients, constant):
        r"""The level :math:`(I - \sum_l A_l)^{-1} c` a constant intercept :math:`c` implies."""
        lag_sum = coefficients.reshape((self.k_endog, self.p, self.k_endog)).sum(axis=1)

        return pt.linalg.solve(pt.eye(self.k_endog) - lag_sum, constant, b_ndim=1)

    @staticmethod
    def _shift_exog_effect(exog_effect):
        """
        Shift ``B x_t`` back by one step for the state intercept.

        The filter applies the intercept at step ``t`` to the transition into ``t + 1``, so the
        row that reaches ``y_t`` must be ``B x_t`` shifted back by one. The final row has no
        successor in the data and repeats the last regressors, which keeps the one-step-ahead
        state prediction finite without asserting anything about the next period.
        """
        return pt.concatenate([exog_effect[1:], exog_effect[-1:]], axis=0)

    def _exogenous_regression(self):
        """
        Return exogenous data and coefficients whose product is the exogenous term.

        Per-series regressors are laid end to end, against a block coefficient matrix whose
        off-blocks are structurally zero.
        """
        if self.exog_state_names is None:
            return None, None

        pymc_model = modelcontext(None)

        if isinstance(self.exog_state_names, list):
            return pymc_model["exogenous_data"], pymc_model["beta_exog"]

        widths = [len(self.exog_state_names.get(name, ())) for name in self.endog_names]
        coefficients = pt.zeros((self.k_endog, sum(widths)), dtype=floatX)
        data, offset = [], 0

        for row, (name, width) in enumerate(zip(self.endog_names, widths)):
            if width == 0:
                continue
            data.append(pymc_model[f"{name}_exogenous_data"])
            coefficients = pt.set_subtensor(
                coefficients[row, offset : offset + width], pymc_model[f"beta_{name}"]
            )
            offset += width

        return pt.concatenate(data, axis=1), coefficients

    def _trend_effect(self):
        r"""
        The polynomial trend's contribution to the first ``k_endog`` rows of the state intercept.

        The filter applies the intercept of step ``t`` to the transition into ``t + 1``, so the
        design starts one period past ``trend_offset`` for the term to reach ``y_t`` at
        :math:`A(t + \text{offset})`. A constant returns a static vector.
        """
        trend_params = self.make_and_register_variable(
            "trend_params", shape=(self.k_endog, len(self.trend_powers)), dtype=floatX
        )
        if self.trend_powers == (0,):
            return trend_params[:, 0]

        design = trend_design(self.trend_powers, self.n_timesteps, self.trend_offset + 1)

        return design @ trend_params.T

    def make_symbolic_graph(self) -> None:
        # Initialize the matrices
        if not self.stationary_initialization:
            # initial states
            x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)
            self.ssm["initial_state"] = x0

            # initial covariance
            P0 = self.make_and_register_variable(
                "P0", shape=(self.k_states, self.k_states), dtype=floatX
            )
            self.ssm["initial_state_cov"] = P0

        # Design matrix is a truncated identity (first k_obs states observed). Written as an
        # indexed update rather than a whole matrix because of #736.
        self.ssm[("design", *np.diag_indices(self.k_endog))] = 1

        # Transition matrix has 4 blocks:
        # Upper left: AR coefs (k_obs, k_obs * min(p, 1))
        # Upper right: MA coefs (k_obs, k_obs * q)
        # Lower left: Truncated identity (k_obs * min(p, 1), k_obs * min(p, 1))
        # Lower right: Shifted identity (k_obs * p, k_obs * q)
        transition = np.zeros((self.k_states, self.k_states))
        if self.p > 1:
            transition[self.k_endog : self.k_endog * self.p, : self.k_endog * (self.p - 1)] = (
                np.eye(self.k_endog * (self.p - 1))
            )

        if self.q > 1:
            transition[-self.k_endog * (self.q - 1) :, -self.k_endog * self.q : -self.k_endog] = (
                np.eye(self.k_endog * (self.q - 1))
            )

        transition = pt.as_tensor_variable(transition)

        if self.p > 0:
            # Register the AR parameter matrix as a (k, p, k), then reshape it and allocate it in the transition matrix
            # This way the user can use 3 dimensions in the prior (clearer?)
            ar_params = self.make_and_register_variable(
                "ar_params", shape=(self.k_endog, self.p, self.k_endog), dtype=floatX
            )

            ar_params = ar_params.reshape((self.k_endog, self.k_endog * self.p))
            transition = pt.set_subtensor(
                transition[: self.k_endog, : self.k_endog * self.p], ar_params
            )

        if self.q > 0:
            # Same as above, register with 3 dimensions then reshape
            ma_params = self.make_and_register_variable(
                "ma_params", shape=(self.k_endog, self.q, self.k_endog), dtype=floatX
            )

            ma_params = ma_params.reshape((self.k_endog, self.k_endog * self.q))
            transition = pt.set_subtensor(
                transition[: self.k_endog, self.k_endog * max(1, self.p) :], ma_params
            )

        self.ssm["transition"] = transition

        # The selection matrix is (k_states, k_obs), with two (k_obs, k_obs) identity
        # matrix blocks inside. One is always on top, the other starts after (k_obs * p) rows
        selection = np.zeros((self.k_states, self.k_endog))
        selection[: self.k_endog, :] = np.eye(self.k_endog)
        if self.q > 0:
            end = -self.k_endog * (self.q - 1) if self.q > 1 else None
            selection[-self.k_endog * self.q : end, :] = np.eye(self.k_endog)
        self.ssm["selection"] = selection

        if self.measurement_error:
            sigma_obs = self.make_and_register_variable(
                "sigma_obs", shape=(self.k_endog,), dtype=floatX
            )
            self.ssm["obs_cov"] = pt.diag(sigma_obs**2)

        state_cov = self.make_and_register_variable(
            "state_cov", shape=(self.k_posdef, self.k_posdef), dtype=floatX
        )
        self.ssm["state_cov"] = state_cov

        state_intercept_terms = []

        if self.exog_state_names is not None:
            if isinstance(self.exog_state_names, list):
                beta_exog = self.make_and_register_variable(
                    "beta_exog", shape=(self.k_posdef, self.k_exog), dtype=floatX
                )
                exog_data = self.make_and_register_data(
                    "exogenous_data", shape=(None, self.k_exog), dtype=floatX
                )

                exog_effect = exog_data @ beta_exog.T

            elif isinstance(self.exog_state_names, dict):
                exog_components = []
                for i, name in enumerate(self.endog_names):
                    if name in self.exog_state_names:
                        k_exog = len(self.exog_state_names[name])
                        beta_exog = self.make_and_register_variable(
                            f"beta_{name}", shape=(k_exog,), dtype=floatX
                        )
                        exog_data = self.make_and_register_data(
                            f"{name}_exogenous_data", shape=(None, k_exog), dtype=floatX
                        )
                        exog_components.append(pt.expand_dims(exog_data @ beta_exog, axis=-1))
                    else:
                        exog_components.append(pt.zeros((1, 1), dtype=floatX))

                # TODO: Replace all of this with pt.concat_with_broadcast once PyMC works with pytensor >= 2.32

                # If there were any zeros, they need to be broadcast against the non-zeros.
                # Core shape is the last dim, the time dim is always broadcast
                non_concat_shape = [1, None]

                # Look for the first non-zero component to get the shape from
                for tensor_inp in exog_components:
                    for i, (bcast, sh) in enumerate(
                        zip(tensor_inp.type.broadcastable, tensor_inp.shape)
                    ):
                        if bcast or i == 1:
                            continue
                        non_concat_shape[i] = sh

                assert non_concat_shape.count(None) == 1

                bcast_tensor_inputs = []
                for tensor_inp in exog_components:
                    non_concat_shape[1] = tensor_inp.shape[1]
                    bcast_tensor_inputs.append(pt.broadcast_to(tensor_inp, non_concat_shape))

                exog_effect = pt.join(1, *bcast_tensor_inputs)

            else:
                raise NotImplementedError()

            if self.exog_in_observation:
                self.ssm["obs_intercept"] = exog_effect
                self.ssm.declare_time_varying("obs_intercept")
            else:
                state_intercept_terms.append(self._shift_exog_effect(exog_effect))

        if self.trend_powers:
            state_intercept_terms.append(self._trend_effect())

        if state_intercept_terms:
            effect = sum(state_intercept_terms)
            padding = [(0, self.k_states - self.k_endog)]
            if effect.ndim == 2:
                padding = [(0, 0), *padding]
                self.ssm.declare_time_varying("state_intercept")
            self.ssm["state_intercept"] = pt.pad(effect, padding)

        if self.stationary_initialization:
            # Solve for matrix quadratic for P0
            T = self.ssm["transition"]
            R = self.ssm["selection"]
            Q = self.ssm["state_cov"]
            c = self.ssm["state_intercept"]
            if "state_intercept" in self.ssm.time_varying_names:
                c = c[0]

            x0 = pt.linalg.solve(pt.eye(T.shape[0]) - T, c, assume_a="gen", check_finite=False)
            P0 = solve_discrete_lyapunov(
                T,
                pt.linalg.matrix_dot(R, Q, R.T),
                method="direct" if self.k_states < 10 else "bilinear",
            )
            self.ssm["initial_state"] = x0
            self.ssm["initial_state_cov"] = P0
