from collections.abc import Sequence
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pytensor import graph_replace
from pytensor.compile.mode import Mode
from pytensor.tensor.slinalg import solve_discrete_lyapunov

from pymc_extras.statespace.core.statespace import PyMCStateSpace, floatX
from pymc_extras.statespace.models.utilities import make_default_coords
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    ETS_SEASONAL_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
)


class BayesianETS(PyMCStateSpace):
    def __init__(
        self,
        order: tuple[str, str, str] | None = None,
        endog_names: str | list[str] | None = None,
        k_endog: int = 1,
        trend: bool = True,
        damped_trend: bool = False,
        seasonal: bool = False,
        seasonal_periods: int | None = None,
        measurement_error: bool = False,
        use_transformed_parameterization: bool = False,
        dense_innovation_covariance: bool = False,
        stationary_initialization: bool = False,
        initialization_dampening: float = 0.8,
        filter_type: str = "standard",
        verbose: bool = True,
        mode: str | Mode | None = None,
    ):
        r"""
        Exponential Smoothing State Space Model

        This class can represent a subset of exponential smoothing state space models, specifically those with additive
        errors. Following .. [1], The general form of the model is:

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
            \epsilon_t &\sim N(0, \sigma)
            \end{align}

        where :math:`l_t` is the level component, :math:`b_t` is the trend component, and :math:`s_t` is the seasonal
        component. These components can be included or excluded, leading to different model specifications. The following
        models are possible:

        * `ETS(A,N,N)`: Simple exponential smoothing

            .. math::

                \begin{align}
                y_t &= l_{t-1} + \epsilon_t \\
                l_t &= l_{t-1} + \alpha \epsilon_t
                \end{align}

        Where :math:`\alpha \in [0, 1]` is a mixing parameter between past observations and current innovations.
        These equations arise by starting from the "component form":

            .. math::

                \begin{align}
                \hat{y}_{t+1 | t} &= l_t \\
                l_t &= \alpha y_t + (1 - \alpha) l_{t-1} \\
                &= l_{t-1} + \alpha (y_t - l_{t-1})
                &= l_{t-1} + \alpha \epsilon_t
                \end{align}

        Where $\epsilon_t$ are the forecast errors, assumed to be IID mean zero and normally distributed. The role of
        :math:`\alpha` is clearest in the second line. The level of the time series at each time is a mixture of
        :math:`\alpha` percent of the incoming data, and :math:`1 - \alpha` percent of the previous level. Recursive
        substitution reveals that the level is a weighted composite of all previous observations; thus the name
        "Exponential Smoothing".

        Additional supposed specifications include:

        * `ETS(A,A,N)`: Holt's linear trend method

            .. math::

                \begin{align}
                y_t &= l_{t-1} + b_{t-1} + \epsilon_t \\
                l_t &= l_{t-1} + b_{t-1} + \alpha \epsilon_t \\
                b_t &= b_{t-1} + \alpha \beta^\star \epsilon_t
                \end{align}

            [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^\star`.

        * `ETS(A,N,A)`: Additive seasonal method

            .. math::

                \begin{align}
                y_t &= l_{t-1} + s_{t-m} + \epsilon_t \\
                l_t &= l_{t-1} + \alpha \epsilon_t \\
                s_t &= s_{t-m} + (1 - \alpha)\gamma^\star \epsilon_t
                \end{align}

            [1]_ also consider an alternative parameterization with :math:`\gamma = (1 - \alpha) \gamma^\star`.

        * `ETS(A,A,A)`: Additive Holt-Winters method

            .. math::

                \begin{align}
                y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
                l_t &= l_{t-1} + \alpha \epsilon_t \\
                b_t &= b_{t-1} + \alpha \beta^\star \epsilon_t \\
                s_t &= s_{t-m} + (1 - \alpha) \gamma^\star \epsilon_t
                \end{align}

            [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^star` and
            :math:`\gamma = (1 - \alpha) \gamma^\star`.

        * `ETS(A, Ad, N)`: Dampened trend method

            .. math::

                \begin{align}
                y_t &= l_{t-1} + b_{t-1} + \epsilon_t \\
                l_t &= l_{t-1} + \alpha \epsilon_t \\
                b_t &= \phi b_{t-1} + \alpha \beta^\star \epsilon_t
                \end{align}

            [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^\star`.

        * `ETS(A, Ad, A)`: Dampened trend with seasonal method

            .. math::

                \begin{align}
                y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
                l_t &= l_{t-1} + \alpha \epsilon_t \\
                b_t &= \phi b_{t-1} + \alpha \beta^\star \epsilon_t \\
                s_t &= s_{t-m} + (1 - \alpha) \gamma^\star \epsilon_t
                \end{align}

            [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^star` and
            :math:`\gamma = (1 - \alpha) \gamma^\star`.


        Parameters
        ----------
        order: tuple of string, Optional
            The exponential smoothing "order". This is a tuple of three strings, each of which should be one of 'A', 'Ad',
            or 'N'.
            If provided, the model will be initialized from the given order, and the `trend`, `damped_trend`, and `seasonal`
            arguments will be ignored.
        endog_names: str or list of str, Optional
            Names associated with observed states. If a list, the length should be equal to the number of time series
            to be estimated.
        k_endog: int, Optional
            Number of time series to estimate. If endog_names are provided, this is ignored and len(endog_names) is
            used instead.
        trend: bool
            Whether to include a trend component. Setting ``trend=True`` is equivalent to ``order[1] == 'A'``.
        damped_trend: bool
            Whether to include a damping parameter on the trend component. Ignored if `trend` is `False`. Setting
            ``trend=True`` and ``damped_trend=True`` is equivalent to order[1] == 'Ad'.
        seasonal: bool
            Whether to include a seasonal component. Setting ``seasonal=True`` is equivalent to ``order[2] = 'A'``.
        seasonal_periods: int
            The number of periods in a complete seasonal cycle. Ignored if `seasonal` is `False`
            (or if ``order[2] == "N"``)
        measurement_error: bool
            Whether to include a measurement error term in the model. Default is `False`.
        use_transformed_parameterization: bool, default False
            If true, use the :math:`\alpha, \beta, \gamma` parameterization, otherwise use the :math:`\alpha, \beta^\star,
            \gamma^\star` parameterization. This will change the admissible region for the priors.

            - Under the **non-transformed** parameterization, all of :math:`\alpha, \beta^\star, \gamma^\star` should be
              between 0 and 1.
            - Under the **transformed**  parameterization, :math:`\alpha \in (0, 1)`, :math:`\beta \in (0, \alpha)`, and
              :math:`\gamma \in (0, 1 - \alpha)`

            The :meth:`param_info` method will change to reflect the suggested intervals based on the value of this
            argument.
        dense_innovation_covariance: bool, default False
            Whether to estimate a dense covariance for statespace innovations. In an ETS models, each observed variable
            has a single source of stochastic variation. If True, these innovations are allowed to be correlated.
            Ignored if ``k_endog == 1``
        stationary_initialization: bool, default False
            If True, the Kalman Filter's initial covariance matrix will be set to an approximate steady-state value.
            The approximation is formed by adding a small dampening factor to each state. Specifically, the level state
            for a ('A', 'N', 'N') model is written:

            .. math::
                \ell_t = \ell_{t-1} + \alpha * e_t

            That this system is not stationary can be understood in ARIMA terms: the level is a random walk; that is,
            :math:`rho = 1`. This can be remedied by pretending that we instead have a dampened system:

            .. math::
                \ell_t = \rho \ell_{t-1} + \alpha * e_t

            With :math:`\rho \approx 1`, the system is stationary, and we can solve for the steady-state covariance
            matrix. This is then used as the initial covariance matrix for the Kalman Filter. This is a heuristic
            method that helps avoid setting a prior on the initial covariance matrix.
        initialization_dampening: float, default 0.8
            Dampening factor to add to non-stationary model components. This is only used for initialization, it does
            *not* add dampening to the model. Ignored if `stationary_initialization` is `False`.
        filter_type: str, default "standard"
            The type of Kalman Filter to use. Options are "standard", "single", "univariate", "steady_state",
            and "cholesky". See the docs for kalman filters for more details.
        verbose: bool, default True
            If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.
        mode: str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
            ``forecast``. The mode does **not** effect calls to ``pm.sample``.

            Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
            to all sampling methods.


        References
        ----------
        .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2018.
        """

        if order is not None:
            if len(order) != 3 or any(not isinstance(o, str) for o in order):
                raise ValueError("Order must be a tuple of three strings.")
            if order[0] != "A":
                raise ValueError("Only additive errors are supported.")
            if order[1] not in {"A", "Ad", "N"}:
                raise ValueError(
                    f"Invalid trend specification. Only 'A' (additive), 'Ad' (additive with dampening), "
                    f"or 'N' (no trend) are allowed. Found {order[1]}"
                )
            if order[2] not in {"A", "N"}:
                raise ValueError(
                    f"Invalid seasonal specification. Only 'A' (additive) or 'N' (no seasonal component) "
                    f"are allowed. Found {order[2]}"
                )

            trend = order[1] != "N"
            damped_trend = order[1] == "Ad"
            seasonal = order[2] == "A"

        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.use_transformed_parameterization = use_transformed_parameterization
        self.stationary_initialization = stationary_initialization

        if not (0.0 < initialization_dampening < 1.0):
            raise ValueError(
                "Dampening term used for initialization must be between 0 and 1 (preferably close to"
                "1.0)"
            )

        self.initialization_dampening = initialization_dampening

        if self.seasonal and self.seasonal_periods is None:
            raise ValueError("If seasonal is True, seasonal_periods must be provided.")

        if endog_names is not None:
            endog_names = list(endog_names)
            k_endog = len(endog_names)
        else:
            endog_names = [f"data_{i}" for i in range(k_endog)] if k_endog > 1 else ["data"]

        self.endog_names = endog_names

        if dense_innovation_covariance and k_endog == 1:
            dense_innovation_covariance = False

        self.dense_innovation_covariance = dense_innovation_covariance

        k_states = (
            2
            + int(trend)
            + int(seasonal) * (seasonal_periods if seasonal_periods is not None else 0)
        ) * k_endog

        k_posdef = k_endog

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
            "initial_level",
            "initial_trend",
            "initial_seasonal",
            "P0",
            "alpha",
            "beta",
            "gamma",
            "phi",
            "sigma_state",
            "state_cov",
            "sigma_obs",
        ]
        if not self.trend:
            names.remove("initial_trend")
            names.remove("beta")
        if not self.damped_trend:
            names.remove("phi")
        if not self.seasonal:
            names.remove("initial_seasonal")
            names.remove("gamma")
        if not self.measurement_error:
            names.remove("sigma_obs")

        if self.dense_innovation_covariance:
            names.remove("sigma_state")
        else:
            names.remove("state_cov")

        if self.stationary_initialization:
            names.remove("P0")

        return names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        info = {
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
            },
            "initial_level": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": None,
            },
            "initial_trend": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": None,
            },
            "initial_seasonal": {"shape": (self.seasonal_periods,), "constraints": None},
            "sigma_obs": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "Positive",
            },
            "sigma_state": {
                "shape": None if self.k_posdef == 1 else (self.k_posdef,),
                "constraints": "Positive",
            },
            "alpha": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "0 < alpha < 1",
            },
            "beta": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "0 < beta < 1"
                if not self.use_transformed_parameterization
                else "0 < beta < alpha",
            },
            "gamma": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "0 < gamma< 1"
                if not self.use_transformed_parameterization
                else "0 < gamma < (1 - alpha)",
            },
            "phi": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "0 < phi < 1",
            },
        }

        if self.dense_innovation_covariance:
            del info["sigma_state"]
            info["state_cov"] = {
                "shape": (self.k_posdef, self.k_posdef),
                "constraints": "Positive Semi-definite",
            }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims.get(name, None)

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        states = ["innovation", "level"]
        if self.trend:
            states += ["trend"]
        if self.seasonal:
            states += ["seasonality"]
            states += [f"L{i}.season" for i in range(1, self.seasonal_periods)]

        if self.k_endog > 1:
            states = [f"{name}_{state}" for name in self.endog_names for state in states]

        return states

    @property
    def observed_states(self):
        return self.endog_names

    @property
    def shock_names(self):
        return (
            ["innovation"]
            if self.k_endog == 1
            else [f"{name}_innovation" for name in self.endog_names]
        )

    @property
    def param_dims(self):
        coord_map = {
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (OBS_STATE_DIM,),
            "initial_level": (OBS_STATE_DIM,),
            "initial_trend": (OBS_STATE_DIM,),
            "initial_seasonal": (ETS_SEASONAL_DIM,),
            "seasonal_param": (ETS_SEASONAL_DIM,),
        }

        if self.dense_innovation_covariance:
            del coord_map["sigma_state"]
            coord_map["state_cov"] = (OBS_STATE_DIM, OBS_STATE_AUX_DIM)

        if self.k_endog == 1:
            coord_map["sigma_state"] = None
            coord_map["sigma_obs"] = None
            coord_map["initial_level"] = None
            coord_map["initial_trend"] = None
        else:
            coord_map["alpha"] = (OBS_STATE_DIM,)
            coord_map["beta"] = (OBS_STATE_DIM,)
            coord_map["gamma"] = (OBS_STATE_DIM,)
            coord_map["phi"] = (OBS_STATE_DIM,)
            coord_map["initial_seasonal"] = (OBS_STATE_DIM, ETS_SEASONAL_DIM)
            coord_map["seasonal_param"] = (OBS_STATE_DIM, ETS_SEASONAL_DIM)

        if not self.measurement_error:
            del coord_map["sigma_obs"]
        if not self.seasonal:
            del coord_map["seasonal_param"]

        return coord_map

    @property
    def coords(self) -> dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.seasonal:
            coords.update({ETS_SEASONAL_DIM: list(range(1, self.seasonal_periods + 1))})

        return coords

    def _stationary_initialization(self, T_stationary):
        # Solve for matrix quadratic for P0
        R = self.ssm["selection"]
        Q = self.ssm["state_cov"]

        # ETS models are not stationary, but we can proceed *as if* the model were stationary by introducing large
        # dampening factors on all components. We then set the initial covariance to the steady-state of that system,
        # which we hope is similar enough to give a good initialization for the non-stationary system.

        T_stationary = pt.specify_shape(T_stationary, (self.k_states, self.k_states))
        P0 = solve_discrete_lyapunov(T_stationary, pt.linalg.matrix_dot(R, Q, R.T))
        P0 = pt.specify_shape(P0, (self.k_states, self.k_states))

        return P0

    def make_symbolic_graph(self) -> None:
        k_states_each = self.k_states // self.k_endog

        initial_level = self.make_and_register_variable(
            "initial_level", shape=(self.k_endog,) if self.k_endog > 1 else (), dtype=floatX
        )

        initial_states = [pt.zeros(k_states_each) for _ in range(self.k_endog)]
        if self.k_endog == 1:
            initial_states = [pt.set_subtensor(initial_states[0][1], initial_level)]
        else:
            initial_states = [
                pt.set_subtensor(initial_state[1], initial_level[i])
                for i, initial_state in enumerate(initial_states)
            ]

        # The shape of R can be pre-allocated, then filled with the required parameters
        R = pt.zeros((self.k_states // self.k_endog, 1))

        alpha = self.make_and_register_variable(
            "alpha", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
        )

        # This is a dummy value for initialization. When we do a stationary initialization, it will be set to a value
        # close to 1. Otherwise, it will be 1. We do not want this value to exist outside of this method.
        stationary_dampening = pt.scalar("dampen_dummy")

        if self.k_endog == 1:
            # The R[0, 0] entry needs to be adjusted for a shift in the time indices. Consider the (A, N, N) model:
            # y_t = l_{t-1} + e_t
            # l_t = l_{t-1} + alpha * e_t
            R_list = [pt.set_subtensor(R[1, 0], alpha)]  # and l_t = ... + alpha * e_t

            # We want the first equation to be in terms of time t on the RHS, because our observation equation is always
            # y_t = Z @ x_t. Re-arranging equation 2, we get l_{t-1} = l_t - alpha * e_t --> y_t = l_t + e_t - alpha * e_t
            # --> y_t = l_t + (1 - alpha) * e_t
            R_list = [pt.set_subtensor(R[0, :], (1 - alpha)) for R in R_list]
        else:
            # If there are multiple endog, clone the basic R matrix and modify the appropriate entries
            R_list = [pt.set_subtensor(R[1, 0], alpha[i]) for i in range(self.k_endog)]
            R_list = [pt.set_subtensor(R[0, :], (1 - alpha[i])) for i, R in enumerate(R_list)]

        # Shock and level component always exists, the base case is e_t = e_t and l_t = l_{t-1}
        T_base = pt.set_subtensor(pt.zeros((2, 2))[1, 1], stationary_dampening)

        if self.trend:
            initial_trend = self.make_and_register_variable(
                "initial_trend", shape=(self.k_endog,) if self.k_endog > 1 else (), dtype=floatX
            )

            if self.k_endog == 1:
                initial_states = [pt.set_subtensor(initial_states[0][2], initial_trend)]
            else:
                initial_states = [
                    pt.set_subtensor(initial_state[2], initial_trend[i])
                    for i, initial_state in enumerate(initial_states)
                ]
            beta = self.make_and_register_variable(
                "beta", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )
            if self.use_transformed_parameterization:
                param = beta
            else:
                param = alpha * beta
            if self.k_endog == 1:
                R_list = [pt.set_subtensor(R[2, 0], param) for R in R_list]
            else:
                R_list = [pt.set_subtensor(R[2, 0], param[i]) for i, R in enumerate(R_list)]

            # If a trend is requested, we have the following transition equations (omitting the shocks):
            # l_t = l_{t-1} + b_{t-1}
            # b_t = b_{t-1}
            T_base = pt.as_tensor_variable(([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]))
            T_base = pt.set_subtensor(T_base[[1, 2], [1, 2]], stationary_dampening)

        if self.damped_trend:
            phi = self.make_and_register_variable(
                "phi", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )
            # We are always in the case where we have a trend, so we can add the dampening parameter to T_base defined
            # in that branch. Transition equations become:
            # l_t = l_{t-1} + phi * b_{t-1}
            # b_t = phi * b_{t-1}
            if self.k_endog > 1:
                T_base = [pt.set_subtensor(T_base[1:, 2], phi[i]) for i in range(self.k_endog)]
            else:
                T_base = pt.set_subtensor(T_base[1:, 2], phi)

        T_components = (
            [T_base for _ in range(self.k_endog)] if not isinstance(T_base, list) else T_base
        )

        if self.seasonal:
            initial_seasonal = self.make_and_register_variable(
                "initial_seasonal",
                shape=(self.seasonal_periods,)
                if self.k_endog == 1
                else (self.k_endog, self.seasonal_periods),
                dtype=floatX,
            )
            if self.k_endog == 1:
                initial_states = [
                    pt.set_subtensor(initial_states[0][2 + int(self.trend) :], initial_seasonal)
                ]
            else:
                initial_states = [
                    pt.set_subtensor(initial_state[2 + int(self.trend) :], initial_seasonal[i])
                    for i, initial_state in enumerate(initial_states)
                ]

            gamma = self.make_and_register_variable(
                "gamma", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )

            param = gamma if self.use_transformed_parameterization else (1 - alpha) * gamma
            # Additional adjustment to the R[0, 0] position is required. Start from:
            # y_t = l_{t-1} + s_{t-m} + e_t
            # l_t = l_{t-1} + alpha * e_t
            # s_t = s_{t-m} + gamma * e_t
            # Solve for l_{t-1} and s_{t-m} in terms of l_t and s_t, then substitute into the observation equation:
            # y_t = l_t + s_t - alpha * e_t - gamma * e_t + e_t --> y_t = l_t + s_t + (1 - alpha - gamma) * e_t

            if self.k_endog == 1:
                R_list = [pt.set_subtensor(R[2 + int(self.trend), 0], param) for R in R_list]
                R_list = [pt.set_subtensor(R[0, 0], R[0, 0] - param) for R in R_list]

            else:
                R_list = [
                    pt.set_subtensor(R[2 + int(self.trend), 0], param[i])
                    for i, R in enumerate(R_list)
                ]
                R_list = [
                    pt.set_subtensor(R[0, 0], R[0, 0] - param[i]) for i, R in enumerate(R_list)
                ]

            # The seasonal component is always going to look like a TimeFrequency structural component, see that
            # docstring for more details
            T_seasonals = [pt.eye(self.seasonal_periods, k=-1) for _ in range(self.k_endog)]
            T_seasonals = [
                pt.set_subtensor(T_seasonal[0, -1], stationary_dampening)
                for T_seasonal in T_seasonals
            ]

            # Organize the components so it goes T1, T_seasonal_1, T2, T_seasonal_2, etc.
            T_components = [
                matrix[i] for i in range(self.k_endog) for matrix in [T_components, T_seasonals]
            ]

        x0 = pt.concatenate(initial_states, axis=0)
        R = pt.linalg.block_diag(*R_list)

        self.ssm["initial_state"] = x0
        self.ssm["selection"] = pt.specify_shape(R, shape=(self.k_states, self.k_posdef))

        T = pt.linalg.block_diag(*T_components)

        # Remove the stationary_dampening dummies before saving the transition matrix
        self.ssm["transition"] = pt.specify_shape(
            graph_replace(T, {stationary_dampening: 1.0}), (self.k_states, self.k_states)
        )

        Zs = [np.zeros((self.k_endog, self.k_states // self.k_endog)) for _ in range(self.k_endog)]
        for i, Z in enumerate(Zs):
            Z[i, 0] = 1.0  # innovation
            Z[i, 1] = 1.0  # level
            if self.seasonal:
                Z[i, 2 + int(self.trend)] = 1.0

        Z = pt.concatenate(Zs, axis=1)

        self.ssm["design"] = Z

        # Set up the state covariance matrix
        if self.dense_innovation_covariance:
            state_cov = self.make_and_register_variable(
                "state_cov", shape=(self.k_posdef, self.k_posdef), dtype=floatX
            )
            self.ssm["state_cov"] = state_cov

        else:
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

        if self.stationary_initialization:
            T_stationary = graph_replace(T, {stationary_dampening: self.initialization_dampening})
            P0 = self._stationary_initialization(T_stationary)

        else:
            P0 = self.make_and_register_variable(
                "P0", shape=(self.k_states, self.k_states), dtype=floatX
            )

        self.ssm["initial_state_cov"] = P0
