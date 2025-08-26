from collections.abc import Sequence
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.mode import Mode
from pytensor.tensor.slinalg import solve_discrete_lyapunov

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models.utilities import make_default_coords
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    EXOGENOUS_DIM,
    MA_PARAM_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    TIME_DIM,
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
            + B_q \varepsilon_{t-q} + \varepsilon_t

    Where :math:`\varepsilon_t = \begin{bmatrix} \varepsilon_{1,t} & \varepsilon_{2,t} & \cdots &
    \varepsilon_{k,t}\end{bmatrix}^T \sim N(0, \Sigma)` is a vector of i.i.d stochastic innovations or shocks that drive
    intertemporal variation in the data. Matrices :math:`A_i, B_i` are :math:`k \times k` coefficient matrices:

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
        k_endog: int | None = None,
        exog_state_names: list[str] | dict[str, list[str]] | None = None,
        k_exog: int | dict[str, int] | None = None,
        stationary_initialization: bool = False,
        filter_type: str = "standard",
        measurement_error: bool = False,
        verbose: bool = True,
        mode: str | Mode | None = None,
    ):
        """
        Create a Bayesian VARMAX model.

        Parameters
        ----------
        order: tuple of (int, int)
            Number of autoregressive (AR) and moving average (MA) terms to include in the model. All terms up to the
            specified order are included. For restricted models, set zeros directly on the priors.

        endog_names: list of str, optional
            Names of the endogenous variables being modeled. Used to generate names for the state and shock coords. If
            None, the state names will simply be numbered.

            Exactly one of either ``endog_names`` or ``k_endog`` must be specified.

        exog_state_names : list[str] or dict[str, list[str]], optional
            Names of the exogenous state variables. If a list, all endogenous variables will share the same exogenous
            variables. If a dict, keys should be the names of the endogenous variables, and values should be lists of the
            exogenous variable names for that endogenous variable. Endogenous variables not included in the dict will
            be assumed to have no exogenous variables. If None, no exogenous variables will be included.

        k_exog : int or dict[str, int], optional
            Number of exogenous variables. If an int, all endogenous variables will share the same number of exogenous
            variables. If a dict, keys should be the names of the endogenous variables, and values should be the number of
            exogenous variables for that endogenous variable. Endogenous variables not included in the dict will be
            assumed to have no exogenous variables. If None, no exogenous variables will be included.

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

        state_structure: str, default "fast"
            How to represent the state-space system. When "interpretable", each element of the state vector will have a
            precise meaning as either lagged data, innovations, or lagged innovations. This comes at the cost of a larger
            state vector, which may hurt performance.

            When "fast", states are combined to minimize the dimension of the state vector, but lags and innovations are
            mixed together as a result. Only the first state (the modeled timeseries) will have an obvious interpretation
            in this case.

        measurement_error: bool, default True
            If true, a measurement error term is added to the model.

        verbose: bool, default True
            If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.

        mode: str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
            ``forecast``. The mode does **not** effect calls to ``pm.sample``.

            Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
            to all sampling methods.

        """
        if (endog_names is None) and (k_endog is None):
            raise ValueError("Must specify either endog_names or k_endog")
        if (endog_names is not None) and (k_endog is None):
            k_endog = len(endog_names)
        if (endog_names is None) and (k_endog is not None):
            endog_names = [f"observed_{i}" for i in range(k_endog)]
        if (endog_names is not None) and (k_endog is not None):
            if len(endog_names) != k_endog:
                raise ValueError("Length of provided endog_names does not match provided k_endog")

        if k_exog is not None and not isinstance(k_exog, int | dict):
            raise ValueError("If not None, k_endog must be either an int or a dict")
        if exog_state_names is not None and not isinstance(exog_state_names, list | dict):
            raise ValueError("If not None, exog_state_names must be either a list or a dict")

        if k_exog is not None and exog_state_names is not None:
            if isinstance(k_exog, int) and isinstance(exog_state_names, list):
                if len(exog_state_names) != k_exog:
                    raise ValueError("Length of exog_state_names does not match provided k_exog")
            elif isinstance(k_exog, int) and isinstance(exog_state_names, dict):
                raise ValueError(
                    "If k_exog is an int, exog_state_names must be a list of the same length (or None)"
                )
            elif isinstance(k_exog, dict) and isinstance(exog_state_names, list):
                raise ValueError(
                    "If k_exog is a dict, exog_state_names must be a dict as well (or None)"
                )
            elif isinstance(k_exog, dict) and isinstance(exog_state_names, dict):
                if set(k_exog.keys()) != set(exog_state_names.keys()):
                    raise ValueError("Keys of k_exog and exog_state_names dicts must match")
                if not all(
                    len(names) == k for names, k in zip(exog_state_names.values(), k_exog.values())
                ):
                    raise ValueError(
                        "If both k_endog and exog_state_names are provided, lengths of exog_state_names "
                        "lists must match corresponding values in k_exog"
                    )

        if k_exog is not None and exog_state_names is None:
            if isinstance(k_exog, int):
                exog_state_names = [f"exogenous_{i}" for i in range(k_exog)]
            elif isinstance(k_exog, dict):
                exog_state_names = {
                    name: [f"{name}_exogenous_{i}" for i in range(k)] for name, k in k_exog.items()
                }

        if k_exog is None and exog_state_names is not None:
            if isinstance(exog_state_names, list):
                k_exog = len(exog_state_names)
            elif isinstance(exog_state_names, dict):
                k_exog = {name: len(names) for name, names in exog_state_names.items()}

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
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
        )

        # Save counts of the number of parameters in each category
        self.param_counts = {
            "x0": k_states * (1 - self.stationary_initialization),
            "P0": k_states**2 * (1 - self.stationary_initialization),
            "AR": k_endog**2 * self.p,
            "MA": k_endog**2 * self.q,
            "state_cov": k_posdef**2,
            "sigma_obs": k_endog * self.measurement_error,
        }

    @property
    def param_names(self):
        names = ["x0", "P0", "ar_params", "ma_params", "state_cov", "sigma_obs"]
        if self.stationary_initialization:
            names.remove("P0")
            names.remove("x0")
        if not self.measurement_error:
            names.remove("sigma_obs")
        if self.p == 0:
            names.remove("ar_params")
        if self.q == 0:
            names.remove("ma_params")

        # Add exogenous regression coefficents rather than remove, since we might have to handle
        # several (if self.exog_state_names is a dict)
        if isinstance(self.exog_state_names, list):
            names.append("beta_exog")
        elif isinstance(self.exog_state_names, dict):
            names.extend([f"beta_{name}" for name in self.exog_state_names.keys()])

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
                "shape": (self.k_endog, self.k_endog),
                "constraints": "Positive Semi-definite",
            },
            "state_cov": {
                "shape": (self.k_posdef, self.k_posdef),
                "constraints": "Positive Semi-definite",
            },
            "ar_params": {
                "shape": (self.k_endog, self.p, self.k_endog),
                "constraints": "None",
            },
            "ma_params": {
                "shape": (self.k_endog, self.q, self.k_endog),
                "constraints": "None",
            },
        }

        if isinstance(self.exog_state_names, list):
            k_exog = len(self.exog_state_names)
            info["beta_exog"] = {
                "shape": (self.k_endog, k_exog),
                "constraints": "None",
            }

        elif isinstance(self.exog_state_names, dict):
            for name, exog_names in self.exog_state_names.items():
                k_exog = len(exog_names)
                info[f"beta_{name}"] = {
                    "shape": (k_exog,),
                    "constraints": "None",
                }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def data_info(self) -> dict[str, dict[str, Any]]:
        info = None

        if isinstance(self.exog_state_names, list):
            info = {
                "exogenous_data": {
                    "dims": (TIME_DIM, EXOGENOUS_DIM),
                    "shape": (None, self.k_exog),
                }
            }

        elif isinstance(self.exog_state_names, dict):
            info = {
                f"{endog_state}_exogenous_data": {
                    "dims": (TIME_DIM, f"{EXOGENOUS_DIM}_{endog_state}"),
                    "shape": (None, len(exog_names)),
                }
                for endog_state, exog_names in self.exog_state_names.items()
            }

        return info

    @property
    def data_names(self) -> list[str]:
        if isinstance(self.exog_state_names, list):
            return ["exogenous_data"]
        elif isinstance(self.exog_state_names, dict):
            return [f"{endog_state}_exogenous_data" for endog_state in self.exog_state_names.keys()]
        return []

    @property
    def state_names(self):
        state_names = self.endog_names.copy()
        state_names += [
            f"L{i + 1}_{state}" for i in range(self.p - 1) for state in self.endog_names
        ]
        state_names += [
            f"L{i + 1}_{state}_innov" for i in range(self.q) for state in self.endog_names
        ]

        return state_names

    @property
    def observed_states(self):
        return self.endog_names

    @property
    def shock_names(self):
        return self.endog_names

    @property
    def default_priors(self):
        raise NotImplementedError

    @property
    def coords(self) -> dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.p > 0:
            coords.update({AR_PARAM_DIM: list(range(1, self.p + 1))})
        if self.q > 0:
            coords.update({MA_PARAM_DIM: list(range(1, self.q + 1))})

        if isinstance(self.exog_state_names, list):
            coords[EXOGENOUS_DIM] = self.exog_state_names
        elif isinstance(self.exog_state_names, dict):
            for name, exog_names in self.exog_state_names.items():
                coords[f"{EXOGENOUS_DIM}_{name}"] = exog_names

        return coords

    @property
    def param_dims(self):
        coord_map = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "state_cov": (SHOCK_DIM, SHOCK_AUX_DIM),
            "ar_params": (OBS_STATE_DIM, AR_PARAM_DIM, OBS_STATE_AUX_DIM),
            "ma_params": (OBS_STATE_DIM, MA_PARAM_DIM, OBS_STATE_AUX_DIM),
        }

        if not self.measurement_error:
            del coord_map["sigma_obs"]
        if self.p == 0:
            del coord_map["ar_params"]
        if self.q == 0:
            del coord_map["ma_params"]
        if self.stationary_initialization:
            del coord_map["P0"]
            del coord_map["x0"]

        if isinstance(self.exog_state_names, list):
            coord_map["beta_exog"] = (OBS_STATE_DIM, EXOGENOUS_DIM)
        elif isinstance(self.exog_state_names, dict):
            # If each state has its own exogenous variables, each parameter needs it own dim, since we expect the
            # dim labels to all be different (otherwise we'd be in the list case).
            for name in self.exog_state_names.keys():
                coord_map[f"beta_{name}"] = (f"{EXOGENOUS_DIM}_{name}",)

        return coord_map

    def add_default_priors(self):
        raise NotImplementedError

    def make_symbolic_graph(self) -> None:
        # Initialize the matrices
        if not self.stationary_initialization:
            # initial states
            x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)
            self.ssm["initial_state", :] = x0

            # initial covariance
            P0 = self.make_and_register_variable(
                "P0", shape=(self.k_states, self.k_states), dtype=floatX
            )
            self.ssm["initial_state_cov", :, :] = P0

        # Design matrix is a truncated identity (first k_obs states observed)
        self.ssm[("design", *np.diag_indices(self.k_endog))] = 1

        # Transition matrix has 4 blocks:
        # Upper left: AR coefs (k_obs, k_obs * min(p, 1))
        # Upper right: MA coefs (k_obs, k_obs * q)
        # Lower left: Truncated identity (k_obs * min(p, 1), k_obs * min(p, 1))
        # Lower right: Shifted identity (k_obs * p, k_obs * q)
        self.ssm["transition"] = np.zeros((self.k_states, self.k_states))
        if self.p > 1:
            idx = (
                slice(self.k_endog, self.k_endog * self.p),
                slice(0, self.k_endog * (self.p - 1)),
            )
            self.ssm[("transition", *idx)] = np.eye(self.k_endog * (self.p - 1))

        if self.q > 1:
            idx = (
                slice(-self.k_endog * (self.q - 1), None),
                slice(-self.k_endog * self.q, -self.k_endog),
            )
            self.ssm[("transition", *idx)] = np.eye(self.k_endog * (self.q - 1))

        if self.p > 0:
            ar_param_idx = ("transition", slice(0, self.k_endog), slice(0, self.k_endog * self.p))

            # Register the AR parameter matrix as a (k, p, k), then reshape it and allocate it in the transition matrix
            # This way the user can use 3 dimensions in the prior (clearer?)
            ar_params = self.make_and_register_variable(
                "ar_params", shape=(self.k_endog, self.p, self.k_endog), dtype=floatX
            )

            ar_params = ar_params.reshape((self.k_endog, self.k_endog * self.p))
            self.ssm[ar_param_idx] = ar_params

        # The selection matrix is (k_states, k_obs), with two (k_obs, k_obs) identity
        # matrix blocks inside. One is always on top, the other starts after (k_obs * p) rows
        self.ssm["selection"] = np.zeros((self.k_states, self.k_endog))
        self.ssm["selection", slice(0, self.k_endog), :] = np.eye(self.k_endog)
        if self.q > 0:
            ma_param_idx = (
                "transition",
                slice(0, self.k_endog),
                slice(self.k_endog * max(1, self.p), None),
            )

            # Same as above, register with 3 dimensions then reshape
            ma_params = self.make_and_register_variable(
                "ma_params", shape=(self.k_endog, self.q, self.k_endog), dtype=floatX
            )

            ma_params = ma_params.reshape((self.k_endog, self.k_endog * self.q))
            self.ssm[ma_param_idx] = ma_params

            end = -self.k_endog * (self.q - 1) if self.q > 1 else None
            self.ssm["selection", slice(self.k_endog * -self.q, end), :] = np.eye(self.k_endog)

        if self.measurement_error:
            obs_cov_idx = ("obs_cov", *np.diag_indices(self.k_endog))
            sigma_obs = self.make_and_register_variable(
                "sigma_obs", shape=(self.k_endog,), dtype=floatX
            )
            self.ssm[obs_cov_idx] = sigma_obs

        state_cov = self.make_and_register_variable(
            "state_cov", shape=(self.k_posdef, self.k_posdef), dtype=floatX
        )
        self.ssm["state_cov", :, :] = state_cov

        if self.exog_state_names is not None:
            if isinstance(self.exog_state_names, list):
                beta_exog = self.make_and_register_variable(
                    "beta_exog", shape=(self.k_posdef, self.k_exog), dtype=floatX
                )
                exog_data = self.make_and_register_data(
                    "exogenous_data", shape=(None, self.k_exog), dtype=floatX
                )

                obs_intercept = exog_data @ beta_exog.T

            elif isinstance(self.exog_state_names, dict):
                obs_components = []
                for i, name in enumerate(self.endog_names):
                    if name in self.exog_state_names:
                        k_exog = len(self.exog_state_names[name])
                        beta_exog = self.make_and_register_variable(
                            f"beta_{name}", shape=(k_exog,), dtype=floatX
                        )
                        exog_data = self.make_and_register_data(
                            f"{name}_exogenous_data", shape=(None, k_exog), dtype=floatX
                        )
                        obs_components.append(pt.expand_dims(exog_data @ beta_exog, axis=-1))
                    else:
                        obs_components.append(pt.zeros((1, 1), dtype=floatX))

                # TODO: Replace all of this with pt.concat_with_broadcast once PyMC works with pytensor >= 2.32

                # If there were any zeros, they need to be broadcast against the non-zeros.
                # Core shape is the last dim, the time dim is always broadcast
                non_concat_shape = [1, None]

                # Look for the first non-zero component to get the shape from
                for tensor_inp in obs_components:
                    for i, (bcast, sh) in enumerate(
                        zip(tensor_inp.type.broadcastable, tensor_inp.shape)
                    ):
                        if bcast or i == 1:
                            continue
                        non_concat_shape[i] = sh

                assert non_concat_shape.count(None) == 1

                bcast_tensor_inputs = []
                for tensor_inp in obs_components:
                    non_concat_shape[1] = tensor_inp.shape[1]
                    bcast_tensor_inputs.append(pt.broadcast_to(tensor_inp, non_concat_shape))

                obs_intercept = pt.join(1, *bcast_tensor_inputs)

            else:
                raise NotImplementedError()

            self.ssm["obs_intercept"] = obs_intercept

        if self.stationary_initialization:
            # Solve for matrix quadratic for P0
            T = self.ssm["transition"]
            R = self.ssm["selection"]
            Q = self.ssm["state_cov"]
            c = self.ssm["state_intercept"]

            x0 = pt.linalg.solve(pt.eye(T.shape[0]) - T, c, assume_a="gen", check_finite=False)
            P0 = solve_discrete_lyapunov(
                T,
                pt.linalg.matrix_dot(R, Q, R.T),
                method="direct" if self.k_states < 10 else "bilinear",
            )
            self.ssm["initial_state", :] = x0
            self.ssm["initial_state_cov", :, :] = P0
