from collections.abc import Sequence
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models.utilities import make_default_coords
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    ERROR_AR_PARAM_DIM,
    FACTOR_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
)

floatX = pytensor.config.floatX


class BayesianDynamicFactor(PyMCStateSpace):
    r"""
    Dynamic Factor Models

    Parameters
    ----------
    k_factors : int
        Number of latent factors.

    factor_order : int
        Order of the VAR process for the latent factors.

    k_endog : int, optional
        Number of observed time series. If not provided, the number of observed series will be inferred from `endog_names`.
        At least one of `k_endog` or `endog_names` must be provided.

    endog_names : list of str, optional
        Names of the observed time series. If not provided, default names will be generated as `endog_1`, `endog_2`, ..., `endog_k` based on `k_endog`.
        At least one of `k_endog` or `endog_names` must be provided.

    exog : array_like, optional
        Array of exogenous regressors for the observation equation (nobs x k_exog).
        Default is None, meaning no exogenous regressors.
        Not implemented yet.

    error_order : int, optional
        Order of the AR process for the observation error component.
        Default is 0, corresponding to white noise errors.

    error_var : bool, optional
        If True, errors are modeled jointly via a VAR process;
        otherwise, each error is modeled separately.

    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
        Structure of the covariance matrix of the observation errors.

    filter_type: str, default "standard"
        The type of Kalman Filter to use. Options are "standard", "single", "univariate", "steady_state",
        and "cholesky". See the docs for kalman filters for more details.

    verbose: bool, default True
        If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.


    Notes
    -----
    The Dynamic Factor Model (DFM) is a multivariate state-space model used to represent high-dimensional time series
    as driven by a smaller set of unobserved dynamic factors. Given a set of observed time series
    :math:`\{y_t\}_{t=0}^T`, with :math:`y_t = \begin{bmatrix} y_{1,t} & y_{2,t} & \cdots & y_{k_endog,t} \end{bmatrix}^T`,
    the DFM assumes that each series is a linear combination of a few latent factors and optional autoregressive errors.

    Specifically, denoting the number of dynamic factors as :math:`k_factors`, the order of the latent factor
    process as :math:`p = \text{factor\_order}`, and the order of the observation error as
    :math:`q = \text{error\_order}`, the model is written as:

    .. math::
        y_t & = \Lambda f_t + B x_t + u_t \\
        f_t & = A_1 f_{t-1} + \dots + A_p f_{t-p} + \eta_t \\
        u_t & = C_1 u_{t-1} + \dots + C_q u_{t-q} + \varepsilon_t


    Where:
    - :math:`f_t` is a vector of latent factors following a VAR(p) process:
    - :math:`\x_t` are optional exogenous vectors (Not implemented yet).
    - :math:`u_t` is a vector of observation errors, possibly VAR(q) if error_var = True otherwise treated as individual autoregressions.
    - :math:`\eta_t` and :math:`\varepsilon_t` are white noise error terms. In order to identify the factors, :math:`Var(\eta_t) = I`.
    Denote :math:`Var(\varepsilon_t) \equiv \Sigma`.


    Internally, this model is represented in state-space form by stacking all current and lagged latent factors and,
    if present, autoregressive observation errors into a single state vector. The full state vector has dimension
    :math:`k_factors \cdot factor_order + k_endog \cdot error_order`, where :math:`k_endog` is the number of observed time series.

    The number of independent shocks in the system (i.e., the number of nonzero diagonal elements in the state noise
    covariance matrix) is equal to the number of latent factors plus the number of observed series if AR errors are present.

    As in other high-dimensional models, identification can be an issue, especially when many observed series load on few
    factors. Careful prior specification is typically required for good estimation.

    Currently, the implementation assumes same factor order for all the factors,
    does not yet support measurement error, exogenous variables and joint (VAR) error modeling.

    Examples
    --------
    The following code snippet estimates a dynamic factor model with 1 latent factors,
    a AR(2) structure on the factor and a AR(1) structure on the errors:

    .. code:: python

        import pymc_extras.statespace as pmss
        import pymc as pm

        # Create DFM Statespace Model
        dfm_mod = pmss.BayesianDynamicFactor(
                k_factors=1,
                factor_order=2,
                endog_names=data.columns,
                error_order=1,
                error_var=False,
                error_cov_type="diagonal",
                filter_type="standard",
                verbose=True
            )

        # Unpack dims and coords
        x0_dims, P0_dims, factor_loadings_dims, factor_sigma_dims, factor_ar_dims, error_ar_dims, error_sigma_dims = dfm_mod.param_dims.values()
        coords = dfm_mod.coords

        with pm.Model(coords=coords) as pymc_mod:
            # Initial state
            x0 = pm.Normal("x0", dims=x0_dims)
            P0 = pm.Normal("P0", dims=P0_dims)
            factor_loadings = pm.Normal("factor_loadings", sigma=1, dims=factor_loadings_dims)
            factor_ar = pm.Normal("factor_ar", sigma=1, dims=factor_ar_dims)
            factor_sigma = pm.Deterministic("factor_sigma", pt.constant([1.0], dtype=float))
            error_ar = pm.Normal("error_ar", sigma=1, dims=error_ar_dims)
            sigmas = pm.HalfNormal("error_sigma", dims=error_sigma_dims)
            # Build symbolic graph
            dfm_mod.build_statespace_graph(data=data, mode="JAX")

        with pymc_mod:
            idata = pm.sample(
            draws=500,
            chains=2,
            nuts_sampler="nutpie",
            nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
        )

    """

    def __init__(
        self,
        k_factors: int,
        factor_order: int,
        k_endog: int | None = None,
        endog_names: Sequence[str] | None = None,
        exog: np.ndarray | None = None,
        error_order: int = 0,
        error_var: bool = False,
        error_cov_type: str = "diagonal",
        filter_type: str = "standard",
        verbose: bool = True,
    ):
        if k_endog is None and endog_names is None:
            raise ValueError("Either k_endog or endog_names must be provided.")
        if k_endog is None:
            k_endog = len(endog_names)
        if endog_names is None:
            endog_names = [f"endog_{i+1}" for i in range(k_endog)]

        if error_var:
            raise NotImplementedError(
                "Joint error modeling (error_var=True) is not yet implemented."
            )
        if exog is not None:
            raise NotImplementedError("Exogenous variables (exog) are not yet implemented.")

        self.endog_names = endog_names
        self.k_endog = k_endog
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var
        self.error_cov_type = error_cov_type
        self.exog = exog
        # TODO add measurement error support
        # TODO add exogenous variables support?

        # Determine the dimension for the latent factor states.
        # For static factors, one might use k_factors.
        # For dynamic factors with lags, the state might include current factors and past lags.
        # TODO: what if we want different factor orders for different factors? (follow suggestions in GitHub)
        k_factor_states = k_factors * factor_order

        # Determine the dimension for the error component.
        # If error_order > 0 then we add additional states for error dynamics, otherwise white noise error.
        k_error_states = k_endog * error_order if error_order > 0 else 0

        # Total state dimension
        k_states = k_factor_states + k_error_states

        # Number of independent shocks.
        # Typically, the latent factors introduce k_factors shocks.
        # If error_order > 0 and errors are modeled jointly or separately, add appropriate count.
        k_posdef = k_factors + (k_endog if error_order > 0 else 0)

        # Initialize the PyMCStateSpace base class.
        super().__init__(
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            filter_type=filter_type,
            verbose=verbose,
            measurement_error=False,
        )

    @property
    def param_names(self):
        names = [
            "x0",
            "P0",
            "factor_loadings",
            "factor_ar",
            "factor_sigma",
            "error_ar",
            "error_sigma",
        ]

        # Handle cases where parameters should be excluded based on model settings
        if self.factor_order == 0:
            names.remove("factor_ar")
        if self.error_order == 0:
            names.remove("error_ar")
        if self.error_cov_type == "unstructured":
            names.remove("error_sigma")
            names.append("error_cov")

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
            "factor_loadings": {
                "shape": (self.k_endog, self.k_factors),
                "constraints": None,
            },
            "factor_ar": {
                "shape": (self.k_factors, self.factor_order),
                "constraints": None,
            },
            "factor_sigma": {
                "shape": (self.k_factors,),
                "constraints": "Positive",
            },
            "error_ar": {
                "shape": (self.k_endog, self.error_order),
                "constraints": None,
            },
            "error_sigma": {
                "shape": (self.k_endog,) if self.error_cov_type == "diagonal" else (),
                "constraints": "Positive",
            },
            "error_cov": {
                "shape": (self.k_endog, self.k_endog),
                "constraints": "Positive Semi-definite",
            },
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self) -> list[str]:
        """
        Returns the names of the hidden states: first factor states (with lags),
        then idiosyncratic error states (with lags).
        """
        names = []
        # TODO adjust notation by looking at the VARMAX implementation
        # Factor states
        for i in range(self.k_factors):
            for lag in range(self.factor_order):
                names.append(f"L{lag}.factor_{i+1}")

        # Idiosyncratic error states
        if self.error_order > 0:
            for i in range(self.k_endog):
                for lag in range(self.error_order):
                    names.append(f"L{lag}.error_{i+1}")

        return names

    @property
    def observed_states(self) -> list[str]:
        """
        Returns the names of the observed states (i.e., the endogenous variables).
        """
        return self.endog_names

    @property
    def coords(self) -> dict[str, Sequence]:
        coords = make_default_coords(self)
        # Add factor dimensions
        coords[FACTOR_DIM] = [f"factor_{i+1}" for i in range(self.k_factors)]

        # AR parameter dimensions - add if needed
        if self.factor_order > 0:
            coords[AR_PARAM_DIM] = list(range(1, self.factor_order + 1))

        # If error_order > 0
        if self.error_order > 0:
            coords[ERROR_AR_PARAM_DIM] = list(range(1, self.error_order + 1))

        return coords

    @property
    def shock_names(self):
        shock_names = []

        # Add names for factor shocks (one per factor)
        for i in range(self.k_factors):
            shock_names.append(f"factor_shock_{i+1}")

        # Add names for idiosyncratic error shocks (one per observed variable)
        if self.error_order > 0:
            for i in range(self.k_endog):
                shock_names.append(f"error_shock_{i+1}")

        return shock_names

    @property
    def param_dims(self):
        coord_map = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_sigma": (FACTOR_DIM,),
        }

        if self.factor_order > 0:
            coord_map["factor_ar"] = (FACTOR_DIM, AR_PARAM_DIM)

        if self.error_order > 0:
            coord_map["error_ar"] = (OBS_STATE_DIM, ERROR_AR_PARAM_DIM)

        if self.error_cov_type in ["scalar"]:
            coord_map["error_sigma"] = ()
        elif self.error_cov_type in ["diagonal"]:
            coord_map["error_sigma"] = (OBS_STATE_DIM,)
        if self.error_cov_type == "unstructured":
            coord_map["error_sigma"] = (OBS_STATE_DIM, OBS_STATE_AUX_DIM)

        return coord_map

    def make_symbolic_graph(self):
        # Initial states
        x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)

        self.ssm["initial_state", :] = x0

        # Initial covariance
        P0 = self.make_and_register_variable(
            "P0", shape=(self.k_states, self.k_states), dtype=floatX
        )

        self.ssm["initial_state_cov", :, :] = P0

        # Design matrix
        factor_loadings = self.make_and_register_variable(
            "factor_loadings", shape=(self.k_endog, self.k_factors), dtype=floatX
        )

        self.ssm["design", :, :] = 0.0

        for j in range(self.k_factors):
            col_idx = j * self.factor_order
            self.ssm["design", :, col_idx] = factor_loadings[:, j]

        for i in range(self.k_endog):
            col_idx = self.k_factors * self.factor_order + i * self.error_order
            self.ssm["design", i, col_idx] = 1.0

        # Transition matrix
        # auxiliary function to build transition matrix block
        def build_ar_block_matrix(ar_coeffs):
            # ar_coeffs: (p,)
            p = ar_coeffs.shape[0]
            top_row = pt.reshape(ar_coeffs, (1, p))
            below = pt.eye(p - 1, p, k=0)
            return pt.concatenate([top_row, below], axis=0)

        transition_blocks = []

        factor_ar = self.make_and_register_variable(
            "factor_ar", shape=(self.k_factors, self.factor_order), dtype=floatX
        )
        for j in range(self.k_factors):
            transition_blocks.append(build_ar_block_matrix(factor_ar[j]))

        if self.error_order > 0:
            error_ar = self.make_and_register_variable(
                "error_ar", shape=(self.k_endog, self.error_order), dtype=floatX
            )
            for j in range(self.k_endog):
                transition_blocks.append(build_ar_block_matrix(error_ar[j]))

        # Final block diagonal transition matrix
        self.ssm["transition", :, :] = pt.linalg.block_diag(*transition_blocks)

        # Selection matrix
        self.ssm["selection", :, :] = 0.0

        for i in range(self.k_factors):
            row = i * self.factor_order
            self.ssm["selection", row, i] = 1.0

        for i in range(self.k_endog):
            row = self.k_factors * self.factor_order + i * self.error_order
            col = self.k_factors + i
            self.ssm["selection", row, col] = 1.0

        # State covariance matrix
        factor_sigma = self.make_and_register_variable(
            "factor_sigma", shape=(self.k_factors,), dtype=floatX
        )
        if self.error_cov_type in ["scalar"]:
            error_sigma = self.make_and_register_variable("error_sigma", shape=(), dtype=floatX)
            error_sigma = error_sigma * np.ones(self.k_endog, dtype=floatX)
        elif self.error_cov_type in ["diagonal"]:
            error_sigma = self.make_and_register_variable(
                "error_sigma", shape=(self.k_endog,), dtype=floatX
            )
        elif self.error_cov_type in ["unstructured"]:
            error_cov = self.make_and_register_variable(
                "error_cov", shape=(self.k_endog, self.k_endog), dtype=floatX
            )

        factor_cov = pt.diag(factor_sigma)

        if self.error_cov_type in ["scalar", "diagonal"]:
            error_cov = pt.diag(error_sigma)
        self.ssm["state_cov", :, :] = (
            pt.linalg.block_diag(factor_cov, error_cov) if self.error_order > 0 else factor_cov
        )

        # Observation covariance matrix
        self.ssm["obs_cov", :, :] = 0.0
