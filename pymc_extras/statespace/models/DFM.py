from typing import Any

import numpy as np

from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    MA_PARAM_DIM,
    OBS_STATE_DIM,
    SHOCK_DIM,
)


class BayesianDynamicFactor(PyMCStateSpace):
    r"""
    Dynamic Factor Models

    Parameters
    ----------
    k_endog : int
        Number of observed time series.

    k_factors : int
        Number of latent factors.

    factor_order : int
        Order of the VAR process for the latent factors.

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

    enforce_stationarity : bool, optional
        Whether to transform AR parameters to enforce stationarity.

    filter_type : str, optional
        Type of Kalman filter to use. See PyMCStateSpace for valid options.

    verbose : bool, optional
        If True, prints model setup details.



    Notes
    -----
    This model implements a dynamic factor model in the spirit of
    statsmodels.tsa.statespace.dynamic_factor.DynamicFactor. The model assumes that
    the observed time series are driven by a set of latent factors that evolve
    according to a VAR process, possibly along with an autoregressive error term.

    Up to now just a draft implementation to test the working of the class and comparing
    with the Custom model done in the Notebook (notebook/Making a Custom DFM.ipynb).
    The model work just with two observations and one factor (k_endog=2, k_factors=1).


    """

    def __init__(
        self,
        k_endog: int,
        k_factors: int,
        factor_order: int,
        exog: np.ndarray | None = None,
        error_order: int = 0,
        error_var: bool = False,
        error_cov_type: str = "diagonal",
        enforce_stationarity: bool = True,
        filter_type: str = "standard",
        verbose: bool = True,
    ):
        self.k_endog = k_endog
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var
        self.error_cov_type = error_cov_type
        self.enforce_stationarity = enforce_stationarity
        self.exog = exog

        # Determine the dimension for the latent factor states.
        # For static factors, one might use k_factors.
        # For dynamic factors with lags, the state might include current factors and past lags.
        k_factor_states = k_factors * (1 + factor_order)

        # Determine the dimension for the error component.
        # If error_order > 0 then we add additional states for error dynamics, otherwise white noise error.
        k_error_states = k_endog * (error_order + 1) if error_order > 0 else 0

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
        if self.error_cov_type in ["diagonal", "scalar"]:
            names.remove("error_sigma")

        return names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        info = {
            "x0": {
                "shape": (self.k_factors,),
                "constraints": None,
            },
            "P0": {
                "shape": (self.k_factors, self.k_factors),
                "constraints": "Positive Semi-definite",
            },
            "factor_loadings": {
                "shape": (self.k_endog, self.k_factors),
                "constraints": None,
            },
            "factor_ar": {
                "shape": (self.k_factors, self.factor_order, self.k_factors),
                "constraints": None,
            },
            "factor_sigma": {
                "shape": (self.k_factors,),
                "constraints": "Positive",
            },
            "error_ar": {
                "shape": (self.k_endog, self.error_order, self.k_endog)
                if self.error_var
                else (self.k_endog, self.error_order),
                "constraints": None,
            },
            "error_sigma": {
                "shape": (self.k_endog,),
                "constraints": "Positive"
                if self.error_cov_type in ["diagonal", "scalar"]
                else "Positive Semi-definite",
            },
            "error_cov": {
                "shape": (self.k_endog, self.k_endog)
                if self.error_cov_type == "unstructured"
                else None,
                "constraints": "Positive Semi-definite"
                if self.error_cov_type == "unstructured"
                else None,
            },
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        state_names = []
        # Add names for the factor loadings (one per observation and factor)
        for i in range(self.k_endog):
            for j in range(self.k_factors):
                state_names.append(f"loading_{i}_{j}")

        # Add names for the factor autoregressive coefficients (for each factor's dynamics)
        for lag in range(1, self.factor_order + 1):
            for i in range(self.k_factors):
                for j in range(self.k_factors):
                    state_names.append(f"factor_ar_{lag}_{i}_{j}")

        # Add names for the error autoregressive coefficients (if error_order > 0)
        if self.error_order > 0:
            if self.error_cov_type == "diagonal":
                # Diagonal error AR, one parameter per series per lag
                for lag in range(1, self.error_order + 1):
                    for i in range(self.k_endog):
                        state_names.append(f"error_ar_{lag}_{i}")
            elif self.error_cov_type == "unstructured":
                # Full covariance error AR (unstructured), one for each pair of endogenous variables
                for lag in range(1, self.error_order + 1):
                    for i in range(self.k_endog):
                        for j in range(i + 1):
                            state_names.append(f"error_ar_{lag}_{i}_{j}")

        # Add names for the factor shocks' variances (one per factor)
        for i in range(self.k_factors):
            state_names.append(f"factor_sigma_{i}")

        # Add names for the error shocks' variances/covariances
        if self.error_order > 0:
            if self.error_cov_type == "diagonal":
                # Diagonal error covariances (one per series)
                for i in range(self.k_endog):
                    state_names.append(f"error_sigma_{i}")
            elif self.error_cov_type == "scalar":
                # Scalar error covariances (shared variance for all errors)
                state_names.append("error_sigma")
            elif self.error_cov_type == "unstructured":
                # Full error covariance matrix
                for i in range(self.k_endog):
                    for j in range(i + 1):
                        state_names.append(f"error_cov_{i}_{j}")

        return state_names

    @property
    def shock_names(self):
        shock_names = []

        # Add names for factor shocks (one per factor)
        for i in range(self.k_factors):
            shock_names.append(f"factor_shock_{i}")

        # Add names for idiosyncratic error shocks (one per observed variable)
        if self.error_order > 0:
            for i in range(self.k_endog):
                shock_names.append(f"error_shock_{i}")

        return shock_names

    @property
    def param_dims(self):
        """
        Define parameter dimensions for the Dynamic Factor Model (DFM).

        Returns
        -------
        dict
            Dictionary mapping parameter names to their respective dimensions.
        """
        coord_map = {
            "x0": (ALL_STATE_DIM,),  # Initial state dimension
            "P0": (ALL_STATE_DIM, ALL_STATE_DIM),  # Initial state covariance dimension
            "factor_loadings": (OBS_STATE_DIM, ALL_STATE_AUX_DIM),  # Factor loadings dimension
            "factor_sigma": (ALL_STATE_DIM,),  # Factor variances dimension
        }

        # Factor AR coefficients if applicable
        if self.factor_order > 0:
            coord_map["factor_ar"] = (AR_PARAM_DIM, SHOCK_DIM, SHOCK_DIM)

        # Error AR coefficients and variances
        if self.error_order > 0:
            if self.error_cov_type == "diagonal":
                coord_map["error_ar"] = (MA_PARAM_DIM, SHOCK_DIM)  # AR for errors
                coord_map["error_sigma"] = (SHOCK_DIM,)  # One variance for each observed variable
            elif self.error_cov_type == "scalar":
                coord_map["error_ar"] = (MA_PARAM_DIM, SHOCK_DIM)
                coord_map["error_sigma"] = None  # Single scalar for error variance
            elif self.error_cov_type == "unstructured":
                coord_map["error_ar"] = (MA_PARAM_DIM, SHOCK_DIM, SHOCK_DIM)  # AR for errors
                coord_map["error_cov_L"] = (
                    SHOCK_DIM,
                    SHOCK_DIM,
                )  # Lower triangular Cholesky factor
                coord_map["error_cov_sd"] = (SHOCK_DIM,)  # Standard deviations for diagonal
            else:
                raise ValueError("Invalid error covariance type.")

        return coord_map

    # def make_symbolic_graph(self):
    # We will implement this in a moment. For now, we need to overwrite it with nothing to avoid a NotImplementedError
    # when we initialize a class instance.
    #    pass

    def make_symbolic_graph(self):
        """
        Create the symbolic graph for the Dynamic Factor Model (DFM).
        This method sets up the state space model, including the design, transition,
        selection, and initial state matrices, as well as the parameters for the model.


        Up to know just a draft implementation to test the working of the class and comparing
        with the Custom model done in the Notebook (notebook/Making a Custom DFM.ipynb).
        """

        # Create symbolic variables for 1D state
        x0 = self.make_and_register_variable("x0", shape=(1,))
        P0 = self.make_and_register_variable("P0", shape=(1, 1))
        factor_loading = self.make_and_register_variable("factor_loadings", shape=(2, 1))

        factor_ar = self.make_and_register_variable("factor_ar", shape=())
        sigma_f = self.make_and_register_variable("factor_sigma", shape=())

        # Initialize matrices with correct dimensions
        self.ssm["design", :, :] = np.array([[0.0], [0.0]])  # 2x1 matrix
        self.ssm["transition", :, :] = np.array([[0.0]])  # 1x1 matrix
        self.ssm["selection", :, :] = np.array([[1.0]])  # 1x1 matrix

        # Set initial state and covariance
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = P0

        # Set design matrix parameters
        self.ssm["design", 0, 0] = factor_loading[0, 0]  # First observation loading
        self.ssm["design", 1, 0] = factor_loading[1, 0]  # Second observation loading

        # Set transition parameter (AR coefficient)
        self.ssm["transition", 0, 0] = factor_ar

        # Set state covariance
        self.ssm["state_cov", 0, 0] = sigma_f
