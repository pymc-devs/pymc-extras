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

    k_endog : int
        Number of observed time series.

    endog_names : Sequence[str], optional
        Names of the observed time series. If not provided, default names will be generated as `endog_1`, `endog_2`, ..., `endog_k`.

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
    This model implements a dynamic factor model in the spirit of
    statsmodels.tsa.statespace.dynamic_factor.DynamicFactor. The model assumes that
    the observed time series are driven by a set of latent factors that evolve
    according to a VAR process, possibly along with an autoregressive error term.



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
        # TODO: what if we want different factor orders for different factors?
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
        if self.error_cov_type in ["unstructured"]:
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
                "shape": (self.k_endog,) if self.error_cov_type in ["diagonal"] else (),
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

        # Factor states
        for i in range(self.k_factors):
            for lag in range(self.factor_order):
                names.append(f"factor_{i+1}_lag{lag}")

        # Idiosyncratic error states
        if self.error_order > 0:
            for i in range(self.k_endog):
                for lag in range(self.error_order):
                    names.append(f"error_{i+1}_lag{lag}")

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

        coords[FACTOR_DIM] = [f"factor_{i+1}" for i in range(self.k_factors)]

        # AR parameter dimensions - add if needed
        if self.factor_order > 0:
            coords[AR_PARAM_DIM] = list(range(1, self.factor_order + 1))

        # If error_order > 0
        if self.error_order > 0:
            coords["error_ar_param"] = list(range(1, self.error_order + 1))

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
            coord_map["error_ar"] = (OBS_STATE_DIM, "error_ar_param")

        if self.error_cov_type in ["scalar"]:
            coord_map["error_sigma"] = ()
        elif self.error_cov_type in ["diagonal"]:
            coord_map["error_sigma"] = (OBS_STATE_DIM,)
        elif self.error_cov_type in ["unstructured"]:
            coord_map["error_sigma"] = (OBS_STATE_DIM, OBS_STATE_AUX_DIM)

        return coord_map

    def make_symbolic_graph(self):
        # initial states
        x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)

        self.ssm["initial_state", :] = x0

        # initial covariance
        P0 = self.make_and_register_variable(
            "P0", shape=(self.k_states, self.k_states), dtype=floatX
        )

        self.ssm["initial_state_cov", :, :] = P0

        # TODO vectorize the design matrix
        # Design matrix
        self.ssm["design", :, :] = 0.0

        factor_loadings = self.make_and_register_variable(
            "factor_loadings", shape=(self.k_endog, self.k_factors), dtype=floatX
        )

        for i in range(self.k_endog):
            for j in range(self.k_factors):
                # Loadings for each observed variable on the latent factors
                self.ssm["design", i, j * self.factor_order] = factor_loadings[i, j]

        for i in range(self.k_endog):
            # Loadings for each observed variable on the latent factors
            self.ssm["design", i, self.k_factors * self.factor_order + i * self.error_order] = 1.0

        # TODO vectorize the transition matrix or use block matrices (reordering states, check the VAR implementation)
        if self.factor_order > 0:
            # Transition matrix
            factor_ar = self.make_and_register_variable(
                "factor_ar", shape=(self.k_factors, self.factor_order), dtype=floatX
            )

            self.ssm["transition", :, :] = 0.0

            for j in range(self.k_factors):
                block_start = j * self.factor_order
                for i in range(self.factor_order):
                    # Assign AR coefficients to the first row of each block
                    self.ssm["transition", block_start, block_start + i] = factor_ar[j, i]

                    # Fill the subdiagonal with ones, only for rows 1 to p-1
                    if i < self.factor_order - 1:
                        self.ssm["transition", block_start + i + 1, block_start + i] = 1.0

        if self.error_order > 0:
            error_ar = self.make_and_register_variable(
                "error_ar", shape=(self.k_endog, self.error_order), dtype=floatX
            )

            for j in range(self.k_endog):
                block_start = self.k_factors * self.factor_order + j * self.error_order
                for i in range(self.error_order):
                    # Set AR coefficients for the top row of each error AR(q) block
                    self.ssm["transition", block_start, block_start + i] = error_ar[j, i]

                    # Set subdiagonal 1.0s, except last row
                    if i < self.error_order - 1:
                        self.ssm["transition", block_start + i + 1, block_start + i] = 1.0

        # TODO vectorize/block matrices (reorder the states accordingly)
        # Selection matrix
        self.ssm["selection", :, :] = 0.0
        for i in range(self.k_factors):
            self.ssm["selection", i * self.factor_order, i] = 1.0

        for i in range(self.k_endog):
            self.ssm[
                "selection",
                self.k_factors * self.factor_order + i * self.error_order,
                self.k_factors + i,
            ] = 1.0

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
