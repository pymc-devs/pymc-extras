import logging
import warnings

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.model import modelcontext
from pymc.util import RandomState
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs
from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.table import Table
from xarray import DataTree

from pymc_extras.statespace.core import dummy_graph, forecast, forward_sampling, irf
from pymc_extras.statespace.core.properties import (
    Coord,
    CoordInfo,
    Data,
    DataInfo,
    Parameter,
    ParameterInfo,
    Shock,
    ShockInfo,
    State,
    StateInfo,
    SymbolicData,
    SymbolicDataInfo,
    SymbolicVariable,
    SymbolicVariableInfo,
)
from pymc_extras.statespace.core.representation import PytensorRepresentation
from pymc_extras.statespace.filters import (
    KalmanSmoother,
    SquareRootFilter,
    StandardFilter,
    UnivariateFilter,
)
from pymc_extras.statespace.filters.distributions import (
    SequenceMvNormal,
)
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_DIMS,
    JITTER_DEFAULT,
    OBS_STATE_DIM,
)
from pymc_extras.statespace.utils.data_tools import register_data_with_pymc

_log = logging.getLogger("pymc.experimental.statespace")

floatX = pytensor.config.floatX
FILTER_FACTORY = {
    "standard": StandardFilter,
    "univariate": UnivariateFilter,
    "cholesky": SquareRootFilter,
}


def _validate_property(props, property_name, expected_type):
    if isinstance(props, expected_type) or props is None:
        return
    elif not isinstance(props, tuple | list):
        raise TypeError(
            f"The {property_name} property must be a {expected_type.__name__} or a "
            f"list/tuple of {expected_type.__name__} instances."
        )

    if not all(isinstance(prop, expected_type) for prop in props):
        raise TypeError(
            f"All elements of the {property_name} property must be instances of "
            f"{expected_type.__name__}."
        )


class PyMCStateSpace:
    r"""
    Base class for Linear Gaussian Statespace models in PyMC.

    Holds a ``PytensorRepresentation`` and ``KalmanFilter``, and provides a mapping between a PyMC model and the
    statespace model.

    Parameters
    ----------
    k_endog : int
        The number of endogenous variables (observed time series).

    k_states : int
        The number of state variables.

    k_posdef : int
        The number of shocks in the model

    filter_type : str, optional
        The type of Kalman filter to use. Valid options are "standard", "univariate", "single", "cholesky", and
        "steady_state". For more information, see the docs for each filter. Default is "standard".

    verbose : bool, optional
        If True, displays information about the initialized model. Defaults to True.

    measurement_error : bool, optional
        If true, the model contains measurement error. Needed by post-estimation sampling methods to decide how to
        compute the observation errors. If False, these errors are deterministically zero; if True, they are sampled
        from a multivariate normal.

    mode: str or Mode, optional
        Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
        ``forecast``. The mode does **not** effect calls to ``pm.sample``.

        Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
        to all sampling methods.

    cov_jitter: float, optional
        Value added to the diagonal of every covariance matrix at each filtering step, for numerical stability.
        Post-estimation graphs -- ``forecast``, the conditional samplers, ``sample_filter_outputs`` -- are built
        with this same value. Default 1e-8, or 1e-6 if ``pytensor.config.floatX`` is float32.

    missing_fill_value: float, optional
        Sentinel used to mask missing observations, so PyMC's imputation machinery is not triggered. Set this only
        if your data legitimately contains the default sentinel. Post-estimation graphs are built with this same
        value. Default -9999.0.

    Notes
    -----
    Based on the statsmodels statespace implementation https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py,
    described in [1].

    All statespace models inherit from this base class, which is responsible for providing an interface between a
    PyMC model and a PytensorRepresentation of a linear statespace model. This is done via the ``update`` method,
    which takes as input a vector of PyMC random variables and assigns them to their correct positions inside the
    underlying ``PytensorRepresentation``. Construction of the parameter vector, called ``theta``, is done
    automatically, but depend on the names provided in the ``param_names`` property.

    To implement a new statespace model, one needs to:

    1. Overload the ``param_names`` property to return a list of parameter names.
    2. Overload the ``update`` method to put these parameters into their respective statespace matrices

    In addition, a number of additional properties can be overloaded to provide users with additional resources
    when writing their PyMC models. For details, see the attributes section of the docs for this class.

    Finally, this class holds post-estimation methods common to all statespace models, which do not need to be
    overloaded when writing a custom statespace model.

    Examples
    --------
    The local level model is a simple statespace model. It is a Gaussian random walk with a drift term that itself also
    follows a Gaussian random walk, as described by the following two equations:

    .. math::
        \begin{align}
            y_{t} &= y_{t-1} + x_t + \nu_t \tag{1} \\
            x_{t} &= x_{t-1} + \eta_t \tag{2}
        \end{align}

    Where :math:`y_t` is the observed data, and :math:`x_t` is an unobserved trend term. The model has two unknown
    parameters, the variances on the two innovations, :math:`sigma_\nu` and :math:`sigma_\eta`. Take the hidden state
    vector to be :math:`\begin{bmatrix} y_t & x_t \end{bmatrix}^T` and the shock vector
    :math:`\varepsilon_t = \begin{bmatrix} \nu_t & \eta_t \end{bmatrix}^T`. Then this model can be cast into state-space
    form with the following matrices:

    .. math::
        \begin{align}
            T &= \begin{bmatrix}1 & 1 \\ 0 & 1 \end{bmatrix} &
            R &= \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} &
            Q &= \begin{bmatrix} \sigma_\nu & 0 \\ 0 & \sigma_\eta \end{bmatrix} &
            Z &= \begin{bmatrix} 1 & 0 \end{bmatrix}
        \end{align}

    With the remaining statespace matrices as zero matrices of the appropriate sizes. The model has two states,
    two shocks, and one observed state. Knowing all this, a very simple local level model can be implemented as
    follows:

    .. code:: python

        from pymc_extras.statespace.core import PyMCStateSpace
        import numpy as np

        class LocalLevel(PyMCStateSpace):
            def __init__():
                # Initialize the superclass. This creates the PytensorRepresentation and the Kalman Filter
                super().__init__(k_endog=1, k_states=2, k_posdef=2)

                # Declare the non-zero, non-parameterized matrices
                self.ssm['transition', :, :] = np.array([[1.0, 1.0], [0.0, 1.0]])
                self.ssm['selection', :, :] = np.eye(2)
                self.ssm['design', :, :] = np.array([[1.0, 0.0]])

            @property
            def param_names(self):
                return ['x0', 'P0', 'sigma_nu', 'sigma_eta']

            def make_symbolic_graph(self):
                # Declare symbolic variables that represent parameters of the model
                # In this case, we have 4: x0 (initial state), P0 (initial state covariance), sigma_nu, and sigma_eta

                x0 = self.make_and_register_variable('x0', shape=(2,))
                P0 = self.make_and_register_variable('P0', shape=(2,2))
                sigma_mu = self.make_and_register_variable('sigma_nu')
                sigma_eta = self.make_and_register_variable('sigma_eta')

                # Next, use these symbolic variables to build the statespace matrices by assigning each parameter
                # to its correct location in the correct matrix

                self.ssm['initial_state', :] = x0
                self.ssm['initial_state_cov', :, :] = P0
                self.ssm['state_cov', 0, 0] = sigma_nu
                self.ssm['state_cov', 1, 1] = sigma_eta

    After defining priors over the named parameters ``P0``, ``x0``, ``sigma_eta``, and ``sigma_nu``, we can sample
    from this model:

    .. code:: python

        import pymc as pm

        ll = LocalLevel()

        with pm.Model() as mod:
            x0 = pm.Normal('x0', shape=(2,))
            P0_diag = pm.Exponential('P0_diag', 1, shape=(2,))
            P0 = pm.Deterministic('P0', pt.diag(P0_diag))
            sigma_nu = pm.Exponential('sigma_nu', 1)
            sigma_eta = pm.Exponential('sigma_eta', 1)

            ll.build_statespace_graph(data = data)
            idata = pm.sample()


    References
    ----------
    .. [1] Fulton, Chad. "Estimating time series models by state space methods in Python: Statsmodels." (2015).
       http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf

    """

    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int,
        filter_type: str = "standard",
        verbose: bool = True,
        measurement_error: bool = False,
        mode: str | None = None,
        cov_jitter: float = JITTER_DEFAULT,
        missing_fill_value: float | None = None,
    ):
        self._fit_coords: dict[str, Sequence[str]] | None = None
        self._fit_dims: dict[str, Sequence[str]] | None = None
        self._fit_data: pt.TensorVariable | None = None
        self._fit_exog_data: dict[str, dict] = {}
        self._shared_timestep: pt.TensorVariable | None = None

        self._needs_exog_data = None
        self._tensor_variable_info = SymbolicVariableInfo()
        self._tensor_data_info = SymbolicDataInfo()

        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef
        self.measurement_error = measurement_error
        self.mode = mode
        self.cov_jitter = cov_jitter
        self.missing_fill_value = missing_fill_value

        self._populate_properties()

        # Placeholder for time-varying matrices that depend on data length
        self._n_timesteps_placeholder = pt.iscalar("n_timesteps")

        # All models contain a state space representation and a Kalman filter
        self.ssm = PytensorRepresentation(k_endog, k_states, k_posdef)

        # This will be populated with PyMC random matrices after calling _insert_random_variables
        self.subbed_ssm: list[pt.TensorVariable] | None = None

        if filter_type.lower() not in FILTER_FACTORY.keys():
            raise NotImplementedError(
                "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
            )

        if filter_type == "single" and self.k_endog > 1:
            raise ValueError('Cannot use filter_type = "single" with multiple observed time series')

        self.kalman_filter = FILTER_FACTORY[filter_type.lower()]()
        self.kalman_smoother = KalmanSmoother()
        self.make_symbolic_graph()

        self.requirement_table = None
        self._populate_prior_requirements()
        self._populate_data_requirements()

        if verbose and self.requirement_table:
            console = Console()
            console.print(self.requirement_table)

    def _populate_properties(self) -> None:
        self._set_parameters()
        self._set_states()
        self._set_shocks()
        self._set_coords()
        self._set_data_info()

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        """
        Provides parameter metadata to the model.

        Optional. Default implementation sets an empty ParameterInfo.
        Child classes can override to define model parameters.
        """
        return

    def _set_parameters(self) -> None:
        param_list = self.set_parameters()
        self._param_info = ParameterInfo(parameters=param_list)

    def set_states(self) -> State | tuple[State, ...] | None:
        """
        Provide state metadata to the model.

        Optional. Default implementation creates generic state names (state_0, state_1, ...).
        Child classes can override to provide meaningful state names.
        """
        hidden_states = [
            State(name=f"hidden_state_{name}", observed=False) for name in range(self.k_states or 0)
        ]
        observed_states = [
            State(name=f"observed_state_{name}", observed=True) for name in range(self.k_endog or 0)
        ]
        return *hidden_states, *observed_states

    def _set_states(self) -> None:
        states = self.set_states()
        _validate_property(states, "states", State)

        if isinstance(states, State):
            states = (states,)

        self._state_info = StateInfo(states=states)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        """
        Provide shock metadata to the model.

        Optional. Default implementation creates generic shock names (shock_0, shock_1, ...).
        Child classes can override to provide meaningful shock names.
        """
        return tuple(Shock(name=f"shock_{i}") for i in range(self.k_posdef))

    def _set_shocks(self) -> None:
        shocks = self.set_shocks()
        _validate_property(shocks, "shocks", Shock)
        if isinstance(shocks, Shock):
            shocks = (shocks,)
        self._shock_info = ShockInfo(shocks=shocks)

    def default_coords(self) -> tuple[Coord, ...]:
        return CoordInfo.default_coords_from_model(self).items

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        """
        Provide coordinates to the model.

        Optional. Default implementation sets an empty CoordInfo.
        Child classes can override to provide model-specific coordinates.
        """
        return self.default_coords()

    def _set_coords(self) -> None:
        coords = self.set_coords()
        _validate_property(coords, "coords", Coord)
        if isinstance(coords, Coord):
            coords = (coords,)
        self._coords_info = CoordInfo(coords=coords)

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        """
        Provide data_info metadata to the model.

        Optional. Default implementation sets an empty DataInfo.
        Child classes can override if the model requires exogenous data.
        """
        return

    def _set_data_info(self) -> None:
        data_info = self.set_data_info()
        _validate_property(data_info, "data_info", Data)
        if isinstance(data_info, Data):
            data_info = (data_info,)
        self._data_info = DataInfo(data=data_info)

    def _populate_prior_requirements(self) -> None:
        """
        Add requirements about priors needed for the model to a rich table, including their names,
        shapes, named dimensions, and any parameter constraints.
        """
        if not self.param_info:
            return

        if self.requirement_table is None:
            self._initialize_requirement_table()

        for param, info in self.param_info.items():
            self.requirement_table.add_row(
                param, str(info["shape"]), info["constraints"], str(info["dims"])
            )

    def _populate_data_requirements(self) -> None:
        """
        Add requirements about the data needed for the model, including their names, shapes, and named dimensions.
        """
        if not self.data_info:
            return

        if self.requirement_table is None:
            self._initialize_requirement_table()
        else:
            self.requirement_table.add_section()

        for data, info in self.data_info.items():
            self.requirement_table.add_row(data, str(info["shape"]), "pm.Data", str(info["dims"]))

    def _initialize_requirement_table(self) -> None:
        self.requirement_table = Table(
            show_header=True,
            show_edge=True,
            box=SIMPLE_HEAD,
            highlight=True,
        )

        self.requirement_table.title = "Model Requirements"
        self.requirement_table.caption = (
            "These parameters should be assigned priors inside a PyMC model block before "
            "calling the build_statespace_graph method."
        )

        self.requirement_table.add_column("Variable", justify="left")
        self.requirement_table.add_column("Shape", justify="left")
        self.requirement_table.add_column("Constraints", justify="left")
        self.requirement_table.add_column("Dimensions", justify="right")

    def _unpack_statespace_with_placeholders(
        self,
    ) -> tuple[
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
        pt.TensorVariable,
    ]:
        """
        Helper function to quickly obtain all statespace matrices in the standard order. Matrices returned by this
        method will include pytensor placeholders.
        """

        a0 = self.ssm["initial_state"]
        P0 = self.ssm["initial_state_cov"]
        c = self.ssm["state_intercept"]
        d = self.ssm["obs_intercept"]
        T = self.ssm["transition"]
        Z = self.ssm["design"]
        R = self.ssm["selection"]
        H = self.ssm["obs_cov"]
        Q = self.ssm["state_cov"]

        return a0, P0, c, d, T, Z, R, H, Q

    def unpack_statespace(self) -> list[pt.TensorVariable]:
        """
        Helper function to quickly obtain all statespace matrices in the standard order.
        """

        if self.subbed_ssm is None:
            raise ValueError(
                "Cannot unpack the complete statespace system until PyMC model variables have been "
                "inserted. To build the random statespace matrices, call build_statespace_graph() inside"
                "a PyMC model context. "
            )

        return self.subbed_ssm

    @property
    def n_timesteps(self) -> Variable:
        """Symbolic placeholder for the number of time steps in the data."""
        return self._n_timesteps_placeholder

    @property
    def param_names(self) -> tuple[str, ...]:
        """
        Names of model parameters

        A list of all parameters expected by the model. Each parameter will be sought inside the active PyMC model
        context when ``build_statespace_graph`` is invoked.
        """
        return self._param_info.names

    @property
    def data_names(self) -> tuple[str, ...]:
        """
        Names of data variables expected by the model.

        This does not include the observed data series, which is automatically handled by PyMC. This property only
        needs to be implemented for models that expect exogenous data.
        """
        return self._data_info.exogenous_names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        """
        Information about parameters needed to declare priors

        A dictionary of param_name: dictionary key-value pairs. The return value is used by the
        ``_print_prior_requirements`` method, to print a message telling users how to define the necessary priors for
        the model. Each dictionary should have the following key-value pairs:
            * key: "shape", value: a tuple of integers
            * key: "constraints", value: a string describing the support of the prior (positive,
              positive semi-definite, etc)
            * key: "dims", value: tuple of strings
        """
        return self._param_info.to_dict()

    @property
    def data_info(self) -> dict[str, dict[str, Any]]:
        """
        Information about Data variables that need to be declared in the PyMC model block.

        Returns a dictionary of data_name: dictionary of property-name:property description pairs. The return value is
        used by the ``_print_data_requirements`` method, to print a message telling users how to define the necessary
        data for the model. Each dictionary should have the following key-value pairs:
            * key: "shape", value: a tuple of integers
            * key: "dims", value: tuple of strings
        """
        return self._data_info.to_dict()

    @property
    def state_names(self) -> tuple[str, ...]:
        """
        A k_states length list of strings, associated with the model's hidden states

        """
        return self._state_info.unobserved_state_names

    @property
    def observed_states(self) -> tuple[str, ...]:
        """
        A k_endog length list of strings, associated with the model's observed states
        """
        return self._state_info.observed_state_names

    @property
    def shock_names(self) -> tuple[str, ...]:
        """
        A k_posdef length list of strings, associated with the model's shock processes

        """
        return self._shock_info.names

    @property
    def default_priors(self) -> dict[str, Callable]:
        """
        Dictionary of parameter names and callable functions to construct default priors for the model

        Returns a dictionary with param_name: Callable key-value pairs. Used by the ``add_default_priors()`` method
        to automatically add priors to the PyMC model.
        """
        raise NotImplementedError("The default_priors property has not been implemented!")

    @property
    def coords(self) -> dict[str, Sequence[str]]:
        """
        PyMC model coordinates

        Returns a dictionary of dimension: coordinate key-value pairs, to be provided to ``pm.Model``. Dimensions
        should come from the default names defined in ``statespace.utils.constants`` for them to be detected by
        sampling methods.
        """
        return self._coords_info.to_dict()

    @property
    def param_dims(self) -> dict[str, tuple[str, ...]]:
        """
        Dictionary of named dimensions for each model parameter

        Returns a dictionary of param_name: dimension key-value pairs, to be provided to the ``dims`` argument of a
        PyMC random variable. Dimensions should come from the default names defined in ``statespace.utils.constants``
        for them to be detected by sampling methods.

        Note: Scalar parameters (with dims=None) are not included in this dictionary.
        """
        return {param.name: param.dims for param in self._param_info if param.dims is not None}

    @property
    def _name_to_variable(self):
        return self._tensor_variable_info.to_dict()

    @property
    def _name_to_data(self):
        return self._tensor_data_info.to_dict()

    def add_default_priors(self) -> None:
        """
        Add default priors to the active PyMC model context
        """
        raise NotImplementedError("The add_default_priors property has not been implemented!")

    def make_and_register_variable(
        self, name, shape: int | tuple[int, ...] | None = None, dtype=floatX
    ) -> pt.TensorVariable:
        """
        Helper function to create a pytensor symbolic variable and register it in the _name_to_variable dictionary

        Parameters
        ----------
        name : str
            The name of the placeholder variable. Must be the name of a model parameter.
        shape : int or tuple of int
            Shape of the parameter
        dtype : str, default pytensor.config.floatX
            dtype of the parameter

        Notes
        -----
        Symbolic pytensor variables are used in the ``make_symbolic_graph`` method as placeholders for PyMC random
        variables. The change is made in the ``_insert_random_variables`` method via ``pytensor.graph_replace``. To
        make the change, a dictionary mapping pytensor variables to PyMC random variables needs to be constructed.

        The purpose of this method is to:
            1.  Create the placeholder symbolic variables
            2.  Register the placeholder variable in the ``_name_to_variable`` dictionary

        The shape provided here will define the shape of the prior that will need to be provided by the user.

        An error is raised if the provided name has already been registered, or if the name is not present in the
        ``param_names`` property.
        """
        if name not in self.param_names:
            raise ValueError(
                f"{name} is not a model parameter. All placeholder variables should correspond to model "
                f"parameters."
            )

        if name in self._tensor_variable_info:
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._tensor_variable_info[name].type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        tensor_var = SymbolicVariable(name=name, symbolic_variable=placeholder)
        self._tensor_variable_info = self._tensor_variable_info.add(tensor_var)
        return placeholder

    def make_and_register_data(
        self, name: str, shape: int | tuple[int], dtype: str = floatX
    ) -> Variable:
        r"""
        Helper function to create a pytensor symbolic variable and register it in the _name_to_data dictionary

        Parameters
        ----------
        name : str
            The name of the placeholder data. Must be the name of an expected data variable.
        shape : int or tuple of int
            Shape of the parameter
        dtype : str, default pytensor.config.floatX
            dtype of the parameter

        Notes
        -----
        See docstring for make_and_register_variable for more details. This function is similar, but handles data
        inputs instead of model parameters.

        An error is raised if the provided name has already been registered, or if the name is not present in the
        ``data_names`` property.
        """
        if name not in self.data_names:
            raise ValueError(
                f"{name} is not a model parameter. All placeholder variables should correspond to model "
                f"parameters."
            )

        if name in self._tensor_data_info:
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._tensor_data_info[name].type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        tensor_data = SymbolicData(name=name, symbolic_data=placeholder)
        self._tensor_data_info = self._tensor_data_info.add(tensor_data)
        return placeholder

    def make_symbolic_graph(self) -> None:
        """
        The purpose of the make_symbolic_graph function is to hide tedious parameter allocations from the user.
        In statespace models, it is extremely rare for an entire matrix to be defined by a single prior distribution.
        Instead, users expect to place priors over single entries of the matrix. The purpose of this function is to
        meet that expectation.

        Every statespace model needs to implement this function.

        Examples
        --------
        As an example, consider an ARMA(2,2) model, which has five parameters (excluding the initial state distribution):
        2 AR parameters (:math:`\rho_1` and :math:`\rho_2`), 2 MA parameters (:math:`\theta_1` and :math:`theta_2`),
        and a single innovation covariance (:math:`\\sigma`). A common way of writing this statespace is:

        ..math::

            \begin{align}
                T &= \begin{bmatrix} \rho_1 & 1 & 0 \\
                                     \rho_2 & 0 & 1 \\
                                     0      & 0 & 0
                      \\end{bmatrix} \\
                R & = \begin{bmatrix} 1 \\ \theta_1 \\ \theta_2 \\end{bmatrix} \\
                Q &= \begin{bmatrix} \\sigma \\end{bmatrix}
            \\end{align}

        To implement this model, we begin by creating the required matrices, and fill in the "fixed" values -- the ones
        at position (0, 1) and (0, 2) in the T matrix, and at position (0, 0) in the R matrix. These are then saved
        to the class's PytensorRepresentation -- called ``ssm``.

        .. code:: python

            T = np.eye(2, k=1)
            R = np.concatenate([np.ones(1,1), np.zeros((2, 1))], axis=0)

            self.ssm['transition'] = T
            self.ssm['selection'] = R

        Next, placeholders need to be inserted for the random variables rho_1, rho_2, theta_1, theta_2, and sigma.
        This can be done many ways: we could define two vectors, rho and theta, and a scalar for sigma, or five
        scalars. Whatever is chosen, the choice needs to be consistent with the ``param_names`` property.

        Suppose the ``param_names`` are ``[rho, theta, sigma]``, then we make one placeholder for each, and insert it
        into the correct ``ssm`` matrix, at the correct location. To create placeholders, use the
        ``make_and_register_variable`` helper method, which will maintain an internal registry of variables.

        .. code:: python
            rho_parmas = self.make_and_register_variable(name='rho', shape=(2,))
            theta_params = self.make_and_register_variable(name='theta', shape=(2,))
            sigma = self.make_and_register_variable(name='sigma', shape=(1,))

            self.ssm['transition', :, 0] = rho_params
            self.ssm['selection', 1:, 0] = theta_params
            self.ssm['state_cov', 0, 0] = sigma
        """
        raise NotImplementedError("The make_symbolic_statespace method has not been implemented!")

    def _save_exogenous_data_info(self):
        """
        Store exogenous data required by posterior sampling functions
        """
        pymc_mod = modelcontext(None)
        for data_name in self.data_names:
            data = pymc_mod[data_name]
            self._fit_exog_data[data_name] = {
                "name": data_name,
                "value": data.get_value(),
                "dims": pymc_mod.named_vars_to_dims.get(data_name, None),
            }

    def _insert_random_variables(self):
        """
        Replace pytensor symbolic variables with PyMC random variables.

        Examples
        --------
        .. code:: python

            ss_mod = pmss.BayesianSARIMAX(order=(2, 0, 2), verbose=False, stationary_initialization=True)
            with pm.Model():
                 x0 = pm.Normal('x0', size=ss_mod.k_states)
                 ar_params = pm.Normal('ar_params', size=ss_mod.p)
                 ma_parama = pm.Normal('ma_params', size=ss_mod.q)
                 sigma_state = pm.Normal('sigma_state')

                 ss_mod._insert_random_variables()
                 matrics = ss_mod.unpack_statespace()

            pm.draw(matrices['transition'], random_seed=RANDOM_SEED)
            >>> array([[-0.90590386,  1.        ,  0.        ],
            >>>        [ 1.25190143,  0.        ,  1.        ],
            >>>        [ 0.        ,  0.        ,  0.        ]])

            pm.draw(matrices['selection'], random_seed=RANDOM_SEED)
            >>> array([[ 1.        ],
            >>>        [-2.46741039],
            >>>        [-0.28947689]])

            pm.draw(matrices['state_cov'], random_seed=RANDOM_SEED)
            >>> array([[-1.69353533]])
        """

        pymc_model = modelcontext(None)
        found_params = []
        with pymc_model:
            for param_name in self.param_names:
                param = getattr(pymc_model, param_name, None)
                if param is not None:
                    found_params.append(param.name)

        missing_params = list(set(self.param_names) - set(found_params))
        if len(missing_params) > 0:
            raise ValueError(
                "The following required model parameters were not found in the PyMC model: "
                + ", ".join(missing_params)
            )

        excess_params = list(set(found_params) - set(self.param_names))
        if len(excess_params) > 0:
            raise ValueError(
                "The following parameters were found in the PyMC model but are not required by the statespace model: "
                + ", ".join(excess_params)
            )

        matrices = list(self._unpack_statespace_with_placeholders())

        replacement_dict = {var: pymc_model[name] for name, var in self._name_to_variable.items()}
        self.subbed_ssm = graph_replace(matrices, replace=replacement_dict, strict=True)

    def _insert_data_variables(self):
        """
        Replace symbolic pytensor variables with PyMC data containers.

        Only used when models require exogenous data. The observed data is not added to the model using this method!
        """
        data_names = self.data_names
        if not data_names:
            return

        pymc_model = modelcontext(None)
        found_data = []
        with pymc_model:
            for data_name in data_names:
                data = getattr(pymc_model, data_name, None)
                if data is not None:
                    found_data.append(data.name)

        missing_data = list(set(data_names) - set(found_data))
        if len(missing_data) > 0:
            raise ValueError(
                "The following required exogenous data were not found in the PyMC model: "
                + ", ".join(missing_data)
            )

        replacement_dict = {data: pymc_model[name] for name, data in self._name_to_data.items()}
        self.subbed_ssm = graph_replace(self.subbed_ssm, replace=replacement_dict, strict=True)

    def _insert_data_shape_into_n_timesteps(self, data):
        """
        Replace any occurrence of the n_timesteps symbolic variable with the length of the data.

        n_timesteps is a special symbolic variable used by graphs with time-varying matrices, whose shapes won't be
        known until the user provides data. We need to collect them and replace them with a single common shared
        variable to define all shapes consistently.
        """
        # This method should only be called after data has been ingested and transformed into a pytensor variable.
        # Otherwise, we don't get symbolic linkage between time-varying matrix shapes and the data when the user calls
        # pm.set_data
        assert isinstance(data, pt.TensorVariable)
        matrices = (
            self.subbed_ssm
            if self.subbed_ssm is not None
            else self._unpack_statespace_with_placeholders()
        )

        n_timestep_variables = tuple(
            variable
            for variable in explicit_graph_inputs(matrices)
            if variable.name == "n_timesteps"
        )

        if n_timestep_variables:
            self._shared_timestep = data.shape[0].astype("int32")
            replacement_dict = {var: self._shared_timestep for var in n_timestep_variables}

            self.subbed_ssm = graph_replace(self.subbed_ssm, replace=replacement_dict, strict=False)

    def _insert_constant_timestep(self, matrices, step: int | pt.TensorVariable):
        """
        Replace any occurrence of the n_timesteps symbolic variable with a constant integer.

        This is used for constructing graphs for prior predictive sampling, where no data is available.
        """
        step = pt.as_tensor_variable(step).astype("int32")

        n_timestep_variables = tuple(
            variable
            for variable in explicit_graph_inputs(matrices)
            if variable.name == "n_timesteps"
        )

        if not n_timestep_variables:
            return matrices

        replacement_dict = {var: step for var in n_timestep_variables}
        return graph_replace(matrices, replace=replacement_dict, strict=False)

    @staticmethod
    def _register_kalman_filter_outputs_with_pymc_model(outputs: tuple[pt.TensorVariable]) -> None:
        mod = modelcontext(None)
        coords = mod.coords

        states, covs = outputs[:4], outputs[4:]

        state_names = [
            "filtered_states",
            "predicted_states",
            "predicted_observed_states",
            "smoothed_states",
        ]
        cov_names = [
            "filtered_covariances",
            "predicted_covariances",
            "predicted_observed_covariances",
            "smoothed_covariances",
        ]

        with mod:
            for var, name in zip(states + covs, state_names + cov_names):
                dim_names = FILTER_OUTPUT_DIMS.get(name, None)
                dims = tuple([dim if dim in coords.keys() else None for dim in dim_names])
                pm.Deterministic(name, var, dims=dims)

    def build_statespace_graph(
        self,
        data: np.ndarray | pd.DataFrame | pt.TensorVariable,
        register_data: bool = True,
        missing_fill_value: float | None = None,
        cov_jitter: float | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        save_kalman_filter_outputs_in_idata: bool = False,
        mode: str | None = None,
    ) -> None:
        """
        Given a parameter vector `theta`, constructs the full computational graph describing the state space model and
        the associated log probability of the data. Hidden states and log probabilities are computed via the Kalman
        Filter.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame, pt.TensorVariable]
            The observed data used to fit the state space model. It can be a NumPy array, a Pandas DataFrame,
            or a Pytensor tensor variable.

        register_data : bool, optional, default=True
            If True, the observed data will be registered with PyMC as a pm.Data variable. In addition,
            a "time" dim will be created an added to the model's coords.

        mode : Optional[str], optional, default=None
            The Pytensor mode used for the computation graph construction. If None, the default mode will be used.
            Other options include "JAX" and "NUMBA".

        missing_fill_value: float, optional
            A value to mask in missing values. NaN values in the data need to be filled with an arbitrary value to
            avoid triggering PyMC's automatic imputation machinery (missing values are instead filled by treating them
            as hidden states during Kalman filtering).

            In general this never needs to be set. But if by a wild coincidence your data includes the value -9999.0,
            you will need to change the missing_fill_value to something else, to avoid incorrectly mark in
            data as missing.

            Overrides the value given to the model constructor, which is also what post-estimation graphs use.
            Default None, meaning the constructor's value is kept.

        cov_jitter: float, optional
            The Kalman filter is known to be numerically unstable, especially at half precision. This value is added to
            the diagonal of every covariance matrix -- predicted, filtered, and smoothed -- at every step, to ensure
            all matrices are strictly positive semi-definite.

            Obviously, if this can be zero, that's best. In general:
                - Having measurement error makes Kalman Filters more robust. A large source of numerical errors come
                  from the Filtered and Smoothed covariance matrices having a zero in the (0, 0) position, which always
                  occurs when there is no measurement error. You can lower this value in the presence of measurement
                  error.

                - The Univariate Filter is more robust than other filters, and can tolerate a lower jitter value

            Overrides the value given to the model constructor, which is also what post-estimation graphs use.
            Default None, meaning the constructor's value is kept.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        save_kalman_filter_outputs_in_idata: bool, optional, default=False
            If True, Kalman Filter outputs will be saved in the model as deterministics. Useful for debugging, but
            should not be necessary for the majority of users.

        mode: str, optional
            Pytensor mode to use when compiling the graph. This will be saved as a model attribute and used when
            compiling sampling functions (e.g. ``sample_conditional_prior``).

            .. deprecated:: 0.2.5
                The `mode` argument is deprecated and will be removed in a future version. Pass ``mode`` to the
                model constructor, or manually specify ``compile_kwargs`` in sampling functions instead.
        """
        if mode is not None:
            warnings.warn(
                "The `mode` argument is deprecated and will be removed in a future version. "
                "Pass `mode` to the model constructor, or manually specify `compile_kwargs` in sampling functions"
                " instead.",
                DeprecationWarning,
            )
            self.mode = mode

        # Recorded on the model so post-estimation graphs are filtered the same way the fit was.
        if cov_jitter is not None:
            self.cov_jitter = cov_jitter
        if missing_fill_value is not None:
            self.missing_fill_value = missing_fill_value

        pm_mod = modelcontext(None)

        self._insert_random_variables()
        self._save_exogenous_data_info()
        self._insert_data_variables()

        obs_coords = pm_mod.coords.get(OBS_STATE_DIM, None)
        self._fit_data = data

        data, nan_mask = register_data_with_pymc(
            data,
            n_obs=self.ssm.k_endog,
            obs_coords=obs_coords,
            register_data=register_data,
            missing_fill_value=self.missing_fill_value,
        )

        # Order is important here: only call _insert_data_shape_into_n_timesteps after data has been registered.
        self._insert_data_shape_into_n_timesteps(data)

        filter_outputs = self.kalman_filter.build_graph(
            pt.as_tensor_variable(data),
            *self.unpack_statespace(),
            missing_fill_value=self.missing_fill_value,
            cov_jitter=self.cov_jitter,
            time_varying_names=self.ssm.time_varying_names,
        )

        logp = filter_outputs.pop(-1)
        states, covs = filter_outputs[:3], filter_outputs[3:]
        filtered_states, predicted_states, observed_states = states
        filtered_covariances, predicted_covariances, observed_covariances = covs
        if save_kalman_filter_outputs_in_idata:
            smooth_states, smooth_covariances = self._build_smoother_graph(
                filtered_states, filtered_covariances, self.unpack_statespace()
            )
            all_kf_outputs = [*states, smooth_states, *covs, smooth_covariances]
            self._register_kalman_filter_outputs_with_pymc_model(all_kf_outputs)

        obs_dims = FILTER_OUTPUT_DIMS["predicted_observed_states"]
        obs_dims = obs_dims if all([dim in pm_mod.coords.keys() for dim in obs_dims]) else None

        SequenceMvNormal(
            "obs",
            mus=observed_states,
            covs=observed_covariances,
            logp=logp,
            observed=data,
            dims=obs_dims,
            method=mvn_method,
        )

        self._fit_coords = pm_mod.coords.copy()
        self._fit_dims = pm_mod.named_vars_to_dims.copy()

    def _build_smoother_graph(
        self,
        filtered_states: pt.TensorVariable,
        filtered_covariances: pt.TensorVariable,
        matrices,
        mode: str | None = None,
        cov_jitter=JITTER_DEFAULT,
    ):
        """
        Build the computation graph for the Kalman smoother.

        This method constructs the computation graph for applying the Kalman smoother to the filtered states
        and covariances obtained from the Kalman filter. The Kalman smoother is used to generate smoothed
        estimates of the latent states and their covariances in a state space model.

        The Kalman smoother provides a more accurate estimate of the latent states by incorporating future
        information in the backward pass, resulting in smoothed state trajectories.

        Parameters
        ----------
        filtered_states : pytensor.tensor.TensorVariable
            The filtered states obtained from the Kalman filter. Returned by the `build_statespace_graph` method.

        filtered_covariances : pytensor.tensor.TensorVariable
            The filtered state covariances obtained from the Kalman filter. Returned by the `build_statespace_graph`
            method.

        mode : Optional[str], default=None
            The mode used by pytensor for the construction of the logp graph. If None, the mode provided to
            `build_statespace_graph` will be used.

        Returns
        -------
        Tuple[pytensor.tensor.TensorVariable, pytensor.tensor.TensorVariable]
            A tuple containing TensorVariables representing the smoothed states and smoothed state covariances
            obtained from the Kalman smoother.
        """

        pymc_model = modelcontext(None)
        with pymc_model:
            *_, T, Z, R, H, Q = matrices

            smooth_states, smooth_covariances = self.kalman_smoother.build_graph(
                T,
                R,
                Q,
                filtered_states,
                filtered_covariances,
                cov_jitter=self.cov_jitter,
                time_varying_names=self.ssm.time_varying_names,
            )
            smooth_states.name = "smooth_states"
            smooth_covariances.name = "smooth_covariances"

            return smooth_states, smooth_covariances

    def _build_dummy_graph(self) -> None:
        return dummy_graph.build_dummy_graph(self)

    def _kalman_filter_outputs_from_dummy_graph(
        self,
        data: pt.TensorLike | None = None,
        data_dims: str | tuple[str] | list[str] | None = None,
        scenario: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
    ) -> tuple[list[pt.TensorVariable], list[tuple[pt.TensorVariable, pt.TensorVariable]]]:
        return dummy_graph.kalman_filter_outputs_from_dummy_graph(
            self, data=data, data_dims=data_dims, scenario=scenario
        )

    def sample_conditional_prior(
        self,
        idata: DataTree,
        random_seed: RandomState | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ) -> DataTree:
        """
        Sample from the conditional prior; that is, given parameter draws from the prior distribution,
        compute Kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the Kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata : DataTree
            DataTree with prior samples for state space matrices x0, P0, c, d, T, Z, R, H, Q.
            Obtained from `pm.sample_prior_predictive` after calling PyMCStateSpace.build_statespace_graph().

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        DataTree
            A DataTree object containing sampled trajectories from the conditional prior.
            The trajectories are stored in the posterior_predictive group with names "filtered_prior",
             "predicted_prior", and "smoothed_prior".
        """

        return forward_sampling.sample_conditional_prior(
            self, idata=idata, random_seed=random_seed, mvn_method=mvn_method, **kwargs
        )

    def sample_conditional_posterior(
        self,
        idata: DataTree,
        random_seed: RandomState | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ):
        """
        Sample from the conditional posterior; that is, given parameter draws from the posterior distribution,
        compute Kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the Kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata : DataTree
            A DataTree object containing the posterior distribution over model parameters.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        DataTree
            A DataTree object containing sampled trajectories from the conditional posterior.
            The trajectories are stored in the posterior_predictive group with names "filtered_posterior",
             "predicted_posterior", and "smoothed_posterior".
        """

        return forward_sampling.sample_conditional_posterior(
            self, idata=idata, random_seed=random_seed, mvn_method=mvn_method, **kwargs
        )

    def sample_unconditional_prior(
        self,
        idata: DataTree,
        steps: int | None = None,
        use_data_time_dim: bool = False,
        random_seed: RandomState | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ) -> DataTree:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the prior
        distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        idata: DataTree
            DataTree with prior samples for state space matrices x0, P0, c, d, T, Z, R, H, Q.
            Obtained from `pm.sample_prior_predictive` after calling PyMCStateSpace.build_statespace_graph().

        steps : Optional[int], default=None
            The number of time steps to sample for the unconditional trajectories. If not provided (None),
            the function will sample trajectories for the entire available time dimension in the posterior.
            Otherwise, it will generate trajectories for the specified number of steps.

        use_data_time_dim : bool, default=False
            If True, the function uses the time dimension present in the provided `idata` object to sample
            unconditional trajectories. If False, a custom time dimension is created based on the number of steps
            specified, or if steps is None, it uses the entire available time dimension in the posterior.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        DataTree
            An Arviz InfereceData with two data variables, prior_latent and prior_observed

            - prior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
              `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

            - prior_observed represents the observed state trajectories `Y[t]`, which is obtained from
              the observation equation: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
        """

        return forward_sampling.sample_unconditional_prior(
            self,
            idata=idata,
            steps=steps,
            use_data_time_dim=use_data_time_dim,
            random_seed=random_seed,
            mvn_method=mvn_method,
            **kwargs,
        )

    def sample_unconditional_posterior(
        self,
        idata: DataTree,
        steps: int | None = None,
        use_data_time_dim: bool = False,
        random_seed: RandomState | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ) -> DataTree:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        posterior distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)
            x[0] ~ N(a0, P0)

        Parameters
        ----------
        idata : DataTree
            A DataTree object with a posterior group containing samples from the
            posterior distribution over model parameters.

        steps : Optional[int], default=None
            The number of time steps to sample for the unconditional trajectories. If not provided (None),
            the function will sample trajectories for the entire available time dimension in the posterior.
            Otherwise, it will generate trajectories for the specified number of steps.

        use_data_time_dim : bool, default=False
            If True, the function uses the time dimension present in the provided `idata` object to sample
            unconditional trajectories. If False, a custom time dimension is created based on the number of steps
            specified, or if steps is None, it uses the entire available time dimension in the posterior.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        Returns
        -------
        DataTree
            An Arviz InfereceData with two groups, posterior_latent and posterior_observed

            - posterior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
              `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

            - posterior_observed represents the observed state trajectories `Y[t]`, which is obtained from
              the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
        """

        return forward_sampling.sample_unconditional_posterior(
            self,
            idata=idata,
            steps=steps,
            use_data_time_dim=use_data_time_dim,
            random_seed=random_seed,
            mvn_method=mvn_method,
            **kwargs,
        )

    def sample_statespace_matrices(
        self, idata, matrix_names: str | list[str] | None, group: str = "posterior", **kwargs
    ):
        """
        Draw samples of requested statespace matrices from provided idata

        Parameters
        ----------
        matrix_names: str, list[str], optional
            Statespace matrices to be sampled. Valid names are short names: x0, P0, c, d, T, Z, R, H, Q, or
             "formal" names: initial_state, initial_state_cov, state_intercept, obs_intercept, transition, design,
                             selection, obs_cov, state_cov
        idata: DataTree
            DataTree from which to sample

        group: str, one of "posterior" or "prior"
            Whether to sample from priors or posteriors

        kwargs:
            Additional keyword arguments are passed to ``pymc.sample_posterior_predictive``

        Returns
        -------
        idata_matrices: az.InterenceData
        """
        return forward_sampling.sample_statespace_matrices(
            self, idata, matrix_names=matrix_names, group=group, **kwargs
        )

    def sample_filter_outputs(
        self,
        idata,
        filter_output_names: str | list[str] | None = None,
        group: str = "posterior",
        **kwargs,
    ):
        return forward_sampling.sample_filter_outputs(
            self,
            idata,
            filter_output_names=filter_output_names,
            group=group,
            **kwargs,
        )

    @staticmethod
    def _validate_forecast_args(
        time_index: pd.RangeIndex | pd.DatetimeIndex,
        start: int | pd.Timestamp,
        periods: int | None = None,
        end: int | pd.Timestamp = None,
        scenario: pd.DataFrame | np.ndarray | None = None,
        use_scenario_index: bool = False,
        verbose: bool = True,
    ):
        return forecast._validate_forecast_args(
            time_index,
            start,
            periods=periods,
            end=end,
            scenario=scenario,
            use_scenario_index=use_scenario_index,
            verbose=verbose,
        )

    def _get_fit_time_index(self) -> pd.RangeIndex | pd.DatetimeIndex:
        return forecast._get_fit_time_index(self)

    def _validate_scenario_data(
        self,
        scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None,
        name: str | None = None,
        verbose=True,
    ):
        return forecast._validate_scenario_data(self, scenario, name=name, verbose=verbose)

    @staticmethod
    def _build_forecast_index(
        time_index: pd.RangeIndex | pd.DatetimeIndex,
        start: int | pd.Timestamp | None = None,
        end: int | pd.Timestamp = None,
        periods: int | None = None,
        use_scenario_index: bool = False,
        scenario: pd.DataFrame | np.ndarray | None = None,
    ) -> tuple[int | pd.Timestamp, pd.RangeIndex | pd.DatetimeIndex]:
        return forecast._build_forecast_index(
            time_index,
            start=start,
            end=end,
            periods=periods,
            use_scenario_index=use_scenario_index,
            scenario=scenario,
        )

    def _finalize_scenario_initialization(
        self,
        scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None,
        forecast_index: pd.RangeIndex | pd.DatetimeIndex,
        name=None,
    ):
        return forecast._finalize_scenario_initialization(self, scenario, forecast_index, name=name)

    def _build_forecast_model(
        self, time_index, t0, forecast_index, scenario, filter_output, mvn_method
    ):
        return forecast._build_forecast_model(
            self, time_index, t0, forecast_index, scenario, filter_output, mvn_method
        )

    def forecast(
        self,
        idata: DataTree,
        start: int | pd.Timestamp | None = None,
        periods: int | None = None,
        end: int | pd.Timestamp = None,
        scenario: pd.DataFrame | np.ndarray | dict[str, pd.DataFrame | np.ndarray] | None = None,
        use_scenario_index: bool = False,
        filter_output="smoothed",
        random_seed: RandomState | None = None,
        verbose: bool = True,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ) -> DataTree:
        """
        Generate forecasts of state space model trajectories into the future.

        This function combines posterior parameter samples in the provided idata with model dynamics to generate
        forecasts for out-of-sample data. The trajectory is initialized using the filter output specified in
        the filter_output argument.

        Parameters
        ----------
        idata : DataTree
            A DataTree object containing the posterior distribution over model parameters.

        start : int or pd.Timestamp, optional
            The starting date index or time step from which to generate the forecasts. If the data provided to
            `PyMCStateSpace.build_statespace_graph` had a datetime index, `start` should be a datetime.
            If using integer time series, `start` should be an integer indicating the starting time step. In either
            case, `start` should be in the data index used to build the statespace graph.

            If start is None, the last value on the data's index will be used.

        periods : int, optional
            The number of time steps to forecast into the future. If `periods` is specified, the `end`
            parameter will be ignored. If `None`, then the `end` parameter must be provided.

        end : int or pd.Timestamp, optional
            The ending date index or time step up to which to generate the forecasts. If the data provided to
            `PyMCStateSpace.build_statespace_graph` had a datetime index, `start` should be a datetime.
            If using integer time series, `end` should be an integer indicating the ending time step.
            If `end` is provided, the `periods` parameter will be ignored.

        scenario: pd.Dataframe or np.ndarray, optional
            Exogenous variables to use for scenario-based forecasting. Must be a 2d array-like, with second dimension
            equal to the number of exogenous variables. If start, end, or periods are specified, the first dimension
            must conform with these settings. Otherwise, the index of the scenario data will be used to set the
            number of forecast steps. If the index of the forecast scenairo is a pandas DateTimeIndex, its frequency
            must match the frequency of the data used to fit the model. Otherwise, dates will be based on the number
            of forecast steps and the data.

        use_scenario_index: bool, default False
            If True, the index of the scenario data will be used to determine the forecast period. In this case,
            the start, end, and periods arguments will be ignored. If True, the scenario data must be a DataFrame,
            otherwise an error will be raised.

        filter_output : str, default="smoothed"
            The type of Kalman Filter output used to initialize the forecasts. The 0th timestep of the forecast will
            be sampled from x[0] ~ N(filter_output_mean[start], filter_output_covariance[start]). Default is "smoothed",
            which uses past and future data to make the best possible hidden state estimate.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        verbose: bool, default=True
            Whether to print diagnostic information about forecasting.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        DataTree
            A DataTree object containing forecast samples for the latent and observed state
            trajectories of the state space model, named  "forecast_latent" and "forecast_observed".

                - forecast_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
                  `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

                - forecast_observed represents the observed state trajectories `Y[t]`, which is obtained from
                  the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.

        """
        return forecast.forecast(
            self,
            idata,
            start=start,
            periods=periods,
            end=end,
            scenario=scenario,
            use_scenario_index=use_scenario_index,
            filter_output=filter_output,
            random_seed=random_seed,
            verbose=verbose,
            mvn_method=mvn_method,
            **kwargs,
        )

    def impulse_response_function(
        self,
        idata,
        n_steps: int = 40,
        use_posterior_cov: bool = True,
        shock_size: float | np.ndarray | None = None,
        shock_cov: np.ndarray | None = None,
        shock_trajectory: np.ndarray | None = None,
        orthogonalize_shocks: bool = False,
        random_seed: RandomState | None = None,
        mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
        **kwargs,
    ):
        """
        Generate impulse response functions (IRF) from state space model dynamics.

        An impulse response function represents the dynamic response of the state space model
        to an instantaneous shock applied to the system. This function calculates the IRF
        based on either provided shock specifications or the posterior state covariance matrix.

        Parameters
        ----------
        idata : DataTree
            A DataTree object containing the posterior distribution over model parameters.

        n_steps: int
            The number of time steps to calculate the impulse response. Default is 40.

            If `shock_trajectory` is provided, the length of the shock trajectory will override this value.

        use_posterior_cov: bool, default=True
            Whether to use the covariance matrix of the posterior distribution to generate the impulse response.

            Only one of `use_posterior_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.

        shock_size : Optional[Union[float, np.ndarray]], default=None
            The size of the shock applied to the system. If specified, it will create a covariance
            matrix for the shock with diagonal elements equal to `shock_size`. If float, the diagonal will be filled
            with `shock_size`. If an array, `shock_size` must match the number of shocks in the state space model.

            Only one of `use_posterior_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.

        shock_cov : Optional[np.ndarray], default=None
            A user-specified covariance matrix for the shocks. It should be a 2D numpy array with
            dimensions (n_shocks, n_shocks), where n_shocks is the number of shocks in the state space model.

            Only one of `use_posterior_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.

        shock_trajectory : Optional[np.ndarray], default=None
            A pre-defined trajectory of shocks applied to the system. It should be a 2D numpy array
            with dimensions (n, n_shocks), where n is the number of time steps and k_posdef is the
            number of shocks in the state space model.

            Only one of `use_posterior_cov`, `shock_cov`, `shock_size`, or `shock_trajectory` can be specified.

        orthogonalize_shocks : bool, default=False
            If True, orthogonalize the shocks using Cholesky decomposition when generating the impulse
            response. This option is ignored if `shock_trajectory` or `shock_size` are used.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        mvn_method: str, default "svd"
            Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
            (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
            to ill-conditioned matrices, while "svd" is slow but extremely robust.

            In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
            recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        DataTree
            A DataTree object containing impulse response function in a variable named "irf".

        Notes
        -----
        For models with time-varying transition matrices, the IRF is computed starting at phase 0 of the
        time-varying cycle. This means the response represents the effect of a shock occurring at the first
        modeled state, T(0).
        """
        return irf.impulse_response_function(
            self,
            idata,
            n_steps=n_steps,
            use_posterior_cov=use_posterior_cov,
            shock_size=shock_size,
            shock_cov=shock_cov,
            shock_trajectory=shock_trajectory,
            orthogonalize_shocks=orthogonalize_shocks,
            random_seed=random_seed,
            mvn_method=mvn_method,
            **kwargs,
        )
