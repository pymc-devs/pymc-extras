import functools as ft
import logging

from abc import ABC
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr

from pytensor import Variable
from pytensor.compile.mode import Mode

from pymc_extras.statespace.core import PytensorRepresentation
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models.utilities import (
    conform_time_varying_and_time_invariant_matrices,
    make_default_coords,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    LONG_MATRIX_NAMES,
    POSITION_DERIVATIVE_NAMES,
    TIME_DIM,
)

_log = logging.getLogger("pymc.experimental.statespace")

floatX = pytensor.config.floatX


def order_to_mask(order):
    if isinstance(order, int):
        return np.ones(order).astype(bool)
    else:
        return np.array(order).astype(bool)


def _frequency_transition_block(s, j):
    lam = 2 * np.pi * j / s

    return pt.stack([[pt.cos(lam), pt.sin(lam)], [-pt.sin(lam), pt.cos(lam)]])


class StructuralTimeSeries(PyMCStateSpace):
    r"""
    Structural Time Series Model

    The structural time series model, named by [1] and presented in statespace form in [2], is a framework for
    decomposing a univariate time series into level, trend, seasonal, and cycle components. It also admits the
    possibility of exogenous regressors. Unlike the SARIMAX framework, the time series is not assumed to be stationary.

    Notes
    -----

    .. math::
         y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    """

    def __init__(
        self,
        ssm: PytensorRepresentation,
        state_names: list[str],
        data_names: list[str],
        shock_names: list[str],
        param_names: list[str],
        exog_names: list[str],
        param_dims: dict[str, tuple[int]],
        coords: dict[str, Sequence],
        param_info: dict[str, dict[str, Any]],
        data_info: dict[str, dict[str, Any]],
        component_info: dict[str, dict[str, Any]],
        measurement_error: bool,
        name_to_variable: dict[str, Variable],
        name_to_data: dict[str, Variable] | None = None,
        name: str | None = None,
        verbose: bool = True,
        filter_type: str = "standard",
        mode: str | Mode | None = None,
    ):
        # Add the initial state covariance to the parameters
        if name is None:
            name = "data"
        self._name = name

        k_states, k_posdef, k_endog = ssm.k_states, ssm.k_posdef, ssm.k_endog
        param_names, param_dims, param_info = self._add_inital_state_cov_to_properties(
            param_names, param_dims, param_info, k_states
        )
        self._state_names = state_names.copy()
        self._data_names = data_names.copy()
        self._shock_names = shock_names.copy()
        self._param_names = param_names.copy()
        self._param_dims = param_dims.copy()

        default_coords = make_default_coords(self)
        coords.update(default_coords)

        self._coords = coords
        self._param_info = param_info.copy()
        self._data_info = data_info.copy()
        self.measurement_error = measurement_error

        super().__init__(
            k_endog,
            k_states,
            max(1, k_posdef),
            filter_type=filter_type,
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
        )
        self.ssm = ssm.copy()

        if k_posdef == 0:
            # If there is no randomness in the model, add dummy matrices to the representation to avoid errors
            # when we go to construct random variables from the matrices
            self.ssm.k_posdef = self.k_posdef
            self.ssm.shapes["state_cov"] = (1, 1, 1)
            self.ssm["state_cov"] = pt.zeros((1, 1, 1))

            self.ssm.shapes["selection"] = (1, self.k_states, 1)
            self.ssm["selection"] = pt.zeros((1, self.k_states, 1))

        self._component_info = component_info.copy()

        self._name_to_variable = name_to_variable.copy()
        self._name_to_data = name_to_data.copy()

        self._exog_names = exog_names.copy()
        self._needs_exog_data = len(exog_names) > 0

        P0 = self.make_and_register_variable("P0", shape=(self.k_states, self.k_states))
        self.ssm["initial_state_cov"] = P0

    @staticmethod
    def _add_inital_state_cov_to_properties(param_names, param_dims, param_info, k_states):
        param_names += ["P0"]
        param_dims["P0"] = (ALL_STATE_DIM, ALL_STATE_AUX_DIM)
        param_info["P0"] = {
            "shape": (k_states, k_states),
            "constraints": "Positive semi-definite",
            "dims": param_dims["P0"],
        }

        return param_names, param_dims, param_info

    @property
    def param_names(self):
        return self._param_names

    @property
    def data_names(self) -> list[str]:
        return self._data_names

    @property
    def state_names(self):
        return self._state_names

    @property
    def observed_states(self):
        return [self._name]

    @property
    def shock_names(self):
        return self._shock_names

    @property
    def param_dims(self):
        return self._param_dims

    @property
    def coords(self) -> dict[str, Sequence]:
        return self._coords

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        return self._param_info

    @property
    def data_info(self) -> dict[str, dict[str, Any]]:
        return self._data_info

    def make_symbolic_graph(self) -> None:
        """
        Assign placeholder pytensor variables among statespace matrices in positions where PyMC variables will go.

        Notes
        -----
        This assignment is handled by the components, so this function is implemented only to avoid the
        NotImplementedError raised by the base class.
        """

        pass

    def _state_slices_from_info(self):
        info = self._component_info.copy()
        comp_states = np.cumsum([0] + [info["k_states"] for info in info.values()])
        state_slices = [slice(i, j) for i, j in pairwise(comp_states)]

        return state_slices

    def _hidden_states_from_data(self, data):
        state_slices = self._state_slices_from_info()
        info = self._component_info
        names = info.keys()
        result = []

        for i, (name, s) in enumerate(zip(names, state_slices)):
            obs_idx = info[name]["obs_state_idx"]
            if obs_idx is None:
                continue

            X = data[..., s]
            if info[name]["combine_hidden_states"]:
                sum_idx = np.flatnonzero(obs_idx)
                result.append(X[..., sum_idx].sum(axis=-1)[..., None])
            else:
                comp_names = self.state_names[s]
                for j, state_name in enumerate(comp_names):
                    result.append(X[..., j, None])

        return np.concatenate(result, axis=-1)

    def _get_subcomponent_names(self):
        state_slices = self._state_slices_from_info()
        info = self._component_info
        names = info.keys()
        result = []

        for i, (name, s) in enumerate(zip(names, state_slices)):
            if info[name]["combine_hidden_states"]:
                result.append(name)
            else:
                comp_names = self.state_names[s]
                result.extend([f"{name}[{comp_name}]" for comp_name in comp_names])
        return result

    def extract_components_from_idata(self, idata: xr.Dataset) -> xr.Dataset:
        r"""
        Extract interpretable hidden states from an InferenceData returned by a PyMCStateSpace sampling method

        Parameters
        ----------
        idata: Dataset
            A Dataset object, returned by a PyMCStateSpace sampling method

        Returns
        -------
        idata: Dataset
            An Dataset object with hidden states transformed to represent only the "interpretable" subcomponents
            of the structural model.

        Notes
        -----
        In general, a structural statespace model can be represented as:

        .. math::
            y_t = \mu_t + \nu_t + \cdots + \gamma_t + c_t + \xi_t + \epsilon_t \tag{1}

        Where:

            - :math:`\mu_t` is the level of the data at time t
            - :math:`\nu_t` is the slope of the data at time t
            - :math:`\cdots` are higher time derivatives of the position (acceleration, jerk, etc) at time t
            - :math:`\gamma_t` is the seasonal component at time t
            - :math:`c_t` is the cycle component at time t
            - :math:`\xi_t` is the autoregressive error at time t
            - :math:`\varepsilon_t` is the measurement error at time t

        In state space form, some or all of these components are represented as linear combinations of other
        subcomponents, making interpretation of the outputs of the outputs difficult. The purpose of this function is
        to take the expended statespace representation and return a "reduced form" of only the components shown in
        equation (1).
        """

        def _extract_and_transform_variable(idata, new_state_names):
            *_, time_dim, state_dim = idata.dims
            state_func = ft.partial(self._hidden_states_from_data)
            new_idata = xr.apply_ufunc(
                state_func,
                idata,
                input_core_dims=[[time_dim, state_dim]],
                output_core_dims=[[time_dim, state_dim]],
                exclude_dims={state_dim},
            )
            new_idata.coords.update({state_dim: new_state_names})
            return new_idata

        var_names = list(idata.data_vars.keys())
        is_latent = [idata[name].shape[-1] == self.k_states for name in var_names]
        new_state_names = self._get_subcomponent_names()

        latent_names = [name for latent, name in zip(is_latent, var_names) if latent]
        dropped_vars = set(var_names) - set(latent_names)
        if len(dropped_vars) > 0:
            _log.warning(
                f'Variables {", ".join(dropped_vars)} do not contain all hidden states (their last dimension '
                f"is not {self.k_states}). They will not be present in the modified idata."
            )
        if len(dropped_vars) == len(var_names):
            raise ValueError(
                "Provided idata had no variables with all hidden states; cannot extract components."
            )

        idata_new = xr.Dataset(
            {
                name: _extract_and_transform_variable(idata[name], new_state_names)
                for name in latent_names
            }
        )
        return idata_new


class Component(ABC):
    r"""
    Base class for a component of a structural timeseries model.

    This base class contains a subset of the class attributes of the PyMCStateSpace class, and none of the class
    methods. The purpose of a component is to allow the partial definition of a structural model. Components are
    assembled into a full model by the StructuralTimeSeries class.

    Parameters
    ----------
    name: str
        The name of the component
    k_endog: int
        Number of endogenous variables being modeled. Currently, must be one because structural models only support
        univariate data.
    k_states: int
        Number of hidden states in the component model
    k_posdef: int
        Rank of the state covariance matrix, or the number of sources of innovations in the component model
    measurement_error: bool
        Whether the observation associated with the component has measurement error. Default is False.
    combine_hidden_states: bool
        Flag for the ``extract_hidden_states_from_data`` method. When ``True``, hidden states from the component model
        are extracted as ``hidden_states[:, np.flatnonzero(Z)]``. Should be True in models where hidden states
        individually have no interpretation, such as seasonal or autoregressive components.
    """

    def __init__(
        self,
        name,
        k_endog,
        k_states,
        k_posdef,
        state_names=None,
        data_names=None,
        shock_names=None,
        param_names=None,
        exog_names=None,
        representation: PytensorRepresentation | None = None,
        measurement_error=False,
        combine_hidden_states=True,
        component_from_sum=False,
        obs_state_idxs=None,
    ):
        self.name = name
        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef
        self.measurement_error = measurement_error

        self.state_names = state_names if state_names is not None else []
        self.data_names = data_names if data_names is not None else []
        self.shock_names = shock_names if shock_names is not None else []
        self.param_names = param_names if param_names is not None else []
        self.exog_names = exog_names if exog_names is not None else []

        self.needs_exog_data = len(self.exog_names) > 0
        self.coords = {}
        self.param_dims = {}

        self.param_info = {}
        self.data_info = {}

        self.param_counts = {}

        if representation is None:
            self.ssm = PytensorRepresentation(k_endog=k_endog, k_states=k_states, k_posdef=k_posdef)
        else:
            self.ssm = representation

        self._name_to_variable = {}
        self._name_to_data = {}

        if not component_from_sum:
            self.populate_component_properties()
            self.make_symbolic_graph()

        self._component_info = {
            self.name: {
                "k_states": self.k_states,
                "k_enodg": self.k_endog,
                "k_posdef": self.k_posdef,
                "combine_hidden_states": combine_hidden_states,
                "obs_state_idx": obs_state_idxs,
            }
        }

    def make_and_register_variable(self, name, shape, dtype=floatX) -> Variable:
        r"""
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

        if name in self._name_to_variable.keys():
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._name_to_variable[name].type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        self._name_to_variable[name] = placeholder
        return placeholder

    def make_and_register_data(self, name, shape, dtype=floatX) -> Variable:
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

        if name in self._name_to_data.keys():
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._name_to_data[name].type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        self._name_to_data[name] = placeholder
        return placeholder

    def make_symbolic_graph(self) -> None:
        raise NotImplementedError

    def populate_component_properties(self):
        raise NotImplementedError

    def _get_combined_shapes(self, other):
        k_states = self.k_states + other.k_states
        k_posdef = self.k_posdef + other.k_posdef
        if self.k_endog != other.k_endog:
            raise NotImplementedError(
                "Merging elements with different numbers of observed states is not supported.>"
            )
        k_endog = self.k_endog

        return k_states, k_posdef, k_endog

    def _combine_statespace_representations(self, other):
        def make_slice(name, x, o_x):
            ndim = max(x.ndim, o_x.ndim)
            return (name,) + (slice(None, None, None),) * ndim

        k_states, k_posdef, k_endog = self._get_combined_shapes(other)

        self_matrices = [self.ssm[name] for name in LONG_MATRIX_NAMES]
        other_matrices = [other.ssm[name] for name in LONG_MATRIX_NAMES]

        x0, P0, c, d, T, Z, R, H, Q = (
            self.ssm[make_slice(name, x, o_x)]
            for name, x, o_x in zip(LONG_MATRIX_NAMES, self_matrices, other_matrices)
        )
        o_x0, o_P0, o_c, o_d, o_T, o_Z, o_R, o_H, o_Q = (
            other.ssm[make_slice(name, x, o_x)]
            for name, x, o_x in zip(LONG_MATRIX_NAMES, self_matrices, other_matrices)
        )

        initial_state = pt.concatenate(conform_time_varying_and_time_invariant_matrices(x0, o_x0))
        initial_state.name = x0.name

        initial_state_cov = pt.linalg.block_diag(P0, o_P0)
        initial_state_cov.name = P0.name

        state_intercept = pt.concatenate(conform_time_varying_and_time_invariant_matrices(c, o_c))
        state_intercept.name = c.name

        obs_intercept = d + o_d
        obs_intercept.name = d.name

        transition = pt.linalg.block_diag(T, o_T)
        transition.name = T.name

        design = pt.concatenate(conform_time_varying_and_time_invariant_matrices(Z, o_Z), axis=-1)
        design.name = Z.name

        selection = pt.linalg.block_diag(R, o_R)
        selection.name = R.name

        obs_cov = H + o_H
        obs_cov.name = H.name

        state_cov = pt.linalg.block_diag(Q, o_Q)
        state_cov.name = Q.name

        new_ssm = PytensorRepresentation(
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initial_state=initial_state,
            initial_state_cov=initial_state_cov,
            state_intercept=state_intercept,
            obs_intercept=obs_intercept,
            transition=transition,
            design=design,
            selection=selection,
            obs_cov=obs_cov,
            state_cov=state_cov,
        )

        return new_ssm

    def _combine_property(self, other, name):
        self_prop = getattr(self, name)
        if isinstance(self_prop, list):
            return self_prop + getattr(other, name)
        elif isinstance(self_prop, dict):
            new_prop = self_prop.copy()
            new_prop.update(getattr(other, name))
            return new_prop

    def _combine_component_info(self, other):
        combined_info = {}
        for key, value in self._component_info.items():
            if not key.startswith("StateSpace"):
                if key in combined_info.keys():
                    raise ValueError(f"Found duplicate component named {key}")
                combined_info[key] = value

        for key, value in other._component_info.items():
            if not key.startswith("StateSpace"):
                if key in combined_info.keys():
                    raise ValueError(f"Found duplicate component named {key}")
                combined_info[key] = value

        return combined_info

    def _make_combined_name(self):
        components = self._component_info.keys()
        name = f'StateSpace[{", ".join(components)}]'
        return name

    def __add__(self, other):
        state_names = self._combine_property(other, "state_names")
        data_names = self._combine_property(other, "data_names")
        param_names = self._combine_property(other, "param_names")
        shock_names = self._combine_property(other, "shock_names")
        param_info = self._combine_property(other, "param_info")
        data_info = self._combine_property(other, "data_info")
        param_dims = self._combine_property(other, "param_dims")
        coords = self._combine_property(other, "coords")
        exog_names = self._combine_property(other, "exog_names")

        _name_to_variable = self._combine_property(other, "_name_to_variable")
        _name_to_data = self._combine_property(other, "_name_to_data")

        measurement_error = any([self.measurement_error, other.measurement_error])

        k_states, k_posdef, k_endog = self._get_combined_shapes(other)
        ssm = self._combine_statespace_representations(other)

        new_comp = Component(
            name="",
            k_endog=1,
            k_states=k_states,
            k_posdef=k_posdef,
            measurement_error=measurement_error,
            representation=ssm,
            component_from_sum=True,
        )
        new_comp._component_info = self._combine_component_info(other)
        new_comp.name = new_comp._make_combined_name()

        names_and_props = [
            ("state_names", state_names),
            ("data_names", data_names),
            ("param_names", param_names),
            ("shock_names", shock_names),
            ("param_dims", param_dims),
            ("coords", coords),
            ("param_dims", param_dims),
            ("param_info", param_info),
            ("data_info", data_info),
            ("exog_names", exog_names),
            ("_name_to_variable", _name_to_variable),
            ("_name_to_data", _name_to_data),
        ]

        for prop, value in names_and_props:
            setattr(new_comp, prop, value)

        return new_comp

    def build(
        self, name=None, filter_type="standard", verbose=True, mode: str | Mode | None = None
    ):
        """
        Build a StructuralTimeSeries statespace model from the current component(s)

        Parameters
        ----------
        name: str, optional
            Name of the exogenous data being modeled. Default is "data"

        filter_type : str, optional
            The type of Kalman filter to use. Valid options are "standard", "univariate", "single", "cholesky", and
            "steady_state". For more information, see the docs for each filter. Default is "standard".

        verbose : bool, optional
            If True, displays information about the initialized model. Defaults to True.

        mode: str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as ``sample_conditional_posterior`` and
            ``forecast``. The mode does **not** effect calls to ``pm.sample``.

            Regardless of whether a mode is specified, it can always be overwritten via the ``compile_kwargs`` argument
            to all sampling methods.

        Returns
        -------
        PyMCStateSpace
            An initialized instance of a PyMCStateSpace, constructed using the system matrices contained in the
            components.
        """

        return StructuralTimeSeries(
            self.ssm,
            name=name,
            state_names=self.state_names,
            data_names=self.data_names,
            shock_names=self.shock_names,
            param_names=self.param_names,
            param_dims=self.param_dims,
            coords=self.coords,
            param_info=self.param_info,
            data_info=self.data_info,
            component_info=self._component_info,
            measurement_error=self.measurement_error,
            exog_names=self.exog_names,
            name_to_variable=self._name_to_variable,
            name_to_data=self._name_to_data,
            filter_type=filter_type,
            verbose=verbose,
            mode=mode,
        )


class LevelTrendComponent(Component):
    r"""
    Level and trend component of a structural time series model

    Parameters
    ----------
    __________
    order : int

        Number of time derivatives of the trend to include in the model. For example, when order=3, the trend will
        be of the form ``y = a + b * t + c * t ** 2``, where the coefficients ``a, b, c`` come from the initial
        state values.

    innovations_order : int or sequence of int, optional

        The number of stochastic innovations to include in the model. By default, ``innovations_order = order``

    Notes
    -----
    This class implements the level and trend components of the general structural time series model. In the most
    general form, the level and trend is described by a system of two time-varying equations.

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \xi_t
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi)
        \end{align}

    Where :math:`\mu_{t+1}` is the mean of the timeseries at time t, and :math:`\nu_t` is the drift or the slope of
    the process. When both innovations :math:`\zeta_t` and :math:`\xi_t` are included in the model, it is known as a
    *local linear trend* model. This system of two equations, corresponding to ``order=2``, can be expanded or
    contracted by adding or removing equations. ``order=3`` would add an acceleration term to the sytsem:

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \eta_t + \xi_t \\
            \eta_{t+1} &= \eta_{t-1} + \omega_t \\
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi) \\
            \omega_t &\sim N(0, \sigma_\omega)
        \end{align}

    After setting all innovation terms to zero and defining initial states :math:`\mu_0, \nu_0, \eta_0`, these equations
    can be collapsed to:

    .. math::
        \mu_t = \mu_0 + \nu_0 \cdot t + \eta_0 \cdot t^2

    Which clarifies how the order and initial states influence the model. In particular, the initial states are the
    coefficients on the intercept, slope, acceleration, and so on.

    In this light, allowing for innovations can be understood as allowing these coefficients to vary over time. Each
    component can be individually selected for time variation by passing a list to the ``innovations_order`` argument.
    For example, a constant intercept with time varying trend and acceleration is specified as ``order=3,
    innovations_order=[0, 1, 1]``.

    By choosing the ``order`` and ``innovations_order``, a large variety of models can be obtained. Notable
    models include:

    * Constant intercept, ``order=1, innovations_order=0``

    .. math::
        \mu_t = \mu

    * Constant linear slope, ``order=2, innovations_order=0``

    .. math::
        \mu_t = \mu_{t-1} + \nu

    * Gaussian Random Walk, ``order=1, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \zeta_t

    * Gaussian Random Walk with Drift, ``order=2, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \nu + \zeta_t

    * Smooth Trend, ``order=2, innovations_order=[0, 1]``

    .. math::
        \begin{align}
            \mu_t &= \mu_{t-1} + \nu_{t-1} \\
            \nu_t &= \nu_{t-1} + \xi_t
        \end{align}

    * Local Level, ``order=2, innovations_order=2``

    [1] notes that the smooth trend model produces more gradually changing slopes than the full local linear trend
    model, and is equivalent to an "integrated trend model".

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.

    """

    def __init__(
        self,
        order: int | list[int] = 2,
        innovations_order: int | list[int] | None = None,
        name: str = "LevelTrend",
    ):
        if innovations_order is None:
            innovations_order = order

        self._order_mask = order_to_mask(order)
        max_state = np.flatnonzero(self._order_mask)[-1].item() + 1

        # If the user passes excess zeros, raise an error. The alternative is to prune them, but this would cause
        # the shape of the state to be different to what the user expects.
        if len(self._order_mask) > max_state:
            raise ValueError(
                f"order={order} is invalid. The highest derivative should not be set to zero. If you want a "
                f"lower order model, explicitly omit the zeros."
            )
        k_states = max_state

        if isinstance(innovations_order, int):
            n = innovations_order
            innovations_order = order_to_mask(k_states)
            if n > 0:
                innovations_order[n:] = False
            else:
                innovations_order[:] = False
        else:
            innovations_order = order_to_mask(innovations_order)

        self.innovations_order = innovations_order[:max_state]
        k_posdef = int(sum(innovations_order))

        super().__init__(
            name,
            k_endog=1,
            k_states=k_states,
            k_posdef=k_posdef,
            measurement_error=False,
            combine_hidden_states=False,
            obs_state_idxs=np.array([1.0] + [0.0] * (k_states - 1)),
        )

    def populate_component_properties(self):
        name_slice = POSITION_DERIVATIVE_NAMES[: self.k_states]
        self.param_names = ["initial_trend"]
        self.state_names = [name for name, mask in zip(name_slice, self._order_mask) if mask]
        self.param_dims = {"initial_trend": ("trend_state",)}
        self.coords = {"trend_state": self.state_names}
        self.param_info = {"initial_trend": {"shape": (self.k_states,), "constraints": None}}

        if self.k_posdef > 0:
            self.param_names += ["sigma_trend"]
            self.shock_names = [
                name for name, mask in zip(name_slice, self.innovations_order) if mask
            ]
            self.param_dims["sigma_trend"] = ("trend_shock",)
            self.coords["trend_shock"] = self.shock_names
            self.param_info["sigma_trend"] = {"shape": (self.k_posdef,), "constraints": "Positive"}

        for name in self.param_names:
            self.param_info[name]["dims"] = self.param_dims[name]

    def make_symbolic_graph(self) -> None:
        initial_trend = self.make_and_register_variable("initial_trend", shape=(self.k_states,))
        self.ssm["initial_state", :] = initial_trend
        triu_idx = np.triu_indices(self.k_states)
        self.ssm[np.s_["transition", triu_idx[0], triu_idx[1]]] = 1

        R = np.eye(self.k_states)
        R = R[:, self.innovations_order]
        self.ssm["selection", :, :] = R

        self.ssm["design", 0, :] = np.array([1.0] + [0.0] * (self.k_states - 1))

        if self.k_posdef > 0:
            sigma_trend = self.make_and_register_variable("sigma_trend", shape=(self.k_posdef,))
            diag_idx = np.diag_indices(self.k_posdef)
            idx = np.s_["state_cov", diag_idx[0], diag_idx[1]]
            self.ssm[idx] = sigma_trend**2


class MeasurementError(Component):
    r"""
    Measurement error term for a structural timeseries model

    Parameters
    ----------
    name: str, optional

        Name of the observed data. Default is "obs".

    Notes
    -----
    This component should only be used in combination with other components, because it has no states. It's only use
    is to add a variance parameter to the model, associated with the observation noise matrix H.

    Examples
    --------
    Create and estimate a deterministic linear trend with measurement error

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=2, innovations_order=0)
        error = st.MeasurementError()
        ss_mod = (trend + error).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')
    """

    def __init__(self, name: str = "MeasurementError"):
        k_endog = 1
        k_states = 0
        k_posdef = 0

        super().__init__(
            name, k_endog, k_states, k_posdef, measurement_error=True, combine_hidden_states=False
        )

    def populate_component_properties(self):
        self.param_names = [f"sigma_{self.name}"]
        self.param_dims = {}
        self.param_info = {
            f"sigma_{self.name}": {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }
        }

    def make_symbolic_graph(self) -> None:
        sigma_shape = ()
        error_sigma = self.make_and_register_variable(f"sigma_{self.name}", shape=sigma_shape)
        diag_idx = np.diag_indices(self.k_endog)
        idx = np.s_["obs_cov", diag_idx[0], diag_idx[1]]
        self.ssm[idx] = error_sigma**2


class AutoregressiveComponent(Component):
    r"""
    Autoregressive timeseries component

    Parameters
    ----------
    order: int or sequence of int

        If int, the number of lags to include in the model.
        If a sequence, an array-like of zeros and ones indicating which lags to include in the model.

    Notes
    -----
    An autoregressive component can be thought of as a way o introducing serially correlated errors into the model.
    The process is modeled:

    .. math::
        x_t = \sum_{i=1}^p \rho_i x_{t-i}

    Where ``p``, the number of autoregressive terms to model, is the order of the process. By default, all lags up to
    ``p`` are included in the model. To disable lags, pass a list of zeros and ones to the ``order`` argumnet. For
    example, ``order=[1, 1, 0, 1]`` would become:

    .. math::
        x_t = \rho_1 x_{t-1} + \rho_2 x_{t-1} + \rho_4 x_{t-1}

    The coefficient :math:`\rho_3` has been constrained to zero.

    .. warning:: This class is meant to be used as a component in a structural time series model. For modeling of
              stationary processes with ARIMA, use ``statespace.BayesianSARIMA``.

    Examples
    --------
    Model a timeseries as an AR(2) process with non-zero mean:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=1, innovations_order=0)
        ar = st.AutoregressiveComponent(2)
        ss_mod = (trend + ar).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            ar_params = pm.Normal('ar_params', dims=ss_mod.param_dims['ar_params'])
            sigma_ar = pm.Exponential('sigma_ar', 1, dims=ss_mod.param_dims['sigma_ar'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')

    """

    def __init__(self, order: int = 1, name: str = "AutoRegressive"):
        order = order_to_mask(order)
        ar_lags = np.flatnonzero(order).ravel().astype(int) + 1
        k_states = len(order)

        self.order = order
        self.ar_lags = ar_lags

        super().__init__(
            name=name,
            k_endog=1,
            k_states=k_states,
            k_posdef=1,
            measurement_error=True,
            combine_hidden_states=True,
            obs_state_idxs=np.r_[[1.0], np.zeros(k_states - 1)],
        )

    def populate_component_properties(self):
        self.state_names = [f"L{i + 1}.data" for i in range(self.k_states)]
        self.shock_names = [f"{self.name}_innovation"]
        self.param_names = ["ar_params", "sigma_ar"]
        self.param_dims = {"ar_params": (AR_PARAM_DIM,)}
        self.coords = {AR_PARAM_DIM: self.ar_lags.tolist()}

        self.param_info = {
            "ar_params": {
                "shape": (self.k_states,),
                "constraints": None,
                "dims": (AR_PARAM_DIM,),
            },
            "sigma_ar": {"shape": (), "constraints": "Positive", "dims": None},
        }

    def make_symbolic_graph(self) -> None:
        k_nonzero = int(sum(self.order))
        ar_params = self.make_and_register_variable("ar_params", shape=(k_nonzero,))
        sigma_ar = self.make_and_register_variable("sigma_ar", shape=())

        T = np.eye(self.k_states, k=-1)
        self.ssm["transition", :, :] = T
        self.ssm["selection", 0, 0] = 1
        self.ssm["design", 0, 0] = 1

        ar_idx = ("transition", np.zeros(k_nonzero, dtype="int"), np.nonzero(self.order)[0])
        self.ssm[ar_idx] = ar_params

        cov_idx = ("state_cov", *np.diag_indices(1))
        self.ssm[cov_idx] = sigma_ar**2


class TimeSeasonality(Component):
    r"""
    Seasonal component, modeled in the time domain

    Parameters
    ----------
    season_length: int
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}]"``

    state_names: list of str, default None
        List of strings for seasonal effect labels. If provided, it must be of length ``season_length``. An example
        would be ``state_names = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']`` when data is daily with a weekly
        seasonal pattern (``season_length = 7``).

        If None, states will be numbered ``[State_0, ..., State_s]``

    remove_first_state: bool, default True
        If True, the first state will be removed from the model. This is done because there are only n-1 degrees of
        freedom in the seasonal component, and one state is not identified. If False, the first state will be
        included in the model, but it will not be identified -- you will need to handle this in the priors (e.g. with
        ZeroSumNormal).

    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" time domain
    representation. The seasonal component can be expressed:

    .. math::
        \gamma_t = -\sum_{i=1}^{s-1} \gamma_{t-i} + \omega_t, \quad \omega_t \sim N(0, \sigma_\gamma)

    Where :math:`s` is the ``seasonal_length`` parameter and :math:`\omega_t` is the (optional) stochastic innovation.
    To give interpretation to the :math:`\gamma` terms, it is helpful to work  through the algebra for a simple
    example. Let :math:`s=4`, and omit the shock term. Define initial conditions :math:`\gamma_0, \gamma_{-1},
    \gamma_{-2}`. The value of the seasonal component for the first 5 timesteps will be:

    .. math::
        \begin{align}
            \gamma_1 &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
             \gamma_2 &= -\gamma_1 - \gamma_0 - \gamma_{-1} \\
                       &= -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 - \gamma_{-1}  \\
                       &= (\gamma_0 - \gamma_0 )+ (\gamma_{-1} - \gamma_{-1}) + \gamma_{-2} \\
                       &= \gamma_{-2} \\
              \gamma_3 &= -\gamma_2 - \gamma_1 - \gamma_0  \\
                       &= -\gamma_{-2} - (-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 \\
                       &=  (\gamma_{-2} - \gamma_{-2}) + \gamma_{-1} + (\gamma_0 - \gamma_0) \\
                       &= \gamma_{-1} \\
              \gamma_4 &= -\gamma_3 - \gamma_2 - \gamma_1 \\
                       &= -\gamma_{-1} - \gamma_{-2} -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) \\
                       &= (\gamma_{-2} - \gamma_{-2}) + (\gamma_{-1} - \gamma_{-1}) + \gamma_0 \\
                       &= \gamma_0 \\
              \gamma_5 &= -\gamma_4 - \gamma_3 - \gamma_2 \\
                       &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
                       &= \gamma_1
        \end{align}

    This exercise shows that, given a list ``initial_conditions`` of length ``s-1``, the effects of this model will be:

        - Period 1: ``-sum(initial_conditions)``
        - Period 2: ``initial_conditions[-1]``
        - Period 3: ``initial_conditions[-2]``
        - ...
        - Period s: ``initial_conditions[0]``
        - Period s+1: ``-sum(initial_condition)``

    And so on. So for interpretation, the ``season_length - 1`` initial states are, when reversed, the coefficients
    associated with ``state_names[1:]``.

    .. warning::
        Although the ``state_names`` argument expects a list of length ``season_length``, only ``state_names[1:]``
        will be saved as model dimensions, since the 1st coefficient is not identified (it is defined as
        :math:`-\sum_{i=1}^{s} \gamma_{t-i}`).

    Examples
    --------
    Estimate monthly with a model with a gaussian random walk trend and monthly seasonality:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt
        import pandas as pd

        # Get month names
        state_names = pd.date_range('1900-01-01', '1900-12-31', freq='MS').month_name().tolist()

        # Build the structural model
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        annual_season = st.TimeSeasonality(season_length=12, name='annual', state_names=state_names, innovations=False)
        ss_mod = (grw + annual_season).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Deterministic('initial_trend', pt.zeros(1), dims=ss_mod.param_dims['initial_trend'])
            annual_coefs = pm.Normal('annual_coefs', sigma=1e-2, dims=ss_mod.param_dims['annual_coefs'])
            trend_sigmas = pm.HalfNormal('trend_sigmas', sigma=1e-6, dims=ss_mod.param_dims['trend_sigmas'])
            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    """

    def __init__(
        self,
        season_length: int,
        innovations: bool = True,
        name: str | None = None,
        state_names: list | None = None,
        remove_first_state: bool = True,
    ):
        if name is None:
            name = f"Seasonal[s={season_length}]"
        if state_names is None:
            state_names = [f"{name}_{i}" for i in range(season_length)]
        else:
            if len(state_names) != season_length:
                raise ValueError(
                    f"state_names must be a list of length season_length, got {len(state_names)}"
                )
            state_names = state_names.copy()
        self.innovations = innovations
        self.remove_first_state = remove_first_state

        if self.remove_first_state:
            # In traditional models, the first state isn't identified, so we can help out the user by automatically
            # discarding it.
            # TODO: Can this be stashed and reconstructed automatically somehow?
            state_names.pop(0)

        k_states = season_length - int(self.remove_first_state)

        super().__init__(
            name=name,
            k_endog=1,
            k_states=k_states,
            k_posdef=int(innovations),
            state_names=state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=np.r_[[1.0], np.zeros(k_states - 1)],
        )

    def populate_component_properties(self):
        self.param_names = [f"{self.name}_coefs"]
        self.param_info = {
            f"{self.name}_coefs": {
                "shape": (self.k_states,),
                "constraints": None,
                "dims": (f"{self.name}_state",),
            }
        }
        self.param_dims = {f"{self.name}_coefs": (f"{self.name}_state",)}
        self.coords = {f"{self.name}_state": self.state_names}

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }
            self.shock_names = [f"{self.name}"]

    def make_symbolic_graph(self) -> None:
        if self.remove_first_state:
            # In this case, parameters are normalized to sum to zero, so the current state is the negative sum of
            # all previous states.
            T = np.eye(self.k_states, k=-1)
            T[0, :] = -1
        else:
            # In this case we assume the user to be responsible for ensuring the states sum to zero, so T is just a
            # circulant matrix that cycles between the states.
            T = np.eye(self.k_states, k=1)
            T[-1, 0] = 1

        self.ssm["transition", :, :] = T
        self.ssm["design", 0, 0] = 1

        initial_states = self.make_and_register_variable(
            f"{self.name}_coefs", shape=(self.k_states,)
        )
        self.ssm["initial_state", np.arange(self.k_states, dtype=int)] = initial_states

        if self.innovations:
            self.ssm["selection", 0, 0] = 1
            season_sigma = self.make_and_register_variable(f"sigma_{self.name}", shape=())
            cov_idx = ("state_cov", *np.diag_indices(1))
            self.ssm[cov_idx] = season_sigma**2


class FrequencySeasonality(Component):
    r"""
    Seasonal component, modeled in the frequency domain

    Parameters
    ----------
    season_length: float
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc. Non-integer seasonal_length is also permitted, for example
        365.2422 days in a (solar) year.

    n: int
        Number of fourier features to include in the seasonal component. Default is ``season_length // 2``, which
        is the maximum possible. A smaller number can be used for a more wave-like seasonal pattern.

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}, n={n}]"``

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" frequency domain
    representation. The seasonal component can be expressed:

    .. math::
        \begin{align}
            \gamma_t &= \sum_{j=1}^{2n} \gamma_{j,t} \\
            \gamma_{j, t+1} &= \gamma_{j,t} \cos \lambda_j + \gamma_{j,t}^\star \sin \lambda_j + \omega_{j, t} \\
            \gamma_{j, t}^\star &= -\gamma_{j,t} \sin \lambda_j + \gamma_{j,t}^\star \cos \lambda_j + \omega_{j,t}^\star
            \lambda_j &= \frac{2\pi j}{s}
        \end{align}

    Where :math:`s` is the ``seasonal_length``.

    Unlike a ``TimeSeasonality`` component, a ``FrequencySeasonality`` component does not require integer season
    length. In addition, for long seasonal periods, it is possible to obtain a more compact state space representation
    by choosing ``n << s // 2``. Using ``TimeSeasonality``, an annual seasonal pattern in daily data requires 364
    states, whereas ``FrequencySeasonality`` always requires ``2 * n`` states, regardless of the ``seasonal_length``.
    The price of this compactness is less representational power. At ``n = 1``, the seasonal pattern will be a pure
    sine wave. At ``n = s // 2``, any arbitrary pattern can be represented.

    One cost of the added flexibility of ``FrequencySeasonality`` is reduced interpretability. States of this model are
    coefficients :math:`\gamma_1, \gamma^\star_1, \gamma_2, \gamma_2^\star ..., \gamma_n, \gamma^\star_n` associated
    with different frequencies in the fourier representation of the seasonal pattern. As a result, it is not possible
    to isolate and identify a "Monday" effect, for instance.
    """

    def __init__(self, season_length, n=None, name=None, innovations=True):
        if n is None:
            n = int(season_length // 2)
        if name is None:
            name = f"Frequency[s={season_length}, n={n}]"

        k_states = n * 2
        self.n = n
        self.season_length = season_length
        self.innovations = innovations

        # If the model is completely saturated (n = s // 2), the last state will not be identified, so it shouldn't
        # get a parameter assigned to it and should just be fixed to zero.
        # Test this way (rather than n == s // 2) to catch cases when n is non-integer.
        self.last_state_not_identified = self.season_length / self.n == 2.0
        self.n_coefs = k_states - int(self.last_state_not_identified)

        obs_state_idx = np.zeros(k_states)
        obs_state_idx[slice(0, k_states, 2)] = 1

        super().__init__(
            name=name,
            k_endog=1,
            k_states=k_states,
            k_posdef=k_states * int(self.innovations),
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
        )

    def make_symbolic_graph(self) -> None:
        self.ssm["design", 0, slice(0, self.k_states, 2)] = 1

        init_state = self.make_and_register_variable(f"{self.name}", shape=(self.n_coefs,))

        init_state_idx = np.arange(self.n_coefs, dtype=int)
        self.ssm["initial_state", init_state_idx] = init_state

        T_mats = [_frequency_transition_block(self.season_length, j + 1) for j in range(self.n)]
        T = pt.linalg.block_diag(*T_mats)
        self.ssm["transition", :, :] = T

        if self.innovations:
            sigma_season = self.make_and_register_variable(f"sigma_{self.name}", shape=())
            self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * sigma_season**2
            self.ssm["selection", :, :] = np.eye(self.k_states)

    def populate_component_properties(self):
        self.state_names = [f"{self.name}_{f}_{i}" for i in range(self.n) for f in ["Cos", "Sin"]]
        self.param_names = [f"{self.name}"]

        self.param_dims = {self.name: (f"{self.name}_state",)}
        self.param_info = {
            f"{self.name}": {
                "shape": (self.k_states - int(self.last_state_not_identified),),
                "constraints": None,
                "dims": (f"{self.name}_state",),
            }
        }

        init_state_idx = np.arange(self.k_states, dtype=int)
        if self.last_state_not_identified:
            init_state_idx = init_state_idx[:-1]
        self.coords = {f"{self.name}_state": [self.state_names[i] for i in init_state_idx]}

        if self.innovations:
            self.shock_names = self.state_names.copy()
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }


class CycleComponent(Component):
    r"""
    A component for modeling longer-term cyclical effects

    Parameters
    ----------
    name: str
        Name of the component. Used in generated coordinates and state names. If None, a descriptive name will be
        used.

    cycle_length: int, optional
        The length of the cycle, in the calendar units of your data. For example, if your data is monthly, and you
        want to model a 12-month cycle, use ``cycle_length=12``. You cannot specify both ``cycle_length`` and
        ``estimate_cycle_length``.

    estimate_cycle_length: bool, default False
        Whether to estimate the cycle length. If True, an additional parameter, ``cycle_length`` will be added to the
        model. You cannot specify both ``cycle_length`` and ``estimate_cycle_length``.

    dampen: bool, default False
        Whether to dampen the cycle by multiplying by a dampening factor :math:`\rho` at every timestep. If true,
        an additional parameter, ``dampening_factor`` will be added to the model.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect. If True, an additional
        parameter, ``sigma_{name}`` will be added to the model.

    Notes
    -----
    The cycle component is very similar in implementation to the frequency domain seasonal component, expect that it
    is restricted to n=1. The cycle component can be expressed:

    .. math::
        \begin{align}
            \gamma_t &= \rho \gamma_{t-1} \cos \lambda + \rho \gamma_{t-1}^\star \sin \lambda + \omega_{t} \\
            \gamma_{t}^\star &= -\rho \gamma_{t-1} \sin \lambda + \rho \gamma_{t-1}^\star \cos \lambda + \omega_{t}^\star \\
            \lambda &= \frac{2\pi}{s}
        \end{align}

    Where :math:`s` is the ``cycle_length``. [1] recommend that this component be used for longer term cyclical
    effects, such as business cycles, and that the seasonal component be used for shorter term effects, such as
    weekly or monthly seasonality.

    Unlike a FrequencySeasonality component, the length of a CycleComponent can be estimated.

    Examples
    --------
    Estimate a business cycle with length between 6 and 12 years:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt
        import pandas as pd
        import numpy as np

        data = np.random.normal(size=(100, 1))

        # Build the structural model
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        cycle = st.CycleComponent('business_cycle', estimate_cycle_length=True, dampen=False)
        ss_mod = (grw + cycle).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states), dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', dims=ss_mod.param_dims['initial_trend'])
            sigma_trend = pm.HalfNormal('sigma_trend', dims=ss_mod.param_dims['sigma_trend'])

            cycle_strength = pm.Normal('business_cycle')
            cycle_length = pm.Uniform('business_cycle_length', lower=6, upper=12)

            sigma_cycle = pm.HalfNormal('sigma_business_cycle', sigma=1)
            ss_mod.build_statespace_graph(data)

            idata = pm.sample(nuts_sampler='numpyro')

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    """

    def __init__(
        self,
        name: str | None = None,
        cycle_length: int | None = None,
        estimate_cycle_length: bool = False,
        dampen: bool = False,
        innovations: bool = True,
    ):
        if cycle_length is None and not estimate_cycle_length:
            raise ValueError("Must specify cycle_length if estimate_cycle_length is False")
        if cycle_length is not None and estimate_cycle_length:
            raise ValueError("Cannot specify cycle_length if estimate_cycle_length is True")
        if name is None:
            cycle = int(cycle_length) if cycle_length is not None else "Estimate"
            name = f"Cycle[s={cycle}, dampen={dampen}, innovations={innovations}]"

        self.estimate_cycle_length = estimate_cycle_length
        self.cycle_length = cycle_length
        self.innovations = innovations
        self.dampen = dampen
        self.n_coefs = 1

        k_states = 2
        k_endog = 1
        k_posdef = 2

        obs_state_idx = np.zeros(k_states)
        obs_state_idx[slice(0, k_states, 2)] = 1

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
        )

    def make_symbolic_graph(self) -> None:
        self.ssm["design", 0, slice(0, self.k_states, 2)] = 1
        self.ssm["selection", :, :] = np.eye(self.k_states)
        self.param_dims = {self.name: (f"{self.name}_state",)}
        self.coords = {f"{self.name}_state": self.state_names}

        init_state = self.make_and_register_variable(f"{self.name}", shape=(self.k_states,))

        self.ssm["initial_state", :] = init_state

        if self.estimate_cycle_length:
            lamb = self.make_and_register_variable(f"{self.name}_length", shape=())
        else:
            lamb = self.cycle_length

        if self.dampen:
            rho = self.make_and_register_variable(f"{self.name}_dampening_factor", shape=())
        else:
            rho = 1

        T = rho * _frequency_transition_block(lamb, j=1)
        self.ssm["transition", :, :] = T

        if self.innovations:
            sigma_cycle = self.make_and_register_variable(f"sigma_{self.name}", shape=())
            self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * sigma_cycle**2

    def populate_component_properties(self):
        self.state_names = [f"{self.name}_{f}" for f in ["Cos", "Sin"]]
        self.param_names = [f"{self.name}"]

        self.param_info = {
            f"{self.name}": {
                "shape": (2,),
                "constraints": None,
                "dims": (f"{self.name}_state",),
            }
        }

        if self.estimate_cycle_length:
            self.param_names += [f"{self.name}_length"]
            self.param_info[f"{self.name}_length"] = {
                "shape": (),
                "constraints": "Positive, non-zero",
                "dims": None,
            }

        if self.dampen:
            self.param_names += [f"{self.name}_dampening_factor"]
            self.param_info[f"{self.name}_dampening_factor"] = {
                "shape": (),
                "constraints": "0 < x ≤ 1",
                "dims": None,
            }

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }
            self.shock_names = self.state_names.copy()


class RegressionComponent(Component):
    def __init__(
        self,
        k_exog: int | None = None,
        name: str | None = "Exogenous",
        state_names: list[str] | None = None,
        innovations=False,
    ):
        self.innovations = innovations
        k_exog = self._handle_input_data(k_exog, state_names, name)

        k_states = k_exog
        k_endog = 1
        k_posdef = k_exog

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            state_names=self.state_names,
            measurement_error=False,
            combine_hidden_states=False,
            exog_names=[f"data_{name}"],
            obs_state_idxs=np.ones(k_states),
        )

    @staticmethod
    def _get_state_names(k_exog: int | None, state_names: list[str] | None, name: str):
        if k_exog is None and state_names is None:
            raise ValueError("Must specify at least one of k_exog or state_names")
        if state_names is not None and k_exog is not None:
            if len(state_names) != k_exog:
                raise ValueError(f"Expected {k_exog} state names, found {len(state_names)}")
        elif k_exog is None:
            k_exog = len(state_names)
        else:
            state_names = [f"{name}_{i + 1}" for i in range(k_exog)]

        return k_exog, state_names

    def _handle_input_data(self, k_exog: int, state_names: list[str] | None, name) -> int:
        k_exog, state_names = self._get_state_names(k_exog, state_names, name)
        self.state_names = state_names

        return k_exog

    def make_symbolic_graph(self) -> None:
        betas = self.make_and_register_variable(f"beta_{self.name}", shape=(self.k_states,))
        regression_data = self.make_and_register_data(
            f"data_{self.name}", shape=(None, self.k_states)
        )

        self.ssm["initial_state", :] = betas
        self.ssm["transition", :, :] = np.eye(self.k_states)
        self.ssm["selection", :, :] = np.eye(self.k_states)
        self.ssm["design"] = pt.expand_dims(regression_data, 1)

        if self.innovations:
            sigma_beta = self.make_and_register_variable(
                f"sigma_beta_{self.name}", (self.k_states,)
            )
            row_idx, col_idx = np.diag_indices(self.k_states)
            self.ssm["state_cov", row_idx, col_idx] = sigma_beta**2

    def populate_component_properties(self) -> None:
        self.shock_names = self.state_names

        self.param_names = [f"beta_{self.name}"]
        self.data_names = [f"data_{self.name}"]
        self.param_dims = {
            f"beta_{self.name}": ("exog_state",),
        }

        self.param_info = {
            f"beta_{self.name}": {
                "shape": (self.k_states,),
                "constraints": None,
                "dims": ("exog_state",),
            },
        }

        self.data_info = {
            f"data_{self.name}": {
                "shape": (None, self.k_states),
                "dims": (TIME_DIM, "exog_state"),
            },
        }
        self.coords = {"exog_state": self.state_names}

        if self.innovations:
            self.param_names += [f"sigma_beta_{self.name}"]
            self.param_dims[f"sigma_beta_{self.name}"] = "exog_state"
            self.param_info[f"sigma_beta_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": ("exog_state",),
            }
