import functools as ft
import logging

from itertools import pairwise
from typing import Any

import numpy as np
import xarray as xr

from pytensor import Mode, Variable, config
from pytensor import tensor as pt

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
from pymc_extras.statespace.core.statespace import PyMCStateSpace, _validate_property
from pymc_extras.statespace.models.utilities import (
    add_tensors_by_dim_labels,
    conform_time_varying_and_time_invariant_matrices,
    join_tensors_by_dim_labels,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    LONG_MATRIX_NAMES,
)

_log = logging.getLogger(__name__)
floatX = config.floatX


class StructuralTimeSeries(PyMCStateSpace):
    r"""
    Structural Time Series Model

    A framework for decomposing a univariate time series into level, trend, seasonal, and cycle
    components, as named by [1]_ and presented in state space form in [2]_.

    This class is not typically instantiated directly. Instead, use ``Component.build()`` to
    construct a model from components combined with the ``+`` operator.

    The model decomposes a time series into interpretable components:

    .. math::

        y_t = \mu_t + \nu_t + \cdots + \gamma_t + c_t + \xi_t + \varepsilon_t

    Where:
        - :math:`\mu_t` is the level component
        - :math:`\nu_t` is the slope/trend component
        - :math:`\cdots` represents higher-order trend components
        - :math:`\gamma_t` is the seasonal component
        - :math:`c_t` is the cycle component
        - :math:`\xi_t` is the autoregressive component
        - :math:`\varepsilon_t` is the measurement error

    Examples
    --------
    Create a model with trend and seasonal components:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=2, innovations_order=1)
        seasonal = st.TimeSeasonality(season_length=12, innovations=True)
        error = st.MeasurementError()

        ss_mod = (trend + seasonal + error).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])

            initial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_trend = pm.HalfNormal('sigma_trend', sigma=1, dims=ss_mod.param_dims['sigma_trend'])

            seasonal_coefs = pm.Normal('params_seasonal', sigma=1, dims=ss_mod.param_dims['params_seasonal'])
            sigma_seasonal = pm.HalfNormal('sigma_seasonal', sigma=1)

            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

    See Also
    --------
    Component : Base class for structural time series components.
    LevelTrendComponent : Component for modeling level and trend.
    TimeSeasonality : Component for seasonal effects.
    CycleComponent : Component for cyclical effects.
    Autoregressive : Component for autoregressive dynamics.

    References
    ----------
    .. [1] Harvey, A. C. (1989). Forecasting, structural time series models and the
           Kalman filter. Cambridge University Press.
    .. [2] Durbin, J., & Koopman, S. J. (2012). Time series analysis by state space
           methods (2nd ed.). Oxford University Press.
    """

    def __init__(
        self,
        ssm: PytensorRepresentation,
        name: str,
        coords_info: CoordInfo,
        param_info: ParameterInfo,
        data_info: DataInfo,
        shock_info: ShockInfo,
        state_info: StateInfo,
        tensor_variable_info: SymbolicVariableInfo,
        tensor_data_info: SymbolicDataInfo,
        component_info: dict[str, dict[str, Any]],
        measurement_error: bool,
        verbose: bool = True,
        filter_type: str = "standard",
        mode: str | Mode | None = None,
    ):
        """
        Initialize a StructuralTimeSeries model.

        This constructor is typically called by ``Component.build()`` rather than directly.

        Parameters
        ----------
        ssm : PytensorRepresentation
            The state space representation containing system matrices.
        name : str
            Name of the model. If None, defaults to "StructuralTimeSeries".
        coords_info : CoordInfo
            Coordinate specifications for model dimensions.
        param_info : ParameterInfo
            Information about model parameters including shapes and constraints.
        data_info : DataInfo
            Information about data variables expected by the model.
        shock_info : ShockInfo
            Information about innovation/shock processes.
        state_info : StateInfo
            Information about hidden and observed states.
        tensor_variable_info : SymbolicVariableInfo
            Mapping from parameter names to PyTensor symbolic variables.
        tensor_data_info : SymbolicDataInfo
            Mapping from data names to PyTensor symbolic variables.
        component_info : dict[str, dict[str, Any]]
            Information about model components used for state extraction.
        measurement_error : bool
            Whether the model includes measurement error.
        verbose : bool, default True
            Whether to print model information during construction.
        filter_type : str, default "standard"
            Type of Kalman filter to use.
        mode : str | Mode | None, default None
            PyTensor compilation mode.
        """
        self._name = name or "StructuralTimeSeries"
        self.measurement_error = measurement_error

        k_states, k_posdef, k_endog = ssm.k_states, ssm.k_posdef, ssm.k_endog

        self._init_info_objects(
            param_info, data_info, shock_info, state_info, coords_info, k_states, k_endog
        )

        super().__init__(
            k_endog,
            k_states,
            max(1, k_posdef),
            filter_type=filter_type,
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
        )

        self._tensor_variable_info = tensor_variable_info
        self._tensor_data_info = tensor_data_info
        self._component_info = component_info.copy()
        self._exog_names = data_info.exogenous_names
        self._needs_exog_data = data_info.needs_exogenous_data

        self._init_ssm(ssm, k_posdef)

    def _init_info_objects(
        self,
        param_info: ParameterInfo,
        data_info: DataInfo,
        shock_info: ShockInfo,
        state_info: StateInfo,
        coords_info: CoordInfo,
        k_states: int,
        k_endog: int,
    ) -> None:
        """Initialize all info objects and set observed state names."""
        self._observed_state_names = state_info.observed_state_names

        param_names, param_dims, param_info = self._add_inital_state_cov_to_properties(
            param_info, k_states
        )
        self._param_dims = param_dims

        self._param_info = param_info
        self._data_info = data_info
        self._shock_info = shock_info
        self._state_info = state_info

        # Stripped names must be set before default_coords_from_model (which accesses state_names)
        self._init_stripped_names(k_endog)

        default_coords = coords_info.default_coords_from_model(self)
        self._coords_info = coords_info.merge(default_coords)

    def _init_stripped_names(self, k_endog: int) -> None:
        """Strip data suffixes from names when k_endog == 1 for cleaner output."""

        def strip(names):
            return self._strip_data_names_if_unambiguous(names, k_endog)

        self._state_names = strip(self._state_info.unobserved_state_names)
        self._data_names = strip([d.name for d in self._data_info if not d.is_exogenous])
        self._shock_names = strip(self._shock_info.names)
        self._param_names = strip(self._param_info.names)

    def _init_ssm(self, ssm: PytensorRepresentation, k_posdef: int) -> None:
        """Initialize state space model representation."""
        self.ssm = ssm.copy()

        if k_posdef == 0:
            self.ssm.k_posdef = self.k_posdef
            self.ssm.shapes["state_cov"] = (1, 1, 1)
            self.ssm["state_cov"] = pt.zeros((1, 1, 1))
            self.ssm.shapes["selection"] = (1, self.k_states, 1)
            self.ssm["selection"] = pt.zeros((1, self.k_states, 1))

        P0 = self.make_and_register_variable("P0", shape=(self.k_states, self.k_states))
        self.ssm["initial_state_cov"] = P0

    def _populate_properties(self) -> None:
        # The base class method needs to be overridden because we directly set properties in
        # the __init__ method.
        pass

    def _strip_data_names_if_unambiguous(self, names: list[str] | tuple[str, ...], k_endog: int):
        """
        State names from components should always be of the form name[data_name], in the case that the component is
        associated with multiple observed states. Not doing so leads to ambiguity -- we might have two level states,
        but which goes to which observed component? So we set `level[data_1]` and `level[data_2]`.

        In cases where there is only one observed state (when k_endog == 1), we can strip the data part and just use
        the state name. This is a bit cleaner.
        """
        if k_endog == 1:
            [data_name] = self._observed_state_names
            return tuple(
                name.replace(f"[{data_name}]", "") if isinstance(name, str) else name
                for name in names
            )

        else:
            return names

    @property
    def state_names(self) -> tuple[str, ...]:
        """Return stripped state names (without [data_name] suffix when k_endog == 1)."""
        return self._state_names

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Return stripped shock names (without [data_name] suffix when k_endog == 1)."""
        return self._shock_names

    @staticmethod
    def _add_inital_state_cov_to_properties(param_info, k_states):
        initial_state_cov_param = Parameter(
            name="P0",
            shape=(k_states, k_states),
            dims=(ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            constraints="Positive semi-definite",
        )

        if param_info is not None:
            param_info = param_info.add(initial_state_cov_param)
        else:
            param_info = ParameterInfo(parameters=(initial_state_cov_param,))

        return param_info.names, [p.dims for p in param_info], param_info

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
                sum_idx_joined = np.flatnonzero(obs_idx)
                sum_idx_split = np.split(sum_idx_joined, info[name]["k_endog"])
                for sum_idx in sum_idx_split:
                    result.append(X[..., sum_idx].sum(axis=-1)[..., None])
            else:
                n_components = len(self.state_names[s])
                for j in range(n_components):
                    result.append(X[..., j, None])

        return np.concatenate(result, axis=-1)

    def _get_subcomponent_names(self):
        state_slices = self._state_slices_from_info()
        info = self._component_info
        names = info.keys()
        result = []

        for i, (name, s) in enumerate(zip(names, state_slices)):
            if info[name]["combine_hidden_states"]:
                if self.k_endog == 1:
                    result.append(name)
                else:
                    # If there are multiple observed states, we will combine per hidden state, preserving the
                    # observed state names. Note this happens even if this *component* has only 1 state for consistency,
                    # as long as the statespace model has multiple observed states.
                    result.extend(
                        [f"{name}[{obs_name}]" for obs_name in info[name]["observed_state_names"]]
                    )
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
            A Dataset object with hidden states transformed to represent only the "interpretable" subcomponents
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

        var_names: list[str] = list(idata.data_vars.keys())  # type: ignore[arg-type]
        is_latent = [idata[name].shape[-1] == self.k_states for name in var_names]
        new_state_names = self._get_subcomponent_names()

        latent_names = [name for latent, name in zip(is_latent, var_names) if latent]
        dropped_vars = set(var_names) - set(latent_names)
        if len(dropped_vars) > 0:
            _log.warning(
                f"Variables {', '.join(sorted(dropped_vars))} do not contain all hidden states (their last dimension "
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


class Component:
    r"""
    Base class for a component of a structural timeseries model.

    This base class contains a subset of the class attributes of the PyMCStateSpace class, and none of the class
    methods. The purpose of a component is to allow the partial definition of a structural model. Components are
    assembled into a full model by the StructuralTimeSeries class.

    Parameters
    ----------
    name : str
        The name of the component.
    k_endog : int
        Number of endogenous (observed) variables being modeled.
    k_states : int
        Number of hidden states in the component model.
    k_posdef : int
        Rank of the state covariance matrix, or the number of sources of innovations
        in the component model.
    base_state_names : list[str] | None, optional
        Base names of hidden states, before any transformations by set_states().
        Subclasses typically transform these (e.g., adding suffixes). If None, defaults to empty list.
    base_observed_state_names : list[str] | None, optional
        Base names of observed states, before any transformations by set_states().
        If None, defaults to empty list.
    representation : PytensorRepresentation | None, optional
        Pre-existing state space representation. If None, creates a new one.
    measurement_error : bool, optional
        Whether the component includes measurement error. Default is False.
    combine_hidden_states : bool, optional
        Whether to combine hidden states when extracting from data. Should be True for
        components where individual states have no interpretation (e.g., seasonal,
        autoregressive). Default is True.
    component_from_sum : bool, optional
        Whether this component is created from combining other components. Default is False.
    obs_state_idxs : np.ndarray | None, optional
        Indices indicating which states contribute to observed variables. If None,
        defaults to None.
    share_states : bool, optional
        Whether states are shared across multiple endogenous variables in multivariate
        models. When True, the same latent states affect all observed variables.
        Default is False.

    Examples
    --------
    Create a simple trend component:

    .. code:: python

        from pymc_extras.statespace import structural as st

        trend = st.LevelTrendComponent(order=2, innovations_order=1)
        seasonal = st.TimeSeasonality(season_length=12, innovations=True)
        model = (trend + seasonal).build()

        print(f"Model has {model.k_states} states and {model.k_posdef} innovations")

    See Also
    --------
    StructuralTimeSeries : The complete model class that combines components.
    LevelTrendComponent : Component for modeling level and trend.
    TimeSeasonality : Component for seasonal effects.
    CycleComponent : Component for cyclical effects.
    RegressionComponent : Component for regression effects.
    """

    def __init__(
        self,
        name,
        k_endog,
        k_states,
        k_posdef,
        base_state_names=None,
        base_observed_state_names=None,
        representation: PytensorRepresentation | None = None,
        measurement_error=False,
        combine_hidden_states=True,
        component_from_sum=False,
        obs_state_idxs=None,
        share_states: bool = False,
    ):
        self.name = name
        self.share_states = share_states
        self.measurement_error = measurement_error

        base_state_names = list(base_state_names) if base_state_names is not None else []
        base_observed_state_names = (
            list(base_observed_state_names) if base_observed_state_names is not None else []
        )

        self._k_posdef = k_posdef
        self._k_endog = len(base_observed_state_names) or k_endog
        self._k_states = k_states
        self.base_state_names = base_state_names
        self.base_observed_state_names = base_observed_state_names

        self._init_ssm(representation, k_endog, k_states, k_posdef)

        self._tensor_variable_info = SymbolicVariableInfo()
        self._tensor_data_info = SymbolicDataInfo()

        if not component_from_sum:
            self.populate_component_properties()
            self.make_symbolic_graph()

            self._component_info = {
                self.name: {
                    "k_states": k_states,
                    "k_endog": k_endog,
                    "k_posdef": k_posdef,
                    "observed_state_names": self._state_info.observed_state_names,
                    "combine_hidden_states": combine_hidden_states,
                    "obs_state_idx": obs_state_idxs,
                    "share_states": self.share_states,
                }
            }

    def _init_ssm(
        self,
        representation: PytensorRepresentation | None,
        k_endog: int,
        k_states: int,
        k_posdef: int,
    ) -> None:
        """Initialize state space model representation."""
        if representation is None:
            self.ssm = PytensorRepresentation(k_endog=k_endog, k_states=k_states, k_posdef=k_posdef)
        else:
            self.ssm = representation

    def populate_component_properties(self) -> None:
        self._set_states()
        self._set_parameters()
        self._set_shocks()
        self._set_data_info()
        self._set_coords()

    def set_states(self) -> State | tuple[State, ...] | None:
        """
        Set default state specification based on number of states and endogenous variables in the component.

        It is encouraged to override this method.
        """
        state_names = self.base_state_names or [i for i in range(self.k_states or 0)]
        observed_state_names = self.base_observed_state_names or [
            i for i in range(self._k_endog or 0)
        ]

        hidden_states = [
            State(name=name, observed=False, shared=self.share_states) for name in state_names
        ]
        observed_states = [
            State(name=name, observed=True, shared=self.share_states)
            for name in observed_state_names
        ]
        return *hidden_states, *observed_states

    def _set_states(self) -> None:
        states = self.set_states()
        _validate_property(states, "states", State)
        if isinstance(states, State):
            states = (states,)
        self._state_info = StateInfo(states=states)

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        """
        Set component parameter specifications. Since different component types will require different specifications,
        you must be override this method.
        """
        return

    def _set_parameters(self) -> None:
        params = self.set_parameters()
        _validate_property(params, "parameters", Parameter)
        if isinstance(params, Parameter):
            params = (params,)
        self._param_info = ParameterInfo(parameters=params)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        """
        Set default shock specifications based on the number of sources of innovations in the component.

        It is encouraged to override this method.
        """
        return tuple(Shock(name=f"shock_{name}") for name in range(self.k_posdef or 0))

    def _set_shocks(self) -> None:
        shocks = self.set_shocks()
        _validate_property(shocks, "shocks", Shock)
        if isinstance(shocks, Shock):
            shocks = (shocks,)
        self._shock_info = ShockInfo(shocks=shocks)

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        """
        Set default data specifications. Since different component types will require different specifications you must be override this method.
        """
        return

    def _set_data_info(self) -> None:
        data_info = self.set_data_info()
        _validate_property(data_info, "data_info", Data)
        if isinstance(data_info, Data):
            data_info = (data_info,)
        self._data_info = DataInfo(data=data_info)

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        """
        Set default coordinate specifications. Since different component types will require different specifications you must be override this method.
        """
        return

    def _set_coords(self) -> None:
        coords = self.set_coords()
        _validate_property(coords, "coords", Coord)
        if isinstance(coords, Coord):
            coords = (coords,)
        self._coords_info = CoordInfo(coords=coords)

    @property
    def state_names(self):
        return self._state_info.unobserved_state_names

    @property
    def observed_state_names(self):
        return self._state_info.observed_state_names

    @property
    def param_names(self):
        return self._param_info.names

    @property
    def param_info(self):
        return self._param_info

    @property
    def shock_names(self):
        return self._shock_info.names

    @property
    def data_names(self):
        return [data.name for data in self._data_info if not data.is_exogenous]

    @property
    def exog_names(self):
        return self._data_info.exogenous_names

    @property
    def coords(self):
        return self._coords_info.to_dict()

    @property
    def param_dims(self):
        return {param.name: param.dims for param in self._param_info if param.dims is not None}

    @property
    def needs_exog_data(self):
        return self._data_info.needs_exogenous_data

    @property
    def k_states(self):
        return self._k_states

    @property
    def k_endog(self):
        return self._k_endog

    @property
    def k_posdef(self):
        return self._k_posdef

    @property
    def _name_to_variable(self):
        return self._tensor_variable_info.to_dict()

    @property
    def _name_to_data(self):
        return self._tensor_data_info.to_dict()

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
        if name not in self._param_info:
            raise ValueError(
                f"{name} is not a model parameter. All placeholder variables should correspond to model "
                f"parameters."
            )

        if name in self._tensor_variable_info:
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._tensor_variable_info[name].symbolic_variable.type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        tensor_var = SymbolicVariable(name=name, symbolic_variable=placeholder)
        self._tensor_variable_info = self._tensor_variable_info.add(tensor_var)
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
        if name not in self._data_info:
            raise ValueError(
                f"{name} is not a model parameter. All placeholder variables should correspond to model "
                f"parameters."
            )

        if name in self._tensor_data_info:
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._tensor_data_info[name].symbolic_data.type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        tensor_data = SymbolicData(name=name, symbolic_data=placeholder)
        tensor_data_info = SymbolicDataInfo(symbolic_data=(tensor_data,))
        self._tensor_data_info = self._tensor_data_info.merge(tensor_data_info)
        return placeholder

    def make_symbolic_graph(self) -> None:
        raise NotImplementedError

    def _get_combined_shapes(self, other):
        k_states = self.k_states + other.k_states
        k_posdef = self.k_posdef + other.k_posdef

        # To count endog states, we have to count unique names between the two components.
        combined_states = self._state_info.merge(other._state_info, overwrite_duplicates=True)
        k_endog = len(combined_states.observed_state_names)

        return k_states, k_posdef, k_endog

    def _combine_statespace_representations(self, other):
        def make_slice(name, x, o_x):
            ndim = max(x.ndim, o_x.ndim)
            return (name,) + (slice(None, None, None),) * ndim

        k_states, k_posdef, k_endog = self._get_combined_shapes(other)

        self_matrices = [self.ssm[name] for name in LONG_MATRIX_NAMES]
        other_matrices = [other.ssm[name] for name in LONG_MATRIX_NAMES]

        self_observed_states = self.observed_state_names
        other_observed_states = other.observed_state_names

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

        obs_intercept = add_tensors_by_dim_labels(
            d,
            o_d,
            labels=list(self_observed_states),
            other_labels=list(other_observed_states),
            labeled_axis=-1,
        )
        obs_intercept.name = d.name

        transition = pt.linalg.block_diag(T, o_T)
        transition = pt.specify_shape(
            transition,
            shape=[
                sum(shapes) if not any([s is None for s in shapes]) else None
                for shapes in zip(*[T.type.shape, o_T.type.shape])
            ],
        )
        transition.name = T.name

        design = join_tensors_by_dim_labels(
            *conform_time_varying_and_time_invariant_matrices(Z, o_Z),
            labels=list(self_observed_states),
            other_labels=list(other_observed_states),
            labeled_axis=-2,
            join_axis=-1,
        )
        design.name = Z.name

        selection = pt.linalg.block_diag(R, o_R)
        selection = pt.specify_shape(
            selection,
            shape=[
                sum(shapes) if not any([s is None for s in shapes]) else None
                for shapes in zip(*[R.type.shape, o_R.type.shape])
            ],
        )
        selection.name = R.name

        obs_cov = add_tensors_by_dim_labels(
            H,
            o_H,
            labels=list(self_observed_states),
            other_labels=list(other_observed_states),
            labeled_axis=(-1, -2),
        )
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
        name = f"StateSpace[{', '.join(components)}]"
        return name

    def __add__(self, other):
        param_info = self._param_info.merge(other._param_info)
        data_info = self._data_info.merge(other._data_info)
        shock_info = self._shock_info.merge(other._shock_info)
        state_info = self._state_info.merge(other._state_info, overwrite_duplicates=True)
        coords_info = self._coords_info.merge(other._coords_info)
        observed_state_names = state_info.observed_state_names
        tensor_variable_info = self._tensor_variable_info.merge(other._tensor_variable_info)
        tensor_data_info = self._tensor_data_info.merge(other._tensor_data_info)

        measurement_error = any([self.measurement_error, other.measurement_error])

        k_states, k_posdef, k_endog = self._get_combined_shapes(other)

        ssm = self._combine_statespace_representations(other)

        new_comp = Component(
            name="",
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            base_observed_state_names=list(observed_state_names),
            measurement_error=measurement_error,
            representation=ssm,
            component_from_sum=True,
        )
        new_comp._component_info = self._combine_component_info(other)
        new_comp.name = new_comp._make_combined_name()

        names_and_props = [
            ("_coords_info", coords_info),
            ("_param_info", param_info),
            ("_data_info", data_info),
            ("_shock_info", shock_info),
            ("_state_info", state_info),
            ("_tensor_variable_info", tensor_variable_info),
            ("_tensor_data_info", tensor_data_info),
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
            coords_info=self._coords_info,
            param_info=self._param_info,
            data_info=self._data_info,
            shock_info=self._shock_info,
            state_info=self._state_info,
            tensor_variable_info=self._tensor_variable_info,
            tensor_data_info=self._tensor_data_info,
            component_info=self._component_info,
            measurement_error=self.measurement_error,
            filter_type=filter_type,
            verbose=verbose,
            mode=mode,
        )
