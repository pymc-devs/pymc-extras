import functools as ft
import logging

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import xarray as xr

from pytensor import Mode, Variable, config
from pytensor import tensor as pt

from pymc_extras.statespace.core import PyMCStateSpace, PytensorRepresentation
from pymc_extras.statespace.models.utilities import (
    add_tensors_by_dim_labels,
    conform_time_varying_and_time_invariant_matrices,
    join_tensors_by_dim_labels,
    make_default_coords,
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

    The structural time series model, named by [1] and presented in statespace form in [2], is a framework for
    decomposing a univariate time series into level, trend, seasonal, and cycle components. It also admits the
    possibility of exogenous regressors. Unlike the SARIMAX framework, the time series is not assumed to be stationary.

    Parameters
    ----------
    ssm : PytensorRepresentation
        The state space representation containing system matrices.
    name : str
        Name of the model. If None, defaults to "StructuralTimeSeries".
    state_names : list[str]
        Names of the hidden states in the model.
    observed_state_names : list[str]
        Names of the observed variables.
    data_names : list[str]
        Names of data variables expected by the model.
    shock_names : list[str]
        Names of innovation/shock processes.
    param_names : list[str]
        Names of model parameters.
    exog_names : list[str]
        Names of exogenous variables.
    param_dims : dict[str, tuple[int]]
        Dimension specifications for parameters.
    coords : dict[str, Sequence]
        Coordinate specifications for the model.
    param_info : dict[str, dict[str, Any]]
        Information about parameters including shapes and constraints.
    data_info : dict[str, dict[str, Any]]
        Information about data variables.
    component_info : dict[str, dict[str, Any]]
        Information about model components.
    measurement_error : bool
        Whether the model includes measurement error.
    name_to_variable : dict[str, Variable]
        Mapping from parameter names to PyTensor variables.
    name_to_data : dict[str, Variable] | None, optional
        Mapping from data names to PyTensor variables. Default is None.
    verbose : bool, optional
        Whether to print model information. Default is True.
    filter_type : str, optional
        Type of Kalman filter to use. Default is "standard".
    mode : str | Mode | None, optional
        PyTensor compilation mode. Default is None.

    Notes
    -----
    The structural time series model decomposes a time series into interpretable components:

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

    The model is built by combining individual components (e.g., LevelTrendComponent,
    TimeSeasonality, CycleComponent) using the addition operator. Each component
    contributes to the overall state space representation.

    Examples
    --------
    Create a model with trend and seasonal components:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=2 innovations_order=1)
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
        state_names: list[str],
        observed_state_names: list[str],
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
        verbose: bool = True,
        filter_type: str = "standard",
        mode: str | Mode | None = None,
    ):
        name = "StructuralTimeSeries" if name is None else name

        self._name = name
        self._observed_state_names = observed_state_names

        k_states, k_posdef, k_endog = ssm.k_states, ssm.k_posdef, ssm.k_endog
        param_names, param_dims, param_info = self._add_inital_state_cov_to_properties(
            param_names, param_dims, param_info, k_states
        )

        self._state_names = self._strip_data_names_if_unambiguous(state_names, k_endog)
        self._data_names = self._strip_data_names_if_unambiguous(data_names, k_endog)
        self._shock_names = self._strip_data_names_if_unambiguous(shock_names, k_endog)
        self._param_names = self._strip_data_names_if_unambiguous(param_names, k_endog)
        self._param_dims = param_dims

        default_coords = make_default_coords(self)
        coords.update(default_coords)

        self._coords = {
            k: self._strip_data_names_if_unambiguous(v, k_endog) for k, v in coords.items()
        }
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

    def _strip_data_names_if_unambiguous(self, names: list[str], k_endog: int):
        """
        State names from components should always be of the form name[data_name], in the case that the component is
        associated with multiple observed states. Not doing so leads to ambiguity -- we might have two level states,
        but which goes to which observed component? So we set `level[data_1]` and `level[data_2]`.

        In cases where there is only one observed state (when k_endog == 1), we can strip the data part and just use
        the state name. This is a bit cleaner.
        """
        if k_endog == 1:
            [data_name] = self.observed_states
            return [
                name.replace(f"[{data_name}]", "") if isinstance(name, str) else name
                for name in names
            ]

        else:
            return names

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
        return self._observed_state_names

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

        var_names = list(idata.data_vars.keys())
        is_latent = [idata[name].shape[-1] == self.k_states for name in var_names]
        new_state_names = self._get_subcomponent_names()

        latent_names = [name for latent, name in zip(is_latent, var_names) if latent]
        dropped_vars = set(var_names) - set(latent_names)
        if len(dropped_vars) > 0:
            _log.warning(
                f"Variables {', '.join(dropped_vars)} do not contain all hidden states (their last dimension "
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
    state_names : list[str] | None, optional
        Names of the hidden states. If None, defaults to empty list.
    observed_state_names : list[str] | None, optional
        Names of the observed states associated with this component. Must have the same
        length as k_endog. If None, defaults to empty list.
    data_names : list[str] | None, optional
        Names of data variables expected by the component. If None, defaults to empty list.
    shock_names : list[str] | None, optional
        Names of innovation/shock processes. If None, defaults to empty list.
    param_names : list[str] | None, optional
        Names of component parameters. If None, defaults to empty list.
    exog_names : list[str] | None, optional
        Names of exogenous variables. If None, defaults to empty list.
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
        state_names=None,
        observed_state_names=None,
        data_names=None,
        shock_names=None,
        param_names=None,
        exog_names=None,
        representation: PytensorRepresentation | None = None,
        measurement_error=False,
        combine_hidden_states=True,
        component_from_sum=False,
        obs_state_idxs=None,
        share_states: bool = False,
    ):
        self.name = name
        self.k_endog = k_endog
        self.k_states = k_states
        self.share_states = share_states
        self.k_posdef = k_posdef
        self.measurement_error = measurement_error

        self.state_names = list(state_names) if state_names is not None else []
        self.observed_state_names = (
            list(observed_state_names) if observed_state_names is not None else []
        )
        self.data_names = list(data_names) if data_names is not None else []
        self.shock_names = list(shock_names) if shock_names is not None else []
        self.param_names = list(param_names) if param_names is not None else []
        self.exog_names = list(exog_names) if exog_names is not None else []

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
                "k_endog": self.k_endog,
                "k_posdef": self.k_posdef,
                "observed_state_names": self.observed_state_names,
                "combine_hidden_states": combine_hidden_states,
                "obs_state_idx": obs_state_idxs,
                "share_states": self.share_states,
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

        # To count endog states, we have to count unique names between the two components.
        combined_states = self._combine_property(
            other, "observed_state_names", allow_duplicates=False
        )
        k_endog = len(combined_states)

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
            d, o_d, labels=self_observed_states, other_labels=other_observed_states, labeled_axis=-1
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
            labels=self_observed_states,
            other_labels=other_observed_states,
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
            labels=self_observed_states,
            other_labels=other_observed_states,
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

    def _combine_property(self, other, name, allow_duplicates=True):
        self_prop = getattr(self, name)
        other_prop = getattr(other, name)

        if not isinstance(self_prop, type(other_prop)):
            raise TypeError(
                f"Property {name} of {self} and {other} are not the same and cannot be combined. Found "
                f"{type(self_prop)} for {self} and {type(other_prop)} for {other}'"
            )

        if not isinstance(self_prop, list | dict):
            raise TypeError(
                f"All component properties are expected to be lists or dicts, but found {type(self_prop)}"
                f"for property {name} of {self} and {type(other_prop)} for {other}'"
            )

        if isinstance(self_prop, list) and allow_duplicates:
            return self_prop + other_prop
        elif isinstance(self_prop, list) and not allow_duplicates:
            return self_prop + [x for x in other_prop if x not in self_prop]
        elif isinstance(self_prop, dict):
            new_prop = self_prop.copy()
            new_prop.update(other_prop)
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
        name = f"StateSpace[{', '.join(components)}]"
        return name

    def __add__(self, other):
        state_names = self._combine_property(other, "state_names")
        data_names = self._combine_property(other, "data_names")
        observed_state_names = self._combine_property(
            other, "observed_state_names", allow_duplicates=False
        )

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
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            observed_state_names=observed_state_names,
            measurement_error=measurement_error,
            representation=ssm,
            component_from_sum=True,
        )
        new_comp._component_info = self._combine_component_info(other)
        new_comp.name = new_comp._make_combined_name()

        names_and_props = [
            ("state_names", state_names),
            ("observed_state_names", observed_state_names),
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
            observed_state_names=self.observed_state_names,
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
