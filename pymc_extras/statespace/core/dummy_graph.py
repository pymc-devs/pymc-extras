from typing import TYPE_CHECKING

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc.model import modelcontext

from pymc_extras.statespace.utils.constants import OBS_STATE_DIM
from pymc_extras.statespace.utils.data_tools import register_data_with_pymc

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace


def build_dummy_graph(ss_mod: "PyMCStateSpace") -> None:
    """
    Build a dummy computation graph for the state space model matrices.

    Create "dummy" ``pm.Flat`` variables representing the deep parameters used in the state space model, so
    post-estimation sampling functions can re-derive the statespace matrices from posterior draws.
    """

    def infer_variable_shape(name):
        shape = ss_mod._name_to_variable[name].type.shape
        if not any(dim is None for dim in shape):
            return shape

        dim_names = ss_mod._fit_dims.get(name, None)
        if dim_names is None:
            raise ValueError(
                f"Could not infer shape for {name}, because it was not given coords during model"
                f"fitting"
            )

        shape_from_coords = tuple([len(ss_mod._fit_coords[dim]) for dim in dim_names])
        return tuple(
            [shape[i] if shape[i] is not None else shape_from_coords[i] for i in range(len(shape))]
        )

    for name in ss_mod.param_names:
        pm.Flat(
            name,
            shape=infer_variable_shape(name),
            dims=ss_mod._fit_dims.get(name, None),
        )


def kalman_filter_outputs_from_dummy_graph(
    ss_mod: "PyMCStateSpace",
    data: pt.TensorLike | None = None,
    data_dims: str | tuple[str] | list[str] | None = None,
    scenario: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
) -> tuple[list[pt.TensorVariable], list[tuple[pt.TensorVariable, pt.TensorVariable]]]:
    """
    Build a Kalman filter graph using "dummy" ``pm.Flat`` distributions for the model variables and sort the
    returns into (mean, covariance) pairs for each of filtered, predicted, and smoothed output.

    Parameters
    ----------
    data : pytensor tensor-like, optional
        Observed data on which to condition the model. If not provided, the data provided when the model was
        built is used.
    data_dims : str or tuple of str, optional
        Dimension names associated with the model data. Defaults to ("time", "obs_state").
    scenario : dict mapping str to pandas.DataFrame, optional
        Out-of-sample scenario data. If provided, it must have values for all data variables in the model;
        ``pm.set_data`` is used to replace training data with the new values.

    Returns
    -------
    matrices : list of tensors
        Statespace matrices with dummy parameters.
    grouped_outputs : list of tuple of tensors
        A list of tuples, each containing the mean and covariance of the filtered, predicted, and smoothed
        states.
    """
    if scenario is None:
        scenario = dict()

    pm_mod = modelcontext(None)
    build_dummy_graph(ss_mod)
    ss_mod._insert_random_variables()

    for name in ss_mod.data_names:
        if name not in pm_mod:
            pm.Data(**ss_mod._fit_exog_data[name])

    ss_mod._insert_data_variables()

    for name in ss_mod.data_names:
        if name in scenario.keys():
            pm.set_data({name: scenario[name]})

    matrices = ss_mod.unpack_statespace()

    if data is None:
        data = ss_mod._fit_data

    # Replace n_timesteps with data length for time-varying matrices
    data_len = data.shape[0] if hasattr(data, "shape") else len(data)
    matrices = ss_mod._insert_constant_timestep(matrices, data_len)
    x0, P0, c, d, T, Z, R, H, Q = matrices

    obs_coords = pm_mod.coords.get(OBS_STATE_DIM, None)

    data, nan_mask = register_data_with_pymc(
        data,
        n_obs=ss_mod.ssm.k_endog,
        obs_coords=obs_coords,
        data_dims=data_dims,
        register_data=True,
    )

    filter_outputs = ss_mod.kalman_filter.build_graph(
        data,
        x0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        time_varying_names=ss_mod.ssm.time_varying_names,
    )

    filter_outputs.pop(-1)
    states, covariances = filter_outputs[:3], filter_outputs[3:]

    filtered_states, predicted_states, _ = states
    filtered_covariances, predicted_covariances, _ = covariances

    [smoothed_states, smoothed_covariances] = ss_mod.kalman_smoother.build_graph(
        T,
        R,
        Q,
        filtered_states,
        filtered_covariances,
        time_varying_names=ss_mod.ssm.time_varying_names,
    )

    grouped_outputs = [
        (filtered_states, filtered_covariances),
        (predicted_states, predicted_covariances),
        (smoothed_states, smoothed_covariances),
    ]

    return [x0, P0, c, d, T, Z, R, H, Q], grouped_outputs
