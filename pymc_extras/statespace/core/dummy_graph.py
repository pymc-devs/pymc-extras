from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc.model import modelcontext

from pymc_extras.statespace.utils.constants import OBS_STATE_DIM
from pymc_extras.statespace.utils.data_tools import register_data_with_pymc

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace


def build_dummy_graph(
    ss_mod: "PyMCStateSpace", *, coords: dict[str, Sequence], dims: dict[str, list[str]]
) -> None:
    """
    Build a dummy computation graph for the state space model matrices.

    Create "dummy" ``pm.Flat`` variables representing the deep parameters used in the state space model, so
    post-estimation sampling functions can re-derive the statespace matrices from posterior draws.

    Parameters
    ----------
    ss_mod : PyMCStateSpace
        Model whose parameters to stand in for.
    coords : dict mapping str to sequence
        Coords the model was fit with, from
        :func:`~pymc_extras.statespace.core.fit_recovery.coords_from_idata`.
    dims : dict mapping str to list of str
        Dims each parameter was fit with, from
        :func:`~pymc_extras.statespace.core.fit_recovery.dims_from_idata`.

    Raises
    ------
    ValueError
        If a parameter's shape is not fully known and no dims were recovered to size it from.
    """

    def infer_variable_shape(name):
        shape = ss_mod._name_to_variable[name].type.shape
        if not any(dim is None for dim in shape):
            return shape

        dim_names = dims.get(name, None)
        if dim_names is None:
            raise ValueError(
                f"Could not infer a shape for {name}, which was given no coords when the model was "
                f"fit. If you did give it one, check its name: a dim called exactly "
                f"{name}_dim_0 cannot be told apart from the name PyMC generates for a variable "
                f"declared without dims, so it is discarded. Rename it."
            )

        shape_from_coords = tuple([len(coords[dim]) for dim in dim_names])
        return tuple(
            [shape[i] if shape[i] is not None else shape_from_coords[i] for i in range(len(shape))]
        )

    for name in ss_mod.param_names:
        pm.Flat(
            name,
            shape=infer_variable_shape(name),
            dims=dims.get(name, None),
        )


def kalman_filter_outputs_from_dummy_graph(
    ss_mod: "PyMCStateSpace",
    data: pt.TensorLike,
    *,
    coords: dict[str, Sequence],
    dims: dict[str, list[str]],
    exog: dict[str, dict],
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
        Statespace matrices with dummy parameters substituted in, still carrying the ``n_timesteps``
        placeholder. Pin it with :meth:`PyMCStateSpace._insert_constant_timestep` for the span you need.
    grouped_outputs : list of tuple of tensors
        A list of tuples, each containing the mean and covariance of the filtered, predicted, and smoothed
        states.
    """
    if scenario is None:
        scenario = dict()

    pm_mod = modelcontext(None)
    build_dummy_graph(ss_mod, coords=coords, dims=dims)
    matrices = ss_mod._insert_random_variables()

    for name in ss_mod.data_names:
        if name not in pm_mod:
            pm.Data(**exog[name])

    matrices = ss_mod._insert_data_variables(matrices)

    for name in ss_mod.data_names:
        if name in scenario.keys():
            pm.set_data({name: scenario[name]})

    # Pinned to the data length only for the filter below; the returned matrices keep the
    # n_timesteps placeholder so a caller can build over a different span.
    data_len = data.shape[0] if hasattr(data, "shape") else len(data)
    x0, P0, c, d, T, Z, R, H, Q = ss_mod._insert_constant_timestep(matrices, data_len)

    obs_coords = pm_mod.coords.get(OBS_STATE_DIM, None)

    data, nan_mask = register_data_with_pymc(
        data,
        n_obs=ss_mod.ssm.k_endog,
        obs_coords=obs_coords,
        data_dims=data_dims,
    )

    kalman_filter, kalman_smoother = ss_mod.make_filters()
    filter_outputs = kalman_filter.build_graph(data, x0, P0, c, d, T, Z, R, H, Q)

    filter_outputs.pop(-1)
    states, covariances = filter_outputs[:3], filter_outputs[3:]

    filtered_states, predicted_states, _ = states
    filtered_covariances, predicted_covariances, _ = covariances

    [smoothed_states, smoothed_covariances] = kalman_smoother.build_graph(
        T,
        R,
        Q,
        filtered_states,
        filtered_covariances,
    )

    grouped_outputs = [
        (filtered_states, filtered_covariances),
        (predicted_states, predicted_covariances),
        (smoothed_states, smoothed_covariances),
    ]

    return matrices, grouped_outputs
