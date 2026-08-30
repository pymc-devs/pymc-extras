from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pymc as pm
import xarray as xr

from pymc.blocking import DictToArrayBijection
from xarray import DataTree

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.idata_utils import make_unpacked_variable_names

# Dims labelled by the guide parameter they index, as opposed to a bare integer range.
_PARAMETER_DIMS = ("rows", "columns")


def add_fit_to_inference_data(
    idata: DataTree,
    guide: AutoGuideModel,
    params: dict[str, np.ndarray],
    model: pm.Model | None = None,
) -> DataTree:
    """Add the fitted guide's mean and covariance to a DataTree, in the ``fit`` group.

    The guide reports the covariance in whatever form it stores, so the group holds a
    marginal standard deviation for a mean-field guide, a Cholesky factor for a full-rank
    one, and a factor plus a diagonal standard deviation for a low-rank one. Which entries
    are present therefore identifies the family.

    Parameters
    ----------
    idata : DataTree
        The tree to add the group to.
    guide : AutoGuideModel
        The fitted guide.
    params : dict of str to ndarray
        Guide parameter values, keyed as in :attr:`AutoGuideModel.params_init_values`.
    model : Model, optional
        The PyMC model the guide approximates. If None, the model is taken from the
        context stack.

    Returns
    -------
    idata : DataTree
        The provided tree, with the ``fit`` group added.
    """
    model = pm.modelcontext(model)
    quantities = dict(guide.fit_quantities(params))
    mean_vector = quantities.pop("mean_vector")

    value_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]
    rows = make_unpacked_variable_names(value_names, model)

    covariance_dims = guide.covariance_dims
    if undeclared := sorted(set(quantities) - set(covariance_dims)):
        raise ValueError(
            f"{type(guide).__name__} reported covariance quantities {undeclared} that it "
            f"declares no dims for. Its covariance_dims names {sorted(covariance_dims)}."
        )

    coords: dict[str, list[str] | np.ndarray] = {"rows": rows}
    data_vars = {"mean_vector": xr.DataArray(mean_vector, dims=["rows"])}

    for name, values in quantities.items():
        dims = covariance_dims[name]
        for axis, dim in enumerate(dims):
            coords.setdefault(
                dim, rows if dim in _PARAMETER_DIMS else np.arange(values.shape[axis])
            )
        data_vars[name] = xr.DataArray(values, dims=list(dims))

    idata["fit"] = DataTree(dataset=xr.Dataset(data_vars, coords=coords))

    return idata


def _split_buffer_name(name: str, parameter_names: Sequence[str]) -> tuple[str | None, str]:
    """Separate an optimizer buffer's name into the parameter it tracks and its kind.

    A per-parameter buffer is named ``{kind}_{parameter}``, so ``adam_m_theta_loc`` tracks
    ``theta_loc`` under the kind ``adam_m``. The longest parameter name wins, so a guide
    holding both ``loc`` and ``theta_loc`` splits unambiguously. A clock such as ``adam_t``
    tracks no parameter and comes back with ``None``.
    """
    for parameter in sorted(parameter_names, key=len, reverse=True):
        suffix = f"_{parameter}"
        if name.endswith(suffix):
            return parameter, name.removesuffix(suffix)
    return None, name


def _unpacked_labels(point_map_info: Sequence[tuple]) -> list[str]:
    """Label every scalar element of a raveled parameter vector, in C order."""
    labels = []
    for name, shape, *_ in point_map_info:
        if not shape:
            labels.append(name)
        else:
            labels.extend(f"{name}[{','.join(map(str, index))}]" for index in np.ndindex(*shape))
    return labels


def add_optimizer_result_to_inference_data(
    idata: DataTree,
    *,
    loss_history: np.ndarray,
    step: int,
    optimizer_state: dict[str, np.ndarray],
    parameter_names: Sequence[str],
) -> DataTree:
    """Add the optimization trace and the optimizer's state to a DataTree.

    The group holds both what a reader wants to see and what a resumed run needs: the ELBO
    over the steps taken so far, and the optimizer's moment buffers and clocks.

    Ravel each kind's per-parameter buffers into one variable over a labelled ``parameter``
    dim. Clocks belong to no parameter and stay scalars.

    Parameters
    ----------
    idata : DataTree
        The tree to add the group to.
    loss_history : ndarray
        Negative ELBO at each step taken so far.
    step : int
        Total number of optimization steps taken.
    optimizer_state : dict of str to ndarray
        The optimizer's shared variable values, keyed by variable name.
    parameter_names : sequence of str
        The guide's parameter names, in the order the raveled vectors follow.

    Returns
    -------
    idata : DataTree
        The provided tree, with the ``optimizer_result`` group added.

    Raises
    ------
    ValueError
        If a buffer kind covers only some of the guide's parameters.
    """
    loss_history = np.asarray(loss_history, dtype=float)

    buffers_by_kind: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    clocks: dict[str, np.ndarray] = {}
    for name, value in optimizer_state.items():
        parameter, kind = _split_buffer_name(name, parameter_names)
        if parameter is None:
            clocks[kind] = np.asarray(value)
        else:
            buffers_by_kind[kind][parameter] = np.asarray(value)

    data_vars: dict[str, xr.DataArray] = {
        "elbo": xr.DataArray(-loss_history, dims=["step"]),
        "step_count": xr.DataArray(np.asarray(step)),
    }
    coords: dict[str, np.ndarray | list[str]] = {"step": np.arange(loss_history.size)}

    for kind, value in clocks.items():
        data_vars[kind] = xr.DataArray(value)

    for kind, buffers in buffers_by_kind.items():
        if set(buffers) != set(parameter_names):
            missing = sorted(set(parameter_names) - set(buffers))
            raise ValueError(
                f"the optimizer's {kind!r} buffers cover only some of the guide's "
                f"parameters, missing {missing}. Every kind must cover all of them to be "
                "raveled into one vector."
            )
        raveled = DictToArrayBijection.map({name: buffers[name] for name in parameter_names})
        if "parameter" not in coords:
            coords["parameter"] = _unpacked_labels(raveled.point_map_info)
        data_vars[kind] = xr.DataArray(raveled.data, dims=["parameter"])

    idata["optimizer_result"] = DataTree(dataset=xr.Dataset(data_vars, coords=coords))

    return idata
