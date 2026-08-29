import numpy as np
import pymc as pm
import xarray as xr

from xarray import DataTree

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.idata_utils import make_unpacked_variable_names

_COVARIANCE_DIMS = {
    "standard_deviation": ("rows",),
    "cholesky_lower": ("rows", "columns"),
    "cov_factor": ("rows", "factors"),
    "cov_diag": ("rows",),
}


def add_fit_to_inference_data(
    idata: DataTree,
    guide: AutoGuideModel,
    params: dict[str, np.ndarray],
    model: pm.Model | None = None,
) -> DataTree:
    """Add the fitted guide's mean and covariance to a DataTree, in the ``fit`` group.

    The guide reports the covariance in whatever form it stores, so the group holds a
    marginal standard deviation for a mean-field guide, a Cholesky factor for a full-rank
    one, and a factor-plus-diagonal pair for a low-rank one. Which entries are present
    therefore identifies the family.

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
    quantities = guide.fit_quantities(params)

    value_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]
    rows = make_unpacked_variable_names(value_names, model)

    coords: dict[str, list[str] | np.ndarray] = {"rows": rows}
    data_vars = {"mean_vector": xr.DataArray(quantities["mean_vector"], dims=["rows"])}

    for name, values in quantities.items():
        if name == "mean_vector":
            continue
        dims = _COVARIANCE_DIMS[name]
        if "columns" in dims:
            coords["columns"] = rows
        if "factors" in dims:
            coords["factors"] = np.arange(values.shape[1])
        data_vars[name] = xr.DataArray(values, dims=list(dims))

    idata["fit"] = DataTree(dataset=xr.Dataset(data_vars, coords=coords))

    return idata


def add_optimizer_result_to_inference_data(
    idata: DataTree,
    *,
    loss_history: np.ndarray,
    step: int,
    optimizer_state: dict[str, np.ndarray],
) -> DataTree:
    """Add the optimization trace and the optimizer's own state to a DataTree.

    The group holds both what a reader wants to see and what a resumed run needs: the ELBO
    over the steps taken so far, and the optimizer's moment buffers and clocks. Each state
    variable keeps its own name and shape, so restoring one is a lookup rather than an
    unpacking.

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

    Returns
    -------
    idata : DataTree
        The provided tree, with the ``optimizer_result`` group added.
    """
    loss_history = np.asarray(loss_history, dtype=float)

    data_vars = {
        "elbo": xr.DataArray(-loss_history, dims=["step"]),
        "step_count": xr.DataArray(np.asarray(step)),
    }
    for name, value in optimizer_state.items():
        value = np.asarray(value)
        dims = [f"{name}_dim_{axis}" for axis in range(value.ndim)]
        data_vars[name] = xr.DataArray(value, dims=dims)

    coords = {"step": np.arange(loss_history.size)}
    idata["optimizer_result"] = DataTree(dataset=xr.Dataset(data_vars, coords=coords))

    return idata
