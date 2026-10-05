import numpy as np
import pymc as pm
import xarray as xr

from xarray import DataTree

from pymc_extras.inference.advi.autoguide import AutoGuideModel
from pymc_extras.inference.idata_utils import make_unpacked_variable_names

# Dims labelled by the free RV element they index, as in the Laplace fit group.
_PARAMETER_DIMS = ("rows", "columns")


def add_fit_to_inference_data(
    idata: DataTree,
    guide: AutoGuideModel,
    params: dict[str, np.ndarray],
    model: pm.Model | None = None,
) -> DataTree:
    """Add the fitted guide's summary to a DataTree, in the ``fit`` group.

    The group holds whatever :meth:`AutoGuideModel.fit_quantities` reports, so its contents
    depend on the guide.

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

    value_names = [model.rvs_to_values[rv].name for rv in model.free_RVs]
    labels = make_unpacked_variable_names(value_names, model)

    fit = xr.Dataset(guide.fit_quantities(params))
    fit = fit.assign_coords({dim: labels for dim in _PARAMETER_DIMS if dim in fit.dims})

    idata["fit"] = DataTree(dataset=fit)

    return idata


def add_fit_stats_to_inference_data(idata: DataTree, loss_history: np.ndarray) -> DataTree:
    """Add the per-step diagnostics of the optimization to a DataTree, in the ``fit_stats`` group.

    Every variable in the group runs over the ``step`` dim.

    Parameters
    ----------
    idata : DataTree
        The tree to add the group to.
    loss_history : ndarray
        Negative ELBO at each step taken so far.

    Returns
    -------
    idata : DataTree
        The provided tree, with the ``fit_stats`` group added.
    """
    loss_history = np.asarray(loss_history, dtype=float)
    fit_stats = xr.Dataset(
        {"elbo": ("step", -loss_history)},
        coords={"step": np.arange(loss_history.size)},
    )

    idata["fit_stats"] = DataTree(dataset=fit_stats)

    return idata
