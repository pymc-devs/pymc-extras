from functools import reduce
from itertools import product
from typing import Any, Literal

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from arviz import dict_to_dataset
from better_optimize.constants import minimize_method
from pymc.backends.arviz import coords_and_dims_for_inferencedata, find_constants, find_observations
from pymc.blocking import RaveledVars
from pymc.util import get_default_varnames
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import LinearOperator


def make_unpacked_variable_names(name, model: pm.Model) -> list[str]:
    coords = model.coords

    value_to_dim = {
        x.name: model.named_vars_to_dims.get(model.values_to_rvs[x].name, None)
        for x in model.value_vars
    }
    value_to_dim = {k: v for k, v in value_to_dim.items() if v is not None}

    rv_to_dim = model.named_vars_to_dims
    dims_dict = rv_to_dim | value_to_dim

    dims = dims_dict.get(name)
    if dims is None:
        return [name]
    labels = product(*(coords[dim] for dim in dims))
    return [f"{name}[{','.join(map(str, label))}]" for label in labels]


def map_results_to_inference_data(results: dict[str, Any], model: pm.Model | None = None):
    """
    Convert a dictionary of results to an InferenceData object.

    Parameters
    ----------
    results: dict
        A dictionary containing the results to convert.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        An InferenceData object containing the results.
    """
    model = pm.modelcontext(model) if model is None else model
    coords, dims = coords_and_dims_for_inferencedata(model)

    idata = az.convert_to_inference_data(results, coords=coords, dims=dims)
    return idata


def laplace_draws_to_inferencedata(
    posterior_draws: list[np.ndarray[float | int]], model: pm.Model | None = None
) -> az.InferenceData:
    """
    Convert draws from a posterior estimated with the Laplace approximation to an InferenceData object.


    Parameters
    ----------
    posterior_draws: list of np.ndarray
        A list of arrays containing the posterior draws. Each array should have shape (chains, draws, *shape), where
        shape is the shape of the variable in the posterior.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples
    """
    model = pm.modelcontext(model) if model is None else model
    chains, draws, *_ = posterior_draws[0].shape

    def make_rv_coords(name):
        coords = {"chain": range(chains), "draw": range(draws)}
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return coords
        return coords | {dim: list(model.coords[dim]) for dim in extra_dims}

    def make_rv_dims(name):
        dims = ["chain", "draw"]
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return dims
        return dims + list(extra_dims)

    names = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    ]
    idata = {
        name: xr.DataArray(
            data=draws,
            coords=make_rv_coords(name),
            dims=make_rv_dims(name),
            name=name,
        )
        for name, draws in zip(names, posterior_draws)
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.convert_to_inference_data(idata, coords=coords, dims=dims)

    return idata


def add_map_posterior_to_inference_data(
    idata: az.InferenceData,
    map_point: dict[str, float | int | np.ndarray],
    model: pm.Model | None = None,
):
    """
    Add the MAP point to an InferenceData object in the posterior group.

    Unlike a typical posterior, the MAP point is a single point estimate rather than a distribution. As a result, it
    does not have a chain or draw dimension, and is stored as a single point in the posterior group.

    Parameters
    ----------
    idata: az.InferenceData
        An InferenceData object to which the MAP point will be added.
    map_point: dict
        A dictionary containing the MAP point estimates for each variable. The keys should be the variable names, and
        the values should be the corresponding MAP estimates.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with the MAP point added to the posterior group.
    """

    model = pm.modelcontext(model) if model is None else model
    coords, dims = coords_and_dims_for_inferencedata(model)

    # The MAP point will have both the transformed and untransformed variables, so we need to ensure that
    # we have the correct dimensions for each variable.
    var_name_to_value_name = {rv.name: value.name for rv, value in model.rvs_to_values.items()}
    dims.update(
        {
            value_name: dims[var_name]
            for var_name, value_name in var_name_to_value_name.items()
            if var_name in dims
        }
    )

    posterior_data = {
        name: xr.DataArray(
            data=np.asarray(value),
            coords={dim: coords[dim] for dim in dims.get(name, [])},
            dims=dims.get(name),
            name=name,
        )
        for name, value in map_point.items()
    }
    idata.add_groups(posterior=xr.Dataset(posterior_data))

    return idata


def add_fit_to_inference_data(
    idata: az.InferenceData, mu: RaveledVars, H_inv: np.ndarray, model: pm.Model | None = None
) -> az.InferenceData:
    """
    Add the mean vector and covariance matrix of the Laplace approximation to an InferenceData object.

    Parameters
    ----------
    idata: az.InfereceData
        An InferenceData object containing the approximated posterior samples.
    mu: RaveledVars
        The MAP estimate of the model parameters.
    H_inv: np.ndarray
        The inverse Hessian matrix of the log-posterior evaluated at the MAP estimate.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with the mean vector and covariance matrix added to the "fit" group.
    """
    model = pm.modelcontext(model) if model is None else model

    variable_names, *_ = zip(*mu.point_map_info)

    unpacked_variable_names = reduce(
        lambda lst, name: lst + make_unpacked_variable_names(name, model), variable_names, []
    )

    mean_dataarray = xr.DataArray(mu.data, dims=["rows"], coords={"rows": unpacked_variable_names})

    data = {"mean_vector": mean_dataarray}

    if H_inv is not None:
        cov_dataarray = xr.DataArray(
            H_inv,
            dims=["rows", "columns"],
            coords={"rows": unpacked_variable_names, "columns": unpacked_variable_names},
        )
        data["covariance_matrix"] = cov_dataarray

    dataset = xr.Dataset(data)
    idata.add_groups(fit=dataset)

    return idata


def add_data_to_inference_data(
    idata: az.InferenceData,
    progressbar: bool = True,
    model: pm.Model | None = None,
    compile_kwargs: dict | None = None,
) -> az.InferenceData:
    """
    Add observed and constant data to an InferenceData object.

    Parameters
    ----------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples.
    progressbar: bool
        Whether to display a progress bar during computations. Default is True.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to pytensor.function.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with observed and constant data added.
    """
    model = pm.modelcontext(model) if model is None else model

    if model.deterministics:
        expand_dims = {}
        if "chain" not in idata.posterior.coords:
            expand_dims["chain"] = [0]
        if "draw" not in idata.posterior.coords:
            expand_dims["draw"] = [0]

        idata.posterior = pm.compute_deterministics(
            idata.posterior.expand_dims(expand_dims),
            model=model,
            merge_dataset=True,
            progressbar=progressbar,
            compile_kwargs=compile_kwargs,
        )

    coords, dims = coords_and_dims_for_inferencedata(model)

    observed_data = dict_to_dataset(
        find_observations(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    constant_data = dict_to_dataset(
        find_constants(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    idata.add_groups(
        {"observed_data": observed_data, "constant_data": constant_data},
        coords=coords,
        dims=dims,
    )

    return idata


def optimizer_result_to_dataset(
    result: OptimizeResult,
    method: minimize_method | Literal["basinhopping"],
    mu: RaveledVars | None = None,
    model: pm.Model | None = None,
) -> xr.Dataset:
    """
    Convert an OptimizeResult object to an xarray Dataset object.

    Parameters
    ----------
    result: OptimizeResult
        The result of the optimization process.
    method: minimize_method or "basinhopping"
        The optimization method used.

    Returns
    -------
    dataset: xr.Dataset
        An xarray Dataset containing the optimization results.
    """
    if not isinstance(result, OptimizeResult):
        raise TypeError("result must be an instance of OptimizeResult")

    model = pm.modelcontext(model) if model is None else model
    variable_names, *_ = zip(*mu.point_map_info)
    unpacked_variable_names = reduce(
        lambda lst, name: lst + make_unpacked_variable_names(name, model), variable_names, []
    )

    data_vars = {}

    if hasattr(result, "lowest_optimization_result"):
        # If we did basinhopping, there's a results inside the results. We want to pop this out and collapse them,
        # overwriting outer keys with the inner keys
        inner_res = result.pop("lowest_optimization_result")
        for key in inner_res.keys():
            result[key] = inner_res[key]

    if hasattr(result, "x"):
        data_vars["x"] = xr.DataArray(
            result.x, dims=["variables"], coords={"variables": unpacked_variable_names}
        )
    if hasattr(result, "fun"):
        data_vars["fun"] = xr.DataArray(result.fun, dims=[])
    if hasattr(result, "success"):
        data_vars["success"] = xr.DataArray(result.success, dims=[])
    if hasattr(result, "message"):
        data_vars["message"] = xr.DataArray(str(result.message), dims=[])
    if hasattr(result, "jac") and result.jac is not None:
        jac = np.asarray(result.jac)
        if jac.ndim == 1:
            data_vars["jac"] = xr.DataArray(
                jac, dims=["variables"], coords={"variables": unpacked_variable_names}
            )
        else:
            data_vars["jac"] = xr.DataArray(
                jac,
                dims=["variables", "variables_aux"],
                coords={
                    "variables": unpacked_variable_names,
                    "variables_aux": unpacked_variable_names,
                },
            )

    if hasattr(result, "hess_inv") and result.hess_inv is not None:
        hess_inv = result.hess_inv
        if isinstance(hess_inv, LinearOperator):
            n = hess_inv.shape[0]
            eye = np.eye(n)
            hess_inv_mat = np.column_stack([hess_inv.matvec(eye[:, i]) for i in range(n)])
            hess_inv = hess_inv_mat
        else:
            hess_inv = np.asarray(hess_inv)
        data_vars["hess_inv"] = xr.DataArray(
            hess_inv,
            dims=["variables", "variables_aux"],
            coords={"variables": unpacked_variable_names, "variables_aux": unpacked_variable_names},
        )

    if hasattr(result, "nit"):
        data_vars["nit"] = xr.DataArray(result.nit, dims=[])
    if hasattr(result, "nfev"):
        data_vars["nfev"] = xr.DataArray(result.nfev, dims=[])
    if hasattr(result, "njev"):
        data_vars["njev"] = xr.DataArray(result.njev, dims=[])
    if hasattr(result, "status"):
        data_vars["status"] = xr.DataArray(result.status, dims=[])

    # Add any other fields present in result
    for key, value in result.items():
        if key in data_vars:
            continue  # already added
        if value is None:
            continue
        arr = np.asarray(value)

        # TODO: We can probably do something smarter here with a dictionary of all possible values and their expected
        #  dimensions.
        dims = [f"{key}_dim_{i}" for i in range(arr.ndim)]
        data_vars[key] = xr.DataArray(
            arr,
            dims=dims,
            coords={f"{key}_dim_{i}": np.arange(arr.shape[i]) for i in range(len(dims))},
        )

    data_vars["method"] = xr.DataArray(np.array(method), dims=[])

    return xr.Dataset(data_vars)


def add_optimizer_result_to_inference_data(
    idata: az.InferenceData,
    result: OptimizeResult,
    method: minimize_method | Literal["basinhopping"],
    mu: RaveledVars | None = None,
    model: pm.Model | None = None,
) -> az.InferenceData:
    """
    Add the optimization result to an InferenceData object.

    Parameters
    ----------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples.
    result: OptimizeResult
        The result of the optimization process.
    method: minimize_method or "basinhopping"
        The optimization method used.
    mu: RaveledVars, optional
        The MAP estimate of the model parameters.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with the optimization results added to the "optimizer" group.
    """
    dataset = optimizer_result_to_dataset(result, method=method, mu=mu, model=model)
    idata.add_groups({"optimizer_result": dataset})

    return idata
