from itertools import product
from typing import Literal

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


def make_default_labels(name: str, shape: tuple[int, ...]) -> list:
    if len(shape) == 0:
        return [name]

    return [list(range(dim)) for dim in shape]


def make_unpacked_variable_names(names: list[str], model: pm.Model) -> list[str]:
    coords = model.coords
    initial_point = model.initial_point()

    value_to_dim = {
        value.name: model.named_vars_to_dims.get(model.values_to_rvs[value].name, None)
        for value in model.value_vars
    }
    value_to_dim = {k: v for k, v in value_to_dim.items() if v is not None}

    rv_to_dim = model.named_vars_to_dims
    dims_dict = rv_to_dim | value_to_dim

    unpacked_variable_names = []
    for name in names:
        shape = initial_point[name].shape
        if shape:
            dims = dims_dict.get(name)
            if dims:
                labels_by_dim = [
                    coords[dim] if shape[i] == len(coords[dim]) else np.arange(shape[i])
                    for i, dim in enumerate(dims)
                ]
            else:
                labels_by_dim = make_default_labels(name, shape)
            labels = product(*labels_by_dim)
            unpacked_variable_names.extend(
                [f"{name}[{','.join(map(str, label))}]" for label in labels]
            )
        else:
            unpacked_variable_names.extend([name])
    return unpacked_variable_names


def map_results_to_inference_data(
    map_point: dict[str, float | int | np.ndarray],
    model: pm.Model | None = None,
    include_transformed: bool = True,
):
    """
    Add the MAP point to an InferenceData object in the posterior group.

    Unlike a typical posterior, the MAP point is a single point estimate rather than a distribution. As a result, it
    does not have a chain or draw dimension, and is stored as a single point in the posterior group.

    Parameters
    ----------
    map_point: dict
        A dictionary containing the MAP point estimates for each variable. The keys should be the variable names, and
        the values should be the corresponding MAP estimates.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.
    include_transformed: bool
        Whether to return transformed (unconstrained) variables in the constrained_posterior group. Default is True.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with the MAP point added to the posterior group.
    """

    model = pm.modelcontext(model) if model is None else model
    coords, dims = coords_and_dims_for_inferencedata(model)
    initial_point = model.initial_point()

    # The MAP point will have both the transformed and untransformed variables, so we need to ensure that
    # we have the correct dimensions for each variable.
    var_name_to_value_name = {
        rv.name: value.name
        for rv, value in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }
    dims.update(
        {
            value_name: dims[var_name]
            for var_name, value_name in var_name_to_value_name.items()
            if var_name in dims and (initial_point[value_name].shape == map_point[var_name].shape)
        }
    )

    constrained_names = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    ]
    all_varnames = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=True)
    ]

    unconstrained_names = set(all_varnames) - set(constrained_names)

    idata = az.from_dict(
        posterior={
            k: np.expand_dims(v, (0, 1)) for k, v in map_point.items() if k in constrained_names
        },
        coords=coords,
        dims=dims,
    )

    if unconstrained_names and include_transformed:
        unconstrained_posterior = az.from_dict(
            posterior={
                k: np.expand_dims(v, (0, 1))
                for k, v in map_point.items()
                if k in unconstrained_names
            },
            coords=coords,
            dims=dims,
        )

        idata["unconstrained_posterior"] = unconstrained_posterior.posterior

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

    unpacked_variable_names = make_unpacked_variable_names(variable_names, model)

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
    unpacked_variable_names = make_unpacked_variable_names(variable_names, model)

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
