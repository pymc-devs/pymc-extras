#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Utilities for converting Pathfinder results to xarray and adding them to InferenceData."""

from __future__ import annotations

import warnings
from dataclasses import asdict

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from pymc.blocking import DictToArrayBijection

from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus
from pymc_extras.inference.pathfinder.pathfinder import (
    MultiPathfinderResult,
    PathfinderConfig,
    PathfinderResult,
    PathStatus,
)


def get_param_coords(model: pm.Model | None, n_params: int) -> list[str]:
    """
    Get parameter coordinate labels from PyMC model.
    
    Parameters
    ----------
    model : pm.Model | None
        PyMC model to extract variable names from. If None, returns numeric indices.
    n_params : int
        Number of parameters (for fallback indexing when model is None)
        
    Returns
    -------
    list[str]
        Parameter coordinate labels
    """
    if model is None:
        return [str(i) for i in range(n_params)]

    ip = model.initial_point()
    bij = DictToArrayBijection.map(ip)

    coords = []
    for var_name, shape, size, _ in bij.point_map_info:
        if size == 1:
            coords.append(var_name)
        else:
            for i in range(size):
                coords.append(f"{var_name}[{i}]")
    return coords


def _status_counter_to_dataarray(counter, status_enum_cls) -> xr.DataArray:
    """Convert a Counter of status values to a dense xarray DataArray."""
    all_statuses = list(status_enum_cls)
    status_names = [s.name for s in all_statuses]
    
    counts = np.array([counter.get(status, 0) for status in all_statuses])
    
    return xr.DataArray(
        counts,
        dims=["status"],
        coords={"status": status_names},
        name="status_counts"
    )


def _extract_scalar(value):
    """Extract scalar from array-like or return as-is."""
    if hasattr(value, 'item'):
        return value.item()
    elif hasattr(value, '__len__') and len(value) == 1:
        return value[0]
    return value


def pathfinder_result_to_xarray(
    result: PathfinderResult,
    model: pm.Model | None = None,
) -> xr.Dataset:
    """
    Convert a PathfinderResult to an xarray Dataset.
    
    Parameters
    ----------
    result : PathfinderResult
        Single pathfinder run result
    model : pm.Model | None
        PyMC model for parameter name extraction
        
    Returns
    -------
    xr.Dataset
        Dataset with pathfinder results
        
    Examples
    --------
    >>> import pymc as pm
    >>> import pymc_extras as pmx
    >>> 
    >>> with pm.Model() as model:
    ...     x = pm.Normal("x", 0, 1)
    ...     y = pm.Normal("y", x, 1, observed=2.0)
    ... 
    >>> # Assuming we have a PathfinderResult from a pathfinder run
    >>> ds = pathfinder_result_to_xarray(result, model=model)
    >>> print(ds.data_vars)  # Shows lbfgs_niter, elbo_argmax, status info, etc.
    >>> print(ds.attrs)      # Shows metadata like lbfgs_status, path_status
    """
    data_vars = {}
    coords = {}
    attrs = {}
    
    n_params = None
    if result.samples is not None:
        n_params = result.samples.shape[-1]
    elif hasattr(result, 'lbfgs_niter') and result.lbfgs_niter is not None:
        if model is not None:
            try:
                ip = model.initial_point()
                n_params = len(DictToArrayBijection.map(ip).data)
            except Exception:
                pass
    
    if n_params is not None:
        coords["param"] = get_param_coords(model, n_params)
    
    if result.lbfgs_niter is not None:
        data_vars["lbfgs_niter"] = xr.DataArray(_extract_scalar(result.lbfgs_niter))
    
    if result.elbo_argmax is not None:
        data_vars["elbo_argmax"] = xr.DataArray(_extract_scalar(result.elbo_argmax))
    
    data_vars["lbfgs_status_code"] = xr.DataArray(result.lbfgs_status.value)
    data_vars["lbfgs_status_name"] = xr.DataArray(result.lbfgs_status.name)
    data_vars["path_status_code"] = xr.DataArray(result.path_status.value)
    data_vars["path_status_name"] = xr.DataArray(result.path_status.name)
    
    if n_params is not None and result.samples is not None:
        if result.samples.ndim >= 2:
            representative_sample = result.samples[0, -1, :]
            data_vars["final_sample"] = xr.DataArray(
                representative_sample,
                dims=["param"],
                coords={"param": coords["param"]}
            )
    
    if result.logP is not None:
        logP = result.logP.flatten() if hasattr(result.logP, 'flatten') else result.logP
        if hasattr(logP, '__len__') and len(logP) > 0:
            data_vars["logP_mean"] = xr.DataArray(np.mean(logP))
            data_vars["logP_std"] = xr.DataArray(np.std(logP))
            data_vars["logP_max"] = xr.DataArray(np.max(logP))
    
    if result.logQ is not None:
        logQ = result.logQ.flatten() if hasattr(result.logQ, 'flatten') else result.logQ
        if hasattr(logQ, '__len__') and len(logQ) > 0:
            data_vars["logQ_mean"] = xr.DataArray(np.mean(logQ))
            data_vars["logQ_std"] = xr.DataArray(np.std(logQ))
            data_vars["logQ_max"] = xr.DataArray(np.max(logQ))
    
    attrs["lbfgs_status"] = result.lbfgs_status.name
    attrs["path_status"] = result.path_status.name
    
    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    
    return ds


def multipathfinder_result_to_xarray(
    result: MultiPathfinderResult,
    model: pm.Model | None = None,
    *,
    store_diagnostics: bool = False,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Convert a MultiPathfinderResult to xarray Datasets.
    
    Parameters
    ----------
    result : MultiPathfinderResult
        Multi-path pathfinder result
    model : pm.Model | None
        PyMC model for parameter name extraction
    store_diagnostics : bool
        Whether to include potentially large diagnostic arrays
        
    Returns
    -------
    tuple[xr.Dataset, xr.Dataset | None]
        Summary dataset and optional per-path dataset
        
    Examples
    --------
    >>> import pymc as pm
    >>> import pymc_extras as pmx
    >>> 
    >>> with pm.Model() as model:
        ...     x = pm.Normal("x", 0, 1)
    ... 
    >>> # Assuming we have a MultiPathfinderResult from multiple pathfinder runs
    >>> summary_ds, paths_ds = multipathfinder_result_to_xarray(result, model=model)
    >>> print("Summary:", summary_ds.data_vars)
    >>> if paths_ds is not None:
    ...     print("Per-path:", paths_ds.data_vars)
    """
    n_params = result.samples.shape[-1] if result.samples is not None else None
    param_coords = get_param_coords(model, n_params) if n_params is not None else None
    
    summary_ds = _build_summary_dataset(
        result, param_coords, LBFGSStatus, PathStatus
    )
    
    paths_ds = None
    if not result.all_paths_failed and result.samples is not None:
        paths_ds = _build_paths_dataset(
            result, param_coords, n_params
        )
    
    return summary_ds, paths_ds


def _determine_num_paths(result: MultiPathfinderResult) -> int:
    """
    Determine the number of paths from per-path arrays.
    
    When importance sampling is applied, result.samples may be collapsed,
    so we use per-path diagnostic arrays to determine the true path count.
    """
    if result.lbfgs_niter is not None:
        return len(result.lbfgs_niter)
    elif result.elbo_argmax is not None:
        return len(result.elbo_argmax)
    elif result.logP is not None:
        return result.logP.shape[0]
    elif result.logQ is not None:
        return result.logQ.shape[0]
    
    if result.lbfgs_status:
        return sum(result.lbfgs_status.values())
    elif result.path_status:
        return sum(result.path_status.values())
    
    if result.samples is not None:
        return result.samples.shape[0]
    
    raise ValueError("Cannot determine number of paths from result")


def _build_summary_dataset(
    result: MultiPathfinderResult,
    param_coords: list[str] | None,
    lbfgs_status_enum,
    path_status_enum,
) -> xr.Dataset:
    """Build the summary dataset with aggregate statistics."""
    data_vars = {}
    coords = {}
    attrs = {}
    
    if param_coords is not None:
        coords["param"] = param_coords
    
    if result.num_paths is not None:
        data_vars["num_paths"] = xr.DataArray(result.num_paths)
    if result.num_draws is not None:
        data_vars["num_draws"] = xr.DataArray(result.num_draws)
    
    if result.compile_time is not None:
        data_vars["compile_time"] = xr.DataArray(result.compile_time)
    if result.compute_time is not None:
        data_vars["compute_time"] = xr.DataArray(result.compute_time)
        if result.compile_time is not None:
            data_vars["total_time"] = xr.DataArray(
                result.compile_time + result.compute_time
            )
    
    data_vars["importance_sampling_method"] = xr.DataArray(
        result.importance_sampling or "none"
    )
    if result.pareto_k is not None:
        data_vars["pareto_k"] = xr.DataArray(result.pareto_k)
    
    if result.lbfgs_status:
        data_vars["lbfgs_status_counts"] = _status_counter_to_dataarray(
            result.lbfgs_status, lbfgs_status_enum
        )
    if result.path_status:
        data_vars["path_status_counts"] = _status_counter_to_dataarray(
            result.path_status, path_status_enum
        )
    
    data_vars["all_paths_failed"] = xr.DataArray(result.all_paths_failed)
    if not result.all_paths_failed and result.samples is not None:
        data_vars["num_successful_paths"] = xr.DataArray(result.samples.shape[0])
    
    if result.lbfgs_niter is not None:
        data_vars["lbfgs_niter_mean"] = xr.DataArray(np.mean(result.lbfgs_niter))
        data_vars["lbfgs_niter_std"] = xr.DataArray(np.std(result.lbfgs_niter))
    
    if result.elbo_argmax is not None:
        data_vars["elbo_argmax_mean"] = xr.DataArray(np.mean(result.elbo_argmax))
        data_vars["elbo_argmax_std"] = xr.DataArray(np.std(result.elbo_argmax))
    
    if result.logP is not None:
        data_vars["logP_mean"] = xr.DataArray(np.mean(result.logP))
        data_vars["logP_std"] = xr.DataArray(np.std(result.logP))
        data_vars["logP_max"] = xr.DataArray(np.max(result.logP))
    
    if result.logQ is not None:
        data_vars["logQ_mean"] = xr.DataArray(np.mean(result.logQ))
        data_vars["logQ_std"] = xr.DataArray(np.std(result.logQ))
        data_vars["logQ_max"] = xr.DataArray(np.max(result.logQ))
    
    if result.pathfinder_config is not None:
        attrs["pathfinder_config"] = asdict(result.pathfinder_config)
    if result.warnings:
        attrs["warnings"] = list(result.warnings)
    
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _build_paths_dataset(
    result: MultiPathfinderResult,
    param_coords: list[str] | None,
    n_params: int | None,
) -> xr.Dataset:
    """Build the per-path dataset with individual path diagnostics."""
    n_paths = _determine_num_paths(result)
    
    coords = {"path": list(range(n_paths))}
    if param_coords is not None:
        coords["param"] = param_coords
    
    data_vars = {}
    
    def _add_path_scalar(name: str, data):
        """Add a per-path scalar array to data_vars."""
        if data is not None:
            data_vars[name] = xr.DataArray(
                data,
                dims=["path"],
                coords={"path": coords["path"]}
            )
    
    _add_path_scalar("lbfgs_niter", result.lbfgs_niter)
    _add_path_scalar("elbo_argmax", result.elbo_argmax)
    
    if result.logP is not None:
        _add_path_scalar("logP_mean", np.mean(result.logP, axis=1))
        _add_path_scalar("logP_max", np.max(result.logP, axis=1))
    
    if result.logQ is not None:
        _add_path_scalar("logQ_mean", np.mean(result.logQ, axis=1))
        _add_path_scalar("logQ_max", np.max(result.logQ, axis=1))
    
    if n_params is not None and result.samples is not None and result.samples.ndim >= 3:
        final_samples = result.samples[:, -1, :]  # (S, N)
        data_vars["final_sample"] = xr.DataArray(
            final_samples,
            dims=["path", "param"],
            coords=coords
        )
    
    return xr.Dataset(data_vars, coords=coords)


def _build_config_dataset(config: "PathfinderConfig") -> xr.Dataset:
    """Build configuration dataset from PathfinderConfig."""
    data_vars = {}
    
    # Convert all config fields to DataArrays
    config_dict = asdict(config)
    for key, value in config_dict.items():
        data_vars[key] = xr.DataArray(value)
    
    return xr.Dataset(data_vars)


def _build_diagnostics_dataset(
    result: "MultiPathfinderResult", 
    model: pm.Model | None = None
) -> xr.Dataset | None:
    """Build diagnostics dataset with detailed diagnostic arrays."""
    data_vars = {}
    coords = {}
    
    n_params = None
    if result.samples is not None:
        n_params = result.samples.shape[-1]
    param_coords = get_param_coords(model, n_params) if n_params is not None else None
    
    if param_coords is not None:
        coords["param"] = param_coords
    
    if result.logP is not None:
        n_paths, n_draws_per_path = result.logP.shape
        coords["path"] = list(range(n_paths))
        coords["draw_per_path"] = list(range(n_draws_per_path))
        
        data_vars["logP_full"] = xr.DataArray(
            result.logP,
            dims=["path", "draw_per_path"],
            coords={"path": coords["path"], "draw_per_path": coords["draw_per_path"]}
        )
    
    if result.logQ is not None:
        if "path" not in coords or "draw_per_path" not in coords:
            n_paths, n_draws_per_path = result.logQ.shape
            coords["path"] = list(range(n_paths))
            coords["draw_per_path"] = list(range(n_draws_per_path))
        
        data_vars["logQ_full"] = xr.DataArray(
            result.logQ,
            dims=["path", "draw_per_path"],
            coords={"path": coords["path"], "draw_per_path": coords["draw_per_path"]}
        )
    
    if (result.samples is not None and 
        result.samples.ndim == 3 and 
        param_coords is not None):
        n_paths, n_draws_per_path, n_params = result.samples.shape
        
        if "path" not in coords or "draw_per_path" not in coords:
            coords["path"] = list(range(n_paths))
            coords["draw_per_path"] = list(range(n_draws_per_path))
        
        data_vars["samples_full"] = xr.DataArray(
            result.samples,
            dims=["path", "draw_per_path", "param"],
            coords=coords
        )
    
    if data_vars:
        return xr.Dataset(data_vars, coords=coords)
    return None


def add_pathfinder_to_inference_data(
    idata: az.InferenceData,
    result: PathfinderResult | MultiPathfinderResult,
    model: pm.Model | None = None,
    *,
    group: str = "pathfinder",
    paths_group: str = "pathfinder_paths", 
    diagnostics_group: str = "pathfinder_diagnostics",
    config_group: str = "pathfinder_config",
    store_diagnostics: bool = False,
) -> az.InferenceData:
    """
    Add pathfinder results to an ArviZ InferenceData object.
    
    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object to modify
    result : PathfinderResult | MultiPathfinderResult
        Pathfinder results to add
    model : pm.Model | None
        PyMC model for parameter name extraction
    group : str
        Name for the main pathfinder group
    paths_group : str
        Name for the per-path group (MultiPathfinderResult only)
    diagnostics_group : str
        Name for diagnostics group (if store_diagnostics=True)
    config_group : str
        Name for configuration group
    store_diagnostics : bool
        Whether to include potentially large diagnostic arrays
        
    Returns
    -------
    az.InferenceData
        Modified InferenceData object with pathfinder groups added
        
    Examples
    --------
    >>> import pymc as pm
    >>> import pymc_extras as pmx
    >>> 
    >>> with pm.Model() as model:
    ...     x = pm.Normal("x", 0, 1)
    ...     idata = pmx.fit(method="pathfinder", model=model, add_pathfinder_groups=False)
    ... 
    >>> # Assuming we have pathfinder results
    >>> idata = add_pathfinder_to_inference_data(idata, results, model=model)
    >>> print(list(idata.groups()))
    """
    groups_to_add = {}
    
    # Detect if this is a multi-path result
    # Use isinstance() as primary check, but fall back to duck typing for compatibility
    # with mocks and testing (MultiPathfinderResult has Counter-type status fields)
    is_multipath = isinstance(result, MultiPathfinderResult) or (
        hasattr(result, 'lbfgs_status') and 
        hasattr(result.lbfgs_status, 'values') and 
        callable(getattr(result.lbfgs_status, 'values'))
    )
    
    if is_multipath:
        summary_ds, paths_ds = multipathfinder_result_to_xarray(
            result, model=model, store_diagnostics=store_diagnostics
        )
        
        if group in idata.groups():
            warnings.warn(f"Group '{group}' already exists in InferenceData, it will be replaced.")
        groups_to_add[group] = summary_ds
        
        if paths_ds is not None:
            if paths_group in idata.groups():
                warnings.warn(f"Group '{paths_group}' already exists in InferenceData, it will be replaced.")
            groups_to_add[paths_group] = paths_ds
        
        if store_diagnostics:
            diagnostics_ds = _build_diagnostics_dataset(result, model)
            if diagnostics_ds is not None:
                if diagnostics_group in idata.groups():
                    warnings.warn(f"Group '{diagnostics_group}' already exists in InferenceData, it will be replaced.")
                groups_to_add[diagnostics_group] = diagnostics_ds
        
        if result.pathfinder_config is not None:
            config_ds = _build_config_dataset(result.pathfinder_config)
            if config_group in idata.groups():
                warnings.warn(f"Group '{config_group}' already exists in InferenceData, it will be replaced.")
            groups_to_add[config_group] = config_ds
            
    else:
        ds = pathfinder_result_to_xarray(
            result, model=model
        )
        
        if group in idata.groups():
            warnings.warn(f"Group '{group}' already exists in InferenceData, it will be replaced.")
        groups_to_add[group] = ds
    
    idata.add_groups(groups_to_add)
    
    return idata
