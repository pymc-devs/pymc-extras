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

from collections.abc import Mapping, Sequence
from typing import Any, Literal
from warnings import warn

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from arviz_base import dict_to_dataset
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection
from pymc.model.fgraph import clone_model
from pymc.util import RandomSeed, _get_seeds_per_chain
from pytensor.graph.traversal import graph_inputs
from xarray import DataTree

from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data

try:  # pragma: no cover - import path exists in supported PyMC versions
    from pymc.data import MinibatchOp
    from pymc.variational.minibatch_rv import MinibatchRandomVariable
except ImportError:  # pragma: no cover
    MinibatchOp = ()
    MinibatchRandomVariable = ()

_SAMPLE_KWARG_BLOCKLIST = {
    "draws",
    "tune",
    "chains",
    "cores",
    "random_seed",
    "progressbar",
    "model",
    "return_inferencedata",
    "compile_kwargs",
    "var_names",
    "discard_tuned_samples",
    "trace",
    "backend",
}


def _validate_subposteriors(subposteriors: np.ndarray, *, diagonal: bool) -> np.ndarray:
    samples = np.asarray(subposteriors, dtype=float)
    if samples.ndim != 3:
        raise ValueError("subposteriors must have shape (num_shards, num_samples, n_params).")
    num_shards, num_samples, n_params = samples.shape
    if num_shards < 2:
        raise ValueError("Consensus MC requires at least two subposteriors.")
    if num_samples < 2:
        raise ValueError("Consensus MC requires at least two samples per subposterior.")
    if n_params < 1:
        raise ValueError("Consensus MC requires at least one flattened parameter.")
    if not np.isfinite(samples).all():
        raise ValueError("subposteriors must contain only finite values.")
    if not diagonal and num_samples <= n_params:
        raise ValueError(
            "Full-covariance consensus MC requires more samples than flattened parameters."
        )
    return samples


def _as_seed(random_seed: RandomSeed = None) -> int:
    return int(_get_seeds_per_chain(random_seed, 1)[0])


def _full_covariance_nodes(samples: pt.TensorVariable, num_shards: int):
    denom = samples.shape[1] - 1
    covs = []
    for k in range(num_shards):
        centered = samples[k] - samples[k].mean(axis=0, keepdims=True)
        covs.append(pt.dot(centered.T, centered) / denom)
    covs = pt.stack(covs)
    precisions = pt.stack([pt.linalg.inv(covs[k]) for k in range(num_shards)])
    total_cov = pt.linalg.inv(precisions.sum(axis=0))
    weights = pt.einsum("ij,kjl->kil", total_cov, precisions)
    return covs, precisions, total_cov, weights


def merge_consensus(
    subposteriors: np.ndarray,
    *,
    draws: int | None = None,
    diagonal: bool = False,
    random_seed: RandomSeed = None,
) -> np.ndarray:
    """Merge subposterior draws with Scott et al. Consensus Monte Carlo."""
    samples_np = _validate_subposteriors(subposteriors, diagonal=diagonal)
    num_shards, num_samples, _ = samples_np.shape
    if draws is not None:
        if draws < 1:
            raise ValueError("draws must be a positive integer.")
        rng = np.random.default_rng(_as_seed(random_seed))
        draw_idxs = rng.integers(0, num_samples, size=(num_shards, draws))
        samples_np = samples_np[np.arange(num_shards)[:, None], draw_idxs]

    samples = pt.tensor3("subposteriors")
    if diagonal:
        precision = 1.0 / pt.var(samples, axis=1, ddof=1)
        normalized = precision / precision.sum(axis=0)
        merged = pt.einsum("kp,knp->np", normalized, samples)
    else:
        _, _, _, weights = _full_covariance_nodes(samples, num_shards)
        merged = pt.einsum("kij,knj->ni", weights, samples)

    return np.asarray(pytensor.function([samples], merged)(samples_np))


def estimate_parametric(
    subposteriors: np.ndarray,
    *,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Gaussian product approximation for Parametric MC."""
    samples_np = _validate_subposteriors(subposteriors, diagonal=diagonal)
    num_shards = samples_np.shape[0]
    samples = pt.tensor3("subposteriors")
    submeans = samples.mean(axis=1)

    if diagonal:
        subvars = pt.var(samples, axis=1, ddof=1)
        precision = 1.0 / subvars
        var = 1.0 / precision.sum(axis=0)
        mean = pt.einsum("kp,kp->p", var * precision, submeans)
        fn = pytensor.function([samples], [mean, var])
    else:
        _, _, cov, weights = _full_covariance_nodes(samples, num_shards)
        mean = pt.einsum("kij,kj->i", weights, submeans)
        fn = pytensor.function([samples], [mean, cov])

    mean_np, cov_np = fn(samples_np)
    return np.asarray(mean_np), np.asarray(cov_np)


def merge_parametric(
    subposteriors: np.ndarray,
    *,
    draws: int,
    diagonal: bool = False,
    random_seed: RandomSeed = None,
) -> np.ndarray:
    """Draw merged samples from the Neiswanger et al. Gaussian approximation."""
    if draws < 1:
        raise ValueError("draws must be a positive integer.")
    mean, cov = estimate_parametric(subposteriors, diagonal=diagonal)
    n_params = mean.shape[0]
    rng = pt.random.default_rng(_as_seed(random_seed))
    if diagonal:
        random_draws = pt.random.normal(
            loc=mean,
            scale=np.sqrt(cov),
            size=(draws, n_params),
            rng=rng,
        )
    else:
        random_draws = pt.random.multivariate_normal(mean, cov, size=(draws,), rng=rng)
    return np.asarray(pytensor.function([], random_draws)())


def _data_var_map(model: pm.Model) -> dict[str, Any]:
    return {var.name: var for var in model.data_vars}


def _validate_settable_data(model: pm.Model, name: str) -> None:
    data_vars = _data_var_map(model)
    if name not in data_vars or not hasattr(data_vars[name], "set_value"):
        raise TypeError(
            f"Consensus MC sharding requires mutable pm.Data containers; '{name}' is not settable."
        )


def _resolve_axis(model: pm.Model, name: str, axis: int | str) -> tuple[int, str]:
    dims = tuple(model.named_vars_to_dims.get(name, ()))
    if isinstance(axis, str):
        if axis not in dims:
            raise ValueError(
                "Automatic splitting requires named model dimensions; pass explicit shards for unnamed axes."
            )
        return dims.index(axis), axis
    if not dims or axis < 0 or axis >= len(dims):
        raise ValueError(
            "Automatic splitting requires named model dimensions; pass explicit shards for unnamed axes."
        )
    return axis, dims[axis]


def _resolve_shards(
    model: pm.Model,
    *,
    shards: Sequence[Mapping[str, Any]] | None,
    shard_coords: Sequence[Mapping[str, Sequence[Any]]] | None,
    split_data: Mapping[str, int | str] | None,
    num_shards: int | None,
    sharded_dims: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Sequence[Any]]], set[str]]:
    if (shards is None) == (split_data is None):
        raise ValueError("Specify exactly one of 'shards' or 'split_data'.")

    if split_data is not None:
        if sharded_dims is not None:
            raise ValueError(
                "sharded_dims is only valid with explicit shards; split_data derives sharded dimensions from named split axes."
            )
        if num_shards is None or num_shards < 2:
            raise ValueError("num_shards is required and must be >= 2 for split_data.")
        split_specs: dict[str, tuple[int, str, np.ndarray]] = {}
        axis_lengths = []
        for name, axis in split_data.items():
            _validate_settable_data(model, name)
            axis_int, dim = _resolve_axis(model, name, axis)
            value = np.asarray(model[name].get_value(borrow=False))
            axis_lengths.append(value.shape[axis_int])
            split_specs[name] = (axis_int, dim, value)
        if len(set(axis_lengths)) > 1:
            raise ValueError(
                "Automatic splitting requires all split axes to have the same length; pass explicit shards for ambiguous models."
            )
        if axis_lengths and axis_lengths[0] < num_shards:
            raise ValueError(
                "Automatic splitting would create empty shards; reduce num_shards or pass explicit non-empty shards."
            )

        resolved_shards = [dict() for _ in range(num_shards)]
        resolved_coords: list[dict[str, Sequence[Any]]] = [dict() for _ in range(num_shards)]
        resolved_dims = set()
        for name, (axis_int, dim, value) in split_specs.items():
            resolved_dims.add(dim)
            parts = np.array_split(value, num_shards, axis=axis_int)
            for k, part in enumerate(parts):
                resolved_shards[k][name] = part
            if dim in model.coords:
                coord_parts = np.array_split(np.asarray(model.coords[dim]), num_shards)
                for k, coord_part in enumerate(coord_parts):
                    resolved_coords[k][dim] = coord_part
        return resolved_shards, resolved_coords, resolved_dims

    assert shards is not None
    resolved_shards = [dict(shard) for shard in shards]
    k_shards = len(resolved_shards)
    if k_shards < 2:
        raise ValueError("Consensus MC requires at least two explicit shards.")
    if num_shards is not None and num_shards != k_shards:
        raise ValueError("num_shards must match len(shards) when explicit shards are provided.")
    keys = set(resolved_shards[0])
    for shard in resolved_shards:
        if set(shard) != keys:
            raise ValueError("All explicit shard dictionaries must have identical keys.")
        for name in shard:
            _validate_settable_data(model, name)

    if shard_coords is None:
        resolved_coords = [dict() for _ in resolved_shards]
    else:
        if len(shard_coords) != k_shards:
            raise ValueError("shard_coords must have the same length as shards.")
        resolved_coords = [dict(coords) for coords in shard_coords]

    if sharded_dims is not None:
        resolved_dims = set(sharded_dims)
    else:
        resolved_dims = set().union(*(coords.keys() for coords in resolved_coords))
        for name in keys:
            original = np.asarray(model[name].get_value(borrow=False))
            dims = tuple(model.named_vars_to_dims.get(name, ()))
            for shard in resolved_shards:
                value = np.asarray(shard[name])
                for axis, dim in enumerate(dims[: value.ndim]):
                    if axis < original.ndim and value.shape[axis] != original.shape[axis]:
                        resolved_dims.add(dim)
        has_non_scalar = any(
            np.asarray(value).ndim > 0 for shard in resolved_shards for value in shard.values()
        )
        if has_non_scalar and not resolved_dims:
            raise ValueError(
                "Explicit shards with non-scalar data require sharded_dims or shard_coords so global and shard-local dimensions are unambiguous."
            )
    return resolved_shards, resolved_coords, resolved_dims


def _walk_apply_nodes(outputs: Sequence[Any]):
    seen = set()
    stack = list(outputs)
    while stack:
        var = stack.pop()
        owner = getattr(var, "owner", None)
        if owner is None or owner in seen:
            continue
        seen.add(owner)
        yield owner
        stack.extend(owner.inputs)


def _check_unscaled_likelihood(model: pm.Model, likelihood_potentials: Sequence[str]) -> None:
    outputs = list(model.observed_RVs) + [model[name] for name in likelihood_potentials]
    minibatch_types = (MinibatchOp, MinibatchRandomVariable)
    for node in _walk_apply_nodes(outputs):
        if isinstance(node.op, minibatch_types):
            raise ValueError(
                "Consensus MC requires unscaled shard likelihoods; rebuild the model without total_size or pm.Minibatch scaling."
            )


def _validate_potentials(
    model: pm.Model,
    prior_potentials: Sequence[str] | None,
    likelihood_potentials: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    potential_names = {p.name for p in model.potentials}
    if not potential_names:
        return [], []
    if prior_potentials is None or likelihood_potentials is None:
        raise ValueError(
            "Models with pm.Potential terms must classify every potential as prior_potentials or likelihood_potentials for consensus prior scaling."
        )
    prior = list(prior_potentials)
    likelihood = list(likelihood_potentials)
    if set(prior).intersection(likelihood) or set(prior).union(likelihood) != potential_names:
        raise ValueError(
            "Models with pm.Potential terms must classify every potential as prior_potentials or likelihood_potentials for consensus prior scaling."
        )
    return prior, likelihood


def _validate_model_supported(model: pm.Model, sharded_dims: set[str]) -> None:
    if not model.free_RVs:
        raise ValueError("Consensus MC requires at least one unobserved continuous variable.")
    if model.discrete_value_vars:
        raise ValueError(
            "Consensus MC only supports continuous unobserved variables because subposterior merging is a Euclidean weighted average."
        )
    for rv in model.free_RVs:
        dims = tuple(model.named_vars_to_dims.get(rv.name, ()))
        for dim in dims:
            if dim in sharded_dims:
                raise ValueError(
                    f"Consensus MC can only merge global parameters; free RV '{rv.name}' is indexed by sharded dimension '{dim}'."
                )
        if sharded_dims and rv.ndim > 0 and not dims:
            raise ValueError(
                f"Consensus MC cannot prove unnamed non-scalar free RV '{rv.name}' is global; give it non-sharded dims or marginalize shard-local variables."
            )
    if any(value is not None for value in model.rvs_to_initial_values.values()):
        raise ValueError(
            "Consensus MC cannot clone models with non-default initvals; remove custom initval entries before calling fit_consensus_mc."
        )


def _validate_likelihood_coverage(
    model: pm.Model,
    *,
    sharded_data_names: set[str],
    global_data: Sequence[str],
    likelihood_potentials: Sequence[str],
) -> None:
    data_vars = _data_var_map(model)
    global_names = set(global_data)
    data_var_values = set(data_vars.values())
    factors: list[tuple[str, Any]] = []
    for rv in model.observed_RVs:
        if model.rvs_to_values[rv] not in data_var_values:
            raise TypeError(
                "Consensus MC requires observed likelihood data to be backed by mutable pm.Data containers."
            )
        for factor in model.logp(vars=[rv], sum=False):
            factors.append((rv.name, factor))
    for name in likelihood_potentials:
        for factor in model.logp(vars=[model[name]], sum=False):
            factors.append((name, factor))

    for factor_name, factor in factors:
        inputs = set(graph_inputs([factor]))
        used_data = {name for name, var in data_vars.items() if var in inputs}
        if not used_data:
            raise TypeError(
                "Consensus MC requires observed likelihood data to be backed by mutable pm.Data containers."
            )
        non_global_unsharded = []
        has_sharded = False
        for name in used_data:
            value = np.asarray(data_vars[name].get_value(borrow=False))
            if name in sharded_data_names:
                has_sharded = True
            elif name in global_names or value.ndim == 0:
                continue
            else:
                non_global_unsharded.append(name)
        if non_global_unsharded:
            names = sorted(non_global_unsharded)
            raise ValueError(
                f"Likelihood factor '{factor_name}' depends on unsharded data variables {names}; include them in shards/split_data or mark true global covariates with global_data."
            )
        if not has_sharded:
            raise ValueError(
                f"Likelihood factor '{factor_name}' has no sharded data input and would be counted once per shard."
            )


def _build_shard_model(
    model: pm.Model,
    *,
    num_shards: int,
    prior_potentials: Sequence[str],
    likelihood_potentials: Sequence[str],
) -> pm.Model:
    prior_potentials, likelihood_potentials = _validate_potentials(
        model, prior_potentials, likelihood_potentials
    )
    _validate_model_supported(model, set())
    _check_unscaled_likelihood(model, likelihood_potentials)

    shard_model = clone_model(model)
    prior_scale = 1.0 / num_shards
    with shard_model:
        prior_logp = shard_model.logp(vars=shard_model.free_RVs, jacobian=False)
        if prior_potentials:
            prior_logp = prior_logp + shard_model.logp(
                vars=[shard_model[name] for name in prior_potentials], jacobian=False
            )
        pm.Potential("__consensus_mc_prior_scale", (prior_scale - 1.0) * prior_logp)
    return shard_model


def _posterior_to_flat_array(
    posterior: xr.Dataset,
    *,
    free_rv_names: Sequence[str],
    point_map_info: tuple | None = None,
) -> tuple[np.ndarray, tuple]:
    sample_count = posterior.sizes["chain"] * posterior.sizes["draw"]
    arrays = []
    first_draw = {}
    for name in free_rv_names:
        values = np.asarray(posterior[name].values)
        flat = values.reshape((sample_count, -1))
        arrays.append(flat)
        first_draw[name] = values.reshape((sample_count, *values.shape[2:]))[0]
    flat_array = np.concatenate(arrays, axis=1)
    if point_map_info is None:
        point_map_info = DictToArrayBijection.map(first_draw).point_map_info
    return flat_array, point_map_info


def _flat_array_to_posterior_dataset(
    flat: np.ndarray,
    *,
    point_map_info: tuple,
    model: pm.Model,
) -> xr.Dataset:
    flat = np.asarray(flat)
    draws = flat.shape[0]
    posterior_dict = {}
    start = 0
    for name, shape, size, dtype in point_map_info:
        stop = start + size
        posterior_dict[name] = (
            flat[:, start:stop].astype(dtype, copy=False).reshape((1, draws, *shape))
        )
        start = stop
    coords, dims = coords_and_dims_for_inferencedata(model)
    return dict_to_dataset(posterior_dict, coords=coords, dims=dims, inference_library=pm)


def _restore_data(
    model: pm.Model,
    originals: Mapping[str, np.ndarray],
    original_coords: Mapping[str, Sequence[Any]],
    sharded_dims: set[str],
) -> None:
    coords = {dim: original_coords[dim] for dim in sharded_dims if dim in original_coords}
    for name, value in originals.items():
        name_dims = set(model.named_vars_to_dims.get(name, ()))
        model.set_data(name, value, coords={dim: coords[dim] for dim in name_dims & coords.keys()})


def _set_shard_data(
    model: pm.Model,
    shard: Mapping[str, Any],
    coords: Mapping[str, Sequence[Any]],
) -> None:
    for name, value in shard.items():
        name_dims = set(model.named_vars_to_dims.get(name, ()))
        model.set_data(name, value, coords={dim: coords[dim] for dim in name_dims & coords.keys()})


def fit_consensus_mc(
    *,
    model: pm.Model | None = None,
    shards: Sequence[Mapping[str, Any]] | None = None,
    shard_coords: Sequence[Mapping[str, Sequence[Any]]] | None = None,
    split_data: Mapping[str, int | str] | None = None,
    sharded_dims: Sequence[str] | None = None,
    num_shards: int | None = None,
    draws: int = 1000,
    sample_draws: int | None = None,
    tune: int = 1000,
    chains: int = 1,
    cores: int = 1,
    merge_method: Literal["consensus", "parametric"] = "consensus",
    diagonal: bool = False,
    random_seed: RandomSeed = None,
    sample_kwargs: Mapping[str, Any] | None = None,
    prior_potentials: Sequence[str] | None = None,
    likelihood_potentials: Sequence[str] | None = None,
    global_data: Sequence[str] | None = None,
    attach_data: bool = True,
    progressbar: bool = True,
    compile_kwargs: dict[str, Any] | None = None,
) -> DataTree:
    """Fit a PyMC model with sequential Consensus Monte Carlo over data shards."""
    model = pm.modelcontext(model)
    sample_draws = draws if sample_draws is None else sample_draws
    sample_kwargs = {} if sample_kwargs is None else dict(sample_kwargs)
    global_data = [] if global_data is None else list(global_data)
    blocked = sorted(set(sample_kwargs).intersection(_SAMPLE_KWARG_BLOCKLIST))
    if blocked:
        raise ValueError(f"sample_kwargs may not override top-level sampling arguments: {blocked}")
    if merge_method not in {"consensus", "parametric"}:
        raise ValueError("merge_method must be 'consensus' or 'parametric'.")

    resolved_shards, resolved_coords, resolved_sharded_dims = _resolve_shards(
        model,
        shards=shards,
        shard_coords=shard_coords,
        split_data=split_data,
        num_shards=num_shards,
        sharded_dims=sharded_dims,
    )
    num_shards = len(resolved_shards)
    prior_potentials, likelihood_potentials = _validate_potentials(
        model, prior_potentials, likelihood_potentials
    )
    _validate_model_supported(model, resolved_sharded_dims)
    _check_unscaled_likelihood(model, likelihood_potentials)
    _validate_likelihood_coverage(
        model,
        sharded_data_names=set(resolved_shards[0]),
        global_data=global_data,
        likelihood_potentials=likelihood_potentials,
    )
    if any(model.rvs_to_transforms.get(rv) is not None for rv in model.free_RVs):
        warn(
            "Consensus MC merges constrained posterior samples in Euclidean space; full-covariance and parametric merges can produce values outside the original support.",
            UserWarning,
            stacklevel=2,
        )

    seeds = _get_seeds_per_chain(random_seed, num_shards + 1)
    shard_model = _build_shard_model(
        model,
        num_shards=num_shards,
        prior_potentials=prior_potentials,
        likelihood_potentials=likelihood_potentials,
    )
    changed_names = set(resolved_shards[0])
    originals = {
        name: np.asarray(shard_model[name].get_value(borrow=False)).copy() for name in changed_names
    }
    original_coords = {
        dim: tuple(shard_model.coords[dim])
        for dim in resolved_sharded_dims
        if dim in shard_model.coords
    }

    free_rv_names = [rv.name for rv in model.free_RVs]
    flat_subposteriors = []
    point_map_info = None
    try:
        for k, (shard, coords) in enumerate(zip(resolved_shards, resolved_coords, strict=True)):
            _set_shard_data(shard_model, shard, coords)
            idata = pm.sample(
                draws=sample_draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=seeds[k],
                progressbar=progressbar,
                model=shard_model,
                return_inferencedata=True,
                compile_kwargs=compile_kwargs,
                **sample_kwargs,
            )
            posterior = idata["posterior"].dataset
            flat, point_map_info = _posterior_to_flat_array(
                posterior,
                free_rv_names=free_rv_names,
                point_map_info=point_map_info,
            )
            if flat_subposteriors:
                first = flat_subposteriors[0]
                if flat.shape != first.shape:
                    raise ValueError(
                        "All shard posterior arrays must have the same sample count and width."
                    )
            flat_subposteriors.append(flat)
    finally:
        _restore_data(shard_model, originals, original_coords, resolved_sharded_dims)

    flat_subposteriors_np = np.stack(flat_subposteriors)
    paired_draws = chains * sample_draws
    if merge_method == "consensus":
        merge_draws = None if draws == paired_draws else draws
        merged_flat = merge_consensus(
            flat_subposteriors_np,
            draws=merge_draws,
            diagonal=diagonal,
            random_seed=seeds[-1],
        )
    else:
        merged_flat = merge_parametric(
            flat_subposteriors_np,
            draws=draws,
            diagonal=diagonal,
            random_seed=seeds[-1],
        )

    assert point_map_info is not None
    posterior_dataset = _flat_array_to_posterior_dataset(
        merged_flat,
        point_map_info=point_map_info,
        model=model,
    )
    idata = DataTree.from_dict({"posterior": posterior_dataset})
    if attach_data:
        add_data_to_inference_data(
            idata,
            progressbar=progressbar,
            model=model,
            compile_kwargs=compile_kwargs,
        )

    idata["consensus_mc"] = DataTree(
        dataset=xr.Dataset(
            {
                "merge_method": xr.DataArray(merge_method),
                "diagonal": xr.DataArray(diagonal),
                "num_shards": xr.DataArray(num_shards),
                "draws": xr.DataArray(draws),
                "sample_draws": xr.DataArray(sample_draws),
                "tune": xr.DataArray(tune),
                "chains": xr.DataArray(chains),
                "cores": xr.DataArray(cores),
                "prior_scale": xr.DataArray(1.0 / num_shards),
                "shard_size": xr.DataArray(
                    [np.asarray(shard[next(iter(shard))]).shape[0] for shard in resolved_shards],
                    dims=["shard"],
                ),
            }
        )
    )
    return idata
