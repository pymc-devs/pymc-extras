import collections
import logging

from typing import Literal

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from numpy.typing import NDArray
from pymc import Model
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.model import modelcontext
from pymc.util import get_default_varnames
from pytensor.compile.mode import FAST_COMPILE
from pytensor.graph import clone_replace, vectorize_graph
from rich.console import Console, Group
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus
from pymc_extras.inference.pathfinder.results import MultiPathfinderResult, PathStatus

logger = logging.getLogger(__name__)


def get_param_coords(model: pm.Model | None, n_params: int) -> list[str]:
    """Return parameter coordinate labels from a PyMC model, or numeric indices if None."""
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
        counts, dims=["status"], coords={"status": status_names}, name="status_counts"
    )


def _determine_num_paths(result: MultiPathfinderResult) -> int:
    """Determine the number of paths from per-path arrays (samples may be collapsed by IS)."""
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


def _pathfinder_dataset(
    result: MultiPathfinderResult,
    model: pm.Model | None = None,
) -> xr.Dataset:
    """Build the ``pathfinder`` idata group: pathfinder-specific stats and config."""
    n_params = result.samples.shape[-1] if result.samples is not None else None
    param_coords = get_param_coords(model, n_params) if n_params is not None else None

    data_vars: dict = {}
    coords: dict = {}
    attrs: dict = {}

    if param_coords is not None:
        coords["param"] = param_coords

    # --- top-level pathfinder summary ---
    if result.num_paths is not None:
        data_vars["num_paths"] = xr.DataArray(result.num_paths)
    if result.num_draws is not None:
        data_vars["num_draws"] = xr.DataArray(result.num_draws)
    data_vars["all_paths_failed"] = xr.DataArray(result.all_paths_failed)
    if not result.all_paths_failed and result.samples is not None:
        data_vars["num_successful_paths"] = xr.DataArray(result.samples.shape[0])

    if result.compile_time is not None:
        data_vars["compile_time"] = xr.DataArray(result.compile_time)
    if result.compute_time is not None:
        data_vars["compute_time"] = xr.DataArray(result.compute_time)
        if result.compile_time is not None:
            data_vars["total_time"] = xr.DataArray(result.compile_time + result.compute_time)

    data_vars["importance_sampling_method"] = xr.DataArray(result.importance_sampling or "none")
    if result.pareto_k is not None:
        data_vars["pareto_k"] = xr.DataArray(result.pareto_k)

    if result.path_status:
        data_vars["path_status_counts"] = _status_counter_to_dataarray(
            result.path_status, PathStatus
        )

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

    # --- per-path arrays ---
    if not result.all_paths_failed and result.samples is not None:
        n_paths = _determine_num_paths(result)
        coords["path"] = list(range(n_paths))

        def _scalar(name: str, data):
            if data is not None:
                data_vars[name] = xr.DataArray(data, dims=["path"], coords={"path": coords["path"]})

        _scalar("elbo_argmax", result.elbo_argmax)
        if result.logP is not None:
            _scalar("paths_logP_mean", np.mean(result.logP, axis=1))
            _scalar("paths_logP_max", np.max(result.logP, axis=1))
        if result.logQ is not None:
            _scalar("paths_logQ_mean", np.mean(result.logQ, axis=1))
            _scalar("paths_logQ_max", np.max(result.logQ, axis=1))

        if n_params is not None and result.inv_hessian_diag is not None:
            data_vars["inv_hessian_diag"] = xr.DataArray(
                result.inv_hessian_diag,
                dims=["path", "param"],
                coords={"path": coords["path"], "param": coords["param"]},
            )

        if n_params is not None and result.samples.ndim >= 3:
            data_vars["final_sample"] = xr.DataArray(
                result.samples[:, -1, :],
                dims=["path", "param"],
                coords={"path": coords["path"], "param": coords["param"]},
            )

    # --- pathfinder-side config (jitter, num_elbo_draws, num_draws_per_path) ---
    if result.pathfinder_config is not None:
        data_vars["num_draws_per_path"] = xr.DataArray(result.pathfinder_config.num_draws)
        data_vars["num_elbo_draws"] = xr.DataArray(result.pathfinder_config.num_elbo_draws)
        data_vars["jitter"] = xr.DataArray(result.pathfinder_config.jitter)

    if result.warnings:
        attrs["warnings"] = list(result.warnings)

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def _lbfgs_dataset(result: MultiPathfinderResult) -> xr.Dataset:
    """Build the ``lbfgs`` idata group: L-BFGS-specific stats and configuration."""
    data_vars: dict = {}
    coords: dict = {}

    if result.lbfgs_status:
        data_vars["status_counts"] = _status_counter_to_dataarray(result.lbfgs_status, LBFGSStatus)

    if result.lbfgs_niter is not None:
        coords["path"] = list(range(len(result.lbfgs_niter)))
        data_vars["niter"] = xr.DataArray(
            result.lbfgs_niter, dims=["path"], coords={"path": coords["path"]}
        )
        data_vars["niter_mean"] = xr.DataArray(np.mean(result.lbfgs_niter))
        data_vars["niter_std"] = xr.DataArray(np.std(result.lbfgs_niter))

    if result.pathfinder_config is not None:
        cfg = result.pathfinder_config
        data_vars["maxcor"] = xr.DataArray(cfg.maxcor)
        data_vars["maxiter"] = xr.DataArray(cfg.maxiter)
        data_vars["ftol"] = xr.DataArray(cfg.ftol)
        data_vars["gtol"] = xr.DataArray(cfg.gtol)
        data_vars["maxls"] = xr.DataArray(cfg.maxls)
        data_vars["epsilon"] = xr.DataArray(cfg.epsilon)

    return xr.Dataset(data_vars, coords=coords)


def add_pathfinder_to_inference_data(
    idata: xr.DataTree,
    result: MultiPathfinderResult,
    model: pm.Model | None = None,
) -> xr.DataTree:
    """Add the ``pathfinder`` and ``lbfgs`` groups to a DataTree.

    Parameters
    ----------
    idata : xr.DataTree
        DataTree to modify in place.
    result : MultiPathfinderResult
        Multi-path pathfinder result.
    model : pm.Model, optional
        PyMC model used to label parameter coords. If None, integer indices are used.

    Returns
    -------
    xr.DataTree
        The same idata, with ``pathfinder`` and ``lbfgs`` groups added.
    """
    idata["pathfinder"] = xr.DataTree(_pathfinder_dataset(result, model=model))
    idata["lbfgs"] = xr.DataTree(_lbfgs_dataset(result))
    return idata


def convert_flat_trace_to_idata(
    samples: NDArray,
    include_transformed: bool = False,
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu",
    inference_backend: Literal["pymc", "blackjax"] = "pymc",
    model: Model | None = None,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
    vectorize: bool = True,
) -> xr.DataTree:
    """convert flattened samples to xarray DataTree format.

    Parameters
    ----------
    samples : NDArray
        flattened samples
    include_transformed : bool
        whether to include transformed variables
    postprocessing_backend : str
        backend for postprocessing transformations, either "cpu" or "gpu"
    inference_backend : str
        backend for inference, either "pymc" or "blackjax"
    model : Model | None
        pymc model for variable transformations
    importance_sampling : str
        importance sampling method used, affects input samples shape
    vectorize : bool, optional
        If True (default), use ``vectorize_graph`` to batch the Deterministic subgraph across
        all draws in one call. If False, iterate draws with ``pytensor.scan``. Set to False when
        memory is a concern, for example, when your model has large intermediate computations.

    Returns
    -------
    InferenceData
        arviz inference data object
    """

    if importance_sampling is None:
        # samples.ndim == 3 in this case, otherwise ndim == 2
        num_paths, num_pdraws, N = samples.shape
        samples = samples.reshape(-1, N)

    model = modelcontext(model)
    ip = model.initial_point()
    ip_point_map_info = DictToArrayBijection.map(ip).point_map_info
    trace = collections.defaultdict(list)
    for sample in samples:
        raveld_vars = RaveledVars(sample, ip_point_map_info)
        point = DictToArrayBijection.rmap(raveld_vars, ip)
        for p, v in point.items():
            trace[p].append(v.tolist())

    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    logger.info("Transforming variables...")

    if inference_backend == "pymc":
        if vectorize:
            # Add a (chain, draw) batch dim across the Deterministic subgraph in one shot.
            new_shapes = [v.ndim * (None,) for v in trace.values()]
            replace = {
                var: pt.tensor(dtype="float64", shape=new_shapes[i])
                for i, var in enumerate(model.value_vars)
            }
            outputs = vectorize_graph(vars_to_sample, replace=replace)

            fn = pytensor.function(
                inputs=[*list(replace.values())],
                outputs=outputs,
                mode=FAST_COMPILE,
                on_unused_input="ignore",
            )
            fn.trust_input = True
            result = fn(*list(trace.values()))
            if not isinstance(result, list):
                result = [result]
        else:
            batched_inputs = [
                pt.tensor(name=f"{v.name}_samples", dtype=v.dtype, shape=(None, *v.type.shape))
                for v in model.value_vars
            ]

            def step(*single_sample_values):
                replace = dict(zip(model.value_vars, single_sample_values, strict=True))
                return clone_replace(vars_to_sample, replace=replace)

            scan_outputs = pytensor.scan(
                fn=step,
                sequences=batched_inputs,
                return_updates=False,
            )
            if not isinstance(scan_outputs, list):
                scan_outputs = [scan_outputs]

            fn = pytensor.function(
                inputs=batched_inputs,
                outputs=scan_outputs,
                mode=FAST_COMPILE,
                on_unused_input="ignore",
            )
            fn.trust_input = True

            # trace values carry a leading chain=1 dim; scan iterates draws, so we squeeze it
            # on the way in and add it back on the way out.
            result = fn(*[trace[v.name][0] for v in model.value_vars])
            if not isinstance(result, list):
                result = [result]
            result = [r[None, ...] for r in result]

        if importance_sampling is None:
            result = [res.reshape(num_paths, num_pdraws, *res.shape[2:]) for res in result]

    elif inference_backend == "blackjax":
        import jax

        from pymc.sampling.jax import get_jaxified_graph

        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
        result = jax.vmap(jax.vmap(jax_fn))(
            *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
        )

    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict({"posterior": trace}, dims=dims, coords=coords)

    return idata


def pathfinder_report(idata: xr.DataTree, *, console: Console | None = None) -> None:
    """Print a rich-table summary built from the ``pathfinder`` and ``lbfgs`` idata groups.

    Both groups are populated automatically by :func:`fit_pathfinder`.
    """
    if "/pathfinder" not in idata.groups:
        raise ValueError(
            "idata is missing the 'pathfinder' group; this idata was not produced by "
            "fit_pathfinder."
        )
    pf = idata["pathfinder"]
    lb = idata["lbfgs"] if "/lbfgs" in idata.groups else None

    table = Table(
        title="Pathfinder Results",
        title_style="none",
        title_justify="left",
        show_header=False,
        box=None,
        padding=(0, 2),
        show_edge=False,
    )
    table.add_column("Description")
    table.add_column("Value")

    if "param" in pf.dims:
        table.add_row("")
        table.add_row("No. model parameters", str(pf.sizes["param"]))

    # Pathfinder-side config
    if "num_draws_per_path" in pf or "num_elbo_draws" in pf or "jitter" in pf or lb is not None:
        table.add_row("")
        table.add_row("Configuration:")
        if "num_draws_per_path" in pf:
            table.add_row("num_draws_per_path", str(pf["num_draws_per_path"].item()))
        if lb is not None:
            if "maxcor" in lb:
                table.add_row("history size (maxcor)", str(lb["maxcor"].item()))
            if "maxiter" in lb:
                table.add_row("max iterations", str(lb["maxiter"].item()))
            if "ftol" in lb:
                table.add_row("ftol", f"{lb['ftol'].item():.2e}")
            if "gtol" in lb:
                table.add_row("gtol", f"{lb['gtol'].item():.2e}")
            if "maxls" in lb:
                table.add_row("max line search", str(lb["maxls"].item()))
            if "epsilon" in lb:
                table.add_row("epsilon", f"{lb['epsilon'].item():.2e}")
        if "jitter" in pf:
            table.add_row("jitter", f"{pf['jitter'].item()}")
        if "num_elbo_draws" in pf:
            table.add_row("ELBO draws", str(pf["num_elbo_draws"].item()))

    # L-BFGS status + iter
    if lb is not None and "status_counts" in lb:
        table.add_row("")
        table.add_row("LBFGS Status:")
        for status_name in lb["status_counts"].coords["status"].values:
            count = int(lb["status_counts"].sel(status=status_name).item())
            if count:
                table.add_row(str(status_name), str(count))
        if "niter_mean" in lb:
            table.add_row(
                "L-BFGS iterations",
                f"mean {lb['niter_mean'].item():.0f} ± std {lb['niter_std'].item():.0f}",
            )

    # Path status
    if "path_status_counts" in pf:
        table.add_row("")
        table.add_row("Path Status:")
        for status_name in pf["path_status_counts"].coords["status"].values:
            count = int(pf["path_status_counts"].sel(status=status_name).item())
            if count:
                table.add_row(str(status_name), str(count))
        if "elbo_argmax_mean" in pf:
            table.add_row(
                "ELBO argmax",
                f"mean {pf['elbo_argmax_mean'].item():.0f} ± "
                f"std {pf['elbo_argmax_std'].item():.0f}",
            )

    # Importance sampling
    if not bool(pf["all_paths_failed"].item()) and "importance_sampling_method" in pf:
        table.add_row("")
        table.add_row("Importance Sampling:")
        table.add_row("Method", str(pf["importance_sampling_method"].item()))
        if "pareto_k" in pf:
            table.add_row("Pareto k", f"{pf['pareto_k'].item():.2f}")

    # Timing
    if "compile_time" in pf:
        table.add_row("")
        table.add_row("Timing (seconds):")
        table.add_row("Compile", f"{pf['compile_time'].item():.2f}")
    if "compute_time" in pf:
        table.add_row("Compute", f"{pf['compute_time'].item():.2f}")
    if "total_time" in pf:
        table.add_row("Total", f"{pf['total_time'].item():.2f}")

    console = console or Console()
    warnings_attr = pf.attrs.get("warnings")
    if warnings_attr:
        warning_text = [
            Text(),
            Text("Warnings:"),
            *(
                Padding(
                    Text("- " + w, no_wrap=False).wrap(console, width=console.width - 6),
                    (0, 0, 0, 2),
                )
                for w in warnings_attr
            ),
        ]
        console.print(Group(table, *warning_text))
    else:
        console.print(table)
