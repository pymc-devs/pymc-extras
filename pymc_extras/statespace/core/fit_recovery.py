import re

from collections.abc import Container, Sequence
from typing import TYPE_CHECKING

import pandas as pd

from xarray import Dataset, DataTree

from pymc_extras.statespace.utils.constants import OBSERVED_DATA_NAME

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace

# ``chain`` and ``draw`` index the samples, not the variable, so they are never model dims.
_SAMPLE_DIMS = frozenset({"chain", "draw"})

# What PyMC names the axes of a variable registered without dims.
_GENERATED_DIM = re.compile(r"^(?P<variable>.+)_dim_\d+$")


def verify_group(group: str) -> None:
    """Raise ValueError unless ``group`` names a group parameters can be drawn from."""
    if group not in ("prior", "posterior"):
        raise ValueError(f'Argument "group" must be one of "prior" or "posterior", found {group}')


def _is_generated(dim: str, variables: Container[str]) -> bool:
    """Report whether ``dim`` is a name PyMC made up for an axis of one of ``variables``."""
    match = _GENERATED_DIM.match(dim)
    return match is not None and match["variable"] in variables


def _require_group(idata: DataTree, group: str) -> Dataset:
    """Return one group of ``idata``, or raise naming the groups it does have."""
    if group not in idata:
        available = ", ".join(sorted(name.lstrip("/") for name in idata.groups if name != "/"))
        raise ValueError(
            f"Inference data has no {group!r} group, so there is nothing to recover from it. "
            f"It holds: {available}."
        )
    return idata[group]


def coords_from_idata(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    group: str,
    coords: dict[str, Sequence] | None = None,
) -> dict[str, Sequence]:
    """
    Rebuild the coords a model was fit with, from its template and one inference data group.

    The two sources are complements. Structural coords -- ``shock``, ``shock_aux``,
    ``observed_state_aux`` and the per-component ones -- are attached to no sampled or observed
    variable, so they reach only the template. Dataset coords, ``time`` above all, are known only
    to the inference data.

    Parameters
    ----------
    ss_mod : PyMCStateSpace
        Model supplying the structural coords.
    idata : DataTree
        Inference data to read.
    group : str
        Group carrying the dataset coords: ``"observed_data"`` for the data a model was fit on,
        or the group a post-estimation result was written to.
    coords : dict mapping str to sequence, optional
        Coords to override the recovered ones with, for inference data that carries none.
        Default None.

    Returns
    -------
    coords : dict mapping str to sequence
        The template's coords, updated with the group's.

    Raises
    ------
    ValueError
        If ``idata`` has no group named ``group``.
    """
    dataset = _require_group(idata, group)

    recovered: dict[str, Sequence] = dict(ss_mod.coords)
    for name, values in dataset.coords.items():
        if name not in _SAMPLE_DIMS and not _is_generated(name, dataset.data_vars):
            # ``to_index`` keeps a datetime coord as timestamps rather than degrading it to
            # ``datetime64``, which is what lets the forecast index recover its frequency.
            recovered[name] = values.to_index()

    return recovered | (coords or {})


def dims_from_idata(ss_mod: "PyMCStateSpace", idata: DataTree, group: str) -> dict[str, list[str]]:
    """
    Rebuild the dims each model parameter was given at fit time.

    A parameter registered without dims is stored under PyMC's generated ``{name}_dim_{k}``
    names, which are indistinguishable from real ones once written to inference data. Those are
    dropped, so a parameter that was given no dims recovers none.

    Parameters
    ----------
    ss_mod : PyMCStateSpace
        Model whose parameters to look up.
    idata : DataTree
        Inference data to read.
    group : str
        Group holding the sampled parameters, usually ``"prior"`` or ``"posterior"``.

    Returns
    -------
    dims : dict mapping str to list of str
        Dims per parameter, omitting parameters that were given none.

    Raises
    ------
    ValueError
        If ``idata`` has no group named ``group``.
    """
    variables = _require_group(idata, group).data_vars

    recovered: dict[str, list[str]] = {}
    for name in ss_mod.param_names:
        if name not in variables:
            continue
        dims = [
            dim
            for dim in variables[name].dims
            if dim not in _SAMPLE_DIMS and not _is_generated(dim, variables)
        ]
        if dims:
            recovered[name] = dims

    return recovered


def data_from_idata(idata: DataTree, group: str) -> pd.DataFrame:
    """
    Rebuild the observed data a model was fit on.

    The recovered values carry the missing-data sentinel where the original held ``nan``, because
    that substitution happens before the data is recorded. The filter derives its missing mask
    from the sentinel, so the two filter identically.

    Parameters
    ----------
    idata : DataTree
        Inference data to read.
    group : str
        Group holding the registered data, ``"constant_data"`` for a fit.

    Returns
    -------
    data : DataFrame
        Observed data, indexed by the time coord it was fit against, with the frequency of a
        datetime index restored.

    Raises
    ------
    ValueError
        If ``idata`` has no group named ``group``, or that group holds no observed data.
    """
    dataset = _require_group(idata, group)
    if OBSERVED_DATA_NAME not in dataset.data_vars:
        raise ValueError(
            f"The {group!r} group holds no {OBSERVED_DATA_NAME!r} variable, so the observed data "
            f"cannot be recovered. It holds: {', '.join(sorted(dataset.data_vars))}."
        )

    data = dataset[OBSERVED_DATA_NAME].to_pandas()
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is None:
        # A round-trip through the inference data drops the frequency, which the data
        # preprocessor warns about and the forecast index needs.
        data.index.freq = data.index.inferred_freq

    return data


def exog_from_idata(ss_mod: "PyMCStateSpace", idata: DataTree, group: str) -> dict[str, dict]:
    """
    Rebuild the ``pm.Data`` arguments for each exogenous variable a model was fit with.

    Parameters
    ----------
    ss_mod : PyMCStateSpace
        Model whose exogenous variables to look up.
    idata : DataTree
        Inference data to read.
    group : str
        Group holding the registered data, ``"constant_data"`` for a fit.

    Returns
    -------
    exog : dict mapping str to dict
        One ``pm.Data`` keyword mapping per exogenous variable, omitting any the group does not
        hold.

    Raises
    ------
    ValueError
        If ``idata`` has no group named ``group``.
    """
    variables = _require_group(idata, group).data_vars

    recovered: dict[str, dict] = {}
    for name in ss_mod.data_names:
        if name not in variables:
            continue
        dims = [dim for dim in variables[name].dims if not _is_generated(dim, variables)]
        recovered[name] = {
            "name": name,
            "value": variables[name].values,
            "dims": dims or None,
        }

    return recovered
