"""
Parsing utilities for component state names in structural time series models.

This module provides functionality to parse complex state names like 'trend[level[observed_state]]'
into structured multi-index coordinates that enable easy component and state selection.

NB: This is still a work in progress, and probably need to be expanded to more complex cases.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

import pandas as pd
import xarray as xr


def parse_component_state_name(state_name: str) -> tuple[str, str]:
    """
    Parse a component state name into its constituent parts.

    Extracts the actual interpretable state name and observed state from
    various component naming formats.

    Parameters
    ----------
    state_name : str
        The state name to parse, e.g., 'trend[level[observed_state]]' or 'ar[observed_state]'

    Returns
    -------
    tuple[str, str]
        A tuple of (component, observed) where component is the interpretable component name
        and observed is the observed state name

    Examples
    --------
    >>> parse_component_state_name('trend[level[chirac2]]')
    ('level', 'chirac2')
    >>> parse_component_state_name('ar[macron]')
    ('ar', 'macron')
    """
    # Handle the nested bracket pattern: component[state[observed]]
    # For these, we want the inner state name (level, trend, etc.)
    # because the first level is redundant with the component name
    nested_pattern = r"^([^[]+)\[([^[]+)\[([^]]+)\]\]$"
    nested_match = re.match(nested_pattern, state_name)

    if nested_match:
        # Return the inner state name and observed state
        return nested_match.group(2), nested_match.group(3)

    # Handle the simple bracket pattern: component[observed]
    # For these, we want the component name directly
    simple_pattern = r"^([^[]+)\[([^]]+)\]$"
    simple_match = re.match(simple_pattern, state_name)

    if simple_match:
        # Return the component name and observed state
        return simple_match.group(1), simple_match.group(2)

    # If no pattern matches, treat the whole string as a state name
    # This is a fallback for edge cases
    return state_name, "default"


def create_component_multiindex(
    state_names: Sequence[str], coord_name: str = "state"
) -> xr.Coordinates:
    """
    Create xarray coordinates with multi-index from component state names.

    Parameters
    ----------
    state_names : Sequence[str]
        List of state names to parse into multi-index
    coord_name : str, default "state"
        Name for the coordinate dimension to transform into a multi-index

    Returns
    -------
    xr.Coordinates
        xarray coordinates with multi-index structure

    Examples
    --------
    >>> state_names = ['trend[level[observed_state]]', 'trend[trend[observed_state]]', 'ar[observed_state]']
    >>> coords = create_component_multiindex(state_names)
    >>> coords.to_index().names
    ['component', 'observed']
    >>> coords.to_index().values
    [('level', 'observed_state'), ('trend', 'observed_state'), ('ar', 'observed_state')]
    """
    tuples = [parse_component_state_name(name) for name in state_names]
    midx = pd.MultiIndex.from_tuples(tuples, names=["component", "observed"])

    return xr.Coordinates.from_pandas_multiindex(midx, dim=coord_name)


def restructure_components_idata(idata: xr.Dataset) -> xr.Dataset:
    """
    Restructure idata with multi-index coordinates for easier component selection.

    Parameters
    ----------
    idata : xr.Dataset
        Dataset with component state names as coordinates

    Returns
    -------
    xr.Dataset
        Dataset with restructured multi-index coordinates

    Examples
    --------
    >>> # After calling extract_components_from_idata from core.py
    >>> restructured = restructure_components_idata(components_idata)
    >>> # Now you can select by component or observed state
    >>> level_data = restructured.sel(component='level')  # All level components
    >>> gdp_data = restructured.sel(observed='gdp')  # All gdp data
    >>> level_gdp = restructured.sel(component='level', observed='gdp')  # Specific combination
    """
    # name of the coordinate containing state names
    # should be `state`, by default, as users don't access it directly
    # would need to be updated if we want to support custom names
    state_coord_name = "state"
    if state_coord_name not in idata.coords:
        raise ValueError(f"Coordinate '{state_coord_name}' not found in dataset")

    state_names = idata.coords[state_coord_name].values
    mindex_coords = create_component_multiindex(state_names, state_coord_name)

    return idata.assign_coords(mindex_coords)
