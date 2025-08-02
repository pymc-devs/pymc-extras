"""
Tests for component state name parsing utilities.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_extras.statespace.utils.component_parsing import (
    create_component_multiindex,
    parse_component_state_name,
    restructure_components_idata,
)


class TestParseComponentStateName:
    """Test the core parsing function for component state names."""

    def test_nested_pattern(self):
        """Test parsing of nested bracket patterns like 'component[state[observed]]'."""
        result = parse_component_state_name("trend[level[chirac2]]")
        assert result == ("level", "chirac2")

        result = parse_component_state_name("seasonal[coef_0[macron]]")
        assert result == ("coef_0", "macron")

    def test_simple_pattern(self):
        """Test parsing of simple bracket patterns like 'component[observed]'."""
        result = parse_component_state_name("ar[macron]")
        assert result == ("ar", "macron")

        result = parse_component_state_name("measurement_error[hollande]")
        assert result == ("measurement_error", "hollande")

    def test_complex_component_names(self):
        """Test parsing with complex component names that might include special characters."""
        # Test with underscores
        result = parse_component_state_name("level_trend[level[data_1]]")
        assert result == ("level", "data_1")

        # Test with numbers
        result = parse_component_state_name("ar2[lag1[series_1]]")
        assert result == ("lag1", "series_1")

    def test_complex_observed_names(self):
        """Test parsing with complex observed variable names."""
        result = parse_component_state_name("trend[level[data_var_1]]")
        assert result == ("level", "data_var_1")

        result = parse_component_state_name("seasonal[coef[obs_2024_Q1]]")
        assert result == ("coef", "obs_2024_Q1")

    def test_fallback_pattern(self):
        """Test the fallback behavior for unusual patterns."""
        result = parse_component_state_name("simple_state_name")
        assert result == ("simple_state_name", "default")

        result = parse_component_state_name("no_brackets")
        assert result == ("no_brackets", "default")

    def test_edge_cases(self):
        """Test edge cases and malformed inputs."""
        # Empty brackets - should fall back to treating as simple component name
        result = parse_component_state_name("component[]")
        assert result == ("component[]", "default")

        # Mismatched brackets - should be parsed as simple pattern component[state[observed]
        result = parse_component_state_name("component[state[observed]")
        assert result == ("component", "state[observed")


class TestCreateComponentMultiindex:
    """Test the multi-index creation functionality."""

    def test_basic_multiindex_creation(self):
        """Test creating a multi-index from basic state names."""
        state_names = [
            "trend[level[chirac2]]",
            "trend[trend[chirac2]]",
            "ar[chirac2]",
            "trend[level[macron]]",
            "ar[macron]",
        ]

        coords = create_component_multiindex(state_names)

        index = coords.to_index()
        assert isinstance(index, pd.MultiIndex)
        assert index.names == ["component", "observed"]
        assert "state" in coords.dims

    def test_custom_coord_name(self):
        """Test creating multi-index with custom coordinate name."""
        state_names = ["trend[level[data]]", "ar[data]"]
        coords = create_component_multiindex(state_names, coord_name="custom_state")

        assert "custom_state" in coords.dims

    def test_mixed_patterns(self):
        """Test with a mix of nested and simple patterns."""
        state_names = [
            "trend[level[obs1]]",
            "ar[obs1]",
            "seasonal[coef_1[obs2]]",
            "measurement_error[obs2]",
        ]

        coords = create_component_multiindex(state_names)
        index = coords.to_index()

        # check we get right structure
        expected_tuples = [
            ("level", "obs1"),
            ("ar", "obs1"),
            ("coef_1", "obs2"),
            ("measurement_error", "obs2"),
        ]

        for i, expected in enumerate(expected_tuples):
            assert index[i] == expected

    def test_empty_input(self):
        """Test with empty state names list."""
        coords = create_component_multiindex([])
        index = coords.to_index()
        assert len(index) == 0
        assert index.names == ["component", "observed"]


class TestRestructureComponentsIdata:
    """Test the idata restructuring functionality."""

    @staticmethod
    def create_sample_idata(state_names):
        n_chains, n_draws, n_time, n_states = 2, 100, 50, len(state_names)
        data = np.random.normal(size=(n_chains, n_draws, n_time, n_states))
        return xr.Dataset(
            {
                "filtered_latent": xr.DataArray(
                    data,
                    dims=["chain", "draw", "time", "state"],
                    coords={
                        "chain": range(n_chains),
                        "draw": range(n_draws),
                        "time": range(n_time),
                        "state": state_names,
                    },
                )
            }
        )

    def test_basic_restructuring(self):
        state_names = [
            "trend[level[chirac2]]",
            "trend[trend[chirac2]]",
            "ar[chirac2]",
            "trend[level[macron]]",
            "ar[macron]",
        ]

        idata = self.create_sample_idata(state_names)
        restructured = restructure_components_idata(idata)

        state_index = restructured.coords["state"].to_index()
        assert isinstance(state_index, pd.MultiIndex)
        assert state_index.names == ["component", "observed"]

        # check we can select by component
        level_data = restructured.sel(component="level")
        assert "level" in restructured.coords["component"].values
        assert (
            "ar" in restructured.coords["component"].values
        )  # ar should be here from ar[chirac2] and ar[macron]

        # check we can select by observed state
        chirac_data = restructured.sel(observed="chirac2")
        assert "chirac2" in restructured.coords["observed"].values

    def test_missing_coordinate_error(self):
        """Test error handling when coordinate doesn't exist."""
        idata = xr.Dataset({"data": xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [1, 2, 3]})})

        with pytest.raises(ValueError, match="Coordinate 'state' not found"):
            restructure_components_idata(idata)


class TestIntegrationScenarios:
    def test_real_world_example(self):
        state_names = [
            "trend[level[chirac2]]",
            "trend[trend[chirac2]]",
            "trend[level[sarkozy]]",
            "trend[trend[sarkozy]]",
            "trend[level[hollande]]",
            "trend[trend[hollande]]",
            "trend[level[macron]]",
            "trend[trend[macron]]",
            "trend[level[macron2]]",
            "trend[trend[macron2]]",
            "ar[chirac2]",
            "ar[sarkozy]",
            "ar[hollande]",
            "ar[macron]",
            "ar[macron2]",
        ]
        n_chains, n_draws, n_time = 4, 500, 100
        data = np.random.normal(size=(n_chains, n_draws, n_time, len(state_names)))
        idata = xr.Dataset(
            {
                "filtered_latent": xr.DataArray(
                    data, dims=["chain", "draw", "time", "state"], coords={"state": state_names}
                )
            }
        )
        restructured = restructure_components_idata(idata)

        macron_data = restructured.sel(observed="macron")
        assert macron_data.filtered_latent.shape == (
            4,
            500,
            100,
            3,
        )  # 3 because trend level, trend trend, ar

        ar_data = restructured.sel(component="ar")
        assert ar_data.filtered_latent.shape == (4, 500, 100, 5)  # 5 observed states

        level_macron = restructured.sel(component="level", observed="macron")
        assert level_macron.filtered_latent.shape == (4, 500, 100)  # single level component
