import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.pathfinder import fit_pathfinder


@pytest.fixture
def simple_model():
    """Create a simple test model for pathfinder testing."""
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=2.0)
    return model


@pytest.fixture
def medium_model():
    """Create a medium-sized test model."""
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1, shape=5)
        y = pm.Normal("y", x.sum(), 1, observed=10.0)
    return model


@pytest.fixture
def hierarchical_model():
    """Create a hierarchical test model."""
    # Generate some synthetic data
    np.random.seed(42)
    n_groups = 3
    n_obs_per_group = 5
    group_effects = [0.5, -0.3, 0.8]

    data = []
    group_idx = []
    for i in range(n_groups):
        group_data = np.random.normal(group_effects[i], 0.5, n_obs_per_group)
        data.extend(group_data)
        group_idx.extend([i] * n_obs_per_group)

    with pm.Model() as model:
        # Hyperpriors
        mu_pop = pm.Normal("mu_pop", 0, 1)
        sigma_pop = pm.HalfNormal("sigma_pop", 1)

        # Group-level parameters
        mu_group = pm.Normal("mu_group", mu_pop, sigma_pop, shape=n_groups)
        sigma_group = pm.HalfNormal("sigma_group", 1)

        # Likelihood
        y = pm.Normal("y", mu_group[group_idx], sigma_group, observed=data)

    return model


def assert_backend_equivalence(model, backend1="pymc", backend2="numba", rtol=1e-1, **kwargs):
    """Test mathematical equivalence between backends.

    Note: Uses relaxed tolerance since we're comparing stochastic sampling results.
    """
    # Default parameters for testing
    test_params = {"num_draws": 50, "num_paths": 2, "random_seed": 42, **kwargs}

    try:
        # Run with first backend
        result1 = fit_pathfinder(model, inference_backend=backend1, **test_params)

        # Run with second backend
        result2 = fit_pathfinder(model, inference_backend=backend2, **test_params)

        # Compare statistical properties (means)
        for var_name in result1.posterior.data_vars:
            mean1 = result1.posterior[var_name].mean().values
            mean2 = result2.posterior[var_name].mean().values

            # Use relative tolerance for comparison
            np.testing.assert_allclose(
                mean1,
                mean2,
                rtol=rtol,
                err_msg=f"Means differ for variable {var_name}: {mean1} vs {mean2}",
            )

        return True, "Backends are statistically equivalent"

    except Exception as e:
        return False, f"Backend comparison failed: {e}"


def get_available_backends():
    """Get list of available backends in current environment."""
    import importlib.util

    available = ["pymc"]  # PyMC should always be available

    if importlib.util.find_spec("jax") is not None:
        available.append("jax")

    if importlib.util.find_spec("numba") is not None:
        available.append("numba")

    if importlib.util.find_spec("blackjax") is not None:
        available.append("blackjax")

    return available


def validate_pathfinder_result(result, expected_draws=None, expected_vars=None):
    """Validate basic properties of pathfinder results."""
    assert result is not None, "Result should not be None"
    assert hasattr(result, "posterior"), "Result should have posterior attribute"

    if expected_draws is not None:
        # Check that we have the expected number of draws
        # Note: pathfinder results have shape (chains, draws)
        for var_name in result.posterior.data_vars:
            draws_shape = result.posterior[var_name].shape
            assert draws_shape[-1] == expected_draws or draws_shape == (
                1,
                expected_draws,
            ), f"Expected {expected_draws} draws, got shape {draws_shape}"

    if expected_vars is not None:
        # Check that expected variables are present
        for var_name in expected_vars:
            assert (
                var_name in result.posterior.data_vars
            ), f"Expected variable {var_name} not found in result"

    # Check that all values are finite
    for var_name in result.posterior.data_vars:
        values = result.posterior[var_name].values
        assert np.all(np.isfinite(values)), f"Non-finite values found in {var_name}: {values}"

    return True
