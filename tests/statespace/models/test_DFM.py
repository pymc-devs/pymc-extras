import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import SHORT_NAME_TO_LONG
from tests.statespace.shared_fixtures import rng

floatX = pytensor.config.floatX


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv(
        "tests/statespace/_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq
    return df


@pytest.mark.parametrize("k_factors", [1, 2])
@pytest.mark.parametrize("factor_order", [0, 1, 2])
@pytest.mark.parametrize("error_order", [0, 1, 2])
def test_dfm_parameter_and_matrix_match(data, k_factors, factor_order, error_order):
    # --- Statsmodels DFM ---
    sm_dfm = DynamicFactor(
        endog=data,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
    )

    # Use deterministic small parameters for reproducibility
    param_array = np.full(len(sm_dfm.param_names), 0.5)
    sm_dfm.update(param_array)

    # Only request matrices that actually exist in ssm.__getitem__
    valid_names = ["design", "obs_cov", "transition", "state_cov", "selection"]
    sm_matrices = {name: np.array(sm_dfm.ssm[name]) for name in valid_names}

    # --- PyMC DFM ---
    mod = BayesianDynamicFactor(
        k_factors=k_factors,
        factor_order=factor_order,
        k_endog=data.shape[1],
        error_order=error_order,
        measurement_error=False,
        verbose=False,
    )

    coords = mod.coords
    with pm.Model(coords=coords):
        k_endog = data.shape[1]
        factor_part = max(1, factor_order) * k_factors
        error_part = error_order * k_endog if error_order > 0 else 0
        k_states = factor_part + error_part

        pm.Deterministic("x0", pt.constant(np.full((k_states,), 0.5), dtype=floatX))
        pm.Deterministic("P0", pt.constant(np.full((k_states, k_states), 0.5), dtype=floatX))
        pm.Deterministic(
            "factor_loadings", pt.constant(np.full((k_endog, k_factors), 0.5), dtype=floatX)
        )

        if factor_order > 0:
            pm.Deterministic(
                "factor_ar",
                pt.constant(np.full((k_factors, factor_order * k_factors), 0.5), dtype=floatX),
            )
        if error_order > 0:
            pm.Deterministic(
                "error_ar", pt.constant(np.full((k_endog, error_order), 0.5), dtype=floatX)
            )
        # pm.Deterministic("factor_sigma", pt.constant(np.full((k_factors,), 0.5), dtype=floatX))
        pm.Deterministic("error_sigma", pt.constant(np.full((k_endog,), 0.5), dtype=floatX))
        pm.Deterministic("sigma_obs", pt.constant(np.full((k_endog,), 0.5), dtype=floatX))

        mod._insert_random_variables()

        pymc_matrices = pm.draw(mod.subbed_ssm)
        pymc_matrices = dict(zip(SHORT_NAME_TO_LONG.values(), pymc_matrices))

    # --- Compare ---
    for mat_name in valid_names:
        assert_allclose(
            pymc_matrices[mat_name],
            sm_matrices[mat_name],
            atol=1e-12,
            err_msg=f"Matrix mismatch: {mat_name} (k_factors={k_factors}, factor_order={factor_order}, error_order={error_order})",
        )


@pytest.mark.parametrize("k_factors", [1, 2])
@pytest.mark.parametrize("factor_order", [0, 1, 2])
@pytest.mark.parametrize("error_order", [1, 2, 3])
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_DFM_update_matches_statsmodels(data, k_factors, factor_order, error_order, rng):
    # --- Fit Statsmodels DynamicFactor with random small params ---
    sm_dfm = DynamicFactor(
        endog=data,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
    )
    param_names = sm_dfm.param_names
    param_dict = {param: getattr(np, floatX)(rng.normal(scale=0.1) ** 2) for param in param_names}
    sm_res = sm_dfm.fit_constrained(param_dict)

    # --- Setup BayesianDynamicFactor ---
    mod = BayesianDynamicFactor(
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
        k_endog=data.shape[1],
        measurement_error=False,
        verbose=False,
    )

    # Convert flat param dict to PyTensor variables as needed
    # Reshape factor_ar and error_ar parameters according to model expected shapes
    factor_ar_shape = (k_factors, factor_order * k_factors)
    error_ar_shape = (data.shape[1], error_order) if error_order > 0 else (0,)

    # Prepare parameter arrays to set as deterministic
    # Extract each group of parameters by name pattern (simplified)
    factor_loadings = np.array([param_dict[p] for p in param_names if "loading" in p]).reshape(
        (data.shape[1], k_factors)
    )

    # Handle factor_ar parameters - need to account for different factor orders
    factor_ar_params = []

    for factor_idx in range(1, k_factors + 1):
        for lag in range(1, factor_order + 1):
            for factor_idx2 in range(1, k_factors + 1):
                param_pattern = f"L{lag}.f{factor_idx2}.f{factor_idx}"
                if param_pattern in param_names:
                    factor_ar_params.append(param_pattern)

    if len(factor_ar_params) > 0:
        factor_ar_values = [param_dict[p] for p in factor_ar_params]
        factor_ar = np.array(factor_ar_values).reshape(factor_ar_shape)
    else:
        factor_ar = np.zeros(factor_ar_shape)

    # factor_sigma = np.array([param_dict[p] for p in param_names if "factor.sigma" in p])

    # Handle error AR parameters - need to account for different error orders and variables
    if error_order > 0:
        error_ar_params = []
        var_names = [col for col in data.columns]  # Get variable names from data

        # Order parameters by variable first, then by lag to match expected shape (n_vars, n_lags)
        for var_name in var_names:
            for lag in range(1, error_order + 1):
                param_pattern = f"L{lag}.e({var_name}).e({var_name})"
                if param_pattern in param_names:
                    error_ar_params.append(param_pattern)

        if len(error_ar_params) > 0:
            error_ar_values = [param_dict[p] for p in error_ar_params]
            error_ar = np.array(error_ar_values).reshape(error_ar_shape)
        else:
            error_ar = np.zeros(error_ar_shape)

    # Handle observation error variances - look for sigma2 pattern
    sigma_obs_params = [p for p in param_names if "sigma2." in p]
    sigma_obs = np.array([param_dict[p] for p in sigma_obs_params])

    # Handle error variances (if needed separately from sigma_obs)
    if error_order > 0:
        error_sigma = sigma_obs  # In this case, error_sigma is the same as sigma_obs

    coords = mod.coords
    with pm.Model(coords=coords) as model:
        k_states = k_factors * max(1, factor_order) + (
            error_order * data.shape[1] if error_order > 0 else 0
        )
        pm.Deterministic("x0", pt.zeros(k_states, dtype=floatX))
        pm.Deterministic("P0", pt.eye(k_states, dtype=floatX))
        # Set deterministic variables with constrained parameter values
        pm.Deterministic("factor_loadings", pt.as_tensor_variable(factor_loadings))
        if factor_order > 0:
            pm.Deterministic("factor_ar", pt.as_tensor_variable(factor_ar))
        # pm.Deterministic("factor_sigma", pt.as_tensor_variable(factor_sigma))
        if error_order > 0:
            pm.Deterministic("error_ar", pt.as_tensor_variable(error_ar))
        pm.Deterministic("error_sigma", pt.as_tensor_variable(error_sigma))
        pm.Deterministic("sigma_obs", pt.as_tensor_variable(sigma_obs))

        mod._insert_random_variables()

        # Draw the substituted state-space matrices from PyMC model
        matrices = pm.draw(mod.subbed_ssm)
        matrix_dict = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))

    # Matrices to check
    matrices_to_check = ["transition", "selection", "state_cov", "obs_cov", "design"]

    # Compare matrices from PyMC and Statsmodels
    for mat_name in matrices_to_check:
        sm_mat = np.array(sm_dfm.ssm[mat_name])
        pm_mat = matrix_dict[mat_name]

        assert_allclose(
            pm_mat,
            sm_mat,
            atol=1e-10,
            err_msg=f"Matrix mismatch: {mat_name} (k_factors={k_factors}, factor_order={factor_order}, error_order={error_order})",
        )
