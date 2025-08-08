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
        pm.Deterministic("factor_sigma", pt.constant(np.full((k_factors,), 0.5), dtype=floatX))
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
