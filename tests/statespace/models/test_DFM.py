from itertools import product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from pytensor.graph.basic import explicit_graph_inputs
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import LONG_MATRIX_NAMES
from tests.statespace.shared_fixtures import rng

floatX = pytensor.config.floatX

# TODO: check test for error_var=True, since there are problems with statsmodels, the matrices looks the same by some experiments done in notebooks
# (FAILED tests/statespace/models/test_DFM.py::test_DFM_update_matches_statsmodels[True-2-2-2] - numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite)


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv(
        "tests/statespace/_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq
    return df


def create_sm_test_values_mapping(
    test_values, data, k_factors, factor_order, error_order, error_var
):
    """Convert PyMC test values to statsmodels parameter format"""
    sm_test_values = {}

    # 1. Factor loadings: PyMC shape (n_endog, k_factors) -> statsmodels individual params
    factor_loadings = test_values["factor_loadings"]
    all_pairs = product(data.columns, range(1, k_factors + 1))
    sm_test_values.update(
        {
            f"loading.f{factor_idx}.{endog_name}": value
            for (endog_name, factor_idx), value in zip(all_pairs, factor_loadings.ravel())
        }
    )

    # 2. Factor AR coefficients: PyMC shape (k_factors, factor_order*k_factors) -> L{lag}.f{to}.f{from}
    if factor_order > 0 and "factor_ar" in test_values:
        factor_ar = test_values["factor_ar"]
        triplets = product(
            range(1, k_factors + 1), range(1, factor_order + 1), range(1, k_factors + 1)
        )
        sm_test_values.update(
            {
                f"L{lag}.f{to_factor}.f{from_factor}": factor_ar[
                    from_factor - 1, (lag - 1) * k_factors + (to_factor - 1)
                ]
                for from_factor, lag, to_factor in triplets
            }
        )

    # 3a. Error AR coefficients: PyMC shape (n_endog, error_order) -> L{lag}.e(var).e(var)
    if error_order > 0 and not error_var and "error_ar" in test_values:
        error_ar = test_values["error_ar"]
        pairs = product(enumerate(data.columns), range(1, error_order + 1))
        sm_test_values.update(
            {
                f"L{lag}.e({endog_name}).e({endog_name})": error_ar[endog_idx, lag - 1]
                for (endog_idx, endog_name), lag in pairs
            }
        )

    # 3b. Error AR coefficients: PyMC shape (n_endog, error_order * n_endog) -> L{lag}.e(var).e(var)
    elif error_order > 0 and error_var and "error_ar" in test_values:
        error_ar = test_values["error_ar"]
        triplets = product(
            enumerate(data.columns), range(1, error_order + 1), enumerate(data.columns)
        )
        sm_test_values.update(
            {
                f"L{lag}.e({from_endog_name}).e({to_endog_name})": error_ar[
                    from_endog_idx, (lag - 1) * data.shape[1] + to_endog_idx
                ]
                for (from_endog_idx, from_endog_name), lag, (
                    to_endog_idx,
                    to_endog_name,
                ) in triplets
            }
        )

    # 4. Observation error variances:
    if "error_sigma" in test_values:
        error_sigma = test_values["error_sigma"]
        sm_test_values.update(
            {
                f"sigma2.{endog_name}": error_sigma[endog_idx]
                for endog_idx, endog_name in enumerate(data.columns)
            }
        )

    return sm_test_values


@pytest.mark.parametrize("k_factors", [1, 2])
@pytest.mark.parametrize("factor_order", [0, 1, 2])
@pytest.mark.parametrize("error_order", [0, 1, 2])
@pytest.mark.parametrize("error_var", [False])
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_DFM_update_matches_statsmodels(data, k_factors, factor_order, error_order, error_var, rng):
    mod = BayesianDynamicFactor(
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
        k_endog=data.shape[1],
        measurement_error=False,
        error_var=error_var,
        verbose=False,
    )
    sm_dfm = DynamicFactor(
        endog=data,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
        error_var=error_var,
    )

    # Generate test values for PyMC model
    test_values = {}
    test_values["x0"] = rng.normal(size=mod.k_states)
    test_values["P0"] = np.eye(mod.k_states)  # Use identity for stability
    test_values["factor_loadings"] = rng.normal(size=(data.shape[1], k_factors))

    if factor_order > 0:
        test_values["factor_ar"] = rng.normal(size=(k_factors, factor_order * k_factors))

    if error_order > 0 and error_var:
        test_values["error_ar"] = rng.normal(size=(data.shape[1], error_order * data.shape[1]))
    elif error_order > 0 and not error_var:
        test_values["error_ar"] = rng.normal(size=(data.shape[1], error_order))

    test_values["error_sigma"] = rng.beta(1, 1, size=data.shape[1])

    # Convert to statsmodels format
    sm_test_values = create_sm_test_values_mapping(
        test_values, data, k_factors, factor_order, error_order, error_var
    )

    # Initialize and constrain statsmodels model
    x0 = test_values["x0"]
    P0 = test_values["P0"]

    sm_dfm.initialize_known(initial_state=x0, initial_state_cov=P0)
    sm_dfm.fit_constrained({name: sm_test_values[name] for name in sm_dfm.param_names})

    # Get PyMC matrices using the same pattern as ETS test
    matrices = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]

    f_matrices = pytensor.function(inputs, matrices)
    test_values_subset = {name: test_values[name] for name in input_names}

    pymc_matrices = f_matrices(**test_values_subset)

    sm_matrices = [sm_dfm.ssm[name] for name in LONG_MATRIX_NAMES[2:]]

    # Compare matrices (skip x0 and P0)
    for matrix, sm_matrix, name in zip(pymc_matrices[2:], sm_matrices, LONG_MATRIX_NAMES[2:]):
        assert_allclose(matrix, sm_matrix, err_msg=f"{name} does not match")
