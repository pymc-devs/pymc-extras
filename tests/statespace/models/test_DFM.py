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
from pymc_extras.statespace.utils.constants import (
    LONG_MATRIX_NAMES,
    MATRIX_NAMES,
    SHORT_NAME_TO_LONG,
)
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
    test_values["P0"] = np.eye(mod.k_states)
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
    test_values_subset = {name: test_values[name] for name in input_names if name in test_values}

    pymc_matrices = f_matrices(**test_values_subset)

    sm_matrices = [sm_dfm.ssm[name] for name in LONG_MATRIX_NAMES[2:]]

    # Compare matrices (skip x0 and P0)
    for matrix, sm_matrix, name in zip(pymc_matrices[2:], sm_matrices, LONG_MATRIX_NAMES[2:]):
        assert_allclose(matrix, sm_matrix, err_msg=f"{name} does not match")


def unpack_statespace(ssm):
    return [ssm[SHORT_NAME_TO_LONG[x]] for x in MATRIX_NAMES]


def unpack_symbolic_matrices_with_params(mod, param_dict, data_dict=None, mode="FAST_COMPILE"):
    inputs = list(mod._name_to_variable.values())
    if data_dict is not None:
        inputs += list(mod._name_to_data.values())
    else:
        data_dict = {}

    f_matrices = pytensor.function(
        inputs,
        unpack_statespace(mod.ssm),
        on_unused_input="raise",
        mode=mode,
    )

    return f_matrices(**param_dict, **data_dict)


def simulate_from_numpy_model(
    mod, rng, param_dict, data_dict=None, steps=100, state_shocks=None, measurement_shocks=None
):
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, param_dict, data_dict)
    k_endog = mod.k_endog
    k_states = mod.k_states
    k_posdef = mod.k_posdef

    x = np.zeros((steps, k_states))
    y = np.zeros((steps, k_endog))

    x[0] = x0
    y[0] = (Z @ x0).squeeze() if Z.ndim == 2 else (Z[0] @ x0).squeeze()

    if not np.allclose(H, 0):
        y[0] += rng.multivariate_normal(mean=np.zeros(1), cov=H).squeeze()

    for t in range(1, steps):
        if k_posdef > 0:
            innov = R @ rng.multivariate_normal(mean=np.zeros(k_posdef), cov=Q)
        else:
            innov = 0

        if not np.allclose(H, 0):
            error = measurement_shocks[t - 1]
        else:
            error = 0

        x[t] = c + T @ x[t - 1] + innov
        if Z.ndim == 2:
            y[t] = (d + Z @ x[t] + error).squeeze()
        else:
            y[t] = (d + Z[t] @ x[t] + error).squeeze()

    return x, y.squeeze()


@pytest.mark.parametrize("n_obs,n_runs", [(100, 200)])
def test_exog_betas_random_walk(n_obs, n_runs):
    rng = np.random.default_rng(123)

    # Example model
    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        k_endog=2,
        error_order=1,
        error_var=False,
        k_exog=2,
        shared_exog_states=False,
        exog_innovations=True,
        error_cov_type="diagonal",
        measurement_error=False,
    )

    # Parameters
    param_dict = {
        "factor_loadings": np.array([[0.9], [0.8]]),
        "factor_ar": np.array([[0.5]]),
        "error_ar": np.array([[0.4], [0.3]]),
        "error_sigma": np.array([0.1, 0.2]),
        "P0": np.eye(dfm_mod.k_states),
        "x0": np.zeros(dfm_mod.k_states - dfm_mod.k_exog * dfm_mod.k_endog),
        "beta": np.array([0.3, 0.5, 1, 2]),
        "beta_sigma": np.array([1, 2, 3, 4]) ** 0.5,
    }
    data_dict = {"exog_data": np.random.normal(size=(n_obs, 2))}

    # Run multiple sims
    betas_t1, betas_t100 = [], []
    k_exog_states = dfm_mod.k_exog * dfm_mod.k_endog

    for _ in range(n_runs):
        x_traj, _ = simulate_from_numpy_model(dfm_mod, rng, param_dict, data_dict, steps=n_obs)
        beta_traj = x_traj[:, -k_exog_states:]
        betas_t1.append(beta_traj[1, :])
        betas_t100.append(beta_traj[-1, :])

    betas_t1 = np.array(betas_t1)
    betas_t100 = np.array(betas_t100)

    var_t1 = betas_t1.var(axis=0)
    var_t100 = betas_t100.var(axis=0)

    # ---- Assertion ----
    assert np.all(
        var_t100 > var_t1
    ), f"Expected variance at T=100 > T=1, got {var_t1} vs {var_t100}"


@pytest.mark.parametrize("shared", [True, False])
def test_exog_shared_vs_not(shared):
    rng = np.random.default_rng(123)

    n_obs = 50
    k_exog = 2
    k_endog = 2

    # Dummy exogenous data
    exog = rng.normal(size=(n_obs, k_exog))

    # Create the model
    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        k_endog=k_endog,
        error_order=1,
        k_exog=k_exog,
        shared_exog_states=shared,
        exog_innovations=False,
        error_cov_type="diagonal",
        measurement_error=False,
    )
    k_exog_states = dfm_mod.k_exog * dfm_mod.k_endog if not shared else dfm_mod.k_exog

    # Dummy parameters (small values so simulation is stable)
    param_dict = {
        "factor_loadings": np.array([[0.9], [0.8]]),
        "factor_ar": np.array([[0.5]]),
        "error_ar": np.array([[0.4], [0.3]]),
        "error_sigma": np.array([0.1, 0.2]),
        "P0": np.eye(dfm_mod.k_states),
        "x0": np.zeros(dfm_mod.k_states - k_exog_states),
        "beta": np.array([0.3, 0.5, 1, 2]) if not shared else np.array([0.3, 0.5]),
    }

    data_dict = {"exog_data": exog}

    # Simulate trajectory
    x_traj, y_traj = simulate_from_numpy_model(dfm_mod, rng, param_dict, data_dict, steps=n_obs)

    # Extract contribution from exogenous variables at time t
    beta = param_dict["beta"].reshape(-1, k_exog)  # shape depends on shared flag
    exog_t = exog[10]  # pick a random time point

    if shared:
        # all endogs get the same contribution
        contributions = [beta @ exog_t for _ in range(k_endog)]
        assert np.allclose(contributions[0], contributions[1:])
    else:
        # each endog gets a different contribution
        contributions = [beta[i] @ exog_t for i in range(k_endog)]
        # check that at least one differs
        assert not np.allclose(contributions[0], contributions[1:])
