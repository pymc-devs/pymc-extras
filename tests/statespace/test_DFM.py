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
@pytest.mark.parametrize("factor_order", [1, 2, 3, 4])
@pytest.mark.parametrize("error_order", [1, 2, 3, 4])
def test_dfm_matrices_match_statsmodels(data, k_factors, factor_order, error_order):
    y = data
    print(data.shape)
    n_obs, k_endog = y.shape
    print(n_obs, k_endog)

    with pm.Model():
        dfm = BayesianDynamicFactor(
            k_factors=k_factors,
            factor_order=factor_order,
            k_endog=k_endog,
            error_order=error_order,
            error_var=False,
            verbose=True,
        )
        factor_part = max(1, factor_order) * k_factors
        error_part = error_order * k_endog if error_order > 0 else 0
        k_states = factor_part + error_part

        pm.Normal("x0", mu=0.0, sigma=1.0, shape=(k_states,))
        pm.HalfNormal("P0", sigma=1.0, shape=(k_states, k_states))

        pm.Normal("factor_loadings", mu=0.0, sigma=1.0, shape=(k_endog, k_factors))
        pm.Normal("factor_ar", mu=0.0, sigma=1.0, shape=(k_factors, factor_order))
        pm.HalfNormal("factor_sigma", sigma=1.0, shape=(k_factors,))

        pm.Normal("error_ar", mu=0.0, sigma=1.0, shape=(k_endog, error_order))
        pm.HalfNormal("error_sigma", sigma=1.0, shape=(k_endog,))

        dfm.build_statespace_graph(data=data)
        ssm_pymc = dfm.unpack_statespace()

    # Build statsmodels DynamicFactor model
    sm_dfm = DynamicFactor(
        endog=y,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
    )

    x0 = np.zeros(k_states)
    P0 = np.eye(k_states)
    sm_dfm.initialize_known(initial_state=x0, initial_state_cov=P0)
    # print(sm_dfm.ssm.valid_names)

    # Extract relevant matrices from statsmodels
    # sm_matrices = [sm_dfm.ssm[m] for m in ["initial_state", "initial_state_cov", "state_intercept", "obs_intercept", "transition", "design", "selection","obs_cov", "state_cov"]]
    sm_matrices = sm_dfm.ssm

    # Matrices arestore in same order in PyMC model and statsmodels model

    # Compile pymc tensors to numpy arrays with dummy inputs
    pymc_inputs = list(pytensor.graph.basic.explicit_graph_inputs(ssm_pymc))
    f_ssm = pytensor.function(pymc_inputs, ssm_pymc)

    # Provide dummy inputs: zeros or identities as appropriate
    test_vals = {}
    for v in pymc_inputs:
        shape = v.type.shape
        if len(shape) == 0:
            test_vals[v.name] = 0.5
        else:
            # For covariance matrices, use identity, else fill with 0.5
            if "cov" in v.name or "state_cov" in v.name or "obs_cov" in v.name:
                test_vals[v.name] = np.eye(shape[0], shape[1], dtype=pytensor.config.floatX)
            else:
                test_vals[v.name] = np.full(shape, 0.5, dtype=pytensor.config.floatX)

    pymc_matrices = f_ssm(**test_vals)

    matrix_names = [
        "initial_state",
        "initial_state_cov",
        "state_intercept",
        "obs_intercept",
        "transition",
        "design",
        "selection",
        "obs_cov",
        "state_cov",
    ]
    for pm_mat, sm_mat, name in zip(pymc_matrices, sm_matrices, matrix_names):
        assert_allclose(pm_mat, sm_mat, atol=1e-5, err_msg=f"Mismatch in matrix {name}")


@pytest.mark.parametrize("k_factors", [1])
@pytest.mark.parametrize("factor_order", [2])
@pytest.mark.parametrize("error_order", [1])
def test_loglike_matches_statsmodels(data, k_factors, factor_order, error_order):
    """
    Tests if the log-likelihood calculated by the PyMC statespace model matches
    the one from statsmodels when given the same parameters.
    """
    k_endog = data.shape[1]
    endog_names = data.columns.tolist()  # ['realgdp', 'realcons', 'realinv']

    sm_dfm = DynamicFactor(
        endog=data,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
    )
    print(sm_dfm.param_names)

    test_params = {
        # Loadings are now separate parameters
        "loading.f1.realgdp": 0.9,
        "loading.f1.realcons": -0.8,
        "loading.f1.realinv": 0.7,
        # Factor AR coefficients
        "L1.f1.f1": 0.5,
        "L2.f1.f1": 0.2,
        # Error AR coefficients use parenthesis
        "L1.e(realgdp).e(realgdp)": 0.4,
        "L1.e(realcons).e(realcons)": 0.2,
        "L1.e(realinv).e(realinv)": -0.1,
        # Error variances
        "sigma2.realgdp": 0.5,
        "sigma2.realcons": 0.4,
        "sigma2.realinv": 0.3,
        # NOTE: 'sigma2.f1' NOT included, as it's fixed to 1 by default.
    }

    assert set(test_params.keys()) == set(sm_dfm.param_names)
    params_vector = np.array([test_params[name] for name in sm_dfm.param_names])

    # Calculate the log-likelihood in statsmodels
    sm_loglike = sm_dfm.loglike(params_vector)

    # Build the PyMC model and compile its logp function
    with pm.Model() as model:
        factor_part = max(1, factor_order) * k_factors
        error_part = error_order * k_endog if error_order > 0 else 0
        k_states = factor_part + error_part

        pm.Normal("x0", mu=0.0, sigma=1.0, shape=(k_states,))
        pm.Deterministic("P0", pt.eye(k_states, k_states))

        pm.Normal("factor_loadings", mu=0.0, sigma=1.0, shape=(k_endog, k_factors))
        pm.Normal("factor_ar", mu=0.0, sigma=1.0, shape=(k_factors, factor_order))
        pm.Deterministic("factor_sigma", pt.ones(k_factors))

        pm.Normal("error_ar", mu=0.0, sigma=1.0, shape=(k_endog, error_order))
        pm.HalfNormal("error_sigma", sigma=1.0, shape=(k_endog,))

        dfm = BayesianDynamicFactor(
            k_factors=k_factors, factor_order=factor_order, k_endog=k_endog, error_order=error_order
        )

        dfm.build_statespace_graph(data, "data_ssm")
        logp_fn = model.compile_logp()

    # Evaluate the PyMC log-likelihood using the same parameters
    pymc_param_values = {
        "x0": np.zeros(k_states),
        #'P0_log__': np.log(np.eye(k_states)),
        "factor_loadings": np.array(
            [
                test_params["loading.f1.realgdp"],
                test_params["loading.f1.realcons"],
                test_params["loading.f1.realinv"],
            ]
        ).reshape(k_endog, k_factors),
        "factor_ar": np.array([test_params["L1.f1.f1"], test_params["L2.f1.f1"]]).reshape(
            k_factors, factor_order
        ),
        #'factor_sigma_log__': np.log(np.array([1.0])),
        "error_ar": np.array(
            [
                test_params["L1.e(realgdp).e(realgdp)"],
                test_params["L1.e(realcons).e(realcons)"],
                test_params["L1.e(realinv).e(realinv)"],
            ]
        ).reshape(k_endog, error_order),
        "error_sigma_log__": np.log(
            np.sqrt(
                np.array(
                    [  # Log-transformed
                        test_params["sigma2.realgdp"],
                        test_params["sigma2.realcons"],
                        test_params["sigma2.realinv"],
                    ]
                )
            )
        ),
    }
    pymc_loglike = logp_fn(pymc_param_values)

    assert_allclose(
        pymc_loglike,
        sm_loglike,
        rtol=1e-5,
        err_msg="PyMC log-likelihood does not match statsmodels for manually specified parameters",
    )
