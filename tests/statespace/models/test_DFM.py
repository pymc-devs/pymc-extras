from itertools import product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from pymc.testing import mock_sample_setup_and_teardown
from pytensor.graph.traversal import explicit_graph_inputs
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    ERROR_AR_PARAM_DIM,
    EXOG_STATE_DIM,
    FACTOR_DIM,
    LONG_MATRIX_NAMES,
    MATRIX_NAMES,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHORT_NAME_TO_LONG,
)
from tests.statespace.shared_fixtures import rng

mock_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)

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


def create_sm_test_values_mapping(
    test_values, data, k_factors, factor_order, error_order, error_var
):
    """Convert PyMC test values to statsmodels parameter format"""
    sm_test_values = {}

    # Factor loadings: PyMC shape (n_endog, k_factors) -> statsmodels individual params
    factor_loadings = test_values["factor_loadings"]
    all_pairs = product(data.columns, range(1, k_factors + 1))
    sm_test_values.update(
        {
            f"loading.f{factor_idx}.{endog_name}": value
            for (endog_name, factor_idx), value in zip(all_pairs, factor_loadings.ravel())
        }
    )

    # Factor AR coefficients: PyMC shape (k_factors, factor_order*k_factors) -> L{lag}.f{to}.f{from}
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

    # Error AR coefficients: PyMC shape (n_endog, error_order) -> L{lag}.e(var).e(var)
    if error_order > 0 and not error_var and "error_ar" in test_values:
        error_ar = test_values["error_ar"]
        pairs = product(enumerate(data.columns), range(1, error_order + 1))
        sm_test_values.update(
            {
                f"L{lag}.e({endog_name}).e({endog_name})": error_ar[endog_idx, lag - 1]
                for (endog_idx, endog_name), lag in pairs
            }
        )

    # Error AR coefficients: PyMC shape (n_endog, error_order * n_endog) -> L{lag}.e(var).e(var)
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

    # Observation error variances:
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
@pytest.mark.parametrize("error_var", [True, False])
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_DFM_update_matches_statsmodels(data, k_factors, factor_order, error_order, error_var, rng):
    if error_var and (factor_order > 0 or error_order > 0):
        pytest.xfail(
            "Statsmodels may be doing something wrong with error_var=True and (factor_order > 0 or error_order > 0) [numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite]"
        )

    mod = BayesianDynamicFactor(
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order,
        endog_names=data.columns.to_list(),
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

    x0 = test_values["x0"]
    P0 = test_values["P0"]

    sm_dfm.initialize_known(initial_state=x0, initial_state_cov=P0)
    sm_dfm.fit_constrained({name: sm_test_values[name] for name in sm_dfm.param_names})

    # Get PyMC matrices
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
    mod,
    rng,
    param_dict,
    data_dict=None,
    steps=100,
    state_shocks=None,
    measurement_shocks=None,
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
def test_DFM_exog_betas_random_walk(n_obs, n_runs):
    rng = np.random.default_rng(123)
    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["endogenous_0", "endogenous_1"],
        error_order=1,
        error_var=False,
        exog_names=["exogenous_0", "exogenous_1"],
        shared_exog_states=False,
        exog_innovations=True,
        error_cov_type="diagonal",
        measurement_error=False,
    )

    # Arbitrary Parameters
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

    assert np.all(var_t100 > var_t1), (
        f"Expected variance at T=100 > T=1, got {var_t1} vs {var_t100}"
    )


@pytest.mark.parametrize("shared", [True, False])
def test_DFM_exog_shared_vs_not(shared):
    rng = np.random.default_rng(123)

    n_obs = 50
    k_exog = 2
    k_endog = 2

    # Dummy exogenous data
    exog = rng.normal(size=(n_obs, k_exog))

    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["endogenous_0", "endogenous_1"],
        error_order=1,
        exog_names=["exogenous_0", "exogenous_1"],
        shared_exog_states=shared,
        exog_innovations=False,
        error_cov_type="diagonal",
        measurement_error=False,
    )

    k_exog_states = dfm_mod.k_exog * dfm_mod.k_endog if not shared else dfm_mod.k_exog

    if shared:
        beta = np.array([0.3, 0.5])
    else:
        beta = np.array([0.3, 0.5, 1.0, 2.0])

    param_dict = {
        "factor_loadings": np.array([[0.9], [0.8]]),
        "factor_ar": np.array([[0.5]]),
        "error_ar": np.array([[0.4], [0.3]]),
        "error_sigma": np.array([0.1, 0.2]),
        "P0": np.eye(dfm_mod.k_states),
        "x0": np.zeros(dfm_mod.k_states - k_exog_states),
        "beta": beta,
    }

    data_dict = {"exog_data": exog}

    # Simulate trajectory
    x_traj, y_traj = simulate_from_numpy_model(dfm_mod, rng, param_dict, data_dict, steps=n_obs)

    # Test 1: Check hidden states
    # Extract exogenous hidden states at time t=10
    t = 10
    exog_states_start = dfm_mod.k_states - k_exog_states
    exog_states_end = dfm_mod.k_states
    exog_hidden_states = x_traj[t, exog_states_start:exog_states_end]

    if shared:
        # When shared=True, there should be k_exog states total
        assert len(exog_hidden_states) == k_exog
    else:
        # When shared=False, there should be k_exog * k_endog states
        assert len(exog_hidden_states) == k_exog * k_endog
        # Each endogenous variable has its own set of exogenous states
        exog_states_reshaped = exog_hidden_states.reshape(k_endog, k_exog)
        assert not np.allclose(exog_states_reshaped[0], exog_states_reshaped[1])

    # Test 2: Check observed contributions
    exog_t = exog[t]

    if shared:
        # All endogenous variables get the same beta * data contribution
        contributions = [beta @ exog_t for _ in range(k_endog)]
        assert np.allclose(contributions[0], contributions[1]), (
            "Expected same contribution for all endog when shared=True"
        )
    else:
        # Each endogenous variable gets different beta * data
        beta_reshaped = beta.reshape(k_endog, k_exog)
        contributions = [beta_reshaped[i] @ exog_t for i in range(k_endog)]
        # Check that contributions are different
        assert not np.allclose(contributions[0], contributions[1]), (
            f"Expected different contributions, got {contributions}"
        )


class TestDFMConfiguration:
    def test_static_factor_no_ar_no_exog_diagonal_error(self):
        mod = BayesianDynamicFactor(
            k_factors=1,
            factor_order=0,
            endog_names=["y0", "y1", "y2"],
            error_order=0,
            error_var=False,
            error_cov_type="diagonal",
            measurement_error=False,
            verbose=False,
        )

        expected_param_names = ("x0", "P0", "factor_loadings", "error_sigma")
        expected_param_dims = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "error_sigma": (OBS_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: ("L0.factor_0",),
            ALL_STATE_AUX_DIM: ("L0.factor_0",),
            FACTOR_DIM: ("factor_1",),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert mod.state_names == ("L0.factor_0",)
        assert mod.observed_states == ("y0", "y1", "y2")
        assert mod.shock_names == ("factor_shock_0",)

    def test_dynamic_factor_ar1_error_diagonal_error(self):
        k_factors = 2
        factor_order = 2
        k_endog = 3
        error_order = 1
        error_var = False

        mod = BayesianDynamicFactor(
            k_factors=k_factors,
            factor_order=factor_order,
            endog_names=["y0", "y1", "y2"],
            error_order=error_order,
            error_var=error_var,
            error_cov_type="diagonal",
            measurement_error=True,
            verbose=False,
        )
        expected_param_names = (
            "x0",
            "P0",
            "factor_loadings",
            "factor_ar",
            "error_ar",
            "error_sigma",
            "sigma_obs",
        )
        expected_param_dims = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_sigma": (OBS_STATE_DIM,),
            "sigma_obs": (OBS_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_0",
                "L1.factor_0",
                "L0.factor_1",
                "L1.factor_1",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_0",
                "L1.factor_0",
                "L0.factor_1",
                "L1.factor_1",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
            ),
            FACTOR_DIM: ("factor_1", "factor_2"),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: (
                tuple(range(1, (error_order * k_endog) + 1))
                if error_var
                else tuple(range(1, error_order + 1))
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert len(mod.state_names) == k_factors * max(factor_order, 1) + k_endog * error_order
        assert mod.observed_states == ("y0", "y1", "y2")
        assert len(mod.shock_names) == k_factors + k_endog

    def test_dynamic_factor_ar2_error_var_unstructured(self):
        k_factors = 1
        factor_order = 1
        k_endog = 3
        error_order = 2
        error_var = True
        mod = BayesianDynamicFactor(
            k_factors=k_factors,
            factor_order=factor_order,
            endog_names=["y0", "y1", "y2"],
            error_order=error_order,
            error_var=error_var,
            error_cov_type="unstructured",
            measurement_error=True,
            verbose=False,
        )
        expected_param_names = (
            "x0",
            "P0",
            "factor_loadings",
            "factor_ar",
            "error_ar",
            "error_cov",
            "sigma_obs",
        )
        expected_param_dims = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_cov": (OBS_STATE_DIM, OBS_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_0",
                "L0.error_0",
                "L1.error_0",
                "L0.error_1",
                "L1.error_1",
                "L0.error_2",
                "L1.error_2",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_0",
                "L0.error_0",
                "L1.error_0",
                "L0.error_1",
                "L1.error_1",
                "L0.error_2",
                "L1.error_2",
            ),
            FACTOR_DIM: ("factor_1",),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: (
                tuple(range(1, (error_order * k_endog) + 1))
                if error_var
                else tuple(range(1, error_order + 1))
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert len(mod.state_names) == k_factors * max(factor_order, 1) + k_endog * error_order
        assert mod.observed_states == ("y0", "y1", "y2")
        assert len(mod.shock_names) == k_factors + k_endog

    def test_exog_shared_exog_states_exog_innovations(self):
        k_factors = 2
        factor_order = 1
        k_endog = 3
        error_order = 1
        k_exog = 2
        error_var = False
        shared_exog_states = True
        mod = BayesianDynamicFactor(
            k_factors=k_factors,
            factor_order=factor_order,
            endog_names=["y0", "y1", "y2"],
            error_order=error_order,
            error_var=error_var,
            exog_names=["x0", "x1"],
            shared_exog_states=shared_exog_states,
            exog_innovations=True,
            error_cov_type="diagonal",
            measurement_error=True,
            verbose=False,
        )
        expected_param_names = (
            "x0",
            "P0",
            "factor_loadings",
            "factor_ar",
            "error_ar",
            "error_sigma",
            "sigma_obs",
            "beta",
            "beta_sigma",
        )
        expected_param_dims = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_sigma": (OBS_STATE_DIM,),
            "sigma_obs": (OBS_STATE_DIM,),
            "beta": (EXOG_STATE_DIM,),
            "beta_sigma": (EXOG_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_0",
                "L0.factor_1",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
                "beta_x0[shared]",
                "beta_x1[shared]",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_0",
                "L0.factor_1",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
                "beta_x0[shared]",
                "beta_x1[shared]",
            ),
            FACTOR_DIM: ("factor_1", "factor_2"),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: (
                tuple(range(1, (error_order * k_endog) + 1))
                if error_var
                else tuple(range(1, error_order + 1))
            ),
            EXOG_STATE_DIM: (
                tuple(range(1, k_exog + 1))
                if shared_exog_states
                else tuple(range(1, k_exog * k_endog + 1))
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert len(mod.state_names) == k_factors * max(factor_order, 1) + k_endog * error_order + (
            k_exog if shared_exog_states else k_exog * k_endog
        )
        assert mod.observed_states == ("y0", "y1", "y2")
        assert len(mod.shock_names) == k_factors + k_endog + (
            k_exog if shared_exog_states else k_exog * k_endog
        )

    def test_exog_not_shared_no_exog_innovations(self):
        k_factors = 1
        factor_order = 2
        k_endog = 3
        error_order = 1
        k_exog = 1
        error_var = False
        shared_exog_states = False
        mod = BayesianDynamicFactor(
            k_factors=k_factors,
            factor_order=factor_order,
            endog_names=["y0", "y1", "y2"],
            error_order=error_order,
            error_var=error_var,
            exog_names=["x0"],
            shared_exog_states=shared_exog_states,
            exog_innovations=False,
            error_cov_type="scalar",
            measurement_error=False,
            verbose=False,
        )
        expected_param_names = (
            "x0",
            "P0",
            "factor_loadings",
            "factor_ar",
            "error_ar",
            "error_sigma",
            "beta",
        )
        expected_param_dims = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_sigma": (),
            "beta": (EXOG_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_0",
                "L1.factor_0",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
                "beta_x0[y0]",
                "beta_x0[y1]",
                "beta_x0[y2]",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_0",
                "L1.factor_0",
                "L0.error_0",
                "L0.error_1",
                "L0.error_2",
                "beta_x0[y0]",
                "beta_x0[y1]",
                "beta_x0[y2]",
            ),
            FACTOR_DIM: ("factor_1",),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: (
                tuple(range(1, (error_order * k_endog) + 1))
                if error_var
                else tuple(range(1, error_order + 1))
            ),
            EXOG_STATE_DIM: (
                tuple(range(1, k_exog + 1))
                if shared_exog_states
                else tuple(range(1, k_exog * k_endog + 1))
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert len(mod.state_names) == k_factors * max(factor_order, 1) + k_endog * error_order + (
            k_exog if shared_exog_states else k_exog * k_endog
        )
        assert mod.observed_states == ("y0", "y1", "y2")
        assert len(mod.shock_names) == k_factors + k_endog + (
            k_exog if shared_exog_states else k_exog * k_endog
        )


def test_dfm_workflow(rng, mock_sample):
    df = pd.read_csv(
        "tests/statespace/_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq

    ss_mod = BayesianDynamicFactor(
        endog_names=df.columns.tolist(),
        k_factors=1,
        factor_order=1,
        error_order=0,
        measurement_error=True,
        verbose=False,
    )

    with pm.Model(coords=ss_mod.coords) as m:
        pm.Normal("x0", dims=["state"])
        P0_diag = pm.Exponential("P0_diag", 1, dims=["state"])
        pm.Deterministic("P0", pt.diag(P0_diag), dims=["state", "state_aux"])

        pm.Normal("factor_loadings", dims=["observed_state", "factor"])
        pm.Normal("factor_ar", dims=["factor", "lag_ar"])
        pm.Exponential("error_sigma", 1, dims=["observed_state"])
        pm.Exponential("sigma_obs", 1, dims=["observed_state"])

        ss_mod.build_statespace_graph(df)

        idata = pm.sample()

    post = ss_mod.sample_conditional_posterior(idata, mvn_method="svd")
    assert "filtered_posterior" in post
    assert "smoothed_posterior" in post
    assert "predicted_posterior" in post

    forecast = ss_mod.forecast(idata, periods=10, random_seed=rng)
    assert "forecast_latent" in forecast
    assert "forecast_observed" in forecast
    assert np.isfinite(forecast.forecast_latent.values).all()
    assert np.isfinite(forecast.forecast_observed.values).all()

    irf = ss_mod.impulse_response_function(idata, n_steps=10, random_seed=rng)
    assert "irf" in irf
    assert np.isfinite(irf.irf.values).all()
