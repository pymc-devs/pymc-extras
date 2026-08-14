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

from pymc_extras.statespace.core.statespace import FILTER_FACTORY
from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    ERROR_AR_PARAM_DIM,
    EXOG_COEF_STATE_DIM,
    EXOG_STATE_DIM,
    FACTOR_DIM,
    LONG_MATRIX_NAMES,
    MATRIX_NAMES,
    NON_EXOG_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHORT_NAME_TO_LONG,
    TIME_DIM,
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

    # statsmodels parameterizes the observation error by its variance, not its standard deviation.
    if "error_sigma" in test_values:
        error_sigma = test_values["error_sigma"]
        sm_test_values.update(
            {
                f"sigma2.{endog_name}": error_sigma[endog_idx] ** 2
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


def simulate_from_matrices(matrices, rng, steps=100):
    x0, P0, c, d, T, Z, R, H, Q = matrices
    k_states, k_posdef = R.shape
    k_endog = Z.shape[-2]
    has_measurement_error = not np.allclose(H, 0)

    x = np.zeros((steps, k_states))
    y = np.zeros((steps, k_endog))

    def measurement_error():
        if not has_measurement_error:
            return 0
        return rng.multivariate_normal(mean=np.zeros(k_endog), cov=H).squeeze()

    x[0] = x0
    y[0] = (Z @ x0).squeeze() if Z.ndim == 2 else (Z[0] @ x0).squeeze()
    y[0] += measurement_error()

    for t in range(1, steps):
        innov = R @ rng.multivariate_normal(mean=np.zeros(k_posdef), cov=Q) if k_posdef > 0 else 0

        x[t] = c + T @ x[t - 1] + innov
        design = Z if Z.ndim == 2 else Z[t]
        y[t] = (d + design @ x[t] + measurement_error()).squeeze()

    return x, y.squeeze()


@pytest.mark.parametrize("n_obs,n_runs", [(100, 200)])
def test_exog_coefficient_random_walk_variance_grows_linearly(n_obs, n_runs):
    """With exog_innovations the coefficients random walk, so Var(beta_t) = t * diag(Q)."""
    rng = np.random.default_rng(123)
    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["endogenous_0", "endogenous_1"],
        error_order=1,
        error_var=False,
        exog_state_names=["exogenous_0", "exogenous_1"],
        shared_exog_states=False,
        exog_innovations=True,
        error_cov_type="diagonal",
        measurement_error=False,
        verbose=False,
    )

    beta_variances = np.array([1.0, 2.0, 3.0, 4.0])
    param_dict = {
        "factor_loadings": np.array([[0.9], [0.8]]),
        "factor_ar": np.array([[0.5]]),
        "error_ar": np.array([[0.4], [0.3]]),
        "error_sigma": np.array([0.1, 0.2]),
        "P0": np.eye(dfm_mod.k_states),
        "x0": np.zeros(dfm_mod.k_states - dfm_mod.k_exog_states),
        "beta": np.array([0.3, 0.5, 1.0, 2.0]),
        "beta_sigma": beta_variances,
    }
    matrices = unpack_symbolic_matrices_with_params(
        dfm_mod, param_dict, {"exog_data": rng.normal(size=(n_obs, 2))}
    )

    first_exog_state = dfm_mod.k_states - dfm_mod.k_exog_states
    coefficient_paths = np.array(
        [
            simulate_from_matrices(matrices, rng, steps=n_obs)[0][:, first_exog_state:]
            for _ in range(n_runs)
        ]
    )

    assert_allclose(coefficient_paths[:, 1].var(axis=0), beta_variances, rtol=0.3)
    assert_allclose(coefficient_paths[:, -1].var(axis=0), (n_obs - 1) * beta_variances, rtol=0.3)


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per_series"])
def test_exog_contribution_is_shared_across_series_only_when_requested(shared):
    rng = np.random.default_rng(123)
    n_obs, k_exog, k_endog = 50, 2, 2
    exog = rng.normal(size=(n_obs, k_exog))

    dfm_mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["endogenous_0", "endogenous_1"],
        error_order=1,
        exog_state_names=["exogenous_0", "exogenous_1"],
        shared_exog_states=shared,
        exog_innovations=False,
        error_cov_type="diagonal",
        measurement_error=False,
        verbose=False,
    )

    beta = np.array([0.3, 0.5]) if shared else np.array([0.3, 0.5, 1.0, 2.0])
    param_dict = {
        "factor_loadings": np.array([[0.9], [0.8]]),
        "factor_ar": np.array([[0.5]]),
        "error_ar": np.array([[0.4], [0.3]]),
        "error_sigma": np.array([0.1, 0.2]),
        "P0": np.eye(dfm_mod.k_states),
        "x0": np.zeros(dfm_mod.k_states - dfm_mod.k_exog_states),
        "beta": beta,
    }

    matrices = unpack_symbolic_matrices_with_params(dfm_mod, param_dict, {"exog_data": exog})
    x_traj, _ = simulate_from_matrices(matrices, rng, steps=n_obs)
    Z = dict(zip(MATRIX_NAMES, matrices))["Z"]

    t = 10
    first_exog_state = dfm_mod.k_states - dfm_mod.k_exog_states

    # Without innovations the coefficients never move off their initial value.
    assert_allclose(x_traj[t, first_exog_state:], beta)

    exog_contribution = Z[t][:, first_exog_state:] @ x_traj[t, first_exog_state:]
    if shared:
        expected = np.full(k_endog, beta @ exog[t])
    else:
        expected = beta.reshape(k_endog, k_exog) @ exog[t]

    assert_allclose(exog_contribution, expected)


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
            ALL_STATE_DIM: ("L0.factor_1",),
            ALL_STATE_AUX_DIM: ("L0.factor_1",),
            FACTOR_DIM: ("factor_1",),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert mod.state_names == ("L0.factor_1",)
        assert mod.observed_states == ("y0", "y1", "y2")
        assert mod.shock_names == ("factor_shock_1",)

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
                "L0.factor_1",
                "L0.factor_2",
                "L1.factor_1",
                "L1.factor_2",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_1",
                "L0.factor_2",
                "L1.factor_1",
                "L1.factor_2",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
            ),
            FACTOR_DIM: ("factor_1", "factor_2"),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: tuple(range(1, error_order + 1)),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
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
                "L0.factor_1",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "L1.error_1",
                "L1.error_2",
                "L1.error_3",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_1",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "L1.error_1",
                "L1.error_2",
                "L1.error_3",
            ),
            FACTOR_DIM: ("factor_1",),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: tuple(range(1, (error_order * k_endog) + 1)),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
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
            exog_state_names=["x0", "x1"],
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
            "x0": (NON_EXOG_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_sigma": (OBS_STATE_DIM,),
            "sigma_obs": (OBS_STATE_DIM,),
            "beta": (EXOG_COEF_STATE_DIM,),
            "beta_sigma": (EXOG_COEF_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_1",
                "L0.factor_2",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "beta_x0[shared]",
                "beta_x1[shared]",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_1",
                "L0.factor_2",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "beta_x0[shared]",
                "beta_x1[shared]",
            ),
            FACTOR_DIM: ("factor_1", "factor_2"),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: tuple(range(1, error_order + 1)),
            EXOG_STATE_DIM: ("x0", "x1"),
            EXOG_COEF_STATE_DIM: ("beta_x0[shared]", "beta_x1[shared]"),
            NON_EXOG_STATE_DIM: (
                "L0.factor_1",
                "L0.factor_2",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert mod.observed_states == ("y0", "y1", "y2")
        assert mod.shock_names == (
            "factor_shock_1",
            "factor_shock_2",
            "error_shock_1",
            "error_shock_2",
            "error_shock_3",
            "exog_shock_x0[shared]",
            "exog_shock_x1[shared]",
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
            exog_state_names=["x0"],
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
            "x0": (NON_EXOG_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "factor_loadings": (OBS_STATE_DIM, FACTOR_DIM),
            "factor_ar": (FACTOR_DIM, AR_PARAM_DIM),
            "error_ar": (OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
            "error_sigma": (),
            "beta": (EXOG_COEF_STATE_DIM,),
        }
        expected_coords = {
            OBS_STATE_DIM: ("y0", "y1", "y2"),
            ALL_STATE_DIM: (
                "L0.factor_1",
                "L1.factor_1",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "beta_x0[y0]",
                "beta_x0[y1]",
                "beta_x0[y2]",
            ),
            ALL_STATE_AUX_DIM: (
                "L0.factor_1",
                "L1.factor_1",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
                "beta_x0[y0]",
                "beta_x0[y1]",
                "beta_x0[y2]",
            ),
            FACTOR_DIM: ("factor_1",),
            AR_PARAM_DIM: tuple(range(1, k_factors * max(factor_order, 1) + 1)),
            ERROR_AR_PARAM_DIM: tuple(range(1, error_order + 1)),
            EXOG_STATE_DIM: ("x0",),
            EXOG_COEF_STATE_DIM: ("beta_x0[y0]", "beta_x0[y1]", "beta_x0[y2]"),
            NON_EXOG_STATE_DIM: (
                "L0.factor_1",
                "L1.factor_1",
                "L0.error_1",
                "L0.error_2",
                "L0.error_3",
            ),
        }

        assert mod.param_names == expected_param_names
        assert mod.param_dims == expected_param_dims
        for k, v in expected_coords.items():
            assert mod.coords[k] == v
        assert mod.observed_states == ("y0", "y1", "y2")
        assert mod.shock_names == (
            "factor_shock_1",
            "error_shock_1",
            "error_shock_2",
            "error_shock_3",
            "exog_shock_x0[y0]",
            "exog_shock_x0[y1]",
            "exog_shock_x0[y2]",
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


def evaluate_matrices(mod, param_values, data_dict=None):
    matrices = unpack_symbolic_matrices_with_params(mod, param_values, data_dict=data_dict)
    return dict(zip(MATRIX_NAMES, matrices))


def random_param_values(mod, rng):
    return {
        name: rng.normal(size=variable.type.shape).astype(floatX)
        for name, variable in mod._name_to_variable.items()
    }


@pytest.mark.parametrize(
    "k_factors, factor_order, error_order, error_var",
    [(2, 3, 0, False), (1, 1, 2, False), (2, 2, 2, False), (1, 1, 2, True), (2, 2, 2, True)],
    ids=["factor_lags", "error_lags", "both", "error_lags_var", "both_var"],
)
def test_lagged_state_names_match_transition_shifts(
    k_factors, factor_order, error_order, error_var, rng
):
    """A state called ``L{lag}.X`` has to be the previous step's ``L{lag - 1}.X``."""
    mod = BayesianDynamicFactor(
        k_factors=k_factors,
        factor_order=factor_order,
        endog_names=["y0", "y1", "y2"],
        error_order=error_order,
        error_var=error_var,
        verbose=False,
    )
    T = evaluate_matrices(mod, random_param_values(mod, rng))["T"]
    name_to_index = {name: i for i, name in enumerate(mod.state_names)}

    lagged_states = [
        (i, name) for i, name in enumerate(mod.state_names) if name.startswith(("L1.", "L2."))
    ]
    assert lagged_states, "this configuration has no lagged states to check"

    for index, name in lagged_states:
        lag, component = name.split(".", 1)
        expected = np.zeros(mod.k_states)
        expected[name_to_index[f"L{int(lag[1:]) - 1}.{component}"]] = 1.0

        assert_allclose(T[index], expected, err_msg=f"state {name!r} does not shift from its lag")


@pytest.mark.parametrize("shared_exog_states", [True, False], ids=["shared", "per_series"])
def test_exog_state_names_match_design_columns(shared_exog_states, rng):
    """A state called ``beta_x[y]`` has to be the column of Z carrying x onto series y."""
    endog_names = ["y0", "y1", "y2"]
    exog_state_names = ["a", "b"]
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=endog_names,
        exog_state_names=exog_state_names,
        shared_exog_states=shared_exog_states,
        verbose=False,
    )

    exog_data = np.array([[2.0, 5.0]], dtype=floatX)
    Z = evaluate_matrices(mod, random_param_values(mod, rng), data_dict={"exog_data": exog_data})[
        "Z"
    ][0]
    exog_values = dict(zip(exog_state_names, exog_data[0]))

    beta_states = [(i, name) for i, name in enumerate(mod.state_names) if name.startswith("beta_")]
    assert len(beta_states) == mod.k_exog_states

    for column, name in beta_states:
        exog_name, endog_name = name.removeprefix("beta_").rstrip("]").split("[")
        value = exog_values[exog_name]

        if endog_name == "shared":
            expected = np.full(len(endog_names), value)
        else:
            expected = np.zeros(len(endog_names))
            expected[endog_names.index(endog_name)] = value

        assert_allclose(Z[:, column], expected, err_msg=f"state {name!r} loads the wrong series")


def test_unstructured_error_cov_without_error_states():
    """With error_order=0 the idiosyncratic error has no state, so it is observation noise."""
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1", "y2"],
        error_order=0,
        error_cov_type="unstructured",
        verbose=False,
    )
    error_cov = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 2.0]], dtype=floatX)
    matrices = evaluate_matrices(
        mod,
        {
            "x0": np.zeros(mod.k_states, dtype=floatX),
            "P0": np.eye(mod.k_states, dtype=floatX),
            "factor_loadings": np.ones((3, 1), dtype=floatX),
            "factor_ar": np.array([[0.5]], dtype=floatX),
            "error_cov": error_cov,
        },
    )

    assert_allclose(matrices["H"], error_cov)
    assert_allclose(matrices["Q"], np.eye(1))


def test_scalar_error_cov_without_error_states():
    """A scalar error standard deviation spreads its variance over the observation covariance."""
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1", "y2"],
        error_order=0,
        error_cov_type="scalar",
        verbose=False,
    )
    matrices = evaluate_matrices(
        mod,
        {
            "x0": np.zeros(mod.k_states, dtype=floatX),
            "P0": np.eye(mod.k_states, dtype=floatX),
            "factor_loadings": np.ones((3, 1), dtype=floatX),
            "factor_ar": np.array([[0.5]], dtype=floatX),
            "error_sigma": np.array(0.75, dtype=floatX),
        },
    )

    assert_allclose(matrices["H"], np.eye(3) * 0.75**2)


def test_measurement_error_adds_to_idiosyncratic_error():
    """error_sigma and sigma_obs are standard deviations, so H sums their variances."""
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1", "y2"],
        error_order=0,
        error_cov_type="diagonal",
        measurement_error=True,
        verbose=False,
    )
    matrices = evaluate_matrices(
        mod,
        {
            "x0": np.zeros(mod.k_states, dtype=floatX),
            "P0": np.eye(mod.k_states, dtype=floatX),
            "factor_loadings": np.ones((3, 1), dtype=floatX),
            "factor_ar": np.array([[0.5]], dtype=floatX),
            "error_sigma": np.array([1.0, 2.0, 3.0], dtype=floatX),
            "sigma_obs": np.array([0.5, 0.25, 0.125], dtype=floatX),
        },
    )

    assert_allclose(matrices["H"], np.diag([1.0**2 + 0.5**2, 2.0**2 + 0.25**2, 3.0**2 + 0.125**2]))


def test_dfm_rejects_unknown_error_cov_type():
    with pytest.raises(ValueError, match="error_cov_type must be one of"):
        BayesianDynamicFactor(
            k_factors=1,
            factor_order=1,
            endog_names=["y0", "y1"],
            error_cov_type="diagnoal",
            verbose=False,
        )


@pytest.mark.parametrize("filter_type", ["standard", "univariate", "cholesky"])
def test_dfm_accepts_filter_type_and_mode(filter_type):
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1"],
        filter_type=filter_type,
        mode="FAST_COMPILE",
        verbose=False,
    )

    kalman_filter, _ = mod.make_filters()
    assert isinstance(kalman_filter, FILTER_FACTORY[filter_type])
    assert mod.mode == "FAST_COMPILE"


@pytest.mark.parametrize("shared_exog_states", [True, False], ids=["shared", "per_series"])
def test_exog_model_builds_from_advertised_dims(shared_exog_states, rng):
    """Every parameter's advertised dims must be as long as the variable it stands for."""
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1", "y2"],
        exog_state_names=["a", "b"],
        shared_exog_states=shared_exog_states,
        exog_innovations=True,
        verbose=False,
    )

    index = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame(rng.normal(size=(20, 3)), columns=["y0", "y1", "y2"], index=index)
    exog = rng.normal(size=(20, 2))

    assert len(mod.coords[EXOG_STATE_DIM]) == exog.shape[1]

    with pm.Model(coords=mod.coords) as pymc_mod:
        pm.Data("exog_data", exog, dims=[TIME_DIM, EXOG_STATE_DIM])
        pm.Normal("x0", dims=mod.param_dims["x0"])
        P0_diag = pm.Exponential("P0_diag", 1, dims=[ALL_STATE_DIM])
        pm.Deterministic("P0", pt.diag(P0_diag), dims=[ALL_STATE_DIM, ALL_STATE_AUX_DIM])
        pm.Normal("factor_loadings", dims=mod.param_dims["factor_loadings"])
        pm.Normal("factor_ar", dims=mod.param_dims["factor_ar"])
        pm.Exponential("error_sigma", 1, dims=mod.param_dims["error_sigma"])
        pm.Normal("beta", dims=mod.param_dims["beta"])
        pm.Exponential("beta_sigma", 1, dims=mod.param_dims["beta_sigma"])

        mod.build_statespace_graph(data)

    assert np.isfinite(pymc_mod.compile_logp()(pymc_mod.initial_point()))


@pytest.mark.parametrize("shared_exog_states", [True, False], ids=["shared", "per_series"])
def test_exog_shock_names_match_the_states_they_drive(shared_exog_states, rng):
    """An exogenous shock is named for the coefficient state its column of R feeds."""
    mod = BayesianDynamicFactor(
        k_factors=1,
        factor_order=1,
        endog_names=["y0", "y1", "y2"],
        exog_state_names=["a", "b"],
        shared_exog_states=shared_exog_states,
        exog_innovations=True,
        verbose=False,
    )
    R = evaluate_matrices(
        mod,
        random_param_values(mod, rng),
        data_dict={"exog_data": np.zeros((1, 2), dtype=floatX)},
    )["R"]

    exog_shocks = [(i, n) for i, n in enumerate(mod.shock_names) if n.startswith("exog_shock_")]
    assert len(exog_shocks) == mod.k_exog_states

    for shock_index, shock_name in exog_shocks:
        driven_states = np.flatnonzero(R[:, shock_index])
        assert len(driven_states) == 1, f"{shock_name!r} does not drive exactly one state"

        state_name = mod.state_names[driven_states[0]]
        assert shock_name == state_name.replace("beta_", "exog_shock_", 1)
