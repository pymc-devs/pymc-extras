from itertools import combinations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose, assert_array_less

from pymc_extras.statespace import BayesianSARIMA
from pymc_extras.statespace.models.utilities import (
    make_harvey_state_names,
    make_SARIMA_transition_matrix,
)
from pymc_extras.statespace.utils.constants import (
    SARIMAX_STATE_STRUCTURES,
    SHORT_NAME_TO_LONG,
)
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import (
    load_nile_test_data,
    make_stationary_params,
    simulate_from_numpy_model,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-6
RTOL = 0 if floatX.endswith("64") else 1e-6

test_state_names = [
    ["data", "state_1", "state_2"],
    ["data", "data_star", "state_star_1", "state_star_2"],
    ["data", "D1.data", "data_star", "state_star_1", "state_star_2"],
    ["data", "D1.data", "D1^2.data", "data_star", "state_star_1", "state_star_2"],
    ["data", "state_1", "state_2", "state_3"],
    ["data", "state_1", "state_2", "state_3", "state_4", "state_5", "state_6", "state_7"],
    [
        "data",
        "state_1",
        "state_2",
        "state_3",
        "state_4",
        "state_5",
        "state_6",
        "state_7",
        "state_8",
        "state_9",
        "state_10",
    ],
    [
        "data",
        "L1.data",
        "L2.data",
        "L3.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
    ],
    [
        "data",
        "L1.data",
        "L2.data",
        "L3.data",
        "D4.data",
        "L1D4.data",
        "L2D4.data",
        "L3D4.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
    ],
    [
        "data",
        "D1.data",
        "L1D1.data",
        "L2D1.data",
        "L3D1.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
        "state_star_4",
        "state_star_5",
    ],
    [
        "data",
        "D1.data",
        "D1^2.data",
        "L1D1^2.data",
        "L2D1^2.data",
        "L3D1^2.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
        "state_star_4",
        "state_star_5",
    ],
    [
        "data",
        "D1.data",
        "D1^2.data",
        "L1D1^2.data",
        "L2D1^2.data",
        "L3D1^2.data",
        "D1^2D4.data",
        "L1D1^2D4.data",
        "L2D1^2D4.data",
        "L3D1^2D4.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
        "state_star_4",
        "state_star_5",
    ],
    [
        "data",
        "D1.data",
        "L1D1.data",
        "L2D1.data",
        "D1D3.data",
        "L1D1D3.data",
        "L2D1D3.data",
        "D1D3^2.data",
        "L1D1D3^2.data",
        "L2D1D3^2.data",
        "data_star",
        "state_star_1",
        "state_star_2",
        "state_star_3",
        "state_star_4",
    ],
    ["data", "data_star"] + [f"state_star_{i+1}" for i in range(26)],
]

test_orders = [
    (2, 0, 2, 0, 0, 0, 0),
    (2, 1, 2, 0, 0, 0, 0),
    (2, 2, 2, 0, 0, 0, 0),
    (2, 3, 2, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 4),
    (0, 0, 0, 2, 0, 1, 4),
    (2, 0, 2, 2, 0, 2, 4),
    (0, 0, 0, 1, 1, 0, 4),
    (0, 0, 0, 1, 2, 0, 4),
    (1, 1, 1, 1, 1, 1, 4),
    (1, 2, 1, 1, 1, 1, 4),
    (1, 2, 1, 1, 2, 1, 4),
    (1, 1, 1, 1, 3, 1, 3),
    (2, 1, 2, 2, 0, 2, 12),
]

ids = [f"p={p},d={d},q={q},P={P},D={D},Q={Q},S={S}" for (p, d, q, P, D, Q, S) in test_orders]


@pytest.fixture
def data():
    return load_nile_test_data()


@pytest.fixture(scope="session")
def arima_mod():
    return BayesianSARIMA(order=(2, 0, 1), stationary_initialization=True, verbose=False)


@pytest.fixture(scope="session")
def pymc_mod(arima_mod):
    data = load_nile_test_data()

    with pm.Model(coords=arima_mod.coords) as pymc_mod:
        # x0  = pm.Normal('x0', dims=['state'])
        # P0_diag = pm.Gamma('P0_diag', alpha=2, beta=1, dims=['state'])
        # P0 = pm.Deterministic('P0', pt.diag(P0_diag), dims=['state', 'state_aux'])
        ar_params = pm.Normal("ar_params", sigma=0.1, dims=["ar_lag"])
        ma_params = pm.Normal("ma_params", sigma=1, dims=["ma_lag"])
        sigma_state = pm.Exponential("sigma_state", 0.5)
        arima_mod.build_statespace_graph(data=data, save_kalman_filter_outputs_in_idata=True)

    return pymc_mod


@pytest.fixture(scope="session")
def arima_mod_interp():
    return BayesianSARIMA(
        order=(3, 0, 3),
        stationary_initialization=False,
        verbose=False,
        state_structure="interpretable",
        measurement_error=True,
    )


@pytest.fixture(scope="session")
def pymc_mod_interp(arima_mod_interp):
    data = load_nile_test_data()

    with pm.Model(coords=arima_mod_interp.coords) as pymc_mod:
        x0 = pm.Normal("x0", dims=["state"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(arima_mod_interp.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        ar_params = pm.Normal("ar_params", sigma=0.1, dims=["ar_lag"])
        ma_params = pm.Normal("ma_params", sigma=1, dims=["ma_lag"])
        sigma_state = pm.Exponential("sigma_state", 0.5)
        sigma_obs = pm.Exponential("sigma_obs", 0.1)

        arima_mod_interp.build_statespace_graph(data=data, save_kalman_filter_outputs_in_idata=True)

    return pymc_mod


def test_mode_argument():
    # Mode argument should be passed to the parent class
    mod = BayesianSARIMA(order=(0, 0, 3), mode="FAST_RUN", verbose=False)
    assert mod.mode == "FAST_RUN"


@pytest.mark.parametrize(
    "p,d,q,P,D,Q,S,expected_names",
    [(*order, name) for order, name in zip(test_orders, test_state_names)],
    ids=ids,
)
def test_harvey_state_names(p, d, q, P, D, Q, S, expected_names):
    if all([x == 0 for x in [p, d, q, P, D, Q, S]]):
        pytest.skip("Skip all zero case")

    k_states = max(p + P * S, q + Q * S + 1) + (S * D + d)
    states = make_harvey_state_names(p, d, q, P, D, Q, S)

    assert len(states) == k_states
    missing_from_expected = set(expected_names) - set(states)
    assert len(missing_from_expected) == 0


@pytest.mark.parametrize("p,d,q,P,D,Q,S", test_orders)
def test_make_SARIMA_transition_matrix(p, d, q, P, D, Q, S):
    T = make_SARIMA_transition_matrix(p, d, q, P, D, Q, S)
    mod = sm.tsa.SARIMAX(np.random.normal(size=100), order=(p, d, q), seasonal_order=(P, D, Q, S))
    T2 = mod.ssm["transition"]

    if D > 2:
        pytest.skip("Statsmodels has a bug when D > 2, skip this test.")
    else:
        assert_allclose(T, T2, err_msg="Transition matrix does not match statsmodels")


@pytest.mark.parametrize("p, d, q, P, D, Q, S", test_orders, ids=ids)
@pytest.mark.filterwarnings(
    "ignore:Non-invertible starting MA parameters found.",
    "ignore:Non-stationary starting autoregressive parameters found",
    "ignore:Non-invertible starting seasonal moving average",
    "ignore:Non-stationary starting seasonal autoregressive",
    "ignore:divide by zero encountered in matmul:RuntimeWarning",
    "ignore:overflow encountered in matmul:RuntimeWarning",
    "ignore:invalid value encountered in matmul:RuntimeWarning",
)
def test_SARIMAX_update_matches_statsmodels(p, d, q, P, D, Q, S, data, rng):
    sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, S))

    param_names = sm_sarimax.param_names
    param_d = {name: getattr(np, floatX)(rng.normal(scale=0.1) ** 2) for name in param_names}

    res = sm_sarimax.fit_constrained(param_d)
    mod = BayesianSARIMA(
        order=(p, d, q), seasonal_order=(P, D, Q, S), verbose=False, stationary_initialization=False
    )

    with pm.Model() as pm_mod:
        x0 = pm.Normal("x0", shape=(mod.k_states,))
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))

        if q > 0:
            pm.Deterministic(
                "ma_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ma.") and "S." not in k])
                ),
            )
        if p > 0:
            pm.Deterministic(
                "ar_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ar.") and "S." not in k])
                ),
            )
        if P > 0:
            pm.Deterministic(
                "seasonal_ar_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ar.S.")])
                ),
            )

        if Q > 0:
            pm.Deterministic(
                "seasonal_ma_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ma.S.")])
                ),
            )

        pm.Deterministic("sigma_state", pt.as_tensor_variable(np.sqrt(param_d["sigma2"])))

        mod._insert_random_variables()
        matrices = pm.draw(mod.subbed_ssm)
        matrix_dict = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))

    for matrix in ["transition", "selection", "state_cov", "obs_cov", "design"]:
        if matrix == "transition" and D > 2:
            pytest.skip("Statsmodels has a bug when D > 2, skip this test.)")
        assert_allclose(matrix_dict[matrix], sm_sarimax.ssm[matrix], err_msg=f"{matrix} not equal")


@pytest.mark.parametrize("filter_output", ["filtered", "predicted", "smoothed"])
def test_all_prior_covariances_are_PSD(filter_output, pymc_mod, rng):
    rv = pymc_mod[f"{filter_output}_covariance"]
    cov_mats = pm.draw(rv, 100, random_seed=rng)
    w, v = np.linalg.eig(cov_mats)
    assert_array_less(0, w, err_msg=f"Smallest eigenvalue: {min(w.ravel())}")


def test_interpretable_raises_if_d_nonzero():
    with pytest.raises(
        ValueError, match="Cannot use interpretable state structure with statespace differencing"
    ):
        BayesianSARIMA(
            order=(2, 1, 1),
            stationary_initialization=True,
            verbose=False,
            state_structure="interpretable",
        )


def test_interpretable_states_are_interpretable(arima_mod_interp, pymc_mod_interp):
    with pymc_mod_interp:
        prior = pm.sample_prior_predictive(draws=10)

    prior_outputs = arima_mod_interp.sample_unconditional_prior(prior)
    ar_lags = prior.prior.coords["ar_lag"].values - 1
    ma_lags = prior.prior.coords["ma_lag"].values - 1

    # Check the first p states are lags of the previous state
    for t, tm1 in zip(ar_lags[1:], ar_lags[:-1]):
        assert_allclose(
            prior_outputs.prior_latent.isel(state=t).values[1:],
            prior_outputs.prior_latent.isel(state=tm1).values[:-1],
            err_msg=f"State {tm1} is not a lagged version of state {t} (AR lags)",
        )

    # Check the next p+q states are lags of the innovations
    n = len(ar_lags)
    for t, tm1 in zip(ma_lags[1:], ma_lags[:-1]):
        assert_allclose(
            prior_outputs.prior_latent.isel(state=n + t).values[1:],
            prior_outputs.prior_latent.isel(state=n + tm1).values[:-1],
            err_msg=f"State {n + tm1} is not a lagged version of state {n + t} (MA lags)",
        )


@pytest.mark.parametrize("p, d, q, P, D, Q, S", test_orders, ids=ids)
@pytest.mark.filterwarnings(
    "ignore:Non-invertible starting MA parameters found.",
    "ignore:Non-stationary starting autoregressive parameters found",
    "ignore:Maximum Likelihood optimization failed to converge.",
    "ignore:divide by zero encountered in matmul:RuntimeWarning",
    "ignore:overflow encountered in matmul:RuntimeWarning",
    "ignore:invalid value encountered in matmul:RuntimeWarning",
)
def test_representations_are_equivalent(p, d, q, P, D, Q, S, data, rng):
    if (d + D) > 0:
        pytest.skip('state_structure = "interpretable" cannot include statespace differences')

    shared_params = make_stationary_params(data, p, d, q, P, D, Q, S)
    test_values = {}

    for representation in SARIMAX_STATE_STRUCTURES:
        rng = np.random.default_rng(sum(map(ord, "representation test")))
        mod = BayesianSARIMA(
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            stationary_initialization=False,
            verbose=False,
            state_structure=representation,
        )
        shared_params.update(
            {
                "x0": np.zeros(mod.k_states, dtype=floatX),
                "initial_state_cov": np.eye(mod.k_states, dtype=floatX) * 100,
            }
        )
        x, y = simulate_from_numpy_model(mod, rng, shared_params)
        test_values[representation] = y

    all_pairs = combinations(SARIMAX_STATE_STRUCTURES, r=2)
    for rep_1, rep_2 in all_pairs:
        assert_allclose(
            test_values[rep_1],
            test_values[rep_2],
            err_msg=f"{rep_1} and {rep_2} are not the same",
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("order, name", [((4, 1, 0, 0), "AR"), ((0, 0, 4, 1), "MA")])
def test_invalid_order_raises(order, name):
    p, P, q, Q = order
    with pytest.raises(ValueError, match=f"The following {name} and seasonal {name} terms overlap"):
        BayesianSARIMA(order=(p, 0, q), seasonal_order=(P, 0, Q, 4))
