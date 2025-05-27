import numpy as np
import pytensor
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from pytensor.graph.basic import explicit_graph_inputs
from scipy import linalg

from pymc_extras.statespace.models.ETS import BayesianETS
from pymc_extras.statespace.utils.constants import LONG_MATRIX_NAMES
from tests.statespace.shared_fixtures import rng
from tests.statespace.test_utilities import load_nile_test_data


@pytest.fixture(scope="session")
def data():
    return load_nile_test_data()


def test_invalid_order_raises():
    # Order must be length 3
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=("A", "N"))

        # Order must be strings
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=(2, 1, 1))

    # Only additive errors allowed
    with pytest.raises(ValueError, match="Only additive errors are supported"):
        BayesianETS(order=("M", "N", "N"))

    # Trend must be A or Ad
    with pytest.raises(ValueError, match="Invalid trend specification"):
        BayesianETS(order=("A", "P", "N"))

    # Seasonal must be A or N
    with pytest.raises(ValueError, match="Invalid seasonal specification"):
        BayesianETS(order=("A", "Ad", "M"))

    # seasonal_periods must be provided if seasonal is requested
    with pytest.raises(ValueError, match="If seasonal is True, seasonal_periods must be provided."):
        BayesianETS(order=("A", "Ad", "A"))


orders = (
    ("A", "N", "N"),
    ("A", "A", "N"),
    ("A", "Ad", "N"),
    ("A", "N", "A"),
    ("A", "A", "A"),
    ("A", "Ad", "A"),
)
order_names = (
    "Basic",
    "Trend",
    "Damped Trend",
    "Seasonal",
    "Trend and Seasonal",
    "Trend, Damped Trend, Seasonal",
)

order_expected_flags = (
    {"trend": False, "damped_trend": False, "seasonal": False},
    {"trend": True, "damped_trend": False, "seasonal": False},
    {"trend": True, "damped_trend": True, "seasonal": False},
    {"trend": False, "damped_trend": False, "seasonal": True},
    {"trend": True, "damped_trend": False, "seasonal": True},
    {"trend": True, "damped_trend": True, "seasonal": True},
)

order_params = (
    ["alpha", "initial_level"],
    ["alpha", "initial_level", "beta", "initial_trend"],
    ["alpha", "initial_level", "beta", "initial_trend", "phi"],
    ["alpha", "initial_level", "gamma", "initial_seasonal"],
    ["alpha", "initial_level", "beta", "initial_trend", "gamma", "initial_seasonal"],
    ["alpha", "initial_level", "beta", "initial_trend", "gamma", "initial_seasonal", "phi"],
)


@pytest.mark.parametrize(
    "order, expected_flags", zip(orders, order_expected_flags), ids=order_names
)
def test_order_flags(order, expected_flags):
    mod = BayesianETS(order=order, seasonal_periods=4)
    for key, value in expected_flags.items():
        assert getattr(mod, key) == value


def test_mode_argument():
    # Mode argument should be passed to the parent class
    mod = BayesianETS(order=("A", "N", "N"), mode="FAST_RUN")
    assert mod.mode == "FAST_RUN"


@pytest.mark.parametrize("order, expected_params", zip(orders, order_params), ids=order_names)
def test_param_info(order: tuple[str, str, str], expected_params):
    mod = BayesianETS(order=order, seasonal_periods=4)

    all_expected_params = [*expected_params, "sigma_state", "P0"]
    assert all(param in mod.param_names for param in all_expected_params)
    assert all(param in all_expected_params for param in mod.param_names)
    assert all(
        mod.param_info[param]["dims"] is None
        for param in expected_params
        if "seasonal" not in param
    )


@pytest.mark.parametrize("order, expected_params", zip(orders, order_params), ids=order_names)
@pytest.mark.parametrize("use_transformed", [True, False], ids=["transformed", "untransformed"])
def test_statespace_matrices(
    rng, order: tuple[str, str, str], expected_params: list[str], use_transformed: bool
):
    seasonal_periods = np.random.randint(3, 12)
    mod = BayesianETS(
        order=order,
        seasonal_periods=seasonal_periods,
        measurement_error=True,
        use_transformed_parameterization=use_transformed,
    )
    expected_states = 2 + int(order[1] != "N") + int(order[2] != "N") * seasonal_periods

    test_values = {
        "alpha": rng.beta(1, 1),
        "beta": rng.beta(1, 1),
        "gamma": rng.beta(1, 1),
        "phi": rng.beta(1, 1),
        "sigma_state": rng.normal() ** 2,
        "sigma_obs": rng.normal() ** 2,
        "initial_level": rng.normal() ** 2,
        "initial_trend": rng.normal() ** 2,
        "initial_seasonal": np.ones(seasonal_periods),
        "initial_state_cov": np.eye(expected_states),
    }

    matrices = x0, P0, c, d, T, Z, R, H, Q = mod._unpack_statespace_with_placeholders()

    assert x0.type.shape == (expected_states,)
    assert P0.type.shape == (expected_states, expected_states)
    assert c.type.shape == (expected_states,)
    assert d.type.shape == (1,)
    assert T.type.shape == (expected_states, expected_states)
    assert Z.type.shape == (1, expected_states)
    assert R.type.shape == (expected_states, 1)
    assert H.type.shape == (1, 1)
    assert Q.type.shape == (1, 1)

    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]
    assert all(name in input_names for name in expected_params)

    f_matrices = pytensor.function(inputs, matrices)
    [x0, P0, c, d, T, Z, R, H, Q] = f_matrices(**{name: test_values[name] for name in input_names})

    assert_allclose(H, np.eye(1) * test_values["sigma_obs"] ** 2)
    assert_allclose(Q, np.eye(1) * test_values["sigma_state"] ** 2)

    R_val = np.zeros((expected_states, 1))
    R_val[0] = 1.0 - test_values["alpha"]
    R_val[1] = test_values["alpha"]

    Z_val = np.zeros((1, expected_states))
    Z_val[0, 0] = 1.0
    Z_val[0, 1] = 1.0

    x0_val = np.zeros((expected_states,))
    x0_val[1] = test_values["initial_level"]

    if order[1] == "N":
        T_val = np.array([[0.0, 0.0], [0.0, 1.0]])
    else:
        x0_val[2] = test_values["initial_trend"]
        R_val[2] = (
            test_values["beta"] if use_transformed else test_values["beta"] * test_values["alpha"]
        )
        T_val = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    if order[1] == "Ad":
        T_val[1:, -1] *= test_values["phi"]

    if order[2] == "A":
        x0_val[2 + int(order[1] != "N") :] = test_values["initial_seasonal"]
        gamma = (
            test_values["gamma"]
            if use_transformed
            else (1 - test_values["alpha"]) * test_values["gamma"]
        )
        R_val[2 + int(order[1] != "N")] = gamma
        R_val[0] = R_val[0] - gamma

        S = np.eye(seasonal_periods, k=-1)
        S[0, -1] = 1.0
        Z_val[0, 2 + int(order[1] != "N")] = 1.0
    else:
        S = np.eye(0)

    T_val = linalg.block_diag(T_val, S)

    assert_allclose(x0, x0_val)
    assert_allclose(T, T_val)
    assert_allclose(R, R_val)
    assert_allclose(Z, Z_val)


@pytest.mark.parametrize("order, params", zip(orders, order_params), ids=order_names)
def test_statespace_matches_statsmodels(rng, order: tuple[str, str, str], params):
    seasonal_periods = rng.integers(3, 12)
    data = rng.normal(size=(100,))
    mod = BayesianETS(
        order=order,
        seasonal_periods=seasonal_periods,
        measurement_error=False,
        use_transformed_parameterization=True,
    )
    sm_mod = sm.tsa.statespace.ExponentialSmoothing(
        data,
        trend=mod.trend,
        damped_trend=mod.damped_trend,
        seasonal=seasonal_periods if mod.seasonal else None,
    )

    simplex_params = ["alpha", "beta", "gamma"]
    test_values = dict(zip(simplex_params, rng.dirichlet(alpha=np.ones(3))))
    test_values["phi"] = rng.beta(1, 1)

    test_values["initial_level"] = rng.normal()
    test_values["initial_trend"] = rng.normal()
    test_values["initial_seasonal"] = rng.normal(size=seasonal_periods)
    test_values["initial_state_cov"] = np.eye(mod.k_states)
    test_values["sigma_state"] = 1.0

    sm_test_values = test_values.copy()
    sm_test_values["smoothing_level"] = test_values["alpha"]
    sm_test_values["smoothing_trend"] = test_values["beta"]
    sm_test_values["smoothing_seasonal"] = test_values["gamma"]
    sm_test_values["damping_trend"] = test_values["phi"]
    sm_test_values["initial_seasonal"] = test_values["initial_seasonal"][0]
    for i in range(1, seasonal_periods):
        sm_test_values[f"initial_seasonal.L{i}"] = test_values["initial_seasonal"][i]

    vals = [
        np.atleast_1d(test_values[name])
        for name in ["initial_level", "initial_trend", "initial_seasonal"]
    ]
    x0 = np.concatenate([[0.0], *vals])

    mask = [True, True, order[1] != "N", *(order[2] != "N",) * seasonal_periods]

    sm_mod.initialize_known(initial_state=x0[mask], initial_state_cov=np.eye(mod.k_states))
    sm_mod.fit_constrained({name: sm_test_values[name] for name in sm_mod.param_names})

    matrices = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]

    f_matrices = pytensor.function(inputs, matrices)
    test_values_subset = {name: test_values[name] for name in input_names}

    matrices = f_matrices(**test_values_subset)
    sm_matrices = [sm_mod.ssm[name] for name in LONG_MATRIX_NAMES[2:]]

    for matrix, sm_matrix, name in zip(matrices[2:], sm_matrices, LONG_MATRIX_NAMES[2:]):
        assert_allclose(matrix, sm_matrix, err_msg=f"{name} does not match")


@pytest.mark.parametrize("order, params", zip(orders, order_params), ids=order_names)
@pytest.mark.parametrize("dense_cov", [True, False], ids=["dense", "diagonal"])
def test_ETS_with_multiple_endog(rng, order, params, dense_cov):
    seasonal_periods = 4
    mod = BayesianETS(
        order=order,
        seasonal_periods=seasonal_periods,
        measurement_error=False,
        use_transformed_parameterization=True,
        dense_innovation_covariance=dense_cov,
        endog_names=["A", "B"],
    )

    single_mod = BayesianETS(
        order=order,
        seasonal_periods=seasonal_periods,
        measurement_error=False,
        use_transformed_parameterization=True,
    )

    simplex_params = ["alpha", "beta", "gamma"]
    test_values = dict(zip(simplex_params, rng.dirichlet(alpha=np.ones(3), size=(mod.k_endog,)).T))
    test_values["phi"] = rng.beta(1, 1, size=(mod.k_endog,))

    test_values["initial_level"] = rng.normal(
        size=mod.k_endog,
    )
    test_values["initial_trend"] = rng.normal(
        size=mod.k_endog,
    )
    test_values["initial_seasonal"] = rng.normal(size=(mod.k_endog, seasonal_periods))
    test_values["initial_state_cov"] = np.eye(mod.k_states)

    if not dense_cov:
        test_values["sigma_state"] = np.ones(
            mod.k_endog,
        )
    else:
        L = np.random.normal(size=(mod.k_endog, mod.k_endog))
        test_values["state_cov"] = L @ L.T

    # Compile functions for the joined model
    matrices_pt = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices_pt))
    input_names = [x.name for x in inputs]

    test_values_subset = {name: test_values[name] for name in input_names}
    f_matrices = pytensor.function(inputs, matrices_pt)

    matrices = f_matrices(**test_values_subset)

    # Compile functions for the single model
    single_matrices_pt = single_mod._unpack_statespace_with_placeholders()
    single_inputs = list(explicit_graph_inputs(single_matrices_pt))
    single_input_names = [x.name for x in single_inputs]

    cursor = 0
    single_test_values_subsets = []
    for i in range(mod.k_endog):
        single_slice = slice(cursor, cursor + single_mod.k_states)
        d = {
            name: (
                test_values[name][i]
                if name != "initial_state_cov"
                else test_values_subset[name][single_slice, single_slice]
            )
            for name in single_input_names
            if name != "sigma_state"
        }
        if dense_cov:
            d["sigma_state"] = np.sqrt(test_values["state_cov"][i, i])
        else:
            d["sigma_state"] = test_values["sigma_state"][i]
        single_test_values_subsets.append(d)
        cursor += single_mod.k_states

    f_single_matrices = pytensor.function(single_inputs, single_matrices_pt)
    single_matrices = [f_single_matrices(**d) for d in single_test_values_subsets]
    names = [x.name for x in matrices_pt]

    for i, (x1, name) in enumerate(zip(matrices, names)):
        cursor = 0
        for j in range(mod.k_endog):
            x2 = single_matrices[j][i]
            state_slice = slice(cursor, cursor + single_mod.k_states)
            obs_slice = slice(j, j + 1)  # Also endog_slice -- it's doing double duty
            if name in ["state_intercept", "initial_state"]:
                assert_allclose(x1[state_slice], x2, err_msg=f"{name} does not match for case {j}")
            elif name in ["P0", "initial_state_cov", "transition"]:
                assert_allclose(
                    x1[state_slice, state_slice], x2, err_msg=f"{name} does not match for case {j}"
                )
            elif name == "selection":
                assert_allclose(
                    x1[state_slice, obs_slice], x2, err_msg=f"{name} does not match for case {j}"
                )
            elif name == "design":
                assert_allclose(
                    x1[obs_slice, state_slice], x2, err_msg=f"{name} does not match for case {j}"
                )
            elif name == "obs_intercept":
                assert_allclose(x1[obs_slice], x2, err_msg=f"{name} does not match for case {j}")
            elif name in ["obs_cov", "state_cov"]:
                assert_allclose(
                    x1[obs_slice, obs_slice], x2, err_msg=f"{name} does not match for case {j}"
                )
            else:
                raise ValueError(f"You forgot {name} !")

            cursor += single_mod.k_states


def test_ETS_stationary_initialization():
    mod = BayesianETS(
        order=("A", "Ad", "A"),
        seasonal_periods=4,
        stationary_initialization=True,
        initialization_dampening=0.66,
    )

    matrices = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]

    # Make sure the stationary_dampening dummy variables was completely rewritten away
    assert "stationary_dampening" not in input_names

    # P0 should have been removed from param names
    assert "P0" not in mod.param_names
    assert "P0" not in mod.param_info.keys()

    f = pytensor.function(inputs, matrices, mode="FAST_COMPILE")
    test_values = f(**{x.name: np.full(x.type.shape, 0.5) for x in inputs})
    outputs = {name: val for name, val in zip(LONG_MATRIX_NAMES, test_values)}

    # Make sure that the transition matrix has ones in the expected positions, not the model dampening factor
    assert outputs["transition"][1, 1] == 1.0
    assert outputs["transition"][2, 2] == 0.5  # phi = 0.5 -- trend is dampened anyway
    assert outputs["transition"][3, -1] == 1.0

    # P0 should be equal to the solution to the Lyapunov equation using the dampening factors in the transition matrix
    T_stationary = outputs["transition"].copy()
    T_stationary[1, 1] = mod.initialization_dampening
    T_stationary[3, -1] = mod.initialization_dampening

    R, Q = outputs["selection"], outputs["state_cov"]
    P0_expected = linalg.solve_discrete_lyapunov(T_stationary, R @ Q @ R.T)

    assert_allclose(outputs["initial_state_cov"], P0_expected, rtol=1e-8, atol=1e-8)
