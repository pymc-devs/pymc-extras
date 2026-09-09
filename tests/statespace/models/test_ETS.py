import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose, assert_array_less
from pymc.logprob.utils import ParameterValueError
from pymc.testing import mock_sample_setup_and_teardown
from pytensor.graph.traversal import explicit_graph_inputs
from scipy import linalg

from pymc_extras.statespace.filters import StandardFilter
from pymc_extras.statespace.filters.distributions import (
    InnovationsStateSpace,
    _innovations_moments,
)
from pymc_extras.statespace.models.ETS import BayesianETS
from pymc_extras.statespace.utils.constants import LONG_MATRIX_NAMES
from tests.statespace.shared_fixtures import rng
from tests.statespace.test_utilities import (
    load_nile_test_data,
    unpack_symbolic_matrices_with_params,
)

floatX = pytensor.config.floatX

mock_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)


@pytest.fixture(scope="session")
def data():
    return load_nile_test_data()


def test_invalid_order_raises():
    # Order must be length 3
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=("A", "N"), endog_names=["y"])

        # Order must be strings
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=(2, 1, 1), endog_names=["y"])

    # Only additive errors allowed
    with pytest.raises(ValueError, match="Only additive errors are supported"):
        BayesianETS(order=("M", "N", "N"), endog_names=["y"])

    # Trend must be A or Ad
    with pytest.raises(ValueError, match="Invalid trend specification"):
        BayesianETS(order=("A", "P", "N"), endog_names=["y"])

    # Seasonal must be A or N
    with pytest.raises(ValueError, match="Invalid seasonal specification"):
        BayesianETS(order=("A", "Ad", "M"), endog_names=["y"])

    # seasonal_periods must be provided if seasonal is requested
    with pytest.raises(
        ValueError, match=r"If seasonal is True, seasonal_periods must be provided."
    ):
        BayesianETS(order=("A", "Ad", "A"), endog_names=["y"])


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
    "order, expected_flags", list(zip(orders, order_expected_flags)), ids=order_names
)
def test_order_flags(order, expected_flags):
    mod = BayesianETS(order=order, endog_names=["y"], seasonal_periods=4)
    for key, value in expected_flags.items():
        assert getattr(mod, key) == value


def test_mode_argument():
    # Mode argument should be passed to the parent class
    mod = BayesianETS(order=("A", "N", "N"), endog_names=["y"], mode="FAST_RUN")
    assert mod.mode == "FAST_RUN"


@pytest.mark.parametrize("order, expected_params", list(zip(orders, order_params)), ids=order_names)
def test_param_info(order: tuple[str, str, str], expected_params):
    mod = BayesianETS(order=order, endog_names=["y"], seasonal_periods=4)

    all_expected_params = [*expected_params, "sigma_state", "P0"]
    assert all(param in mod.param_names for param in all_expected_params)
    assert all(param in all_expected_params for param in mod.param_names)
    assert all(
        mod.param_info[param]["dims"] is None
        for param in expected_params
        if "seasonal" not in param
    )


@pytest.mark.parametrize("order, expected_params", list(zip(orders, order_params)), ids=order_names)
@pytest.mark.parametrize("use_transformed", [True, False], ids=["transformed", "untransformed"])
def test_statespace_matrices(
    rng, order: tuple[str, str, str], expected_params: list[str], use_transformed: bool
):
    seasonal_periods = np.random.randint(3, 12)
    mod = BayesianETS(
        order=order,
        endog_names=["y"],
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
        "P0": np.eye(expected_states),
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


@pytest.mark.parametrize("order, params", list(zip(orders, order_params)), ids=order_names)
def test_statespace_matches_statsmodels(rng, order: tuple[str, str, str], params):
    seasonal_periods = rng.integers(3, 12)
    data = rng.normal(size=(100,))
    mod = BayesianETS(
        order=order,
        endog_names=["y"],
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
    test_values["P0"] = np.eye(mod.k_states)
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


@pytest.mark.parametrize("order, params", list(zip(orders, order_params)), ids=order_names)
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
        endog_names=["y"],
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
    test_values["P0"] = np.eye(mod.k_states)

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
                if name != "P0"
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
        endog_names=["y"],
        seasonal_periods=4,
        stationary_initialization=True,
    )

    matrices = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices))

    # P0 should have been removed from param names
    assert "P0" not in mod.param_names
    assert "P0" not in mod.param_info.keys()

    f = pytensor.function(inputs, matrices, mode="FAST_COMPILE")
    test_values = f(**{x.name: np.full(x.type.shape, 0.5) for x in inputs})
    outputs = {name: val for name, val in zip(LONG_MATRIX_NAMES, test_values)}

    # The transition matrix carries ones where the model is undampened
    assert outputs["transition"][1, 1] == 1.0
    assert outputs["transition"][2, 2] == 0.5  # phi = 0.5 -- trend is dampened anyway
    assert outputs["transition"][3, -1] == 1.0

    R, Q = outputs["selection"], outputs["state_cov"]

    assert_allclose(outputs["initial_state_cov"], R @ Q @ R.T, rtol=1e-8, atol=1e-8)


def test_ETS_stationary_initialization_holds_the_filter_steady(rng):
    """
    A single source of error leaves the state known, so the filter never moves off ``R Q R'``.

    This is what the initialization is for: without it the predicted covariance runs a transient
    before settling, and the density over the first few observations is not the model's.
    """
    mod = BayesianETS(
        order=("A", "N", "N"), endog_names=["y"], stationary_initialization=True, verbose=False
    )
    params = {
        "initial_level": np.array(1.0),
        "alpha": np.array(0.4),
        "sigma_state": np.array(2.0),
    }
    matrices = unpack_symbolic_matrices_with_params(mod, params)
    data = rng.normal(size=(50, 1)).astype(floatX)

    _, _, _, filtered_covariances, predicted_covariances, _, _ = [
        output.eval()
        for output in StandardFilter(cov_jitter=0.0).build_graph(
            pt.specify_shape(pt.as_tensor_variable(data), (50, 1)),
            *[pt.as_tensor_variable(matrix) for matrix in matrices],
        )
    ]

    assert_allclose(predicted_covariances[1:], predicted_covariances[:-1], atol=1e-12)
    assert_allclose(filtered_covariances, 0.0, atol=1e-12)


def test_ets_workflow(mock_sample):
    data = load_nile_test_data()

    ss_mod = BayesianETS(
        order=("A", "Ad", "N"),
        endog_names=["height"],
        stationary_initialization=True,
        measurement_error=True,
    )

    with pm.Model(coords=ss_mod.coords) as m:
        pm.Normal("initial_level", 0, 1)
        pm.Normal("initial_trend", 0, 1)
        pm.Beta("alpha", 1, 1)
        pm.Beta("beta", 1, 1)
        pm.Beta("phi", 1, 1)

        pm.Exponential("sigma_state", 1)
        pm.Exponential("sigma_obs", 1)

        ss_mod.build_statespace_graph(data)

        idata = pm.sample()

    # Not "cholesky": a single source of error makes P0 singular, so it has no Cholesky factor.
    post = ss_mod.sample_conditional_posterior(idata, mvn_method="eigh")
    assert "filtered_posterior" in post
    assert "smoothed_posterior" in post
    assert "predicted_posterior" in post

    forecast = ss_mod.forecast(idata, periods=10, random_seed=42)
    assert "forecast_latent" in forecast
    assert "forecast_observed" in forecast
    assert np.isfinite(forecast.forecast_latent.values).all()
    assert np.isfinite(forecast.forecast_observed.values).all()

    irf = ss_mod.impulse_response_function(idata, n_steps=10, random_seed=42)
    assert "irf" in irf
    assert np.isfinite(irf.irf.values).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": ("A", "N", "N"), "endog_names": ["y"]},
        {"order": ("A", "A", "N"), "endog_names": ["y"]},
        {"order": ("A", "Ad", "A"), "seasonal_periods": 4, "endog_names": ["y"]},
        {"order": ("A", "N", "N"), "endog_names": ["a", "b"]},
    ],
    ids=["ANN", "AAN", "AAdA_seasonal", "ANN_multivariate"],
)
def test_ETS_recursion_matches_the_kalman_filter(kwargs, rng):
    """
    The recursion carries no covariance, so agreeing with the filter is the whole claim.

    A single source of error leaves the state determined by the data, which is what lets the
    density come from the one-step-ahead errors alone.
    """
    defaults = {"alpha": 0.4, "beta": 0.2, "gamma": 0.1, "phi": 0.9, "sigma_state": 2.0}
    mod = BayesianETS(verbose=False, stationary_initialization=True, **kwargs)
    params = {
        name: np.full(info["shape"], defaults.get(name, 0.5))
        for name, info in mod.param_info.items()
    }
    data = rng.normal(size=(60, mod.k_endog)).astype(floatX)

    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)
    recursion = pm.logp(
        InnovationsStateSpace.dist(x0, T, Z, R, Q, data), pt.as_tensor_variable(data)
    ).eval()
    *_, loglike = StandardFilter(cov_jitter=0.0).build_graph(
        pt.specify_shape(pt.as_tensor_variable(data), data.shape),
        *[pt.as_tensor_variable(matrix) for matrix in (x0, P0, c, d, T, Z, R, H, Q)],
    )

    assert_allclose(recursion, loglike.sum().eval(), atol=1e-8)


def test_ETS_recursion_draws_come_from_the_predictive_moments(rng):
    """The sampling path uses the same means the density does, and a constant covariance."""
    n_draws, n_timesteps = 4000, 12
    mod = BayesianETS(
        order=("A", "N", "N"), endog_names=["y"], stationary_initialization=True, verbose=False
    )
    params = {"initial_level": np.array(1.0), "alpha": np.array(0.4), "sigma_state": np.array(2.0)}
    data = rng.normal(size=(n_timesteps, 1)).astype(floatX)

    x0, _, _, _, T, Z, R, _, Q = unpack_symbolic_matrices_with_params(mod, params)
    draws = pm.draw(InnovationsStateSpace.dist(x0, T, Z, R, Q, data), draws=n_draws, random_seed=13)
    means = _innovations_moments(
        *(pt.as_tensor_variable(m) for m in (x0, T, Z, R)), pt.as_tensor_variable(data)
    ).eval()

    assert draws.shape == (n_draws, n_timesteps, 1)
    assert_array_less(np.abs(draws.mean(0) - means), 5 * np.sqrt(Q[0, 0] / n_draws))
    assert_allclose(draws.var(0), Q[0, 0], rtol=0.15)


def test_innovations_state_space_rejects_an_unrecoverable_innovation(rng):
    """
    ``design @ selection`` must be the identity, or the error is not the innovation.

    Every batteries-included model that reaches this distribution satisfies it, so the guard
    exists for the ones that do not: without it the density is finite, smooth, and wrong.
    """
    x0 = np.zeros(2, dtype=floatX)
    transition = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=floatX)
    design = np.array([[1.0, 1.0]], dtype=floatX)
    state_cov = np.array([[2.0]], dtype=floatX)
    data = rng.normal(size=(30, 1)).astype(floatX)

    selection = np.array([[0.6], [0.4]], dtype=floatX)
    assert np.allclose(design @ selection, np.eye(1))
    dist = InnovationsStateSpace.dist(x0, transition, design, selection, state_cov, data)
    assert np.isfinite(pm.logp(dist, pt.as_tensor_variable(data)).eval())

    selection = np.array([[0.6], [0.1]], dtype=floatX)
    dist = InnovationsStateSpace.dist(x0, transition, design, selection, state_cov, data)

    with pytest.raises(ParameterValueError, match="design @ selection"):
        pm.logp(dist, pt.as_tensor_variable(data)).eval()
