import numpy as np
import pytensor

from numpy.testing import assert_allclose
from pytensor import config
from pytensor.graph.basic import explicit_graph_inputs
from scipy import linalg

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.structural.utils import _frequency_transition_block
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import assert_pattern_repeats, simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


cycle_test_vals = zip([None, None, 3, 5, 10], [False, True, True, False, False])


def test_cycle_component_deterministic(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False
    )
    params = {"params_cycle": np.array([1.0, 1.0], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    assert_pattern_repeats(y, 12, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_dampening(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False, dampen=True
    )
    params = {
        "params_cycle": np.array([10.0, 10.0], dtype=config.floatX),
        "dampening_factor_cycle": 0.75,
    }
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=100)

    # check that cycle dampens to zero over time
    assert_allclose(y[-1], 0.0, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_innovations_and_cycle_length(rng):
    cycle = st.CycleComponent(
        name="cycle", estimate_cycle_length=True, innovations=True, dampen=True
    )
    params = {
        "params_cycle": np.array([1.0, 1.0], dtype=config.floatX),
        "length_cycle": 12.0,
        "dampening_factor_cycle": 0.95,
        "sigma_cycle": 1.0,
    }
    x, y = simulate_from_numpy_model(cycle, rng, params)

    cycle.build(verbose=False)
    _assert_basic_coords_correct(cycle)


def test_cycle_multivariate_deterministic(rng):
    """Test multivariate cycle component with deterministic cycles."""
    cycle = st.CycleComponent(
        name="cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {"params_cycle": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    # Check that each variable has a cyclical pattern with the expected period
    for i in range(3):
        assert_pattern_repeats(y[:, i], 12, atol=ATOL, rtol=RTOL)

    # Check that the cycles have different amplitudes (different initial states)
    assert np.std(y[:, 0]) > 0
    assert np.std(y[:, 1]) > 0
    assert np.std(y[:, 2]) > 0
    # The second and third variables should have larger amplitudes due to larger initial states
    assert np.std(y[:, 1]) > np.std(y[:, 0])
    assert np.std(y[:, 2]) > np.std(y[:, 0])

    # check design, transition, selection matrices
    Z, T, R = pytensor.function(
        [],
        [cycle.ssm["design"], cycle.ssm["transition"], cycle.ssm["selection"]],
        mode="FAST_COMPILE",
    )()

    # each block is [1, 0] for design
    expected_Z = np.zeros((3, 6))
    expected_Z[0, 0] = 1.0
    expected_Z[1, 2] = 1.0
    expected_Z[2, 4] = 1.0
    np.testing.assert_allclose(Z, expected_Z)

    # each block is 2x2 frequency transition matrix for given cycle length (12 here)
    block = _frequency_transition_block(12, 1).eval()
    expected_T = np.zeros((6, 6))
    for i in range(3):
        expected_T[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    np.testing.assert_allclose(T, expected_T)

    # each block is 2x2 identity for selection
    expected_R = np.zeros((6, 6))
    for i in range(3):
        expected_R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.eye(2)
    np.testing.assert_allclose(R, expected_R)


def test_multivariate_cycle_with_shared(rng):
    cycle = st.CycleComponent(
        name="cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        observed_state_names=["data_1", "data_2", "data_3"],
        share_states=True,
    )

    assert cycle.state_names == ["Cos_cycle[shared]", "Sin_cycle[shared]"]
    assert cycle.shock_names == []
    assert cycle.param_names == ["params_cycle"]

    params = {"params_cycle": np.array([1.0, 2.0], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    np.testing.assert_allclose(y[:, 0], y[:, 1], atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(y[:, 0], y[:, 2], atol=ATOL, rtol=RTOL)


def test_cycle_multivariate_with_dampening(rng):
    """Test multivariate cycle component with dampening."""
    cycle = st.CycleComponent(
        name="cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        dampen=True,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {
        "params_cycle": np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=config.floatX),
        "dampening_factor_cycle": 0.75,
    }
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=100)

    # Check that all cycles dampen to zero over time
    for i in range(3):
        assert_allclose(y[-1, i], 0.0, atol=ATOL, rtol=RTOL)

    # Check that the dampening pattern is consistent across variables
    # The variables should dampen at the same rate but with different initial amplitudes
    for i in range(1, 3):
        # The ratio of final to initial values should be similar across variables
        ratio_0 = abs(y[-1, 0] / y[0, 0]) if y[0, 0] != 0 else 0
        ratio_i = abs(y[-1, i] / y[0, i]) if y[0, i] != 0 else 0
        assert_allclose(ratio_0, ratio_i, atol=1e-2, rtol=1e-2)


def test_cycle_multivariate_with_innovations_and_cycle_length(rng):
    """Test multivariate cycle component with innovations and estimated cycle length."""
    cycle = st.CycleComponent(
        name="cycle",
        estimate_cycle_length=True,
        innovations=True,
        dampen=True,
        observed_state_names=["data_1", "data_2", "data_3"],
    )
    params = {
        "params_cycle": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=config.floatX),
        "length_cycle": 12.0,
        "dampening_factor_cycle": 0.95,
        "sigma_cycle": np.array([0.5, 1.0, 1.5]),  # different innov variances per var
    }
    x, y = simulate_from_numpy_model(cycle, rng, params)

    cycle.build(verbose=False)
    _assert_basic_coords_correct(cycle)

    assert cycle.coords["state_cycle"] == ["Cos_cycle", "Sin_cycle"]
    assert cycle.coords["endog_cycle"] == ["data_1", "data_2", "data_3"]

    assert cycle.k_endog == 3
    assert cycle.k_states == 6  # 2 states per variable
    assert cycle.k_posdef == 6  # 2 innovations per variable

    # Check that the data has the expected shape
    assert y.shape[1] == 3  # 3 variables

    # Check that each variable shows some variation (due to innovations)
    for i in range(3):
        assert np.std(y[:, i]) > 0

    # check design, transition, selection & state_cov matrices
    Z, T, R, Q = pytensor.function(
        [
            cycle._name_to_variable["length_cycle"],
            cycle._name_to_variable["dampening_factor_cycle"],
            cycle._name_to_variable["sigma_cycle"],
        ],
        [
            cycle.ssm["design"],
            cycle.ssm["transition"],
            cycle.ssm["selection"],
            cycle.ssm["state_cov"],
        ],
        mode="FAST_COMPILE",
    )(params["length_cycle"], params["dampening_factor_cycle"], params["sigma_cycle"])

    # each block is [1, 0] for design
    expected_Z = np.zeros((3, 6))
    expected_Z[0, 0] = 1.0
    expected_Z[1, 2] = 1.0
    expected_Z[2, 4] = 1.0
    np.testing.assert_allclose(Z, expected_Z)

    # each block is 2x2 frequency transition matrix for given cycle length (12 here),
    # scaled by dampening factor
    block = _frequency_transition_block(12, 1).eval() * params["dampening_factor_cycle"]
    expected_T = np.zeros((6, 6))
    for i in range(3):
        expected_T[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    np.testing.assert_allclose(T, expected_T)

    # each block is 2x2 identity for selection
    expected_R = np.zeros((6, 6))
    for i in range(3):
        expected_R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.eye(2)
    np.testing.assert_allclose(R, expected_R)

    # each block is sigma^2 * I_2 for state_cov
    sigmas = params["sigma_cycle"]
    expected_Q = np.zeros((6, 6))
    for i in range(3):
        expected_Q[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.eye(2) * sigmas[i] ** 2
    np.testing.assert_allclose(Q, expected_Q)


def test_add_multivariate_cycle_components_with_different_observed():
    """
    Test adding two multivariate CycleComponents with different observed_state_names.
    Ensures that combining two multivariate CycleComponents with different observed state names
    results in the correct block-diagonal state space matrices and state naming.
    """
    cycle1 = st.CycleComponent(
        name="cycle1",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=False,
        observed_state_names=["a1", "a2"],
    )
    cycle2 = st.CycleComponent(
        name="cycle2",
        cycle_length=6,
        estimate_cycle_length=False,
        innovations=False,
        observed_state_names=["b1", "b2"],
    )
    mod = (cycle1 + cycle2).build(verbose=False)

    # check dimensions
    assert mod.k_endog == 4
    assert mod.k_states == 8
    assert mod.k_posdef == 2 * mod.k_endog  # 2 innovations per variable

    # check state names and coords
    expected_state_names = [
        "Cos_cycle1[a1]",
        "Sin_cycle1[a1]",
        "Cos_cycle1[a2]",
        "Sin_cycle1[a2]",
        "Cos_cycle2[b1]",
        "Sin_cycle2[b1]",
        "Cos_cycle2[b2]",
        "Sin_cycle2[b2]",
    ]
    assert mod.state_names == expected_state_names

    assert mod.coords["state_cycle1"] == ["Cos_cycle1", "Sin_cycle1"]
    assert mod.coords["state_cycle2"] == ["Cos_cycle2", "Sin_cycle2"]
    assert mod.coords["endog_cycle1"] == ["a1", "a2"]
    assert mod.coords["endog_cycle2"] == ["b1", "b2"]

    # evaluate design, transition, selection matrices
    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    # design: each row selects first state of its block
    expected_Z = np.zeros((4, 8))
    expected_Z[0, 0] = 1.0  # "a1" -> Cos_cycle1[a1]
    expected_Z[1, 2] = 1.0  # "a2" -> Cos_cycle1[a2]
    expected_Z[2, 4] = 1.0  # "b1" -> Cos_cycle2[b1]
    expected_Z[3, 6] = 1.0  # "b2" -> Cos_cycle2[b2]
    assert_allclose(Z, expected_Z)

    # transition: block diagonal, each block is 2x2 frequency transition matrix
    block1 = _frequency_transition_block(12, 1).eval()
    block2 = _frequency_transition_block(6, 1).eval()
    expected_T = np.zeros((8, 8))
    for i in range(2):
        expected_T[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block1
    for i in range(2):
        expected_T[4 + 2 * i : 4 + 2 * i + 2, 4 + 2 * i : 4 + 2 * i + 2] = block2
    assert_allclose(T, expected_T)

    # selection: block diagonal, each block is 2x2 identity
    expected_R = np.zeros((8, 8))
    for i in range(4):
        expected_R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.eye(2)
    assert_allclose(R, expected_R)


def test_add_multivariate_shared_and_not_shared():
    cycle_shared = st.CycleComponent(
        name="shared_cycle",
        cycle_length=12,
        estimate_cycle_length=False,
        innovations=True,
        observed_state_names=["gdp", "inflation", "unemployment"],
        share_states=True,
    )
    cycle_individual = st.CycleComponent(
        name="individual_cycle",
        estimate_cycle_length=True,
        innovations=False,
        observed_state_names=["gdp", "inflation", "unemployment"],
        dampen=True,
    )
    mod = (cycle_shared + cycle_individual).build(verbose=False)

    assert mod.k_endog == 3
    assert mod.k_states == 2 + 3 * 2
    assert mod.k_posdef == 2 + 3 * 2

    expected_states = [
        "Cos_shared_cycle[shared]",
        "Sin_shared_cycle[shared]",
        "Cos_individual_cycle[gdp]",
        "Sin_individual_cycle[gdp]",
        "Cos_individual_cycle[inflation]",
        "Sin_individual_cycle[inflation]",
        "Cos_individual_cycle[unemployment]",
        "Sin_individual_cycle[unemployment]",
    ]

    assert mod.state_names == expected_states
    assert mod.shock_names == expected_states[:2]

    assert mod.param_names == [
        "params_shared_cycle",
        "sigma_shared_cycle",
        "params_individual_cycle",
        "length_individual_cycle",
        "dampening_factor_individual_cycle",
        "P0",
    ]

    assert "endog_shared_cycle" not in mod.coords
    assert mod.coords["state_shared_cycle"] == ["Cos_shared_cycle", "Sin_shared_cycle"]
    assert mod.coords["state_individual_cycle"] == ["Cos_individual_cycle", "Sin_individual_cycle"]
    assert mod.coords["endog_individual_cycle"] == ["gdp", "inflation", "unemployment"]

    assert mod.param_info["params_shared_cycle"]["dims"] == ("state_shared_cycle",)
    assert mod.param_info["params_shared_cycle"]["shape"] == (2,)

    assert mod.param_info["sigma_shared_cycle"]["dims"] is None
    assert mod.param_info["sigma_shared_cycle"]["shape"] == ()

    assert mod.param_info["params_individual_cycle"]["dims"] == (
        "endog_individual_cycle",
        "state_individual_cycle",
    )
    assert mod.param_info["params_individual_cycle"]["shape"] == (3, 2)

    params = {
        "length_individual_cycle": 12.0,
        "dampening_factor_individual_cycle": 0.95,
    }
    outputs = [mod.ssm["transition"], mod.ssm["design"], mod.ssm["selection"]]
    T, Z, R = pytensor.function(
        list(explicit_graph_inputs(outputs)),
        outputs,
        mode="FAST_COMPILE",
    )(**params)

    lamb = 2 * np.pi / 12  # dampening factor for individual cycle
    transition_block = np.array(
        [[np.cos(lamb), np.sin(lamb)], [-np.sin(lamb), np.cos(lamb)]], dtype=config.floatX
    )
    T_expected = linalg.block_diag(transition_block, *[0.95 * transition_block] * 3)
    np.testing.assert_allclose(T, T_expected)

    np.testing.assert_allclose(
        Z, np.array([[1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0]])
    )

    np.testing.assert_allclose(R, np.eye(8, dtype=config.floatX))
