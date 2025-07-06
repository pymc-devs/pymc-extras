import numpy as np
import pytensor
import pytest

from pytensor import config
from pytensor.graph.basic import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import assert_pattern_repeats, simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


@pytest.mark.parametrize("s", [10, 25, 50])
@pytest.mark.parametrize("innovations", [True, False])
@pytest.mark.parametrize("remove_first_state", [True, False])
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in matmul:RuntimeWarning",
    "ignore:overflow encountered in matmul:RuntimeWarning",
    "ignore:invalid value encountered in matmul:RuntimeWarning",
)
def test_time_seasonality(s, innovations, remove_first_state, rng):
    def random_word(rng):
        return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(5))

    state_names = [random_word(rng) for _ in range(s)]
    mod = st.TimeSeasonality(
        season_length=s,
        innovations=innovations,
        name="season",
        state_names=state_names,
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros(mod.k_states, dtype=config.floatX)
    x0[0] = 1

    params = {"season_coefs": x0}
    if innovations:
        params["sigma_season"] = 0.0

    x, y = simulate_from_numpy_model(mod, rng, params)
    y = y.ravel()
    if not innovations:
        assert_pattern_repeats(y, s, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    test_slice = slice(1, None) if remove_first_state else slice(None)
    assert mod.coords["season_state"] == state_names[test_slice]


@pytest.mark.parametrize(
    "remove_first_state", [True, False], ids=["remove_first_state", "keep_first_state"]
)
def test_time_seasonality_multiple_observed(rng, remove_first_state):
    s = 3
    state_names = [f"state_{i}" for i in range(s)]
    mod = st.TimeSeasonality(
        season_length=s,
        innovations=True,
        name="season",
        state_names=state_names,
        observed_state_names=["data_1", "data_2"],
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros((mod.k_endog, mod.k_states // mod.k_endog), dtype=config.floatX)

    expected_states = [
        f"state_{i}[data_{j}]" for j in range(1, 3) for i in range(int(remove_first_state), s)
    ]
    assert mod.state_names == expected_states
    assert mod.shock_names == ["season[data_1]", "season[data_2]"]

    x0[0, 0] = 1
    x0[1, 0] = 2.0

    params = {"season_coefs": x0, "sigma_season": np.array([0.0, 0.0], dtype=config.floatX)}

    x, y = simulate_from_numpy_model(mod, rng, params, steps=123)
    assert_pattern_repeats(y[:, 0], s, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], s, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    x0, *_, T, Z, R, _, Q = mod._unpack_statespace_with_placeholders()

    input_vars = explicit_graph_inputs([x0, T, Z, R, Q])

    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0, T, Z, R, Q],
        mode="FAST_COMPILE",
    )

    params["sigma_season"] = np.array([0.1, 0.8], dtype=config.floatX)
    x0, T, Z, R, Q = fn(**params)

    if remove_first_state:
        expected_x0 = np.array([1.0, 0.0, 2.0, 0.0])

        expected_T = np.array(
            [
                [-1.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        expected_R = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        expected_Z = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    else:
        expected_x0 = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        expected_T = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        expected_R = np.array(
            [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
        )
        expected_Z = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

    expected_Q = np.array([[0.1**2, 0.0], [0.0, 0.8**2]])

    for matrix, expected in zip(
        [x0, T, Z, R, Q],
        [expected_x0, expected_T, expected_Z, expected_R, expected_Q],
    ):
        np.testing.assert_allclose(matrix, expected)


def test_add_two_time_seasonality_different_observed(rng):
    mod1 = st.TimeSeasonality(
        season_length=3,
        innovations=True,
        name="season1",
        state_names=[f"state_{i}" for i in range(3)],
        observed_state_names=["data_1"],
        remove_first_state=False,
    )
    mod2 = st.TimeSeasonality(
        season_length=5,
        innovations=True,
        name="season2",
        state_names=[f"state_{i}" for i in range(5)],
        observed_state_names=["data_2"],
    )

    mod = (mod1 + mod2).build(verbose=False)

    params = {
        "season1_coefs": np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        "season2_coefs": np.array([3.0, 0.0, 0.0, 0.0], dtype=config.floatX),
        "sigma_season1": np.array(0.0, dtype=config.floatX),
        "sigma_season2": np.array(0.0, dtype=config.floatX),
        "initial_state_cov": np.eye(mod.k_states, dtype=config.floatX),
    }

    x, y = simulate_from_numpy_model(mod, rng, params, steps=3 * 5 * 5)
    assert_pattern_repeats(y[:, 0], 3, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 5, atol=ATOL, rtol=RTOL)

    assert mod.state_names == [
        "state_0[data_1]",
        "state_1[data_1]",
        "state_2[data_1]",
        "state_1[data_2]",
        "state_2[data_2]",
        "state_3[data_2]",
        "state_4[data_2]",
    ]

    assert mod.shock_names == ["season1[data_1]", "season2[data_2]"]

    x0, *_, T = mod._unpack_statespace_with_placeholders()[:5]
    input_vars = explicit_graph_inputs([x0, T])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0, T],
        mode="FAST_COMPILE",
    )

    x0, T = fn(
        season1_coefs=np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        season2_coefs=np.array([3.0, 0.0, 0.0, 1.2], dtype=config.floatX),
    )

    np.testing.assert_allclose(
        np.array([1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.2]), x0, atol=ATOL, rtol=RTOL
    )

    np.testing.assert_allclose(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
        T,
        atol=ATOL,
        rtol=RTOL,
    )


def get_shift_factor(s):
    s_str = str(s)
    if "." not in s_str:
        return 1
    _, decimal = s_str.split(".")
    return 10 ** len(decimal)


@pytest.mark.parametrize("n", [*np.arange(1, 6, dtype="int").tolist(), None])
@pytest.mark.parametrize("s", [5, 10, 25, 25.2])
def test_frequency_seasonality(n, s, rng):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="season")
    x0 = rng.normal(size=mod.n_coefs).astype(config.floatX)
    params = {"season": x0, "sigma_season": 0.0}
    k = get_shift_factor(s)
    T = int(s * k)

    x, y = simulate_from_numpy_model(mod, rng, params, steps=2 * T)
    assert_pattern_repeats(y, T, atol=ATOL, rtol=RTOL)

    # Check coords
    mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    if n is None:
        n = int(s // 2)
    states = [f"season_{f}_{i}" for i in range(n) for f in ["Cos", "Sin"]]

    # Remove the last state when the model is completely saturated
    if s / n == 2.0:
        states.pop()
    assert mod.coords["season_state"] == states
