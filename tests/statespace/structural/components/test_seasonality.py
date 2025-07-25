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
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("innovations", [True, False])
@pytest.mark.parametrize("remove_first_state", [True, False])
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in matmul:RuntimeWarning",
    "ignore:overflow encountered in matmul:RuntimeWarning",
    "ignore:invalid value encountered in matmul:RuntimeWarning",
)
def test_time_seasonality(s, d, innovations, remove_first_state, rng):
    def random_word(rng):
        return "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz")) for _ in range(5))

    state_names = [random_word(rng) for _ in range(s * d)]
    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=innovations,
        name="season",
        state_names=state_names,
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros(mod.k_states // mod.duration, dtype=config.floatX)
    x0[0] = 1

    params = {"coefs_season": x0}
    if innovations:
        params["sigma_season"] = 0.0

    x, y = simulate_from_numpy_model(mod, rng, params, steps=100 * mod.duration)
    y = y.ravel()
    if not innovations:
        assert_pattern_repeats(y, s * d, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    test_slice = slice(d, None) if remove_first_state else slice(None)
    assert mod.coords["state_season"] == state_names[test_slice]


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize(
    "remove_first_state", [True, False], ids=["remove_first_state", "keep_first_state"]
)
def test_time_seasonality_multiple_observed(rng, d, remove_first_state):
    s = 3
    state_names = [f"state_{i}_{j}" for i in range(s) for j in range(d)]
    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=True,
        name="season",
        state_names=state_names,
        observed_state_names=["data_1", "data_2"],
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros((mod.k_endog, mod.k_states // mod.k_endog // mod.duration), dtype=config.floatX)

    expected_states = [
        f"state_{i}_{j}[data_{k}]"
        for k in range(1, 3)
        for i in range(int(remove_first_state), s)
        for j in range(d)
    ]
    assert mod.state_names == expected_states
    assert mod.shock_names == ["season[data_1]", "season[data_2]"]

    x0[0, 0] = 1
    x0[1, 0] = 2.0

    params = {"coefs_season": x0, "sigma_season": np.array([0.0, 0.0], dtype=config.floatX)}

    x, y = simulate_from_numpy_model(mod, rng, params, steps=123 * d)
    assert_pattern_repeats(y[:, 0], s * d, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], s * d, atol=ATOL, rtol=RTOL)

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

    # Because the dimension of the observed states is 2,
    # the expected T is the diagonal block matrix [[T0, 0], [0, T0]]
    # where T0 is the transition matrix we would have if the
    # seasonality were not multiple observed.
    mod0 = st.TimeSeasonality(season_length=s, duration=d, remove_first_state=remove_first_state)
    T0 = mod0.ssm["transition"].eval()

    if remove_first_state:
        expected_x0 = np.repeat(np.array([1.0, 0.0, 2.0, 0.0]), d)
        expected_T = np.block(
            [[T0, np.zeros((d * (s - 1), d * (s - 1)))], [np.zeros((d * (s - 1), d * (s - 1))), T0]]
        )
        expected_R = np.array(
            [[1.0, 1.0]] + [[0.0, 0.0]] * (2 * d - 1) + [[1.0, 1.0]] + [[0.0, 0.0]] * (2 * d - 1)
        )
        Z0 = np.zeros((2, d * (s - 1)))
        Z0[0, 0] = 1
        Z1 = np.zeros((2, d * (s - 1)))
        Z1[1, 0] = 1
        expected_Z = np.block([[Z0, Z1]])

    else:
        expected_x0 = np.repeat(np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]), d)
        expected_T = np.block([[T0, np.zeros((s * d, s * d))], [np.zeros((s * d, s * d)), T0]])
        expected_R = np.array(
            [[1.0, 1.0]] + [[0.0, 0.0]] * (s * d - 1) + [[1.0, 1.0]] + [[0.0, 0.0]] * (s * d - 1)
        )
        Z0 = np.zeros((2, s * d))
        Z0[0, 0] = 1
        Z1 = np.zeros((2, s * d))
        Z1[1, 0] = 1
        expected_Z = np.block([[Z0, Z1]])

    expected_Q = np.array([[0.1**2, 0.0], [0.0, 0.8**2]])

    for matrix, expected in zip(
        [x0, T, Z, R, Q],
        [expected_x0, expected_T, expected_Z, expected_R, expected_Q],
    ):
        np.testing.assert_allclose(matrix, expected)


@pytest.mark.parametrize("d1, d2", [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_add_two_time_seasonality_different_observed(rng, d1, d2):
    mod1 = st.TimeSeasonality(
        season_length=3,
        duration=d1,
        innovations=True,
        name="season1",
        state_names=[f"state_{i}_{j}" for i in range(3) for j in range(d1)],
        observed_state_names=["data_1"],
        remove_first_state=False,
    )
    mod2 = st.TimeSeasonality(
        season_length=5,
        duration=d2,
        innovations=True,
        name="season2",
        state_names=[f"state_{i}_{j}" for i in range(5) for j in range(d2)],
        observed_state_names=["data_2"],
    )

    mod = (mod1 + mod2).build(verbose=False)

    params = {
        "coefs_season1": np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        "coefs_season2": np.array([3.0, 0.0, 0.0, 0.0], dtype=config.floatX),
        "sigma_season1": np.array(0.0, dtype=config.floatX),
        "sigma_season2": np.array(0.0, dtype=config.floatX),
        "initial_state_cov": np.eye(mod.k_states, dtype=config.floatX),
    }

    x, y = simulate_from_numpy_model(mod, rng, params, steps=3 * 5 * 5 * d1 * d2)
    assert_pattern_repeats(y[:, 0], 3 * d1, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 5 * d2, atol=ATOL, rtol=RTOL)

    assert mod.state_names == [
        item
        for sublist in [
            [f"state_0_{j}[data_1]" for j in range(d1)],
            [f"state_1_{j}[data_1]" for j in range(d1)],
            [f"state_2_{j}[data_1]" for j in range(d1)],
            [f"state_1_{j}[data_2]" for j in range(d2)],
            [f"state_2_{j}[data_2]" for j in range(d2)],
            [f"state_3_{j}[data_2]" for j in range(d2)],
            [f"state_4_{j}[data_2]" for j in range(d2)],
        ]
        for item in sublist
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
        coefs_season1=np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        coefs_season2=np.array([3.0, 0.0, 0.0, 1.2], dtype=config.floatX),
    )

    np.testing.assert_allclose(
        np.repeat(np.array([1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.2]), [d1, d1, d1, d2, d2, d2, d2]),
        x0,
        atol=ATOL,
        rtol=RTOL,
    )

    # The transition matrix T of mod is expected to be [[T1, 0], [0, T2]],
    # where T1 and T2 are the transition matrices of mod1 and mod2, respectively.
    T1 = mod1.ssm["transition"].eval()
    T2 = mod2.ssm["transition"].eval()
    np.testing.assert_allclose(
        np.block(
            [[T1, np.zeros((T1.shape[0], T2.shape[1]))], [np.zeros((T2.shape[0], T1.shape[1])), T2]]
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
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    if n is None:
        n = int(s // 2)
    states = [f"{f}_season_{i}" for i in range(n) for f in ["Cos", "Sin"]]

    # Remove the last state when the model is completely saturated
    if s / n == 2.0:
        states.pop()
    assert mod.coords["state_season"] == states


def test_frequency_seasonality_multiple_observed(rng):
    observed_state_names = ["data_1", "data_2"]
    season_length = 4
    mod = st.FrequencySeasonality(
        season_length=season_length,
        n=None,
        name="season",
        innovations=True,
        observed_state_names=observed_state_names,
    )
    expected_state_names = [
        "Cos_season_0[data_1]",
        "Sin_season_0[data_1]",
        "Cos_season_1[data_1]",
        "Sin_season_1[data_1]",
        "Cos_season_0[data_2]",
        "Sin_season_0[data_2]",
        "Cos_season_1[data_2]",
        "Sin_season_1[data_2]",
    ]
    assert mod.state_names == expected_state_names
    assert mod.shock_names == [
        "Cos_season_0[data_1]",
        "Sin_season_0[data_1]",
        "Cos_season_1[data_1]",
        "Sin_season_1[data_1]",
        "Cos_season_0[data_2]",
        "Sin_season_0[data_2]",
        "Cos_season_1[data_2]",
        "Sin_season_1[data_2]",
    ]

    # Simulate
    x0 = np.zeros((2, 3), dtype=config.floatX)
    x0[0, 0] = 1.0
    x0[1, 0] = 2.0
    params = {"season": x0, "sigma_season": np.zeros(2, dtype=config.floatX)}
    x, y = simulate_from_numpy_model(mod, rng, params, steps=12)

    # Check periodicity for each observed series
    assert_pattern_repeats(y[:, 0], 4, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 4, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    assert list(mod.coords["state_season"]) == [
        "Cos_season_0[data_1]",
        "Sin_season_0[data_1]",
        "Cos_season_1[data_1]",
        "Cos_season_0[data_2]",
        "Sin_season_0[data_2]",
        "Cos_season_1[data_2]",
    ]

    x0_sym, *_, T_sym, Z_sym, R_sym, _, Q_sym = mod._unpack_statespace_with_placeholders()
    input_vars = explicit_graph_inputs([x0_sym, T_sym, Z_sym, R_sym, Q_sym])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0_sym, T_sym, Z_sym, R_sym, Q_sym],
        mode="FAST_COMPILE",
    )
    params["sigma_season"] = np.array([0.1, 0.8], dtype=config.floatX)
    x0_v, T_v, Z_v, R_v, Q_v = fn(**params)

    # x0 should be raveled into a single vector, with data_1 states first, then data_2 states
    np.testing.assert_allclose(
        x0_v, np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]), atol=ATOL, rtol=RTOL
    )

    # T_v shape: (8, 8) (k_endog * k_states)
    # The transition matrix is block diagonal, each block is:
    # For n=2, season_length=4:
    # lambda_1 = 2*pi*1/4 = pi/2, cos(pi/2)=0, sin(pi/2)=1
    # lambda_2 = 2*pi*2/4 = pi,   cos(pi)=-1, sin(pi)=0
    # Block 1 (Cos_0, Sin_0):
    # [[cos(pi/2), sin(pi/2)],
    #  [-sin(pi/2), cos(pi/2)]] = [[0, 1], [-1, 0]]
    # Block 2 (Cos_1, Sin_1):
    # [[-1, 0], [0, -1]]
    expected_T_block1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    expected_T_block2 = np.array([[-1.0, 0.0], [0.0, -1.0]])
    expected_T = np.zeros((8, 8))
    # data_1
    expected_T[0:2, 0:2] = expected_T_block1
    expected_T[2:4, 2:4] = expected_T_block2
    # data_2
    expected_T[4:6, 4:6] = expected_T_block1
    expected_T[6:8, 6:8] = expected_T_block2
    np.testing.assert_allclose(T_v, expected_T, atol=ATOL, rtol=RTOL)

    # Only the first two states (one sin and one cos component) of each observed series are observed
    expected_Z = np.zeros((2, 8))
    expected_Z[0, 0] = 1.0
    expected_Z[0, 2] = 1.0
    expected_Z[1, 4] = 1.0
    expected_Z[1, 6] = 1.0
    np.testing.assert_allclose(Z_v, expected_Z, atol=ATOL, rtol=RTOL)

    np.testing.assert_allclose(R_v, np.eye(8), atol=ATOL, rtol=RTOL)

    Q_diag = np.diag(Q_v)
    expected_Q_diag = np.r_[np.full(4, 0.1**2), np.full(4, 0.8**2)]
    np.testing.assert_allclose(Q_diag, expected_Q_diag, atol=ATOL, rtol=RTOL)


def test_add_two_frequency_seasonality_different_observed(rng):
    mod1 = st.FrequencySeasonality(
        season_length=4,
        n=2,  # saturated
        name="freq1",
        innovations=True,
        observed_state_names=["data_1"],
    )
    mod2 = st.FrequencySeasonality(
        season_length=6,
        n=1,  # unsaturated
        name="freq2",
        innovations=True,
        observed_state_names=["data_2"],
    )

    mod = (mod1 + mod2).build(verbose=False)

    params = {
        "freq1": np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        "freq2": np.array([3.0, 0.0], dtype=config.floatX),
        "sigma_freq1": np.array(0.0, dtype=config.floatX),
        "sigma_freq2": np.array(0.0, dtype=config.floatX),
        "initial_state_cov": np.eye(mod.k_states, dtype=config.floatX),
    }

    x, y = simulate_from_numpy_model(mod, rng, params, steps=4 * 6 * 3)

    assert_pattern_repeats(y[:, 0], 4, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 6, atol=ATOL, rtol=RTOL)

    assert mod.state_names == [
        "Cos_freq1_0[data_1]",
        "Sin_freq1_0[data_1]",
        "Cos_freq1_1[data_1]",
        "Sin_freq1_1[data_1]",
        "Cos_freq2_0[data_2]",
        "Sin_freq2_0[data_2]",
    ]

    assert mod.shock_names == [
        "Cos_freq1_0[data_1]",
        "Sin_freq1_0[data_1]",
        "Cos_freq1_1[data_1]",
        "Sin_freq1_1[data_1]",
        "Cos_freq2_0[data_2]",
        "Sin_freq2_0[data_2]",
    ]

    x0, *_, T = mod._unpack_statespace_with_placeholders()[:5]
    input_vars = explicit_graph_inputs([x0, T])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0, T],
        mode="FAST_COMPILE",
    )

    x0_v, T_v = fn(
        freq1=np.array([1.0, 0.0, 1.2], dtype=config.floatX),
        freq2=np.array([3.0, 0.0], dtype=config.floatX),
    )

    # Make sure the extra 0 in from the first component (the saturated state) is there!
    np.testing.assert_allclose(np.array([1.0, 0.0, 1.2, 0.0, 3.0, 0.0]), x0_v, atol=ATOL, rtol=RTOL)

    # Transition matrix is block diagonal: 4x4 for freq1, 2x2 for freq2
    # freq1: n=4, lambdas = 2*pi*1/6, 2*pi*2/6
    lam1 = 2 * np.pi * 1 / 4
    lam2 = 2 * np.pi * 2 / 4
    freq1_T1 = np.array([[np.cos(lam1), np.sin(lam1)], [-np.sin(lam1), np.cos(lam1)]])
    freq1_T2 = np.array([[np.cos(lam2), np.sin(lam2)], [-np.sin(lam2), np.cos(lam2)]])
    freq1_T = np.zeros((4, 4))

    # freq2: n=4, lambdas = 2*pi*1/6
    lam3 = 2 * np.pi * 1 / 6
    freq2_T = np.array([[np.cos(lam3), np.sin(lam3)], [-np.sin(lam3), np.cos(lam3)]])

    freq1_T[0:2, 0:2] = freq1_T1
    freq1_T[2:4, 2:4] = freq1_T2

    expected_T = np.zeros((6, 6))
    expected_T[0:4, 0:4] = freq1_T
    expected_T[4:6, 4:6] = freq2_T

    np.testing.assert_allclose(expected_T, T_v, atol=ATOL, rtol=RTOL)
