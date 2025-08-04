import numpy as np
import pytensor
import pytest

from pytensor import config
from pytensor.graph.basic import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.structural.components.seasonality import FrequencySeasonality
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

    params = {"params_season": x0}
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

    params = {"params_season": x0, "sigma_season": np.array([0.0, 0.0], dtype=config.floatX)}

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


def test_time_seasonality_shared_states():
    mod = st.TimeSeasonality(
        season_length=3,
        duration=1,
        innovations=True,
        name="season",
        state_names=["season_1", "season_2", "season_3"],
        observed_state_names=["data_1", "data_2"],
        remove_first_state=False,
        share_states=True,
    )

    assert mod.k_endog == 2
    assert mod.k_states == 3
    assert mod.k_posdef == 1

    assert mod.coords["state_season"] == ["season_1", "season_2", "season_3"]

    assert mod.state_names == [
        "season_1[season_shared]",
        "season_2[season_shared]",
        "season_3[season_shared]",
    ]
    assert mod.shock_names == ["season[shared]"]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), Z)

    np.testing.assert_allclose(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), T)

    np.testing.assert_allclose(np.array([[1.0], [0.0], [0.0]]), R)


def test_add_mixed_shared_not_shared_time_seasonality():
    shared_season = st.TimeSeasonality(
        season_length=3,
        duration=1,
        innovations=True,
        name="shared",
        state_names=["season_1", "season_2", "season_3"],
        observed_state_names=["data_1", "data_2"],
        remove_first_state=False,
        share_states=True,
    )
    individual_season = st.TimeSeasonality(
        season_length=3,
        duration=1,
        innovations=False,
        name="individual",
        state_names=["season_1", "season_2", "season_3"],
        observed_state_names=["data_1", "data_2"],
        remove_first_state=True,
        share_states=False,
    )
    mod = (shared_season + individual_season).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 7
    assert mod.k_posdef == 1

    assert mod.coords["state_shared"] == ["season_1", "season_2", "season_3"]
    assert mod.coords["state_individual"] == ["season_2", "season_3"]

    assert mod.state_names == [
        "season_1[shared_shared]",
        "season_2[shared_shared]",
        "season_3[shared_shared]",
        "season_2[data_1]",
        "season_3[data_1]",
        "season_2[data_2]",
        "season_3[data_2]",
    ]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(
        np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]), Z
    )

    np.testing.assert_allclose(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
        T,
    )

    np.testing.assert_allclose(np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]), R)


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
        "params_season1": np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        "params_season2": np.array([3.0, 0.0, 0.0, 0.0], dtype=config.floatX),
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
        params_season1=np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        params_season2=np.array([3.0, 0.0, 0.0, 1.2], dtype=config.floatX),
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
    assert mod.param_info["sigma_season"]["shape"] == ()  # scalar for univariate
    assert mod.param_info["sigma_season"]["dims"] is None
    assert len(mod.coords["state_season"]) == mod.n_coefs

    x0 = rng.normal(size=mod.n_coefs).astype(config.floatX)
    params = {"params_season": x0, "sigma_season": 0.0}
    k = get_shift_factor(s)
    T = int(s * k)

    x, y = simulate_from_numpy_model(mod, rng, params, steps=2 * T)
    assert_pattern_repeats(y, T, atol=ATOL, rtol=RTOL)

    # check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    if n is None:
        n = int(s // 2)
    states = [f"{f}_{i}_season" for i in range(n) for f in ["Cos", "Sin"]]

    # remove last state when model is completely saturated
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
    assert mod.param_info["params_season"]["shape"] == (mod.k_endog, mod.n_coefs)
    assert mod.param_info["params_season"]["dims"] == ("endog_season", "state_season")
    assert mod.param_dims["sigma_season"] == ("endog_season",)

    expected_state_names = [
        "Cos_0_season[data_1]",
        "Sin_0_season[data_1]",
        "Cos_1_season[data_1]",
        "Sin_1_season[data_1]",
        "Cos_0_season[data_2]",
        "Sin_0_season[data_2]",
        "Cos_1_season[data_2]",
        "Sin_1_season[data_2]",
    ]
    assert mod.state_names == expected_state_names
    assert mod.shock_names == [
        "Cos_0_season[data_1]",
        "Sin_0_season[data_1]",
        "Cos_1_season[data_1]",
        "Sin_1_season[data_1]",
        "Cos_0_season[data_2]",
        "Sin_0_season[data_2]",
        "Cos_1_season[data_2]",
        "Sin_1_season[data_2]",
    ]

    x0 = np.zeros((2, 3), dtype=config.floatX)
    x0[0, 0] = 1.0
    x0[1, 0] = 2.0
    params = {"params_season": x0, "sigma_season": np.zeros(2, dtype=config.floatX)}
    x, y = simulate_from_numpy_model(mod, rng, params, steps=12)

    # check periodicity for each observed series
    assert_pattern_repeats(y[:, 0], 4, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 4, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    assert list(mod.coords["state_season"]) == [
        "Cos_0_season",
        "Sin_0_season",
        "Cos_1_season",
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


def test_frequency_seasonality_multivariate_shared_states():
    mod = st.FrequencySeasonality(
        season_length=4,
        n=1,
        name="season",
        innovations=True,
        observed_state_names=["data_1", "data_2"],
        share_states=True,
    )

    assert mod.k_endog == 2
    assert mod.k_states == 2
    assert mod.k_posdef == 2

    assert mod.state_names == ["Cos_0_season[shared]", "Sin_0_season[shared]"]
    assert mod.shock_names == ["Cos_0_season[shared]", "Sin_0_season[shared]"]

    assert mod.coords["state_season"] == ["Cos_0_season", "Sin_0_season"]

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(np.array([[1.0, 0.0], [1.0, 0.0]]), Z)

    np.testing.assert_allclose(np.array([[1.0, 0.0], [0.0, 1.0]]), R)

    lam = 2 * np.pi * 1 / 4
    np.testing.assert_allclose(
        np.array([[np.cos(lam), np.sin(lam)], [-np.sin(lam), np.cos(lam)]]), T
    )


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
        "params_freq1": np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        "params_freq2": np.array([3.0, 0.0], dtype=config.floatX),
        "sigma_freq1": np.array(0.0, dtype=config.floatX),
        "sigma_freq2": np.array(0.0, dtype=config.floatX),
        "initial_state_cov": np.eye(mod.k_states, dtype=config.floatX),
    }

    x, y = simulate_from_numpy_model(mod, rng, params, steps=4 * 6 * 3)

    assert_pattern_repeats(y[:, 0], 4, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 6, atol=ATOL, rtol=RTOL)

    assert mod.state_names == [
        "Cos_0_freq1[data_1]",
        "Sin_0_freq1[data_1]",
        "Cos_1_freq1[data_1]",
        "Sin_1_freq1[data_1]",
        "Cos_0_freq2[data_2]",
        "Sin_0_freq2[data_2]",
    ]

    assert mod.shock_names == [
        "Cos_0_freq1[data_1]",
        "Sin_0_freq1[data_1]",
        "Cos_1_freq1[data_1]",
        "Sin_1_freq1[data_1]",
        "Cos_0_freq2[data_2]",
        "Sin_0_freq2[data_2]",
    ]

    x0, *_, T = mod._unpack_statespace_with_placeholders()[:5]
    input_vars = explicit_graph_inputs([x0, T])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0, T],
        mode="FAST_COMPILE",
    )

    x0_v, T_v = fn(
        params_freq1=np.array([1.0, 0.0, 1.2], dtype=config.floatX),
        params_freq2=np.array([3.0, 0.0], dtype=config.floatX),
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


def test_add_frequency_seasonality_shared_and_not_shared():
    shared_season = st.FrequencySeasonality(
        season_length=4,
        n=1,
        name="shared_season",
        innovations=True,
        observed_state_names=["data_1", "data_2"],
        share_states=True,
    )

    individual_season = st.FrequencySeasonality(
        season_length=4,
        n=2,
        name="individual_season",
        innovations=True,
        observed_state_names=["data_1", "data_2"],
        share_states=False,
    )

    mod = (shared_season + individual_season).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 10
    assert mod.k_posdef == 10

    assert mod.coords["state_shared_season"] == [
        "Cos_0_shared_season",
        "Sin_0_shared_season",
    ]
    assert mod.coords["state_individual_season"] == [
        "Cos_0_individual_season",
        "Sin_0_individual_season",
        "Cos_1_individual_season",
    ]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "single_endog_non_saturated",
            "season_length": 12,
            "n": 2,
            "observed_state_names": ["data1"],
            "expected_shape": (4,),
        },
        {
            "name": "single_endog_saturated",
            "season_length": 12,
            "n": 6,
            "observed_state_names": ["data1"],
            "expected_shape": (11,),
        },
        {
            "name": "multiple_endog_non_saturated",
            "season_length": 12,
            "n": 2,
            "observed_state_names": ["data1", "data2"],
            "expected_shape": (2, 4),
        },
        {
            "name": "multiple_endog_saturated",
            "season_length": 12,
            "n": 6,
            "observed_state_names": ["data1", "data2"],
            "expected_shape": (2, 11),
        },
        {
            "name": "small_n",
            "season_length": 12,
            "n": 1,
            "observed_state_names": ["data1"],
            "expected_shape": (2,),
        },
        {
            "name": "many_endog",
            "season_length": 12,
            "n": 2,
            "observed_state_names": ["data1", "data2", "data3", "data4"],
            "expected_shape": (4, 4),
        },
    ],
    ids=lambda x: x["name"],
)
def test_frequency_seasonality_coordinates(test_case):
    model_name = f"season_{test_case['name'].split('_')[0]}"

    season = FrequencySeasonality(
        season_length=test_case["season_length"],
        n=test_case["n"],
        name=model_name,
        observed_state_names=test_case["observed_state_names"],
    )
    season.populate_component_properties()

    # assert parameter shape
    assert season.param_info[f"params_{model_name}"]["shape"] == test_case["expected_shape"]

    # generate expected state names based on actual model name
    expected_state_names = [
        f"{f}_{i}_{model_name}" for i in range(test_case["n"]) for f in ["Cos", "Sin"]
    ][: test_case["expected_shape"][-1]]

    # assert coordinate structure
    if len(test_case["observed_state_names"]) == 1:
        assert len(season.coords[f"state_{model_name}"]) == test_case["expected_shape"][0]
        assert season.coords[f"state_{model_name}"] == expected_state_names
    else:
        assert len(season.coords[f"endog_{model_name}"]) == test_case["expected_shape"][0]
        assert len(season.coords[f"state_{model_name}"]) == test_case["expected_shape"][1]
        assert season.coords[f"state_{model_name}"] == expected_state_names

    # Check coords match the expected shape
    param_shape = season.param_info[f"params_{model_name}"]["shape"]
    state_coords = season.coords[f"state_{model_name}"]
    endog_coords = season.coords.get(f"endog_{model_name}")

    assert len(state_coords) == param_shape[-1]
    if endog_coords:
        assert len(endog_coords) == param_shape[0]
