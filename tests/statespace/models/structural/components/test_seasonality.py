import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config
from pytensor.graph.traversal import explicit_graph_inputs

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.structural.components.seasonality import FrequencySeasonality
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import assert_pattern_repeats, simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


@pytest.mark.parametrize("s", [2, 10, 25])
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("innovations", [True, False])
@pytest.mark.parametrize("remove_first_state", [True, False])
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in matmul:RuntimeWarning",
    "ignore:overflow encountered in matmul:RuntimeWarning",
    "ignore:invalid value encountered in matmul:RuntimeWarning",
)
def test_time_seasonality(s, d, innovations, remove_first_state, rng):
    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=innovations,
        name="season",
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros(mod.n_seasons, dtype=config.floatX)
    x0[0] = 1

    params = {"params_season": x0}
    if innovations:
        params["sigma_season"] = 0.0

    x, y = simulate_from_numpy_model(mod, rng, params, steps=100 * mod.duration)
    y = y.ravel()
    if not innovations:
        assert_pattern_repeats(y, s * d, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)

    test_slice = slice(1, None) if remove_first_state else slice(None)
    expected_states = tuple(f"season_{i}" for i in range(s))
    assert mod.coords["state_season"] == expected_states[test_slice]


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("start_state", [0, 2, "state_2"])
def test_time_seasonality_start_state(d, start_state, rng):
    s = 4
    state_names = [f"state_{i}" for i in range(s)]

    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=False,
        name="season",
        state_names=state_names,
        remove_first_state=True,
        start_state=start_state,
    )

    params = np.array([1.0, 2.0, 3.0], dtype=config.floatX)
    implied_gamma0 = -params.sum()
    default_seasons = [implied_gamma0, params[0], params[1], params[2]]

    start_idx = state_names.index(start_state) if isinstance(start_state, str) else start_state
    expected_seasons = default_seasons[start_idx:] + default_seasons[:start_idx]

    param_dict = {"params_season": params}
    x, y = simulate_from_numpy_model(mod, rng, param_dict, steps=s * d * 2)
    y = y.ravel()

    for season_idx, expected_val in enumerate(expected_seasons):
        for duration_step in range(d):
            t = season_idx * d + duration_step
            np.testing.assert_allclose(y[t], expected_val, atol=ATOL, rtol=RTOL)


@pytest.mark.filterwarnings("ignore:No time index found:UserWarning")
def test_time_seasonality_prior_sampling():
    s, d = 4, 2
    state_names = ["Q1", "Q2", "Q3", "Q4"]

    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=False,
        name="quarterly",
        state_names=state_names,
        remove_first_state=True,
    )
    ss_mod = mod.build(verbose=False)

    assert len(ss_mod.coords["state"]) == d * (s - 1)
    assert len(ss_mod.coords["state_quarterly"]) == s - 1

    with pm.Model(coords=ss_mod.coords) as model:
        P0 = pm.Deterministic("P0", pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims["P0"])
        params = pm.Normal(
            "params_quarterly", mu=0, sigma=1, dims=ss_mod.param_dims["params_quarterly"]
        )
        ss_mod.build_statespace_graph(np.zeros((20, 1)))
        prior = pm.sample_prior_predictive(draws=5)

    assert prior.prior["params_quarterly"].shape == (1, 5, s - 1)


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("remove_first_state", [True, False])
def test_time_seasonality_multiple_observed(rng, d, remove_first_state):
    s = 3
    state_names = [f"state_{i}" for i in range(s)]
    mod = st.TimeSeasonality(
        season_length=s,
        duration=d,
        innovations=True,
        name="season",
        state_names=state_names,
        observed_state_names=["data_1", "data_2"],
        remove_first_state=remove_first_state,
    )
    x0 = np.zeros((mod.k_endog, mod.n_seasons), dtype=config.floatX)
    x0[0, 0] = 1
    x0[1, 0] = 2.0

    params = {"params_season": x0, "sigma_season": np.array([0.0, 0.0], dtype=config.floatX)}
    x, y = simulate_from_numpy_model(mod, rng, params, steps=123 * d)
    assert_pattern_repeats(y[:, 0], s * d, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], s * d, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    x0_sym, *_, T, Z, R, _, Q = mod._unpack_statespace_with_placeholders()

    input_vars = explicit_graph_inputs([x0_sym, T, Z, R, Q])
    fn = pytensor.function(
        inputs=list(input_vars), outputs=[x0_sym, T, Z, R, Q], mode="FAST_COMPILE"
    )

    params["sigma_season"] = np.array([0.1, 0.8], dtype=config.floatX)
    x0_v, T_v, Z_v, R_v, Q_v = fn(**params)

    mod0 = st.TimeSeasonality(season_length=s, duration=d, remove_first_state=remove_first_state)
    T0 = mod0.ssm["transition"].eval()

    if remove_first_state:
        k_states_per_endog = d * (s - 1)
        expected_T = np.block(
            [
                [T0, np.zeros((k_states_per_endog, k_states_per_endog))],
                [np.zeros((k_states_per_endog, k_states_per_endog)), T0],
            ]
        )
    else:
        expected_T = np.block([[T0, np.zeros((s * d, s * d))], [np.zeros((s * d, s * d)), T0]])

    np.testing.assert_allclose(T_v, expected_T, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(Q_v, np.array([[0.1**2, 0.0], [0.0, 0.8**2]]), atol=ATOL, rtol=RTOL)


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
    assert mod.coords["state_season"] == ("season_1", "season_2", "season_3")

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(Z, np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(T, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(R, np.array([[1.0], [0.0], [0.0]]))


def test_add_mixed_shared_not_shared_time_seasonality():
    shared = st.TimeSeasonality(
        season_length=3,
        duration=1,
        innovations=True,
        name="shared",
        state_names=["season_1", "season_2", "season_3"],
        observed_state_names=["data_1", "data_2"],
        remove_first_state=False,
        share_states=True,
    )
    individual = st.TimeSeasonality(
        season_length=3,
        duration=1,
        innovations=False,
        name="individual",
        state_names=["season_1", "season_2", "season_3"],
        observed_state_names=["data_1", "data_2"],
        remove_first_state=True,
        share_states=False,
    )
    mod = (shared + individual).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 7
    assert mod.k_posdef == 1
    assert mod.coords["state_shared"] == ("season_1", "season_2", "season_3")
    assert mod.coords["state_individual"] == ("season_2", "season_3")


@pytest.mark.parametrize("d1, d2", [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_add_two_time_seasonality_different_observed(rng, d1, d2):
    mod1 = st.TimeSeasonality(
        season_length=3,
        duration=d1,
        innovations=True,
        name="season1",
        state_names=[f"state_{i}" for i in range(3)],
        observed_state_names=["data_1"],
        remove_first_state=False,
    )
    mod2 = st.TimeSeasonality(
        season_length=5,
        duration=d2,
        innovations=True,
        name="season2",
        state_names=[f"state_{i}" for i in range(5)],
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

    T1 = mod1.ssm["transition"].eval()
    T2 = mod2.ssm["transition"].eval()

    x0, *_, T = mod._unpack_statespace_with_placeholders()[:5]
    input_vars = explicit_graph_inputs([x0, T])
    fn = pytensor.function(inputs=list(input_vars), outputs=[x0, T], mode="FAST_COMPILE")
    x0_v, T_v = fn(
        params_season1=np.array([1.0, 0.0, 0.0], dtype=config.floatX),
        params_season2=np.array([3.0, 0.0, 0.0, 1.2], dtype=config.floatX),
    )

    expected_T = np.block(
        [[T1, np.zeros((T1.shape[0], T2.shape[1]))], [np.zeros((T2.shape[0], T1.shape[1])), T2]]
    )
    np.testing.assert_allclose(T_v, expected_T, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [1, 2, 3, None])
@pytest.mark.parametrize("s", [5, 10, 25.2])
def test_frequency_seasonality(n, s, rng):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="season")
    assert mod.param_info["sigma_season"].shape == ()
    assert mod.param_info["sigma_season"].dims is None
    assert len(mod.coords["state_season"]) == mod.n_coefs

    x0 = rng.normal(size=mod.n_coefs).astype(config.floatX)
    params = {"params_season": x0, "sigma_season": 0.0}

    decimal = s_str.split(".") if "." in (s_str := str(s)) else "0"
    T = int(s * 10 ** len(decimal))

    x, y = simulate_from_numpy_model(mod, rng, params, steps=2 * T)
    assert_pattern_repeats(y, T, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)


def test_frequency_seasonality_multiple_observed(rng):
    observed_state_names = ["data_1", "data_2"]
    mod = st.FrequencySeasonality(
        season_length=4,
        n=None,
        name="season",
        innovations=True,
        observed_state_names=observed_state_names,
    )
    assert mod.param_info["params_season"].shape == (mod.k_endog, mod.n_coefs)
    assert mod.param_info["params_season"].dims == ("endog_season", "state_season")
    assert mod.param_info["sigma_season"].dims == ("endog_season",)

    x0 = np.zeros((2, 3), dtype=config.floatX)
    x0[0, 0] = 1.0
    x0[1, 0] = 2.0
    params = {"params_season": x0, "sigma_season": np.zeros(2, dtype=config.floatX)}
    x, y = simulate_from_numpy_model(mod, rng, params, steps=12)

    assert_pattern_repeats(y[:, 0], 4, atol=ATOL, rtol=RTOL)
    assert_pattern_repeats(y[:, 1], 4, atol=ATOL, rtol=RTOL)

    mod = mod.build(verbose=False)

    x0_sym, *_, T_sym, Z_sym, R_sym, _, Q_sym = mod._unpack_statespace_with_placeholders()
    input_vars = explicit_graph_inputs([x0_sym, T_sym, Z_sym, R_sym, Q_sym])
    fn = pytensor.function(
        inputs=list(input_vars),
        outputs=[x0_sym, T_sym, Z_sym, R_sym, Q_sym],
        mode="FAST_COMPILE",
    )
    params["sigma_season"] = np.array([0.1, 0.8], dtype=config.floatX)
    x0_v, T_v, Z_v, R_v, Q_v = fn(**params)

    np.testing.assert_allclose(
        x0_v, np.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]), atol=ATOL, rtol=RTOL
    )
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
    assert mod.coords["state_season"] == ("Cos_0_season", "Sin_0_season")

    Z, T, R = pytensor.function(
        [], [mod.ssm["design"], mod.ssm["transition"], mod.ssm["selection"]], mode="FAST_COMPILE"
    )()

    np.testing.assert_allclose(Z, np.array([[1.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(R, np.array([[1.0, 0.0], [0.0, 1.0]]))

    lam = 2 * np.pi * 1 / 4
    np.testing.assert_allclose(
        T, np.array([[np.cos(lam), np.sin(lam)], [-np.sin(lam), np.cos(lam)]])
    )


def test_add_two_frequency_seasonality_different_observed(rng):
    mod1 = st.FrequencySeasonality(
        season_length=4,
        n=2,
        name="freq1",
        innovations=True,
        observed_state_names=["data_1"],
    )
    mod2 = st.FrequencySeasonality(
        season_length=6,
        n=1,
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


def test_add_frequency_seasonality_shared_and_not_shared():
    shared = st.FrequencySeasonality(
        season_length=4,
        n=1,
        name="shared_season",
        innovations=True,
        observed_state_names=["data_1", "data_2"],
        share_states=True,
    )
    individual = st.FrequencySeasonality(
        season_length=4,
        n=2,
        name="individual_season",
        innovations=True,
        observed_state_names=["data_1", "data_2"],
        share_states=False,
    )

    mod = (shared + individual).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 10
    assert mod.k_posdef == 10
    assert mod.coords["state_shared_season"] == ("Cos_0_shared_season", "Sin_0_shared_season")
    assert mod.coords["state_individual_season"] == (
        "Cos_0_individual_season",
        "Sin_0_individual_season",
        "Cos_1_individual_season",
    )


@pytest.mark.parametrize(
    "n,observed,expected_shape",
    [
        (2, ("data1",), (4,)),
        (6, ("data1",), (11,)),
        (2, ("data1", "data2"), (2, 4)),
        (6, ("data1", "data2"), (2, 11)),
        (1, ("data1",), (2,)),
        (2, ("data1", "data2", "data3", "data4"), (4, 4)),
    ],
)
def test_frequency_seasonality_coordinates(n, observed, expected_shape):
    season = FrequencySeasonality(
        season_length=12,
        n=n,
        name="season",
        observed_state_names=observed,
    )
    season.populate_component_properties()

    assert season.param_info["params_season"].shape == expected_shape

    state_coords = season.coords["state_season"]
    assert len(state_coords) == expected_shape[-1]

    if len(observed) > 1:
        endog_coords = season.coords["endog_season"]
        assert len(endog_coords) == expected_shape[0]
