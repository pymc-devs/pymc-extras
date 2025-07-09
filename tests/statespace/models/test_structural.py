import functools as ft
import warnings

from collections import defaultdict

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from scipy import linalg

from pymc_extras.statespace import structural as st
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    SHORT_NAME_TO_LONG,
)
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import (
    assert_pattern_repeats,
    simulate_from_numpy_model,
    unpack_symbolic_matrices_with_params,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 0 if floatX.endswith("64") else 1e-6


def _assert_all_statespace_matrices_match(mod, params, sm_mod):
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)

    sm_x0, sm_H0, sm_P0 = sm_mod.initialization()

    if len(x0) > 0:
        assert_allclose(x0, sm_x0)

    for name, matrix in zip(["T", "R", "Z", "Q"], [T, R, Z, Q]):
        long_name = SHORT_NAME_TO_LONG[name]
        if np.any([x == 0 for x in matrix.shape]):
            continue
        assert_allclose(
            sm_mod.ssm[long_name],
            matrix,
            err_msg=f"matrix {name} does not match statsmodels",
            atol=ATOL,
            rtol=RTOL,
        )


def _assert_coord_shapes_match_matrices(mod, params):
    if "initial_state_cov" not in params:
        params["initial_state_cov"] = np.eye(mod.k_states)

    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)

    n_states = len(mod.coords[ALL_STATE_DIM])

    # There will always be one shock dimension -- dummies are inserted into fully deterministic models to avoid errors
    # in the state space representation.
    n_shocks = max(1, len(mod.coords[SHOCK_DIM]))
    n_obs = len(mod.coords[OBS_STATE_DIM])

    assert x0.shape[-1:] == (
        n_states,
    ), f"x0 expected to have shape (n_states, ), found {x0.shape[-1:]}"
    assert P0.shape[-2:] == (
        n_states,
        n_states,
    ), f"P0 expected to have shape (n_states, n_states), found {P0.shape[-2:]}"
    assert c.shape[-1:] == (
        n_states,
    ), f"c expected to have shape (n_states, ), found {c.shape[-1:]}"
    assert d.shape[-1:] == (n_obs,), f"d expected to have shape (n_obs, ), found {d.shape[-1:]}"
    assert T.shape[-2:] == (
        n_states,
        n_states,
    ), f"T expected to have shape (n_states, n_states), found {T.shape[-2:]}"
    assert Z.shape[-2:] == (
        n_obs,
        n_states,
    ), f"Z expected to have shape (n_obs, n_states), found {Z.shape[-2:]}"
    assert R.shape[-2:] == (
        n_states,
        n_shocks,
    ), f"R expected to have shape (n_states, n_shocks), found {R.shape[-2:]}"
    assert H.shape[-2:] == (
        n_obs,
        n_obs,
    ), f"H expected to have shape (n_obs, n_obs), found {H.shape[-2:]}"
    assert Q.shape[-2:] == (
        n_shocks,
        n_shocks,
    ), f"Q expected to have shape (n_shocks, n_shocks), found {Q.shape[-2:]}"


def _assert_basic_coords_correct(mod):
    assert mod.coords[ALL_STATE_DIM] == mod.state_names
    assert mod.coords[ALL_STATE_AUX_DIM] == mod.state_names
    assert mod.coords[SHOCK_DIM] == mod.shock_names
    assert mod.coords[SHOCK_AUX_DIM] == mod.shock_names
    assert mod.coords[OBS_STATE_DIM] == ["data"]
    assert mod.coords[OBS_STATE_AUX_DIM] == ["data"]


def _assert_keys_match(test_dict, expected_dict):
    expected_keys = list(expected_dict.keys())
    param_keys = list(test_dict.keys())
    key_diff = set(expected_keys) - set(param_keys)
    assert len(key_diff) == 0, f'{", ".join(key_diff)} were not found in the test_dict keys.'

    key_diff = set(param_keys) - set(expected_keys)
    assert (
        len(key_diff) == 0
    ), f'{", ".join(key_diff)} were keys of the tests_dict not in expected_dict.'


def _assert_param_dims_correct(param_dims, expected_dims):
    if len(expected_dims) == 0 and len(param_dims) == 0:
        return

    _assert_keys_match(param_dims, expected_dims)
    for param, dims in expected_dims.items():
        assert dims == param_dims[param], f"dims for parameter {param} do not match"


def _assert_coords_correct(coords, expected_coords):
    if len(coords) == 0 and len(expected_coords) == 0:
        return

    _assert_keys_match(coords, expected_coords)
    for dim, labels in expected_coords.items():
        assert labels == coords[dim], f"labels on dimension {dim} do not match"


def _assert_params_info_correct(param_info, coords, param_dims):
    for param in param_info.keys():
        info = param_info[param]

        dims = info["dims"]
        labels = [coords[dim] for dim in dims] if dims is not None else None
        if labels is not None:
            assert param in param_dims.keys()
            inferred_dims = param_dims[param]
        else:
            inferred_dims = None

        shape = tuple(len(label) for label in labels) if labels is not None else ()

        assert info["shape"] == shape
        assert dims == inferred_dims


def create_structural_model_and_equivalent_statsmodel(
    rng,
    level: bool | None = False,
    trend: bool | None = False,
    seasonal: int | None = None,
    freq_seasonal: list[dict] | None = None,
    cycle: bool = False,
    autoregressive: int | None = None,
    exog: np.ndarray | None = None,
    irregular: bool | None = False,
    stochastic_level: bool | None = True,
    stochastic_trend: bool | None = False,
    stochastic_seasonal: bool | None = True,
    stochastic_freq_seasonal: list[bool] | None = None,
    stochastic_cycle: bool | None = False,
    damped_cycle: bool | None = False,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ft.partial(
            sm.tsa.UnobservedComponents,
            level=level,
            trend=trend,
            seasonal=seasonal,
            freq_seasonal=freq_seasonal,
            cycle=cycle,
            autoregressive=autoregressive,
            exog=exog,
            irregular=irregular,
            stochastic_level=stochastic_level,
            stochastic_trend=stochastic_trend,
            stochastic_seasonal=stochastic_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            stochastic_cycle=stochastic_cycle,
            damped_cycle=damped_cycle,
            mle_regression=False,
        )

    params = {}
    sm_params = {}
    sm_init = {}
    expected_param_dims = defaultdict(tuple)
    expected_coords = defaultdict(list)
    expected_param_dims["P0"] += ("state", "state_aux")

    default_states = [
        ALL_STATE_DIM,
        ALL_STATE_AUX_DIM,
        OBS_STATE_DIM,
        OBS_STATE_AUX_DIM,
        SHOCK_DIM,
        SHOCK_AUX_DIM,
    ]
    default_values = [[], [], ["data"], ["data"], [], []]
    for dim, value in zip(default_states, default_values):
        expected_coords[dim] += value

    components = []

    if irregular:
        sigma2 = np.abs(rng.normal()).astype(floatX).item()
        params["sigma_irregular"] = np.sqrt(sigma2)
        sm_params["sigma2.irregular"] = sigma2

        comp = st.MeasurementError("irregular")
        components.append(comp)

    level_trend_order = [0, 0]
    level_trend_innov_order = [0, 0]

    if level:
        level_trend_order[0] = 1
        expected_coords["trend_state"] += [
            "level",
        ]
        expected_coords[ALL_STATE_DIM] += [
            "level",
        ]
        expected_coords[ALL_STATE_AUX_DIM] += [
            "level",
        ]
        if stochastic_level:
            level_trend_innov_order[0] = 1
            expected_coords["trend_shock"] += ["level"]
            expected_coords[SHOCK_DIM] += [
                "level",
            ]
            expected_coords[SHOCK_AUX_DIM] += [
                "level",
            ]

    if trend:
        level_trend_order[1] = 1
        expected_coords["trend_state"] += [
            "trend",
        ]
        expected_coords[ALL_STATE_DIM] += [
            "trend",
        ]
        expected_coords[ALL_STATE_AUX_DIM] += [
            "trend",
        ]

        if stochastic_trend:
            level_trend_innov_order[1] = 1
            expected_coords["trend_shock"] += ["trend"]
            expected_coords[SHOCK_DIM] += ["trend"]
            expected_coords[SHOCK_AUX_DIM] += ["trend"]

    if level or trend:
        expected_param_dims["initial_trend"] += ("trend_state",)
        level_value = np.where(
            level_trend_order,
            rng.normal(
                size=2,
            ).astype(floatX),
            np.zeros(2, dtype=floatX),
        )
        sigma_level_value2 = np.abs(rng.normal(size=(2,)))[
            np.array(level_trend_innov_order, dtype="bool")
        ]
        max_order = np.flatnonzero(level_value)[-1].item() + 1
        level_trend_order = level_trend_order[:max_order]

        params["initial_trend"] = level_value[:max_order]
        sm_init["level"] = level_value[0]
        sm_init["trend"] = level_value[1]

        if sum(level_trend_innov_order) > 0:
            expected_param_dims["sigma_trend"] += ("trend_shock",)
            params["sigma_trend"] = np.sqrt(sigma_level_value2)

        sigma_level_value = sigma_level_value2.tolist()
        if stochastic_level:
            sigma = sigma_level_value.pop(0)
            sm_params["sigma2.level"] = sigma
        if stochastic_trend:
            sigma = sigma_level_value.pop(0)
            sm_params["sigma2.trend"] = sigma

        comp = st.LevelTrendComponent(
            name="level", order=level_trend_order, innovations_order=level_trend_innov_order
        )
        components.append(comp)

    if seasonal is not None:
        state_names = [f"seasonal_{i}" for i in range(seasonal)][1:]
        seasonal_coefs = rng.normal(size=(seasonal - 1,)).astype(floatX)
        params["seasonal_coefs"] = seasonal_coefs
        expected_param_dims["seasonal_coefs"] += ("seasonal_state",)

        expected_coords["seasonal_state"] += tuple(state_names)
        expected_coords[ALL_STATE_DIM] += state_names
        expected_coords[ALL_STATE_AUX_DIM] += state_names

        seasonal_dict = {
            "seasonal" if i == 0 else f"seasonal.L{i}": c for i, c in enumerate(seasonal_coefs)
        }
        sm_init.update(seasonal_dict)

        if stochastic_seasonal:
            sigma2 = np.abs(rng.normal()).astype(floatX)
            params["sigma_seasonal"] = np.sqrt(sigma2)
            sm_params["sigma2.seasonal"] = sigma2
            expected_coords[SHOCK_DIM] += [
                "seasonal",
            ]
            expected_coords[SHOCK_AUX_DIM] += [
                "seasonal",
            ]

        comp = st.TimeSeasonality(
            name="seasonal", season_length=seasonal, innovations=stochastic_seasonal
        )
        components.append(comp)

    if freq_seasonal is not None:
        state_count = 0
        for d, has_innov in zip(freq_seasonal, stochastic_freq_seasonal):
            n = d["harmonics"]
            s = d["period"]
            last_state_not_identified = (s / n) == 2.0
            n_states = 2 * n - int(last_state_not_identified)
            state_names = [f"seasonal_{s}_{f}_{i}" for i in range(n) for f in ["Cos", "Sin"]]

            seasonal_params = rng.normal(size=n_states).astype(floatX)

            params[f"seasonal_{s}"] = seasonal_params
            expected_param_dims[f"seasonal_{s}"] += (f"seasonal_{s}_state",)
            expected_coords[ALL_STATE_DIM] += state_names
            expected_coords[ALL_STATE_AUX_DIM] += state_names
            expected_coords[f"seasonal_{s}_state"] += (
                tuple(state_names[:-1]) if last_state_not_identified else tuple(state_names)
            )

            for param in seasonal_params:
                sm_init[f"freq_seasonal.{state_count}"] = param
                state_count += 1
            if last_state_not_identified:
                sm_init[f"freq_seasonal.{state_count}"] = 0.0
                state_count += 1

            if has_innov:
                sigma2 = np.abs(rng.normal()).astype(floatX)
                params[f"sigma_seasonal_{s}"] = np.sqrt(sigma2)
                sm_params[f"sigma2.freq_seasonal_{s}({n})"] = sigma2
                expected_coords[SHOCK_DIM] += state_names
                expected_coords[SHOCK_AUX_DIM] += state_names

            comp = st.FrequencySeasonality(
                name=f"seasonal_{s}", season_length=s, n=n, innovations=has_innov
            )
            components.append(comp)

    if cycle:
        cycle_length = np.random.choice(np.arange(2, 12)).astype(floatX)

        # Statsmodels takes the frequency not the cycle length, so convert it.
        sm_params["frequency.cycle"] = 2.0 * np.pi / cycle_length
        params["cycle_length"] = cycle_length

        init_cycle = rng.normal(size=(2,)).astype(floatX)
        params["cycle"] = init_cycle
        expected_param_dims["cycle"] += ("cycle_state",)

        state_names = ["cycle_Cos", "cycle_Sin"]
        expected_coords["cycle_state"] += state_names
        expected_coords[ALL_STATE_DIM] += state_names
        expected_coords[ALL_STATE_AUX_DIM] += state_names

        sm_init["cycle"] = init_cycle[0]
        sm_init["cycle.auxilliary"] = init_cycle[1]

        if stochastic_cycle:
            sigma2 = np.abs(rng.normal()).astype(floatX)
            params["sigma_cycle"] = np.sqrt(sigma2)
            expected_coords[SHOCK_DIM] += state_names
            expected_coords[SHOCK_AUX_DIM] += state_names

            sm_params["sigma2.cycle"] = sigma2

        if damped_cycle:
            rho = rng.beta(1, 1)
            params["cycle_dampening_factor"] = rho
            sm_params["damping.cycle"] = rho

        comp = st.CycleComponent(
            name="cycle",
            dampen=damped_cycle,
            innovations=stochastic_cycle,
            estimate_cycle_length=True,
        )

        components.append(comp)

    if autoregressive is not None:
        ar_names = [f"L{i+1}.data" for i in range(autoregressive)]
        ar_params = rng.normal(size=(autoregressive,)).astype(floatX)
        if autoregressive == 1:
            ar_params = ar_params.item()
        sigma2 = np.abs(rng.normal()).astype(floatX)

        params["ar_params"] = ar_params
        params["sigma_ar"] = np.sqrt(sigma2)
        expected_param_dims["ar_params"] += (AR_PARAM_DIM,)
        expected_coords[AR_PARAM_DIM] += tuple(list(range(1, autoregressive + 1)))
        expected_coords[ALL_STATE_DIM] += ar_names
        expected_coords[ALL_STATE_AUX_DIM] += ar_names
        expected_coords[SHOCK_DIM] += ["ar_innovation"]
        expected_coords[SHOCK_AUX_DIM] += ["ar_innovation"]

        sm_params["sigma2.ar"] = sigma2
        for i, rho in enumerate(ar_params):
            sm_init[f"ar.L{i+1}"] = 0
            sm_params[f"ar.L{i+1}"] = rho

        comp = st.AutoregressiveComponent(name="ar", order=autoregressive)
        components.append(comp)

    if exog is not None:
        names = [f"x{i + 1}" for i in range(exog.shape[1])]
        betas = rng.normal(size=(exog.shape[1],)).astype(floatX)
        params["beta_exog"] = betas
        params["data_exog"] = exog
        expected_param_dims["beta_exog"] += ("exog_state",)
        expected_param_dims["data_exog"] += ("time", "exog_data")

        expected_coords["exog_state"] += tuple(names)

        for i, beta in enumerate(betas):
            sm_params[f"beta.x{i + 1}"] = beta
            sm_init[f"beta.x{i+1}"] = beta
        comp = st.RegressionComponent(name="exog", state_names=names)
        components.append(comp)

    st_mod = components.pop(0)
    for comp in components:
        st_mod += comp
    return mod, st_mod, params, sm_params, sm_init, expected_param_dims, expected_coords


@pytest.mark.parametrize(
    "level, trend, stochastic_level, stochastic_trend, irregular",
    [
        (False, False, False, False, True),
        (True, True, True, True, True),
        (True, True, False, True, False),
    ],
)
@pytest.mark.parametrize("autoregressive", [None, 3])
@pytest.mark.parametrize("seasonal, stochastic_seasonal", [(None, False), (12, False), (12, True)])
@pytest.mark.parametrize(
    "freq_seasonal, stochastic_freq_seasonal",
    [
        (None, None),
        ([{"period": 12, "harmonics": 2}], [False]),
        ([{"period": 12, "harmonics": 6}], [True]),
    ],
)
@pytest.mark.parametrize(
    "cycle, damped_cycle, stochastic_cycle",
    [(False, False, False), (True, False, True), (True, True, True)],
)
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.SpecificationWarning")
def test_structural_model_against_statsmodels(
    level,
    trend,
    stochastic_level,
    stochastic_trend,
    irregular,
    autoregressive,
    seasonal,
    stochastic_seasonal,
    freq_seasonal,
    stochastic_freq_seasonal,
    cycle,
    damped_cycle,
    stochastic_cycle,
    rng,
):
    retvals = create_structural_model_and_equivalent_statsmodel(
        rng,
        level=level,
        trend=trend,
        seasonal=seasonal,
        freq_seasonal=freq_seasonal,
        cycle=cycle,
        damped_cycle=damped_cycle,
        autoregressive=autoregressive,
        irregular=irregular,
        stochastic_level=stochastic_level,
        stochastic_trend=stochastic_trend,
        stochastic_seasonal=stochastic_seasonal,
        stochastic_freq_seasonal=stochastic_freq_seasonal,
        stochastic_cycle=stochastic_cycle,
    )
    f_sm_mod, mod, params, sm_params, sm_init, expected_dims, expected_coords = retvals

    data = rng.normal(size=(100,)).astype(floatX)
    sm_mod = f_sm_mod(data)

    if len(sm_init) > 0:
        init_array = np.concatenate(
            [np.atleast_1d(sm_init[k]).ravel() for k in sm_mod.state_names if k != "dummy"]
        )
        sm_mod.initialize_known(init_array, np.eye(sm_mod.k_states))
    else:
        sm_mod.initialize_default()

    if len(sm_params) > 0:
        param_array = np.concatenate(
            [np.atleast_1d(sm_params[k]).ravel() for k in sm_mod.param_names]
        )
        sm_mod.update(param_array, transformed=True)

    _assert_all_statespace_matrices_match(mod, params, sm_mod)

    built_model = mod.build(verbose=False, mode="FAST_RUN")
    assert built_model.mode == "FAST_RUN"

    _assert_coord_shapes_match_matrices(built_model, params)
    _assert_param_dims_correct(built_model.param_dims, expected_dims)
    _assert_coords_correct(built_model.coords, expected_coords)
    _assert_params_info_correct(built_model.param_info, built_model.coords, built_model.param_dims)


def test_level_trend_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["trend_state"] == ["level", "trend"]


def test_measurement_error(rng):
    mod = st.MeasurementError("obs") + st.LevelTrendComponent(order=2)
    mod = mod.build(verbose=False)

    _assert_basic_coords_correct(mod)
    assert "sigma_obs" in mod.param_names


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
    ar = st.AutoregressiveComponent(order=order)
    params = {
        "ar_params": np.full((sum(ar.order),), 0.5, dtype=floatX),
        "sigma_ar": 0.0,
    }

    x, y = simulate_from_numpy_model(ar, rng, params, steps=100)

    # Check coords
    ar.build(verbose=False)
    _assert_basic_coords_correct(ar)
    lags = np.arange(len(order) if isinstance(order, list) else order, dtype="int") + 1
    if isinstance(order, list):
        lags = lags[np.flatnonzero(order)]
    assert_allclose(ar.coords["ar_lag"], lags)


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
    x0 = np.zeros(mod.k_states, dtype=floatX)
    x0[0] = 1

    params = {"season_coefs": x0}
    if mod.innovations:
        params["sigma_season"] = 0.0

    x, y = simulate_from_numpy_model(mod, rng, params)
    y = y.ravel()
    if not innovations:
        assert_pattern_repeats(y, s, atol=ATOL, rtol=RTOL)

    # Check coords
    mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    test_slice = slice(1, None) if remove_first_state else slice(None)
    assert mod.coords["season_state"] == state_names[test_slice]


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
    x0 = rng.normal(size=mod.n_coefs).astype(floatX)
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


cycle_test_vals = zip([None, None, 3, 5, 10], [False, True, True, False, False])


def test_cycle_component_deterministic(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False
    )
    params = {"cycle": np.array([1.0, 1.0], dtype=floatX)}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=12 * 12)

    assert_pattern_repeats(y, 12, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_dampening(rng):
    cycle = st.CycleComponent(
        name="cycle", cycle_length=12, estimate_cycle_length=False, innovations=False, dampen=True
    )
    params = {"cycle": np.array([10.0, 10.0], dtype=floatX), "cycle_dampening_factor": 0.75}
    x, y = simulate_from_numpy_model(cycle, rng, params, steps=100)

    # Check that the cycle dampens to zero over time
    assert_allclose(y[-1], 0.0, atol=ATOL, rtol=RTOL)


def test_cycle_component_with_innovations_and_cycle_length(rng):
    cycle = st.CycleComponent(
        name="cycle", estimate_cycle_length=True, innovations=True, dampen=True
    )
    params = {
        "cycle": np.array([1.0, 1.0], dtype=floatX),
        "cycle_length": 12.0,
        "cycle_dampening_factor": 0.95,
        "sigma_cycle": 1.0,
    }

    x, y = simulate_from_numpy_model(cycle, rng, params)

    cycle.build(verbose=False)
    _assert_basic_coords_correct(cycle)


def test_exogenous_component(rng):
    data = rng.normal(size=(100, 2)).astype(floatX)
    mod = st.RegressionComponent(state_names=["feature_1", "feature_2"], name="exog")

    params = {"beta_exog": np.array([1.0, 2.0], dtype=floatX)}
    exog_data = {"data_exog": data}
    x, y = simulate_from_numpy_model(mod, rng, params, exog_data)

    # Check that the generated data is just a linear regression
    assert_allclose(y, data @ params["beta_exog"], atol=ATOL, rtol=RTOL)

    mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["exog_state"] == ["feature_1", "feature_2"]


def test_adding_exogenous_component(rng):
    data = rng.normal(size=(100, 2)).astype(floatX)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    ll = st.LevelTrendComponent(name="level")

    seasonal = st.FrequencySeasonality(name="annual", season_length=12, n=4)
    mod = reg + ll + seasonal

    assert mod.ssm["design"].eval({"data_exog": data}).shape == (100, 1, 2 + 2 + 8)
    assert_allclose(mod.ssm["design", 5, 0, :2].eval({"data_exog": data}), data[5])


def test_add_components():
    ll = st.LevelTrendComponent(order=2)
    se = st.TimeSeasonality(name="seasonal", season_length=12)
    mod = ll + se

    ll_params = {
        "initial_trend": np.zeros(2, dtype=floatX),
        "sigma_trend": np.ones(2, dtype=floatX),
    }
    se_params = {
        "seasonal_coefs": np.ones(11, dtype=floatX),
        "sigma_seasonal": 1.0,
    }
    all_params = ll_params.copy()
    all_params.update(se_params)

    (ll_x0, ll_P0, ll_c, ll_d, ll_T, ll_Z, ll_R, ll_H, ll_Q) = unpack_symbolic_matrices_with_params(
        ll, ll_params
    )
    (se_x0, se_P0, se_c, se_d, se_T, se_Z, se_R, se_H, se_Q) = unpack_symbolic_matrices_with_params(
        se, se_params
    )
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, all_params)

    for property in ["param_names", "shock_names", "param_info", "coords", "param_dims"]:
        assert [x in getattr(mod, property) for x in getattr(ll, property)]
        assert [x in getattr(mod, property) for x in getattr(se, property)]

    ll_mats = [ll_T, ll_R, ll_Q]
    se_mats = [se_T, se_R, se_Q]
    all_mats = [T, R, Q]

    for ll_mat, se_mat, all_mat in zip(ll_mats, se_mats, all_mats):
        assert_allclose(all_mat, linalg.block_diag(ll_mat, se_mat), atol=ATOL, rtol=RTOL)

    ll_mats = [ll_x0, ll_c, ll_Z]
    se_mats = [se_x0, se_c, se_Z]
    all_mats = [x0, c, Z]
    axes = [0, 0, 1]

    for ll_mat, se_mat, all_mat, axis in zip(ll_mats, se_mats, all_mats, axes):
        assert_allclose(all_mat, np.concatenate([ll_mat, se_mat], axis=axis), atol=ATOL, rtol=RTOL)


def test_filter_scans_time_varying_design_matrix(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    mod = reg.build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.Data("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])

        mod.build_statespace_graph(y)
        x0, P0, c, d, T, Z, R, H, Q = mod.unpack_statespace()
        pm.Deterministic("Z", Z)

        prior = pm.sample_prior_predictive(draws=10)

    prior_Z = prior.prior.Z.values
    assert prior_Z.shape == (1, 10, 100, 1, 2)
    assert_allclose(prior_Z[0, :, :, 0, :], data.values[None].repeat(10, axis=0))


@pytest.mark.skipif(floatX.endswith("32"), reason="Prior covariance not PSD at half-precision")
def test_extract_components_from_idata(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    ll = st.LevelTrendComponent()
    season = st.FrequencySeasonality(name="seasonal", season_length=12, n=2, innovations=False)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    me = st.MeasurementError("obs")
    mod = (ll + season + reg + me).build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.Data("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])
        initial_trend = pm.Normal("initial_trend", dims=["trend_state"])
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["trend_shock"])
        seasonal_coefs = pm.Normal("seasonal", dims=["seasonal_state"])
        sigma_obs = pm.Exponential("sigma_obs", 1)

        mod.build_statespace_graph(y)

        x0, P0, c, d, T, Z, R, H, Q = mod.unpack_statespace()
        prior = pm.sample_prior_predictive(draws=10)

    filter_prior = mod.sample_conditional_prior(prior)
    comp_prior = mod.extract_components_from_idata(filter_prior)
    comp_states = comp_prior.filtered_prior.coords["state"].values
    expected_states = ["LevelTrend[level]", "LevelTrend[trend]", "seasonal", "exog[a]", "exog[b]"]
    missing = set(comp_states) - set(expected_states)

    assert len(missing) == 0, missing
