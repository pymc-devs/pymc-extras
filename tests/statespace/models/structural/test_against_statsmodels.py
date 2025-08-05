import functools as ft
import warnings

from collections import defaultdict

import numpy as np
import pytensor
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose

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


def _assert_keys_match(test_dict, expected_dict):
    expected_keys = list(expected_dict.keys())
    param_keys = list(test_dict.keys())
    key_diff = set(expected_keys) - set(param_keys)
    assert len(key_diff) == 0, f"{', '.join(key_diff)} were not found in the test_dict keys."

    key_diff = set(param_keys) - set(expected_keys)
    assert (
        len(key_diff) == 0
    ), f"{', '.join(key_diff)} were keys of the tests_dict not in expected_dict."


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
        expected_coords["state_level"] += [
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
            expected_coords["shock_level"] += ["level"]
            expected_coords[SHOCK_DIM] += [
                "level",
            ]
            expected_coords[SHOCK_AUX_DIM] += [
                "level",
            ]

    if trend:
        level_trend_order[1] = 1
        expected_coords["state_level"] += [
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
            expected_coords["shock_level"] += ["trend"]
            expected_coords[SHOCK_DIM] += ["trend"]
            expected_coords[SHOCK_AUX_DIM] += ["trend"]

    if level or trend:
        expected_param_dims["initial_level"] += ("state_level",)
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

        params["initial_level"] = level_value[:max_order]
        sm_init["level"] = level_value[0]
        sm_init["trend"] = level_value[1]

        if sum(level_trend_innov_order) > 0:
            expected_param_dims["sigma_level"] += ("shock_level",)
            params["sigma_level"] = np.sqrt(sigma_level_value2)

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
        params["params_seasonal"] = seasonal_coefs
        expected_param_dims["params_seasonal"] += ("state_seasonal",)

        expected_coords["state_seasonal"] += tuple(state_names)
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
            state_names = [f"{f}_{i}_seasonal_{s}" for i in range(n) for f in ["Cos", "Sin"]]

            seasonal_params = rng.normal(size=n_states).astype(floatX)

            params[f"params_seasonal_{s}"] = seasonal_params
            expected_param_dims[f"params_seasonal_{s}"] += (f"state_seasonal_{s}",)
            expected_coords[ALL_STATE_DIM] += state_names
            expected_coords[ALL_STATE_AUX_DIM] += state_names
            expected_coords[f"state_seasonal_{s}"] += (
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
        params["length_cycle"] = cycle_length

        init_cycle = rng.normal(size=(2,)).astype(floatX)
        params["params_cycle"] = init_cycle
        expected_param_dims["params_cycle"] += ("state_cycle",)

        state_names = ["Cos_cycle", "Sin_cycle"]
        expected_coords["state_cycle"] += state_names
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
            params["dampening_factor_cycle"] = rho
            sm_params["damping.cycle"] = rho

        comp = st.CycleComponent(
            name="cycle",
            dampen=damped_cycle,
            innovations=stochastic_cycle,
            estimate_cycle_length=True,
        )

        components.append(comp)

    if autoregressive is not None:
        ar_names = [f"L{i + 1}_ar" for i in range(autoregressive)]
        params_ar = rng.normal(size=(autoregressive,)).astype(floatX)
        if autoregressive == 1:
            params_ar = params_ar.item()
        sigma2 = np.abs(rng.normal()).astype(floatX)

        params["params_ar"] = params_ar
        params["sigma_ar"] = np.sqrt(sigma2)
        expected_param_dims["params_ar"] += (AR_PARAM_DIM,)
        expected_coords[AR_PARAM_DIM] += tuple(list(range(1, autoregressive + 1)))
        expected_coords[ALL_STATE_DIM] += ar_names
        expected_coords[ALL_STATE_AUX_DIM] += ar_names
        expected_coords[SHOCK_DIM] += ["ar"]
        expected_coords[SHOCK_AUX_DIM] += ["ar"]

        sm_params["sigma2.ar"] = sigma2
        for i, rho in enumerate(params_ar):
            sm_init[f"ar.L{i + 1}"] = 0
            sm_params[f"ar.L{i + 1}"] = rho

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
            sm_init[f"beta.x{i + 1}"] = beta
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
