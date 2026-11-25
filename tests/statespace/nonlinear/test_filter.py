import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace import structural as st
from pymc_extras.statespace.filters.kalman_filter import BaseFilter, StandardFilter
from pymc_extras.statespace.filters.kalman_smoother import DisturbanceSmoother
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    filter_step,
    fisher_curvature,
    observed_state_names,
    run_filter,
    scan_filter,
)
from pymc_extras.statespace.utils.constants import FILTER_OUTPUT_DIMS


def test_step_matches_standard_filter_linear_case():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
        + st.MeasurementError(name="noise", observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    X_data = pt.matrix("X")
    pointwise_logp, fisher_terms = fused.observation_logp(
        parameters=theta, data={"y": y, "X": X_data}
    )["y"]
    _, _, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters={
            "initial_level_trend": np.zeros((2, 1)),
            "sigma_level_trend": np.full((2, 1), 0.3),
            "params_auto_regressive": np.array([[0.7], [-0.4]]),
            "sigma_auto_regressive": np.array([0.5, 0.1]),
            "sigma_noise": np.array([0.2, 0.4]),
            "P0": np.eye(4),
        },
        statespace_data={},
    )
    c = c + np.array([0.2, -0.1, 0.0, 0.3])
    d = d + np.array([0.1, -0.3])

    predicted_mean = pt.vector("predicted_mean", shape=(4,))
    predicted_covariance = pt.matrix("predicted_covariance", shape=(4, 4))
    filtered_mean, filtered_covariance, next_mean, next_covariance, mode, _ = filter_step(
        logp=pointwise_logp.sum(),
        curvature=fisher_curvature(
            information=fisher_terms.information,
            distribution_parameters=fisher_terms.distribution_parameters,
            parameters=theta,
        ),
        parameters=theta,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        c=c,
        d=d,
        T=T,
        Z=Z,
        R=R,
        H=H,
        Q=Q,
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )

    reference_filter = StandardFilter(cov_jitter=0.0)
    reference_filtered_mean, reference_filtered_covariance, *_ = reference_filter.update(
        a=predicted_mean,
        P=predicted_covariance,
        y=y,
        d=X_data @ d,
        Z=X_data @ Z,
        H=0.25 * pt.eye(y.shape[0]) + X_data @ H @ X_data.T,
        nan_mask=pt.zeros_like(y, dtype=bool),
    )
    reference_next_mean, reference_next_covariance = BaseFilter.predict(
        a=reference_filtered_mean, P=reference_filtered_covariance, c=c, T=T, R=R, Q=Q
    )

    rng = np.random.default_rng(0)
    point = {
        predicted_mean: np.array([0.4, -0.2, 0.1, 0.3]),
        predicted_covariance: np.diag([1.0, 0.5, 0.8, 0.6]) + 0.2,
        y: rng.normal(size=4),
        X_data: rng.normal(size=(4, 2)),
    }
    evaluate = pytensor.function(
        list(point),
        [
            mode.converged,
            filtered_mean,
            filtered_covariance,
            next_mean,
            next_covariance,
            reference_filtered_mean,
            reference_filtered_covariance,
            reference_next_mean,
            reference_next_covariance,
        ],
    )
    converged, *values = evaluate(*point.values())
    assert converged
    for fused_value, reference_value in zip(values[:4], values[4:], strict=True):
        assert_allclose(fused_value, reference_value, atol=1e-10)


@pytest.mark.parametrize("curvature", ["fisher", "exact"])
def test_filter_matches_standard_filter_over_ragged_dates(curvature):
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
        + st.MeasurementError(name="noise", observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature=curvature,
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "params_auto_regressive": np.array([[0.7], [-0.4]]),
        "sigma_auto_regressive": np.array([0.5, 0.1]),
        "sigma_noise": np.array([0.2, 0.4]),
        "P0": np.eye(4),
    }

    rng = np.random.default_rng(0)
    sizes = [3, 1, 5]
    X_dates = [rng.normal(size=(size, 2)) for size in sizes]
    y_dates = [rng.normal(size=size) for size in sizes]
    filtered = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={"X": X_dates, "y": y_dates},
        dates=np.arange(3),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)

    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    a = pt.vector("a", shape=(4,))
    P = pt.matrix("P", shape=(4, 4))
    y = pt.vector("y")
    X_data = pt.matrix("X")
    reference_filter = StandardFilter(cov_jitter=0.0)
    reference_filtered_mean, reference_filtered_covariance, _, _, reference_loglike = (
        reference_filter.update(
            a=a,
            P=P,
            y=y,
            d=X_data @ d,
            Z=X_data @ Z,
            H=0.25 * pt.eye(y.shape[0]) + X_data @ H @ X_data.T,
            nan_mask=pt.zeros_like(y, dtype=bool),
        )
    )
    reference_step = pytensor.function(
        [a, P, y, X_data],
        [
            reference_filtered_mean,
            reference_filtered_covariance,
            reference_loglike,
            *BaseFilter.predict(
                a=reference_filtered_mean, P=reference_filtered_covariance, c=c, T=T, R=R, Q=Q
            ),
        ],
    )
    mean, covariance = x0.eval(), P0.eval()
    design, observed_intercept, observed_covariance = Z.eval(), d.eval(), H.eval()
    for date in range(len(sizes)):
        assert_allclose(outputs["predicted_states"].values[date], mean, atol=1e-10)
        assert_allclose(outputs["predicted_covariances"].values[date], covariance, atol=1e-10)
        assert_allclose(
            outputs["predicted_observed_states"].values[date],
            design @ mean + observed_intercept,
            atol=1e-10,
        )
        assert_allclose(
            outputs["predicted_observed_covariances"].values[date],
            design @ covariance @ design.T + observed_covariance,
            atol=1e-10,
        )
        filtered_mean, filtered_covariance, loglike, mean, covariance = reference_step(
            mean, covariance, y_dates[date], X_dates[date]
        )
        assert_allclose(outputs["filtered_states"].values[date], filtered_mean, atol=1e-10)
        assert_allclose(
            outputs["filtered_covariances"].values[date], filtered_covariance, atol=1e-10
        )
        assert_allclose(filtered["diagnostics/loglike_obs"].values[date], loglike, atol=1e-10)
    assert filtered["diagnostics/converged"].values.all()


@pytest.mark.parametrize("curvature", ["fisher", "exact"])
def test_weighted_gaussian_prior_is_a_pseudo_observation_of_the_states(curvature):
    prior_mean, prior_sigma, prior_weight = np.array([1.0, -0.5]), np.array([2.0, 0.8]), 3.0
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", mu=prior_mean, sigma=prior_sigma, dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature=curvature,
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=prior_weight,
    )
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "P0": np.eye(2),
    }

    rng = np.random.default_rng(0)
    sizes = [3, 0, 5]
    X_dates = [rng.normal(size=(size, 2)) for size in sizes]
    y_dates = [rng.normal(size=size) for size in sizes]
    filtered = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={"X": X_dates, "y": y_dates},
        dates=np.arange(3),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)

    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    a = pt.vector("a", shape=(2,))
    P = pt.matrix("P", shape=(2, 2))
    y = pt.vector("y")
    X_data = pt.matrix("X")
    # The weighted prior density is the density of the prior mean observed with variance
    # prior_sigma**2 / prior_weight, up to a constant.
    pseudo_design = pt.concatenate([X_data, pt.eye(2)], axis=0)
    pseudo_observations = pt.concatenate([y, prior_mean])
    pseudo_noise = pt.linalg.block_diag(
        0.25 * pt.eye(y.shape[0]), pt.diag(prior_sigma**2 / prior_weight)
    )
    reference_filtered_mean, reference_filtered_covariance, _, _, reference_loglike = (
        StandardFilter(cov_jitter=0.0).update(
            a=a,
            P=P,
            y=pseudo_observations,
            d=pseudo_design @ d,
            Z=pseudo_design @ Z,
            H=pseudo_noise + pseudo_design @ H @ pseudo_design.T,
            nan_mask=pt.zeros_like(pseudo_observations, dtype=bool),
        )
    )
    reference_step = pytensor.function(
        [a, P, y, X_data],
        [
            reference_filtered_mean,
            reference_filtered_covariance,
            reference_loglike,
            *BaseFilter.predict(
                a=reference_filtered_mean, P=reference_filtered_covariance, c=c, T=T, R=R, Q=Q
            ),
        ],
    )
    normalizing_difference = np.sum(
        -prior_weight / 2 * np.log(2 * np.pi * prior_sigma**2)
        + np.log(2 * np.pi * prior_sigma**2 / prior_weight) / 2
    )
    mean, covariance = x0.eval(), P0.eval()
    for date in range(len(sizes)):
        filtered_mean, filtered_covariance, loglike, mean, covariance = reference_step(
            mean, covariance, y_dates[date], X_dates[date].reshape(-1, 2)
        )
        assert_allclose(outputs["filtered_states"].values[date], filtered_mean, atol=1e-10)
        assert_allclose(
            outputs["filtered_covariances"].values[date], filtered_covariance, atol=1e-10
        )
        assert_allclose(
            filtered["diagnostics/loglike_obs"].values[date],
            loglike + normalizing_difference,
            atol=1e-10,
        )
    assert filtered["diagnostics/converged"].values.all()


def test_missing_date_is_predict_only():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "params_auto_regressive": np.array([[0.7], [-0.4]]),
        "sigma_auto_regressive": np.array([0.5, 0.1]),
        "P0": np.eye(4),
    }

    rng = np.random.default_rng(0)
    filtered = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={
            "X": [rng.normal(size=(3, 2)), np.zeros((0, 2)), rng.normal(size=(2, 2))],
            "y": [rng.normal(size=3), np.zeros(0), rng.normal(size=2)],
        },
        dates=np.arange(3),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)

    _, _, c, _, T, _, R, _, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    assert_allclose(outputs["filtered_states"].values[1], outputs["predicted_states"].values[1])
    assert_allclose(
        outputs["filtered_covariances"].values[1],
        outputs["predicted_covariances"].values[1],
    )
    next_mean, next_covariance = BaseFilter.predict(
        a=outputs["filtered_states"].values[1],
        P=outputs["filtered_covariances"].values[1],
        c=c,
        T=T,
        R=R,
        Q=Q,
    )
    assert_allclose(outputs["predicted_states"].values[2], next_mean.eval())
    assert_allclose(outputs["predicted_covariances"].values[2], next_covariance.eval())


def test_time_varying_statespace_matrices_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Regression(
        name="exog", state_names=["x1"], observed_state_names=["alpha"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"Statespace matrices \['Z'\] vary over time"):
        run_filter(
            fused=fused,
            statespace_parameters={"beta_exog": [0.5], "P0": [[1.0]]},
            statespace_data={"data_exog": np.ones((2, 1))},
            data={"y": [np.zeros(3), np.zeros(2)]},
            dates=np.arange(2),
        )


def test_fisher_curvature_without_closed_form_information_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.MvNormal("w", mu=alpha + np.zeros(2), cov=np.eye(2), observed=np.zeros((4, 2)))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(
        ValueError, match=r"Observed variables \['w'\] have no closed-form Fisher information"
    ):
        run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"w": [np.zeros((4, 2))]},
            dates=np.arange(1),
        )


def test_fisher_curvature_ignores_observations_that_do_not_depend_on_the_states():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))
        pm.Normal("z", 0, sigma, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={"sigma": 1.0},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    filtered = run_filter(
        fused=fused,
        statespace_parameters={
            "initial_level_trend": [0.0],
            "sigma_level_trend": [0.3],
            "P0": [[1.0]],
        },
        statespace_data={},
        data={"y": [np.ones(3)], "z": [np.ones(2)]},
        dates=np.arange(1),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)
    assert_allclose(outputs["filtered_states"].values, [[0.75]])
    assert_allclose(outputs["filtered_covariances"].values, [[[0.25]]])


def test_filter_output_coords():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    filtered = run_filter(
        fused=fused,
        statespace_parameters={
            "initial_level_trend": np.zeros((2, 1)),
            "sigma_level_trend": np.full((2, 1), 0.3),
            "P0": np.eye(2),
        },
        statespace_data={},
        data={"X": [np.ones((3, 2)), np.ones((1, 2))], "y": [np.zeros(3), np.zeros(1)]},
        dates=dates,
    )

    outputs = filtered["posterior_predictive"]
    assert {name: outputs[name].dims for name in outputs.data_vars} == {
        name: ("chain", "draw", *dims) for name, dims in FILTER_OUTPUT_DIMS.items()
    }
    assert outputs.sizes["chain"] == outputs.sizes["draw"] == 1
    pd.testing.assert_index_equal(outputs.indexes["time"], dates, check_names=False)
    state_names = ["level[beta[x1]]", "level[beta[x2]]"]
    assert list(outputs.coords["state"].values) == state_names
    assert list(outputs.coords["state_aux"].values) == state_names
    assert list(outputs.coords["observed_state"].values) == ["beta[x1]", "beta[x2]"]
    assert list(outputs.coords["observed_state_aux"].values) == ["beta[x1]", "beta[x2]"]

    diagnostics = filtered["diagnostics"]
    assert set(diagnostics.data_vars) == {
        "loglike_obs",
        "iterations",
        "evaluations",
        "search_failed",
        "invalid_direction",
        "converged",
    }
    assert all(variable.dims == ("time",) for variable in diagnostics.data_vars.values())


def test_data_with_another_number_of_dates_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"each of the 3 dates for each name: \{'y': 2\}"):
        run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": [np.zeros(3), np.zeros(2)]},
            dates=np.arange(3),
        )


def test_smoother_matches_linear_smoother_with_a_missing_date():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((3, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(3))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "params_auto_regressive": np.array([[0.7], [-0.4]]),
        "sigma_auto_regressive": np.array([0.5, 0.1]),
        "P0": np.eye(4),
    }

    rng = np.random.default_rng(0)
    X_dates = rng.normal(size=(4, 3, 2))
    y_dates = rng.normal(size=(4, 3))
    y_dates[2] = np.nan
    filtered = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={
            "X": [X_dates[0], X_dates[1], np.zeros((0, 2)), X_dates[3]],
            "y": [y_dates[0], y_dates[1], np.zeros(0), y_dates[3]],
        },
        dates=np.arange(4),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)

    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    time_varying_names = ["design", "obs_intercept"]
    observation_matrices = [
        x0,
        P0,
        c,
        pt.as_tensor(X_dates) @ d,
        T,
        pt.as_tensor(X_dates) @ Z,
        R,
        0.25 * pt.eye(3),
        Q,
    ]
    data = pt.as_tensor(y_dates)
    filter_outputs = StandardFilter(
        time_varying_names=time_varying_names, cov_jitter=0.0
    ).build_graph(data, *observation_matrices)
    smoothed_states, smoothed_covariances = DisturbanceSmoother(
        time_varying_names=time_varying_names, cov_jitter=0.0
    ).build_graph(data, observation_matrices, filter_outputs)

    assert_allclose(outputs["smoothed_states"].values, smoothed_states.eval(), atol=1e-10)
    assert_allclose(outputs["smoothed_covariances"].values, smoothed_covariances.eval(), atol=1e-10)


def test_exact_and_fisher_curvatures_differ_at_the_same_mode_for_a_noncanonical_link():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", pm.math.exp(alpha), 1, observed=np.zeros(3))

    statespace = st.LevelTrend(order=1, innovations_order=1, observed_state_names=["alpha"]).build(
        verbose=False
    )

    y = np.array([1.5, 1.2, 1.4])
    modes, variances = {}, {}
    for curvature in ["exact", "fisher"]:
        fused = FusedStateSpace(
            model=model,
            statespace=statespace,
            rv_names=["alpha"],
            frozen={},
            curvature=curvature,
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=50,
            prior_weight=0.0,
        )
        filtered = run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": [y]},
            dates=np.arange(1),
        )
        assert filtered["diagnostics/converged"].values.all()
        outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)
        modes[curvature] = outputs["filtered_states"].values[0, 0]
        variances[curvature] = outputs["filtered_covariances"].values[0, 0, 0]

    mode = modes["exact"]
    assert_allclose(modes["fisher"], mode, atol=1e-8)
    residual_term = np.sum((y - np.exp(mode)) * np.exp(mode))
    assert_allclose(mode, residual_term, atol=1e-8)
    assert_allclose(variances["fisher"], 1 / (1 + 3 * np.exp(2 * mode)))
    assert_allclose(variances["exact"], 1 / (1 + 3 * np.exp(2 * mode) - residual_term))


def test_fisher_curvature_equals_the_exact_one_for_a_canonical_link():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Poisson("y", pm.math.exp(alpha), observed=np.zeros(5, dtype=int))

    statespace = st.LevelTrend(order=1, innovations_order=1, observed_state_names=["alpha"]).build(
        verbose=False
    )

    counts = np.array([0, 0, 3, 0, 1])
    covariances = {}
    for curvature in ["exact", "fisher"]:
        fused = FusedStateSpace(
            model=model,
            statespace=statespace,
            rv_names=["alpha"],
            frozen={},
            curvature=curvature,
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=50,
            prior_weight=0.0,
        )
        filtered = run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": [counts]},
            dates=np.arange(1),
        )
        assert filtered["diagnostics/converged"].values.all()
        covariances[curvature] = filtered["posterior_predictive"]["filtered_covariances"].values

    assert_allclose(covariances["fisher"], covariances["exact"], atol=1e-8)


def test_date_out_of_iterations_is_not_converged():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", pm.math.exp(alpha), 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=1,
        prior_weight=0.0,
    )

    filtered = run_filter(
        fused=fused,
        statespace_parameters={
            "initial_level_trend": [0.0],
            "sigma_level_trend": [0.3],
            "P0": [[1.0]],
        },
        statespace_data={},
        data={"y": [np.array([2.0, 3.0, 2.5])]},
        dates=np.arange(1),
    )
    assert filtered["diagnostics/iterations"].values.tolist() == [1]
    assert filtered["diagnostics/converged"].values.tolist() == [False]
    assert filtered["diagnostics/search_failed"].values.tolist() == [False]
    assert filtered["diagnostics/invalid_direction"].values.tolist() == [False]

    # With one state equal to the parameter and no parameter noise, the filtered mean is the
    # iterate itself, and the covariance is the Laplace posterior's at that iterate, whose
    # Fisher information for three unit-variance observations of exp(alpha) is 3 exp(2 alpha).
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)
    iterate = outputs["filtered_states"].values[0, 0]
    assert_allclose(
        outputs["filtered_covariances"].values[0, 0, 0],
        1 / (1 / 1.0 + 3 * np.exp(2 * iterate)),
        atol=1e-10,
    )


def test_non_finite_observations_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"non-finite values at \(name, date\) \[\('y', 1\)\]"):
        run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": [np.array([1.0]), np.array([np.nan, 1.0])]},
            dates=np.arange(2),
        )


def test_data_names_that_are_not_model_variables_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"data names \['bogus'\] are not variables"):
        run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": [np.zeros(1)], "bogus": [np.zeros(1)]},
            dates=np.arange(1),
        )


def test_integer_observations_take_whole_numbers_and_reject_fractions():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Poisson("counts", pm.math.exp(alpha), observed=np.array([1, 2, 3]))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )
    statespace_parameters = {
        "initial_level_trend": [0.0],
        "sigma_level_trend": [0.3],
        "P0": [[1.0]],
    }

    filtered = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={"counts": [np.array([1.0, 2.0]), np.zeros(0)]},
        dates=np.arange(2),
    )
    outputs = filtered["posterior_predictive"].isel(chain=0, draw=0)
    assert_allclose(outputs["filtered_states"].values[1], outputs["predicted_states"].values[1])

    with pytest.raises(ValueError, match=r"\[\('counts', 0\)\] are not exactly representable"):
        run_filter(
            fused=fused,
            statespace_parameters=statespace_parameters,
            statespace_data={},
            data={"counts": [np.array([1.5])]},
            dates=np.arange(1),
        )


@pytest.mark.parametrize("prior_weight", [0.0, 1.0])
def test_scan_matches_driver(prior_weight):
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
        + st.MeasurementError(name="noise", observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=prior_weight,
    )
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "params_auto_regressive": np.array([[0.7], [-0.4]]),
        "sigma_auto_regressive": np.array([0.5, 0.1]),
        "sigma_noise": np.array([0.2, 0.4]),
        "P0": np.eye(4),
    }

    rng = np.random.default_rng(0)
    sizes = [3, 1, 5, 0]
    X_dates = [rng.normal(size=(size, 2)) for size in sizes]
    y_dates = [rng.poisson(lam=2.0, size=size) for size in sizes]
    driver = run_filter(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data={},
        data={"X": X_dates, "y": y_dates},
        dates=np.arange(len(sizes)),
    )

    width = max(sizes)
    X_padded = np.ones((len(sizes), width, 2))
    y_padded = np.zeros((len(sizes), width), dtype=int)
    y_mask = np.zeros((len(sizes), width), dtype=bool)
    for date, size in enumerate(sizes):
        X_padded[date, :size] = X_dates[date]
        y_padded[date, :size] = y_dates[date]
        y_mask[date, :size] = True
    sigma_level_trend = pt.matrix("sigma_level_trend", shape=(2, 1))
    scanned = scan_filter(
        fused=fused,
        statespace_parameters={**statespace_parameters, "sigma_level_trend": sigma_level_trend},
        statespace_data={},
        data={"X": X_padded, "y": y_padded},
        observed_masks={"y": y_mask},
    )
    evaluate = pytensor.function([sigma_level_trend], list(scanned.values()))
    values = dict(zip(scanned, evaluate(statespace_parameters["sigma_level_trend"]), strict=True))

    driver_outputs = driver["posterior_predictive"].isel(chain=0, draw=0)
    for name in [
        "predicted_states",
        "predicted_covariances",
        "filtered_states",
        "filtered_covariances",
        "predicted_observed_states",
        "predicted_observed_covariances",
    ]:
        assert_allclose(values[name], driver_outputs[name].values, atol=1e-10)
    assert_allclose(values["loglike_obs"], driver["diagnostics/loglike_obs"].values, atol=1e-10)
    for name in ["iterations", "search_failed", "invalid_direction", "converged"]:
        assert_allclose(values[name], driver[f"diagnostics/{name}"].values)
    assert values["converged"].all()


def test_masks_for_other_observed_variables_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"mask for exactly \['y'\]"):
        scan_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"y": np.zeros((2, 3))},
            observed_masks={"z": np.ones((2, 3), dtype=bool)},
        )


@pytest.mark.parametrize(
    "base_covariance",
    [[[1.0, 0.3], [0.3, 0.5]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
    ids=["full_rank", "isotropic", "singular"],
)
def test_loglike_gradient_matches_finite_differences(base_covariance):
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sigma = pt.vector("sigma", shape=(2,))
    initial_scale = pt.scalar("initial_scale")
    scanned = scan_filter(
        fused=fused,
        statespace_parameters={
            "initial_level_trend": np.zeros((2, 1)),
            "sigma_level_trend": sigma[:, None],
            "P0": initial_scale**2 * pt.as_tensor(np.array(base_covariance)),
        },
        statespace_data={},
        data={"X": rng.normal(size=(3, 4, 2)), "y": rng.poisson(lam=2.0, size=(3, 4))},
        observed_masks={"y": np.ones((3, 4), dtype=bool)},
    )
    loglike = scanned["loglike_obs"].sum()
    inputs = [sigma, initial_scale]
    loglike_and_gradients = pytensor.function(inputs, [loglike, *pytensor.grad(loglike, inputs)])

    point = [np.array([0.3, 0.3]), np.array(0.8)]
    _, *gradients = loglike_and_gradients(*point)

    step = 1e-6
    for position, gradient in enumerate(gradients):
        finite_differences = np.zeros_like(point[position])
        for index in np.ndindex(point[position].shape):
            shifted = [value.copy() for value in point]
            shifted[position][index] += step
            loglike_up = loglike_and_gradients(*shifted)[0]
            shifted[position][index] -= 2 * step
            loglike_down = loglike_and_gradients(*shifted)[0]
            finite_differences[index] = (loglike_up - loglike_down) / (2 * step)
        assert_allclose(gradient, finite_differences, atol=1e-6)


def test_loglike_gradient_with_parameters_sharing_a_state():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"], share_states=True
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="fisher",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sigma = pt.vector("sigma", shape=(1,))
    scanned = scan_filter(
        fused=fused,
        statespace_parameters={
            "initial_level_trend": np.zeros(1),
            "sigma_level_trend": sigma,
            "P0": np.eye(1),
        },
        statespace_data={},
        data={"X": rng.normal(size=(3, 4, 2)), "y": rng.poisson(lam=2.0, size=(3, 4))},
        observed_masks={"y": np.ones((3, 4), dtype=bool)},
    )
    loglike = scanned["loglike_obs"].sum()
    loglike_and_gradient = pytensor.function([sigma], [loglike, pytensor.grad(loglike, sigma)])

    point = np.array([0.3])
    _, gradient = loglike_and_gradient(point)

    step = 1e-6
    loglike_up = loglike_and_gradient(point + step)[0]
    loglike_down = loglike_and_gradient(point - step)[0]
    assert_allclose(gradient, (loglike_up - loglike_down) / (2 * step), atol=1e-6)
