import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace import structural as st
from pymc_extras.statespace.filters.kalman_filter import BaseFilter, StandardFilter
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    filter_step,
    gauss_newton_curvature,
    observed_state_names,
    run_filter,
)


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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    X_data = pt.matrix("X")
    pointwise_logp, distribution_parameters = fused.observation_logp(
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
    filtered_mean, filtered_covariance, next_mean, next_covariance, mode = filter_step(
        logp=pointwise_logp.sum(),
        curvature=gauss_newton_curvature(
            pointwise_logp=pointwise_logp,
            distribution_parameters=distribution_parameters,
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


@pytest.mark.parametrize("curvature", ["gauss_newton", "exact"])
def test_filter_matches_standard_filter_over_ragged_dates(curvature):
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=X @ beta, sigma=0.5, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=names["beta"])
        + st.Autoregressive(order=1, observed_state_names=names["beta"])
    ).build(verbose=False)
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})
    statespace_parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.full((2, 1), 0.3),
        "params_auto_regressive": np.array([[0.7], [-0.4]]),
        "sigma_auto_regressive": np.array([0.5, 0.1]),
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
        curvature=curvature,
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )

    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    a = pt.vector("a", shape=(4,))
    P = pt.matrix("P", shape=(4, 4))
    y = pt.vector("y")
    X_data = pt.matrix("X")
    reference_filter = StandardFilter(cov_jitter=0.0)
    reference_filtered_mean, reference_filtered_covariance, *_ = reference_filter.update(
        a=a,
        P=P,
        y=y,
        d=X_data @ d,
        Z=X_data @ Z,
        H=0.25 * pt.eye(y.shape[0]),
        nan_mask=pt.zeros_like(y, dtype=bool),
    )
    reference_step = pytensor.function(
        [a, P, y, X_data],
        [
            reference_filtered_mean,
            reference_filtered_covariance,
            *BaseFilter.predict(
                a=reference_filtered_mean, P=reference_filtered_covariance, c=c, T=T, R=R, Q=Q
            ),
        ],
    )
    mean, covariance = x0.eval(), P0.eval()
    for date in range(len(sizes)):
        assert_allclose(filtered["predicted_mean"][date], mean, atol=1e-10)
        assert_allclose(filtered["predicted_covariance"][date], covariance, atol=1e-10)
        filtered_mean, filtered_covariance, mean, covariance = reference_step(
            mean, covariance, y_dates[date], X_dates[date]
        )
        assert_allclose(filtered["filtered_mean"][date], filtered_mean, atol=1e-10)
        assert_allclose(filtered["filtered_covariance"][date], filtered_covariance, atol=1e-10)
    assert filtered["converged"].all()


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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )

    _, _, c, _, T, _, R, _, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data={}
    )
    assert_allclose(filtered["filtered_mean"][1], filtered["predicted_mean"][1])
    assert_allclose(filtered["filtered_covariance"][1], filtered["predicted_covariance"][1])
    next_mean, next_covariance = BaseFilter.predict(
        a=filtered["filtered_mean"][1], P=filtered["filtered_covariance"][1], c=c, T=T, R=R, Q=Q
    )
    assert_allclose(filtered["predicted_mean"][2], next_mean.eval())
    assert_allclose(filtered["predicted_covariance"][2], next_covariance.eval())


def test_time_varying_statespace_matrices_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Regression(
        name="exog", state_names=["x1"], observed_state_names=["alpha"]
    ).build(verbose=False)
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["alpha"], frozen={})

    with pytest.raises(ValueError, match=r"Statespace matrices \['Z'\] vary over time"):
        run_filter(
            fused=fused,
            statespace_parameters={"beta_exog": [0.5], "P0": [[1.0]]},
            statespace_data={"data_exog": np.ones((2, 1))},
            data={"y": [np.zeros(3), np.zeros(2)]},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
        )


def test_gauss_newton_without_distribution_parameters_raises():
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
    )

    with pytest.raises(ValueError, match=r"Observed variables \['w'\] have no distribution"):
        run_filter(
            fused=fused,
            statespace_parameters={
                "initial_level_trend": [0.0],
                "sigma_level_trend": [0.3],
                "P0": [[1.0]],
            },
            statespace_data={},
            data={"w": [np.zeros((4, 2))]},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
        )


def test_gauss_newton_ignores_observations_that_do_not_depend_on_the_states():
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    assert_allclose(filtered["filtered_mean"], [[0.75]])
    assert_allclose(filtered["filtered_covariance"], [[[0.25]]])
