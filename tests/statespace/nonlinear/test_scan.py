import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace import structural as st
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    observed_state_names,
    run_filter,
    scan_filter,
)


def test_scan_matches_driver():
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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
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
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})

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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["beta"], frozen={})

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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
    )
    loglike = scanned["loglike_obs"].sum()
    loglike_and_gradient = pytensor.function([sigma], [loglike, pytensor.grad(loglike, sigma)])

    point = np.array([0.3])
    _, gradient = loglike_and_gradient(point)

    step = 1e-6
    loglike_up = loglike_and_gradient(point + step)[0]
    loglike_down = loglike_and_gradient(point - step)[0]
    assert_allclose(gradient, (loglike_up - loglike_down) / (2 * step), atol=1e-6)
