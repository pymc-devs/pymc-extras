import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from numpy.testing import assert_allclose
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace import structural as st
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    observed_state_names,
    run_filter,
    scan_filter,
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
        curvature="gauss_newton",
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
        curvature="gauss_newton",
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
        curvature="gauss_newton",
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
        curvature="gauss_newton",
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


def test_model_logp_matches_build_statespace_graph():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=beta, sigma=0.5, observed=np.zeros(2))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )
    linear = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=["x1", "x2"])
        + st.MeasurementError(name="noise", observed_state_names=["x1", "x2"])
    ).build(verbose=False)

    y = np.random.default_rng(0).normal(size=(6, 2))
    with pm.Model() as fused_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        fused.build_statespace_graph(data={"y": list(y)}, dates=np.arange(6))
    with pm.Model() as linear_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        pm.Deterministic("sigma_noise", pt.as_tensor([0.5, 0.5]))
        linear.build_statespace_graph(data=y)

    fused_logp, linear_logp = fused_model.compile_logp(), linear_model.compile_logp()
    point = {
        "initial_level_trend": np.array([[0.3], [-0.2]]),
        "sigma_level_trend_log__": np.log([[0.4], [0.2]]),
    }
    assert_allclose(fused_logp(point), linear_logp(point), atol=1e-10)


def test_dlogp_vanishes_at_stored_mode():
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    beta_path = np.cumsum(np.array([0.2, 0.4]) * rng.normal(size=(20, 2)), axis=0)
    X_dates = rng.normal(scale=0.5, size=(20, 20, 2))
    y_dates = rng.poisson(lam=np.exp(np.einsum("tnk,tk->tn", X_dates, beta_path)))
    with pm.Model() as fused_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        fused.build_statespace_graph(
            data={"X": list(X_dates), "y": list(y_dates)}, dates=np.arange(20)
        )

    # The mode find_MAP reached with BFGS at tol=1e-12. Shifting both log scales by 0.01 moves
    # their gradients to 0.04 or more.
    mode = {
        "initial_level_trend": np.array([[0.063079986435], [0.019299647647]]),
        "sigma_level_trend_log__": np.log([[0.185623519401], [0.140406251664]]),
    }
    dlogp = fused_model.compile_dlogp(jacobian=False)
    assert_allclose(dlogp(mode), 0.0, atol=1e-6)


def test_conditional_posterior_draws_match_run_filter():
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [3, 1, 0, 4]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.poisson(lam=2.0, size=size) for size in sizes],
    }
    parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.array([[0.3], [0.2]]),
        "P0": np.eye(2),
    }
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(4))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    n_draws = 4000
    idata["posterior"] = xr.Dataset(
        {
            name: (
                ("chain", "draw", *fused.param_dims[name]),
                np.broadcast_to(value, (1, n_draws, *value.shape)),
            )
            for name, value in parameters.items()
        }
    )
    conditional = fused.sample_conditional_posterior(idata, random_seed=1, progressbar=False)
    reference = run_filter(
        fused=fused,
        statespace_parameters=parameters,
        statespace_data={},
        data=data,
        dates=np.arange(4),
    )["posterior_predictive"].isel(chain=0, draw=0)

    for kind in ["filtered", "smoothed"]:
        draws = conditional[f"{kind}_posterior"].values[0]
        mean = reference[f"{kind}_states"].values
        sd = np.sqrt(np.diagonal(reference[f"{kind}_covariances"].values, axis1=1, axis2=2))
        assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * sd / np.sqrt(n_draws))
        assert_allclose(draws.std(axis=0), sd, rtol=0.05)


def test_conditional_observations_follow_the_observation_model_at_each_draw():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=X @ beta, sigma=sigma, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={"sigma": 1e-8},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [3, 1, 2]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.normal(size=size) for size in sizes],
    }
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(3))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    theta = rng.normal(size=(1, 5, 3, 2))
    conditional = xr.Dataset(
        {"smoothed_posterior_observed": (("chain", "draw", "time", "observed_state"), theta)},
        coords={"chain": [0], "draw": np.arange(5), "time": np.arange(3)},
    )
    draws = fused.sample_conditional_observations(
        idata=idata, conditional=conditional, kind="smoothed", random_seed=1
    )["y"].values

    for date, (X_date, size) in enumerate(zip(data["X"], sizes, strict=True)):
        assert_allclose(draws[:, :, date, :size], theta[:, :, date] @ X_date.T, atol=1e-6)
        assert np.isnan(draws[:, :, date, size:]).all()


def test_conditional_observations_carry_the_observation_noise():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=X @ beta, sigma=sigma, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={"sigma": 0.5},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    data = {"X": [rng.normal(size=(2, 2))], "y": [rng.normal(size=2)]}
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(1))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    n_draws = 20000
    theta = np.array([0.4, -0.7])
    conditional = xr.Dataset(
        {
            "smoothed_posterior_observed": (
                ("chain", "draw", "time", "observed_state"),
                np.broadcast_to(theta, (1, n_draws, 1, 2)),
            )
        },
        coords={"chain": [0], "draw": np.arange(n_draws), "time": np.arange(1)},
    )
    draws = fused.sample_conditional_observations(
        idata=idata, conditional=conditional, kind="smoothed", random_seed=1
    )["y"].values[0, :, 0]

    assert np.all(np.abs(draws.mean(axis=0) - data["X"][0] @ theta) < 4 * 0.5 / np.sqrt(n_draws))
    assert_allclose(draws.std(axis=0), 0.5, rtol=0.03)


@pytest.mark.xfail(
    reason="statespace_matrices cannot convert float32 parameter values to the statespace's types",
    raises=TypeError,
    strict=True,
)
@pytensor.config.change_flags(floatX="float32")
def test_large_integer_observations_survive_packing_in_float32():
    with pm.Model(coords={"coef": ["x1"]}) as model:
        X = pm.Data("X", np.zeros((2, 1)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(2, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    # 2**24 + 1 is the smallest positive integer float32 cannot represent.
    counts = np.array([2**24 + 1, 2**25 + 3])
    with pm.Model(coords=fused.coords) as fused_model:
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(1), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(
            data={"X": [np.ones((2, 1))], "y": [counts]}, dates=np.arange(1)
        )

    observations = fused_model.rvs_to_values[fused_model["obs"]].data
    np.testing.assert_array_equal(observations[0], counts)


def test_prior_predictive_observations_have_the_statespace_moments():
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    design = np.array([[1.0, 0.5], [1.0, -1.0], [0.0, 1.0]])
    mean = np.array([0.5, -0.3])
    initial_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    sigma = np.array([0.4, 0.3])
    with pm.Model(coords=fused.coords):
        pm.Deterministic("initial_level_trend", pt.as_tensor(mean[:, None]))
        pm.Deterministic("sigma_level_trend", pt.as_tensor(sigma[:, None]))
        pm.Deterministic("P0", pt.as_tensor(initial_covariance))
        fused.build_statespace_graph(
            data={"X": [design, design], "y": [np.zeros(3, dtype=int)] * 2}, dates=np.arange(2)
        )
        draws = pm.sample_prior_predictive(draws=40_000, random_seed=1).prior_predictive["obs"]

    # The coefficients on the first date have the initial covariance, and on the second the
    # random walk has added its variance, so each station's count has a lognormal Poisson mean.
    for date, covariance in enumerate([initial_covariance, initial_covariance + np.diag(sigma**2)]):
        expected = np.exp(
            design @ mean + 0.5 * np.einsum("ik,kl,il->i", design, covariance, design)
        )
        assert_allclose(draws.values[0, :, date].mean(axis=0), expected, rtol=0.02)


def test_per_date_logp_with_several_observed_variables_matches_run_filter():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        x = pm.Data("x", np.zeros(3))
        pm.Poisson("counts", mu=pm.math.exp(alpha + x), observed=np.zeros(3, dtype=int))
        pm.Normal("sizes", mu=alpha, sigma=0.5, observed=np.zeros(2))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="exact",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )
    rng = np.random.default_rng(0)
    counts_sizes, sizes_sizes = [3, 1, 0, 2], [2, 0, 1, 2]
    data = {
        "x": [rng.normal(size=size) for size in counts_sizes],
        "counts": [rng.poisson(lam=2.0, size=size) for size in counts_sizes],
        "sizes": [rng.normal(size=size) for size in sizes_sizes],
    }
    parameters = {"initial_level_trend": [0.2], "sigma_level_trend": [0.3], "P0": [[1.0]]}

    with pm.Model(coords=fused.coords) as fused_model:
        for name, value in parameters.items():
            pm.Deterministic(name, pt.as_tensor(np.array(value)))
        fused.build_statespace_graph(data=data, dates=np.arange(4))
    [per_date_logp] = fused_model.compile_logp(vars=[fused_model["obs"]], sum=False)({})

    reference = run_filter(
        fused=fused,
        statespace_parameters=parameters,
        statespace_data={},
        data=data,
        dates=np.arange(4),
    )
    assert_allclose(per_date_logp, reference["diagnostics/loglike_obs"].values, atol=1e-10)


def test_model_gradient_matches_finite_differences():
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
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [6, 4, 0, 5, 6]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.poisson(lam=3.0, size=size) for size in sizes],
    }
    with pm.Model(coords=fused.coords) as fused_model:
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", sigma=0.5, dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(len(sizes)))

    # PyMC rewrites the log-probability before differentiating it, which is what this checks.
    point = {
        "initial_level_trend": np.array([[0.4], [-0.3]]),
        "sigma_level_trend_log__": np.log([[0.3], [0.2]]),
    }
    logp, dlogp = fused_model.compile_logp(), fused_model.compile_dlogp()
    flat = np.concatenate([value.ravel() for value in point.values()])

    def logp_at(vector):
        return logp(
            {
                "initial_level_trend": vector[:2].reshape(2, 1),
                "sigma_level_trend_log__": vector[2:].reshape(2, 1),
            }
        )

    step = 1e-6
    finite_differences = [
        (logp_at(flat + step * unit) - logp_at(flat - step * unit)) / (2 * step)
        for unit in np.eye(flat.size)
    ]
    assert_allclose(dlogp(point), finite_differences, atol=1e-5)
