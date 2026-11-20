import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import hessian
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace.nonlinear import (
    gauss_newton_curvature,
    implicit_mode,
    laplace_mode,
    parameter_posterior,
    state_posterior,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 1e-8 if floatX.endswith("64") else 1e-4

NOISE_SIGMA = 0.5


def regression_data(*, n_observations, n_parameters, seed):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(n_observations, n_parameters)).astype(floatX)
    observed = rng.normal(size=n_observations).astype(floatX)
    predictive_mean = rng.normal(size=n_parameters).astype(floatX)
    return design, observed, predictive_mean


def test_laplace_mode_matches_conjugate_posterior():
    design, observed, predictive_mean = regression_data(n_observations=10, n_parameters=3, seed=1)
    root = np.random.default_rng(2).normal(size=(3, 3))
    predictive_covariance = (root @ root.T + np.eye(3)).astype(floatX)

    parameters = pt.vector("parameters", shape=(3,))
    residuals = observed - design @ parameters
    logp = -0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_covariance=pt.as_tensor(predictive_covariance),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    mean, covariance = parameter_posterior(
        whitened=mode.whitened,
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
    )
    mean, covariance, iterations, evaluations, converged = pytensor.function(
        [], [mean, covariance, mode.iterations, mode.evaluations, mode.converged]
    )()

    precision = np.linalg.inv(predictive_covariance) + design.T @ design / NOISE_SIGMA**2
    expected_covariance = np.linalg.inv(precision)
    expected_mean = expected_covariance @ (
        np.linalg.solve(predictive_covariance, predictive_mean)
        + design.T @ observed / NOISE_SIGMA**2
    )
    assert_allclose(mean, expected_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=RTOL)
    assert converged
    assert iterations == 1
    assert evaluations == 1


def test_laplace_mode_handles_singular_predictive_covariance():
    design, observed, predictive_mean = regression_data(n_observations=10, n_parameters=2, seed=3)
    direction = np.array([0.6, 0.8], dtype=floatX)
    direction_sd = 1.5

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - design @ parameters
    logp = -0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2

    # The predictive confines the parameters to predictive_mean + direction * z with
    # z ~ N(0, direction_sd**2), so the posterior is the conjugate one for the scalar z.
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_covariance=pt.as_tensor(
            (direction_sd**2 * np.outer(direction, direction)).astype(floatX)
        ),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    mean, covariance = parameter_posterior(
        whitened=mode.whitened,
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
    )
    mean, covariance, converged = pytensor.function([], [mean, covariance, mode.converged])()

    projected_design = design @ direction
    z_variance = 1 / (1 / direction_sd**2 + projected_design @ projected_design / NOISE_SIGMA**2)
    z_mean = z_variance * projected_design @ (observed - design @ predictive_mean) / NOISE_SIGMA**2
    assert converged
    assert_allclose(mean, predictive_mean + z_mean * direction, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, z_variance * np.outer(direction, direction), atol=ATOL, rtol=RTOL)


def test_line_search_decreases_objective_far_from_mode():
    nu = 3.0
    predictive_sd = 5.0
    design, observed, _ = regression_data(n_observations=20, n_parameters=2, seed=4)
    predictive_mean = np.array([4.0, -4.0], dtype=floatX)

    def objective(whitened):
        residuals = observed - design @ (predictive_mean + predictive_sd * whitened)
        return 0.5 * (nu + 1) * np.sum(np.log1p(residuals**2 / nu)) + 0.5 * whitened @ whitened

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - design @ parameters

    # A hundredth of the Student-t Fisher information: positive definite, but so small that the
    # full Newton step overshoots.
    curvature = 0.01 * (nu + 1) / (nu + 3) * design.T @ design
    mode = laplace_mode(
        logp=-0.5 * (nu + 1) * pt.sum(pt.log1p(residuals**2 / nu)),
        parameters=parameters,
        curvature=pt.as_tensor(curvature),
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_covariance=predictive_sd**2 * pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=1,
    )
    damped, search_failed = pytensor.function([], [mode.whitened, mode.search_failed])()

    start_residuals = observed - design @ predictive_mean
    logp_gradient = design.T @ ((nu + 1) * start_residuals / (nu + start_residuals**2))
    newton_direction = np.linalg.solve(
        np.eye(2) + predictive_sd**2 * curvature, predictive_sd * logp_gradient
    )
    step_size = damped @ newton_direction / (newton_direction @ newton_direction)

    assert objective(newton_direction) > objective(np.zeros(2))
    assert objective(damped) < objective(np.zeros(2))
    assert not search_failed
    assert 0.0 < step_size < 1.0
    assert_allclose(damped, step_size * newton_direction, atol=ATOL, rtol=RTOL)


def test_laplace_mode_reports_failed_line_search():
    nu = 3.0
    design, observed, _ = regression_data(n_observations=20, n_parameters=2, seed=4)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - design @ parameters

    # The same overshooting curvature, with a search allowed one trial: the full step fails it.
    curvature = 0.01 * (nu + 1) / (nu + 3) * design.T @ design
    mode = laplace_mode(
        logp=-0.5 * (nu + 1) * pt.sum(pt.log1p(residuals**2 / nu)),
        parameters=parameters,
        curvature=pt.as_tensor(curvature),
        predictive_mean=pt.as_tensor(np.array([4.0, -4.0], dtype=floatX)),
        predictive_covariance=25.0 * pt.eye(2),
        line_search=zoom_line_search(max_steps=1),
        tolerance=1e-10,
        max_iterations=10,
    )
    iterations, search_failed, converged = pytensor.function(
        [], [mode.iterations, mode.search_failed, mode.converged]
    )()

    assert iterations == 1
    assert search_failed
    assert not converged


def test_laplace_mode_converges_to_student_t_mode():
    nu = 3.0
    predictive_variance = 25.0
    design, _, _ = regression_data(n_observations=20, n_parameters=2, seed=5)
    noise = np.random.default_rng(6).standard_t(df=nu, size=20)
    observed = (design @ np.array([2.0, -1.0]) + noise).astype(floatX)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - design @ parameters
    logp = -0.5 * (nu + 1) * pt.sum(pt.log1p(residuals**2 / nu))

    # The iteratively reweighted least-squares weights: positive, and dependent on the parameters,
    # so the returned curvature has to be the one at the mode.
    weights = (nu + 1) / (nu + residuals**2)
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=(design.T * weights) @ design,
        predictive_mean=pt.zeros(2),
        predictive_covariance=predictive_variance * pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
    )
    mean, covariance = parameter_posterior(
        whitened=mode.whitened,
        predictive_mean=pt.zeros(2),
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
    )
    mean, covariance, iterations, converged, search_failed = pytensor.function(
        [],
        [
            mean,
            covariance,
            mode.iterations,
            mode.converged,
            mode.search_failed,
        ],
    )()

    mode_residuals = observed - design @ mean
    logp_gradient = design.T @ ((nu + 1) * mode_residuals / (nu + mode_residuals**2))
    mode_weights = (nu + 1) / (nu + mode_residuals**2)
    expected_covariance = np.linalg.inv(
        np.eye(2) / predictive_variance + (design.T * mode_weights) @ design
    )

    assert converged
    assert not search_failed
    assert iterations > 1
    assert_allclose(logp_gradient - mean / predictive_variance, 0.0, atol=1e-6)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=1e-6)


def test_laplace_mode_reports_unconverged_at_iteration_budget():
    nu = 3.0
    design, _, _ = regression_data(n_observations=20, n_parameters=2, seed=5)
    noise = np.random.default_rng(6).standard_t(df=nu, size=20)
    observed = (design @ np.array([2.0, -1.0]) + noise).astype(floatX)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - design @ parameters
    weights = (nu + 1) / (nu + residuals**2)
    mode = laplace_mode(
        logp=-0.5 * (nu + 1) * pt.sum(pt.log1p(residuals**2 / nu)),
        parameters=parameters,
        curvature=(design.T * weights) @ design,
        predictive_mean=pt.zeros(2),
        predictive_covariance=25.0 * pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=1,
    )
    iterations, converged, search_failed = pytensor.function(
        [], [mode.iterations, mode.converged, mode.search_failed]
    )()

    assert iterations == 1
    assert not converged
    assert not search_failed


def test_laplace_mode_stays_at_predictive_mean_when_it_is_the_mode():
    predictive_mean = np.array([0.3, -0.2], dtype=floatX)

    parameters = pt.vector("parameters", shape=(2,))
    logp = -0.5 * pt.sum((parameters - predictive_mean) ** 2)
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.as_tensor(predictive_mean),
        predictive_covariance=pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    whitened, iterations, evaluations, converged, search_failed = pytensor.function(
        [], [mode.whitened, mode.iterations, mode.evaluations, mode.converged, mode.search_failed]
    )()

    assert_allclose(whitened, 0.0)
    assert iterations == 1
    assert evaluations == 1
    assert converged
    assert not search_failed


def test_laplace_mode_stops_on_indefinite_curvature():
    parameters = pt.vector("parameters", shape=(2,))
    logp = -0.25 * pt.sum((parameters**2 - 1.0) ** 2)

    # Near the origin the double-well logp is convex, so its negative Hessian is indefinite.
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.as_tensor(np.array([0.3, -0.2], dtype=floatX)),
        predictive_covariance=4.0 * pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-8,
        max_iterations=20,
    )
    iterations, invalid_direction, converged = pytensor.function(
        [], [mode.iterations, mode.invalid_direction, mode.converged]
    )()

    assert iterations == 1
    assert invalid_direction
    assert not converged


def test_gauss_newton_equals_hessian_for_normal_mean():
    design, observed, point = regression_data(n_observations=10, n_parameters=3, seed=7)

    parameters = pt.vector("parameters", shape=(3,))
    mean = design @ parameters
    pointwise_logp = -0.5 * ((observed - mean) / NOISE_SIGMA) ** 2
    curvature = gauss_newton_curvature(
        pointwise_logp=pointwise_logp,
        distribution_parameters=[mean],
        parameters=parameters,
    )

    gauss_newton, exact = pytensor.function(
        [parameters], [curvature, -hessian(pointwise_logp.sum(), parameters)]
    )(point)
    assert_allclose(gauss_newton, exact, atol=ATOL, rtol=RTOL)


def test_gauss_newton_psd_far_from_mode():
    nu = 3.0
    design, observed, _ = regression_data(n_observations=20, n_parameters=2, seed=8)
    far_point = np.array([4.0, -4.0], dtype=floatX)

    parameters = pt.vector("parameters", shape=(2,))
    mean = design @ parameters
    pointwise_logp = -0.5 * (nu + 1) * pt.log1p((observed - mean) ** 2 / nu)
    curvature = gauss_newton_curvature(
        pointwise_logp=pointwise_logp,
        distribution_parameters=[mean],
        parameters=parameters,
    )
    gauss_newton, exact = pytensor.function(
        [parameters], [curvature, -hessian(pointwise_logp.sum(), parameters)]
    )(far_point)

    # The Student-t log-likelihood is concave in the location only for residuals below sqrt(nu),
    # so each observation's curvature block is clipped at zero beyond that.
    residuals = observed - design @ far_point
    location_curvature = (nu + 1) * (nu - residuals**2) / (nu + residuals**2) ** 2
    expected = (design.T * np.maximum(location_curvature, 0.0)) @ design

    assert np.linalg.eigvalsh(exact).min() < 0
    assert_allclose(gauss_newton, expected, atol=ATOL, rtol=RTOL)


def test_gauss_newton_moves_log_sigma():
    n_observations = 50
    observed = np.random.default_rng(9).normal(loc=1.5, scale=2.0, size=n_observations)
    observed = observed.astype(floatX)
    predictive_variance = 4.0

    # The parameters are the mean and the log standard deviation, shared by every observation.
    parameters = pt.vector("parameters", shape=(2,))
    mean = pt.broadcast_to(parameters[0], (n_observations,))
    sigma = pt.broadcast_to(pt.exp(parameters[1]), (n_observations,))
    pointwise_logp = -0.5 * ((observed - mean) / sigma) ** 2 - pt.log(sigma)
    mode = laplace_mode(
        logp=pointwise_logp.sum(),
        parameters=parameters,
        curvature=gauss_newton_curvature(
            pointwise_logp=pointwise_logp,
            distribution_parameters=[mean, sigma],
            parameters=parameters,
        ),
        predictive_mean=pt.zeros(2),
        predictive_covariance=predictive_variance * pt.eye(2),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
    )
    posterior_mean, _ = parameter_posterior(
        whitened=mode.whitened,
        predictive_mean=pt.zeros(2),
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
    )
    (location, log_scale), converged = pytensor.function([], [posterior_mean, mode.converged])()

    residuals = observed - location
    scale = np.exp(log_scale)
    location_gradient = residuals.sum() / scale**2 - location / predictive_variance
    log_scale_gradient = (residuals**2 / scale**2 - 1).sum() - log_scale / predictive_variance

    assert converged
    assert log_scale > 0.5
    assert_allclose([location_gradient, log_scale_gradient], 0.0, atol=1e-6)


def test_gauss_newton_rejects_constant_distribution_parameter():
    parameters = pt.vector("parameters", shape=(2,))
    mean = pt.broadcast_to(parameters[0], (5,))
    sigma = pt.ones(5)
    pointwise_logp = -0.5 * ((np.zeros(5) - mean) / sigma) ** 2

    with pytest.raises(ValueError, match="positions \\[1\\]"):
        gauss_newton_curvature(
            pointwise_logp=pointwise_logp,
            distribution_parameters=[mean, sigma],
            parameters=parameters,
        )


def test_gauss_newton_projects_location_scale_blocks():
    scale_slope = 0.7
    design, observed, point = regression_data(n_observations=8, n_parameters=2, seed=10)

    parameters = pt.vector("parameters", shape=(2,))
    mean = design @ parameters
    sigma = pt.broadcast_to(pt.exp(scale_slope * parameters[1]), (8,))
    pointwise_logp = -0.5 * ((observed - mean) / sigma) ** 2 - pt.log(sigma)
    curvature = gauss_newton_curvature(
        pointwise_logp=pointwise_logp,
        distribution_parameters=[mean, sigma],
        parameters=parameters,
    )
    gauss_newton = pytensor.function([parameters], curvature)(point)

    # Each observation's negative Hessian in (mean, sigma) has a negative determinant, so every
    # block is indefinite and its projection keeps only the positive eigenvalue.
    residuals = observed - design @ point
    scale = np.exp(scale_slope * point[1])
    expected = np.zeros((2, 2))
    for design_row, residual in zip(design, residuals):
        negative_hessian = np.array(
            [
                [1 / scale**2, 2 * residual / scale**3],
                [2 * residual / scale**3, 3 * residual**2 / scale**4 - 1 / scale**2],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(negative_hessian)
        projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        jacobian = np.array([design_row, [0.0, scale_slope * scale]])
        expected += jacobian.T @ projected @ jacobian

    assert_allclose(gauss_newton, expected, atol=ATOL, rtol=RTOL)


def kalman_update(
    *,
    state_mean,
    state_covariance,
    design,
    offset,
    parameter_noise,
    observation_design,
    observed,
    noise_sigma,
):
    """Return the exact state posterior for observations that are linear-Gaussian in the parameters.

    ``observed ~ N(observation_design @ theta, noise_sigma**2 I)`` with
    ``theta = design @ x + offset + eta`` and ``eta ~ N(0, parameter_noise)``.
    """
    full_design = observation_design @ design
    innovation_covariance = (
        full_design @ state_covariance @ full_design.T
        + observation_design @ parameter_noise @ observation_design.T
        + noise_sigma**2 * np.eye(len(observed))
    )
    gain = state_covariance @ full_design.T @ np.linalg.inv(innovation_covariance)
    innovation = observed - observation_design @ (design @ state_mean + offset)
    return state_mean + gain @ innovation, state_covariance - gain @ full_design @ state_covariance


def lifted_posterior(*, logp, parameters, state_mean, state_covariance, design, offset, noise):
    """Return the state posterior from laplace_mode followed by state_posterior."""
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.as_tensor(design @ state_mean + offset),
        predictive_covariance=pt.as_tensor(design @ state_covariance @ design.T + noise),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    mean, covariance = state_posterior(
        whitened=mode.whitened,
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
        state_mean=pt.as_tensor(state_mean),
        state_covariance=pt.as_tensor(state_covariance),
        design=pt.as_tensor(design),
    )
    return pytensor.function([], [mean, covariance])()


def test_state_posterior_matches_kalman_update_with_parameter_noise():
    rng = np.random.default_rng(11)
    state_mean = rng.normal(size=3).astype(floatX)
    root = rng.normal(size=(3, 3))
    state_covariance = (root @ root.T + np.eye(3)).astype(floatX)
    design = rng.normal(size=(2, 3)).astype(floatX)
    offset = rng.normal(size=2).astype(floatX)
    parameter_noise = np.diag([0.3, 0.5]).astype(floatX)
    observation_design, observed, _ = regression_data(n_observations=6, n_parameters=2, seed=12)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - observation_design @ parameters
    mean, covariance = lifted_posterior(
        logp=-0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2,
        parameters=parameters,
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        noise=parameter_noise,
    )

    expected_mean, expected_covariance = kalman_update(
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        parameter_noise=parameter_noise,
        observation_design=observation_design,
        observed=observed,
        noise_sigma=NOISE_SIGMA,
    )
    assert_allclose(mean, expected_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=RTOL)


def test_state_posterior_handles_parameters_sharing_a_state():
    state_mean = np.array([0.4], dtype=floatX)
    state_covariance = np.array([[2.0]], dtype=floatX)
    # Both parameters load on the one state with no parameter noise, so their predictive is rank 1.
    design = np.array([[0.6], [0.8]], dtype=floatX)
    offset = np.zeros(2, dtype=floatX)
    parameter_noise = np.zeros((2, 2), dtype=floatX)
    observation_design, observed, _ = regression_data(n_observations=6, n_parameters=2, seed=13)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - observation_design @ parameters
    mean, covariance = lifted_posterior(
        logp=-0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2,
        parameters=parameters,
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        noise=parameter_noise,
    )

    expected_mean, expected_covariance = kalman_update(
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        parameter_noise=parameter_noise,
        observation_design=observation_design,
        observed=observed,
        noise_sigma=NOISE_SIGMA,
    )
    assert_allclose(mean, expected_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=RTOL)


def test_state_posterior_unchanged_by_uninformative_observations():
    rng = np.random.default_rng(14)
    state_mean = rng.normal(size=3).astype(floatX)
    root = rng.normal(size=(3, 3))
    state_covariance = (root @ root.T + np.eye(3)).astype(floatX)
    design = rng.normal(size=(2, 3)).astype(floatX)

    parameters = pt.vector("parameters", shape=(2,))
    mean, covariance = lifted_posterior(
        logp=0.0 * pt.sum(parameters),
        parameters=parameters,
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=np.zeros(2, dtype=floatX),
        noise=np.diag([0.3, 0.5]).astype(floatX),
    )

    assert_allclose(mean, state_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, state_covariance, atol=ATOL, rtol=RTOL)


def test_state_posterior_handles_parameter_with_zero_predictive_variance():
    state_mean = np.array([0.4, -0.3], dtype=floatX)
    state_covariance = np.array([[2.0, 0.0], [0.0, 1.5]], dtype=floatX)
    # The second parameter loads on no state and has no noise, so it is known exactly and its
    # column of the predictive factor is exactly zero.
    design = np.array([[1.0, 0.5], [0.0, 0.0]], dtype=floatX)
    offset = np.array([0.0, 0.7], dtype=floatX)
    parameter_noise = np.diag([0.2, 0.0]).astype(floatX)
    observation_design, observed, _ = regression_data(n_observations=6, n_parameters=2, seed=15)

    parameters = pt.vector("parameters", shape=(2,))
    residuals = observed - observation_design @ parameters
    mean, covariance = lifted_posterior(
        logp=-0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2,
        parameters=parameters,
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        noise=parameter_noise,
    )

    expected_mean, expected_covariance = kalman_update(
        state_mean=state_mean,
        state_covariance=state_covariance,
        design=design,
        offset=offset,
        parameter_noise=parameter_noise,
        observation_design=observation_design,
        observed=observed,
        noise_sigma=NOISE_SIGMA,
    )
    assert_allclose(mean, expected_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=RTOL)


def test_gauss_newton_partial_in_dependent_distribution_parameters():
    scale_slope = 0.5
    design, observed, point = regression_data(n_observations=8, n_parameters=2, seed=16)

    # The scale is computed from the mean, so the curvature block has to hold partial derivatives
    # in (mean, sigma) while the Jacobian carries sigma's dependence on the mean.
    parameters = pt.vector("parameters", shape=(2,))
    mean = design @ parameters
    sigma = pt.exp(scale_slope * mean)
    pointwise_logp = -0.5 * ((observed - mean) / sigma) ** 2 - pt.log(sigma)
    curvature = gauss_newton_curvature(
        pointwise_logp=pointwise_logp,
        distribution_parameters=[mean, sigma],
        parameters=parameters,
    )
    gauss_newton = pytensor.function([parameters], curvature)(point)

    location = design @ point
    residuals = observed - location
    scale = np.exp(scale_slope * location)
    expected = np.zeros((2, 2))
    for design_row, residual, row_scale in zip(design, residuals, scale):
        negative_hessian = np.array(
            [
                [1 / row_scale**2, 2 * residual / row_scale**3],
                [2 * residual / row_scale**3, 3 * residual**2 / row_scale**4 - 1 / row_scale**2],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(negative_hessian)
        projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        jacobian = np.array([design_row, scale_slope * row_scale * design_row])
        expected += jacobian.T @ projected @ jacobian

    assert_allclose(gauss_newton, expected, atol=ATOL, rtol=RTOL)


def test_gauss_newton_rejects_logp_outside_distribution_parameters():
    parameters = pt.vector("parameters", shape=(2,))
    mean = pt.broadcast_to(parameters[0], (5,))
    pointwise_logp = -0.5 * mean**2 + parameters[1]

    with pytest.raises(ValueError, match="other than through distribution_parameters"):
        gauss_newton_curvature(
            pointwise_logp=pointwise_logp,
            distribution_parameters=[mean],
            parameters=parameters,
        )


def test_laplace_mode_reports_invalid_direction_after_a_step():
    parameters = pt.vector("parameters", shape=(1,))
    logp = -0.5 * pt.sum((parameters - 3.0) ** 2)

    # The curvature is positive at the predictive mean and turns negative a short step away, so the
    # first step is taken and the direction at the new iterate is invalid.
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=pt.atleast_2d(9.0 - 200.0 * parameters[0] ** 2),
        predictive_mean=pt.zeros(1),
        predictive_covariance=pt.eye(1),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    whitened, iterations, invalid_direction, converged = pytensor.function(
        [], [mode.whitened, mode.iterations, mode.invalid_direction, mode.converged]
    )()

    assert_allclose(whitened, [0.3], atol=ATOL, rtol=RTOL)
    assert iterations == 1
    assert invalid_direction
    assert not converged


def test_laplace_mode_factor_has_orthogonal_columns():
    root = np.random.default_rng(17).normal(size=(3, 3))
    predictive_covariance = (root @ root.T).astype(floatX)

    parameters = pt.vector("parameters", shape=(3,))
    logp = -0.5 * pt.sum(parameters**2)
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=pt.zeros(3),
        predictive_covariance=pt.as_tensor(predictive_covariance),
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
    )
    factor = pytensor.function([], mode.predictive_factor)()

    gram = factor.T @ factor
    assert_allclose(gram - np.diag(np.diag(gram)), 0.0, atol=ATOL)
    assert_allclose(factor @ factor.T, predictive_covariance, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "base_covariance",
    [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
    ids=["isotropic", "singular"],
)
def test_implicit_gradient_matches_finite_differences(base_covariance):
    rng = np.random.default_rng(19)
    design = rng.normal(scale=0.5, size=(6, 2)).astype(floatX)
    weights = rng.normal(size=2).astype(floatX)

    predictive_mean = pt.vector("predictive_mean", shape=(2,))
    scale = pt.scalar("scale")
    observed = pt.vector("observed", shape=(6,))
    predictive_covariance = scale**2 * pt.as_tensor(np.array(base_covariance, dtype=floatX))

    parameters = pt.vector("parameters", shape=(2,))
    logp = -0.5 * pt.sum((observed - pt.exp(design @ parameters)) ** 2) / NOISE_SIGMA**2
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
        line_search=zoom_line_search(),
        tolerance=1e-12,
        max_iterations=50,
    )
    differentiable_mode = implicit_mode(
        mode=predictive_mean + mode.predictive_factor @ mode.whitened,
        logp=logp,
        parameters=parameters,
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
    )
    loss = weights @ differentiable_mode
    inputs = [predictive_mean, scale, observed]
    loss_and_gradients = pytensor.function(inputs, [loss, *pytensor.grad(loss, inputs)])

    point = [
        np.array([0.2, -0.3], dtype=floatX),
        np.array(0.8, dtype=floatX),
        rng.normal(loc=1.5, size=6).astype(floatX),
    ]
    _, *gradients = loss_and_gradients(*point)

    step = 1e-5
    for position, gradient in enumerate(gradients):
        finite_differences = np.zeros_like(point[position])
        for index in np.ndindex(point[position].shape):
            shifted = [value.copy() for value in point]
            shifted[position][index] += step
            loss_up = loss_and_gradients(*shifted)[0]
            shifted[position][index] -= 2 * step
            loss_down = loss_and_gradients(*shifted)[0]
            finite_differences[index] = (loss_up - loss_down) / (2 * step)
        assert_allclose(gradient, finite_differences, atol=1e-7)


def test_implicit_mode_keeps_the_value_of_an_unconverged_mode():
    rng = np.random.default_rng(23)
    design = rng.normal(scale=0.5, size=(6, 2)).astype(floatX)
    observed = rng.normal(loc=3.0, size=6).astype(floatX)
    predictive_mean = pt.as_tensor(np.zeros(2, dtype=floatX))
    predictive_covariance = pt.as_tensor(np.eye(2, dtype=floatX))

    parameters = pt.vector("parameters", shape=(2,))
    logp = -0.5 * pt.sum((observed - pt.exp(design @ parameters)) ** 2) / NOISE_SIGMA**2
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
        line_search=zoom_line_search(),
        tolerance=1e-12,
        max_iterations=1,
    )
    iterate = predictive_mean + mode.predictive_factor @ mode.whitened
    differentiable_mode = implicit_mode(
        mode=iterate,
        logp=logp,
        parameters=parameters,
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
    )
    iterate, differentiable_mode, converged = pytensor.function(
        [], [iterate, differentiable_mode, mode.converged]
    )()

    assert not converged
    np.testing.assert_array_equal(differentiable_mode, iterate)
