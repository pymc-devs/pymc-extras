import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import hessian
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace.nonlinear import (
    gauss_newton_curvature,
    laplace_mode,
    parameter_posterior,
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
