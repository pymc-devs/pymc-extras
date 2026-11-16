import numpy as np
import pytensor
import pytensor.tensor as pt

from numpy.testing import assert_allclose
from pytensor.gradient import hessian
from pytensor_ml.optim import zoom_line_search

from pymc_extras.statespace.nonlinear import laplace_mode, newton_step, parameter_posterior

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 1e-8 if floatX.endswith("64") else 1e-4

NOISE_SIGMA = 0.5


def one_step_posterior(*, design, observed, predictive_mean, predictive_factor):
    """Return the parameter posterior and the line-search trial count after one exact Newton step.

    The objective is quadratic, so one step reaches the mode from any start. Starting away from
    zero exercises the prior term of the whitened gradient.
    """
    k = predictive_mean.shape[0]
    start = np.linspace(-1.0, 1.0, k).astype(floatX)
    parameters = pt.vector("parameters", shape=(k,))
    whitened = pt.vector("whitened", shape=(k,))
    residuals = observed - design @ parameters
    logp = -0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2
    predictive_mean = pt.as_tensor(predictive_mean)
    predictive_factor = pt.as_tensor(predictive_factor)

    next_whitened, whitened_curvature, search = newton_step(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        whitened=whitened,
        predictive_mean=predictive_mean,
        predictive_factor=predictive_factor,
        line_search=zoom_line_search(),
    )
    mean, covariance = parameter_posterior(
        whitened=next_whitened,
        predictive_mean=predictive_mean,
        predictive_factor=predictive_factor,
        whitened_curvature=whitened_curvature,
    )
    posterior = pytensor.function([whitened], [mean, covariance, search.evaluations])
    return posterior(start)


def regression_data(*, n_observations, n_parameters, seed):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(n_observations, n_parameters)).astype(floatX)
    observed = rng.normal(size=n_observations).astype(floatX)
    predictive_mean = rng.normal(size=n_parameters).astype(floatX)
    return design, observed, predictive_mean


def test_newton_step_matches_conjugate_posterior():
    design, observed, predictive_mean = regression_data(n_observations=10, n_parameters=3, seed=1)
    root = np.random.default_rng(2).normal(size=(3, 3))
    predictive_covariance = (root @ root.T + np.eye(3)).astype(floatX)

    mean, covariance, evaluations = one_step_posterior(
        design=design,
        observed=observed,
        predictive_mean=predictive_mean,
        predictive_factor=np.linalg.cholesky(predictive_covariance),
    )

    precision = np.linalg.inv(predictive_covariance) + design.T @ design / NOISE_SIGMA**2
    expected_covariance = np.linalg.inv(precision)
    expected_mean = expected_covariance @ (
        np.linalg.solve(predictive_covariance, predictive_mean)
        + design.T @ observed / NOISE_SIGMA**2
    )
    assert_allclose(mean, expected_mean, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, expected_covariance, atol=ATOL, rtol=RTOL)
    assert evaluations == 1


def test_rank_deficient_predictive_keeps_posterior_in_its_span():
    design, observed, predictive_mean = regression_data(n_observations=10, n_parameters=2, seed=3)
    direction = np.array([0.6, 0.8], dtype=floatX)
    direction_sd = 1.5

    # The second column is zero, so the predictive confines the parameters to
    # predictive_mean + direction * z with z ~ N(0, direction_sd**2).
    predictive_factor = np.column_stack([direction_sd * direction, np.zeros(2)]).astype(floatX)

    mean, covariance, _ = one_step_posterior(
        design=design,
        observed=observed,
        predictive_mean=predictive_mean,
        predictive_factor=predictive_factor,
    )

    projected_design = design @ direction
    z_variance = 1 / (1 / direction_sd**2 + projected_design @ projected_design / NOISE_SIGMA**2)
    z_mean = z_variance * projected_design @ (observed - design @ predictive_mean) / NOISE_SIGMA**2
    assert_allclose(mean, predictive_mean + z_mean * direction, atol=ATOL, rtol=RTOL)
    assert_allclose(covariance, z_variance * np.outer(direction, direction), atol=ATOL, rtol=RTOL)


def test_line_search_decreases_objective_far_from_mode():
    nu = 3.0
    predictive_sd = 5.0
    design, observed, _ = regression_data(n_observations=20, n_parameters=2, seed=4)
    start = np.array([0.8, -0.8], dtype=floatX)

    def objective(whitened):
        residuals = observed - design @ (predictive_sd * whitened)
        return 0.5 * (nu + 1) * np.sum(np.log1p(residuals**2 / nu)) + 0.5 * whitened @ whitened

    parameters = pt.vector("parameters", shape=(2,))
    whitened = pt.vector("whitened", shape=(2,))
    residuals = observed - design @ parameters
    logp = -0.5 * (nu + 1) * pt.sum(pt.log1p(residuals**2 / nu))

    # A hundredth of the Student-t Fisher information: positive definite, but so small that the
    # full Newton step overshoots.
    curvature = 0.01 * (nu + 1) / (nu + 3) * design.T @ design
    next_whitened, _, search = newton_step(
        logp=logp,
        parameters=parameters,
        curvature=pt.as_tensor(curvature),
        whitened=whitened,
        predictive_mean=pt.zeros(2),
        predictive_factor=predictive_sd * pt.eye(2),
        line_search=zoom_line_search(),
    )
    damped, step_size = pytensor.function([whitened], [next_whitened, search.step_size])(start)

    start_residuals = observed - design @ (predictive_sd * start)
    logp_gradient = design.T @ ((nu + 1) * start_residuals / (nu + start_residuals**2))
    newton_direction = np.linalg.solve(
        np.eye(2) + predictive_sd**2 * curvature, predictive_sd * logp_gradient - start
    )

    assert objective(start + newton_direction) > objective(start)
    assert objective(damped) < objective(start)
    assert step_size < 1.0
    assert_allclose(damped, start + step_size * newton_direction, atol=ATOL, rtol=RTOL)


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
