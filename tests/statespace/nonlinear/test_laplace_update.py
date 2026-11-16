import numpy as np
import pytensor
import pytensor.tensor as pt

from numpy.testing import assert_allclose
from pytensor.gradient import hessian

from pymc_extras.statespace.nonlinear import newton_step, parameter_posterior

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 1e-8 if floatX.endswith("64") else 1e-4

NOISE_SIGMA = 0.5


def one_step_posterior(*, design, observed, predictive_mean, predictive_factor):
    """Return the parameter posterior after one exact Newton step.

    The objective is quadratic, so one step reaches the mode from any start. Starting away from
    zero exercises the prior term of the whitened gradient.
    """
    k = predictive_mean.shape[0]
    start = np.linspace(-1.0, 1.0, k).astype(floatX)
    parameters = pt.vector("parameters", shape=(k,))
    residuals = observed - design @ parameters
    logp = -0.5 * pt.sum(residuals**2) / NOISE_SIGMA**2
    predictive_mean = pt.as_tensor(predictive_mean)
    predictive_factor = pt.as_tensor(predictive_factor)

    next_whitened, whitened_curvature = newton_step(
        logp=logp,
        parameters=parameters,
        curvature=-hessian(logp, parameters),
        whitened=pt.as_tensor(start),
        predictive_mean=predictive_mean,
        predictive_factor=predictive_factor,
    )
    mean, covariance = parameter_posterior(
        whitened=next_whitened,
        predictive_mean=predictive_mean,
        predictive_factor=predictive_factor,
        whitened_curvature=whitened_curvature,
    )
    return mean.eval(), covariance.eval()


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

    mean, covariance = one_step_posterior(
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


def test_rank_deficient_predictive_keeps_posterior_in_its_span():
    design, observed, predictive_mean = regression_data(n_observations=10, n_parameters=2, seed=3)
    direction = np.array([0.6, 0.8], dtype=floatX)
    direction_sd = 1.5

    # The second column is zero, so the predictive confines the parameters to
    # predictive_mean + direction * z with z ~ N(0, direction_sd**2).
    predictive_factor = np.column_stack([direction_sd * direction, np.zeros(2)]).astype(floatX)

    mean, covariance = one_step_posterior(
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
