from typing import NamedTuple

import pytensor
import pytensor.tensor as pt

from pytensor.assumptions import assume
from pytensor.gradient import grad
from pytensor.graph.replace import graph_replace
from pytensor.scan.utils import until
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch, SearchResult, search_along

from pymc_extras.statespace.filters.utilities import quad_form_sym


class LaplaceMode(NamedTuple):
    r"""
    The mode of the parameters' log posterior in whitened coordinates, and how it was found.

    Attributes
    ----------
    whitened : TensorVariable
        Final iterate :math:`\hat u`, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L` of the predictive covariance, shape ``(k, k)``.
    whitened_curvature : TensorVariable
        :math:`B` at :math:`\hat u`, shape ``(k, k)``, marked positive definite.
    iterations : TensorVariable
        Number of loop iterations, each of which searches along one Newton direction.
    evaluations : TensorVariable
        Line-search trials summed over all iterations.
    search_failed : TensorVariable
        Whether any line search along a direction larger than the tolerance failed.
    invalid_direction : TensorVariable
        Whether a Newton direction was not finite. A ``curvature`` that is not positive
        semidefinite makes :math:`B` indefinite and produces one, and then ``whitened_curvature``
        is not positive definite either.
    converged : TensorVariable
        Whether the Newton direction at :math:`\hat u` is within the tolerance and the last search
        did not fail.
    """

    whitened: TensorVariable
    predictive_factor: TensorVariable
    whitened_curvature: TensorVariable
    iterations: TensorVariable
    evaluations: TensorVariable
    search_failed: TensorVariable
    invalid_direction: TensorVariable
    converged: TensorVariable


def laplace_mode(
    *,
    logp: TensorVariable,
    parameters: TensorVariable,
    curvature: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_covariance: TensorVariable,
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> LaplaceMode:
    r"""
    Find the mode of the parameters' log posterior by damped Newton steps from the predictive mean.

    The parameters :math:`\theta` have the Gaussian predictive :math:`N(\mu, L L^\top)` and the
    observations contribute :math:`\ell(\theta)`. Writing :math:`\theta = \mu + L u`, the iterations
    minimize :math:`J(u) = -\ell(\mu + L u) + \tfrac{1}{2} u^\top u`, whose curvature
    :math:`B = I + L^\top \Lambda L` is at least the identity whenever :math:`\Lambda` is positive
    semidefinite [1]_. Each iteration searches along the Newton direction
    :math:`d = -B^{-1} \nabla J(u)`, starting from the full step.

    The iterations run inside the graph as a ``scan`` that stops when the largest entry of
    :math:`d` at the current iterate is at most ``tolerance``, when a line search along a larger
    direction fails, when :math:`d` is not finite, or after ``max_iterations`` iterations. A
    direction within the tolerance is not taken. A ``scan`` that stops early cannot be compiled to
    JAX, so this runs on the numba and C backends. The predictive covariance is factored once, by
    an eigendecomposition, so it may be singular.

    Parameters
    ----------
    logp : TensorVariable
        Scalar log-likelihood of the observations, a graph of ``parameters``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``, that ``logp`` and ``curvature`` are built on.
    curvature : TensorVariable
        Positive semidefinite approximation :math:`\Lambda` to the negative Hessian of ``logp``, a
        graph of ``parameters``, shape ``(k, k)``.
    predictive_mean : TensorVariable
        Predictive mean :math:`\mu` of the parameters, shape ``(k,)``.
    predictive_covariance : TensorVariable
        Symmetric positive semidefinite predictive covariance of the parameters, shape ``(k, k)``.
    line_search : LineSearch
        Search along each Newton direction, for example
        :func:`pytensor_ml.optim.zoom_line_search`.
    tolerance : float
        Largest absolute entry of the undamped whitened Newton direction at which the iterations
        stop.
    max_iterations : int
        Most loop iterations.

    Returns
    -------
    mode : LaplaceMode
        The final iterate, the curvature there, and the iteration diagnostics.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Section 3.4.3.
    """
    eigenvalues, eigenvectors = pt.linalg.eigh(predictive_covariance)
    predictive_factor = eigenvectors * pt.sqrt(pt.maximum(eigenvalues, 0.0))

    gradient = grad(logp, parameters)

    def newton_terms_at(whitened):
        at_iterate = {parameters: predictive_mean + predictive_factor @ whitened}
        logp_at, gradient_at, curvature_at = graph_replace([logp, gradient, curvature], at_iterate)
        objective = -logp_at + 0.5 * whitened @ whitened
        objective_gradient = whitened - predictive_factor.T @ gradient_at
        whitened_curvature = assume(
            pt.eye(whitened.shape[0]) + predictive_factor.T @ curvature_at @ predictive_factor,
            positive_definite=True,
        )
        direction = pt.linalg.solve(whitened_curvature, -objective_gradient)
        return objective, objective_gradient, whitened_curvature, direction

    def iterate(whitened, direction):
        objective, objective_gradient, *_ = newton_terms_at(whitened)
        search = search_along(
            loss=objective,
            parameters=[whitened],
            gradients=[objective_gradient],
            directions=[direction],
            line_search=line_search,
        )
        moving = pt.max(pt.abs(direction)) > tolerance
        next_whitened = pt.switch(moving, whitened + search.step_size * direction, whitened)
        failed = search.failed & moving

        *_, next_curvature, next_direction = newton_terms_at(next_whitened)
        invalid = ~pt.all(pt.isfinite(next_direction))
        stop = (pt.max(pt.abs(next_direction)) <= tolerance) | failed | invalid
        outputs = (
            next_whitened,
            next_direction,
            next_curvature,
            search.evaluations,
            failed,
            invalid,
        )
        return outputs, until(stop)

    initial_whitened = pt.zeros_like(predictive_mean)
    *_, initial_direction = newton_terms_at(initial_whitened)

    (
        whitened_trace,
        direction_trace,
        curvature_trace,
        evaluations,
        failures,
        invalid_directions,
    ) = pytensor.scan(
        iterate,
        outputs_info=[initial_whitened, initial_direction, None, None, None, None],
        n_steps=max_iterations,
        name="laplace_mode",
        return_updates=False,
    )

    return LaplaceMode(
        whitened=whitened_trace[-1],
        predictive_factor=predictive_factor,
        whitened_curvature=assume(curvature_trace[-1], positive_definite=True),
        iterations=whitened_trace.shape[0],
        evaluations=evaluations.sum(),
        search_failed=failures.any(),
        invalid_direction=invalid_directions.any(),
        converged=(pt.max(pt.abs(direction_trace[-1])) <= tolerance) & ~failures[-1],
    )


def newton_step(
    *,
    logp: TensorVariable,
    parameters: TensorVariable,
    curvature: TensorVariable,
    whitened: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_factor: TensorVariable,
    line_search: LineSearch,
) -> tuple[TensorVariable, TensorVariable, SearchResult]:
    r"""
    Take one damped Newton step on the parameters' log posterior in whitened coordinates.

    The parameters :math:`\theta` have the Gaussian predictive :math:`N(\mu, L L^\top)` and the
    observations contribute :math:`\ell(\theta)`. Writing :math:`\theta = \mu + L u`, the step
    minimizes :math:`J(u) = -\ell(\mu + L u) + \tfrac{1}{2} u^\top u`, whose curvature
    :math:`B = I + L^\top \Lambda L` is at least the identity whenever :math:`\Lambda` is positive
    semidefinite [1]_. :math:`\Lambda` may be singular, and :math:`L` may have zero columns where the
    predictive is singular. ``line_search`` sets the step length along the Newton direction,
    starting from the full step. When the search fails, the step is the best trial that met
    sufficient decrease, or zero.

    Parameters
    ----------
    logp : TensorVariable
        Scalar log-likelihood of the observations, a graph of ``parameters``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``, that ``logp`` and ``curvature`` are built on.
    curvature : TensorVariable
        Positive semidefinite approximation :math:`\Lambda` to the negative Hessian of ``logp``, a
        graph of ``parameters``, shape ``(k, k)``.
    whitened : TensorVariable
        Current iterate :math:`u`, shape ``(k,)``.
    predictive_mean : TensorVariable
        Predictive mean :math:`\mu` of the parameters, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L` of the predictive covariance, shape ``(k, k)``.
    line_search : LineSearch
        Search along the Newton direction, for example
        :func:`pytensor_ml.optim.zoom_line_search`.

    Returns
    -------
    next_whitened : TensorVariable
        Next iterate, shape ``(k,)``.
    whitened_curvature : TensorVariable
        :math:`B` at the current iterate, shape ``(k, k)``, marked positive definite.
    search : SearchResult
        The accepted step length, whether the search failed, and how many trials it took.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Section 3.4.3.
    """
    gradient = grad(logp, parameters)
    at_iterate = {parameters: predictive_mean + predictive_factor @ whitened}
    logp, gradient, curvature = graph_replace([logp, gradient, curvature], at_iterate)

    objective = -logp + 0.5 * whitened @ whitened
    objective_gradient = whitened - predictive_factor.T @ gradient
    whitened_curvature = assume(
        pt.eye(whitened.shape[0]) + predictive_factor.T @ curvature @ predictive_factor,
        positive_definite=True,
    )
    direction = pt.linalg.solve(whitened_curvature, -objective_gradient)

    search = search_along(
        loss=objective,
        parameters=[whitened],
        gradients=[objective_gradient],
        directions=[direction],
        line_search=line_search,
    )
    return whitened + search.step_size * direction, whitened_curvature, search


def parameter_posterior(
    *,
    whitened: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_factor: TensorVariable,
    whitened_curvature: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Return the Gaussian approximation to the parameters' posterior at a whitened iterate.

    The covariance is :math:`L B^{-1} L^\top`, with :math:`B` used as given. It is the Laplace
    covariance when ``whitened_curvature`` is evaluated at ``whitened``, which holds at a converged
    iterate.

    Parameters
    ----------
    whitened : TensorVariable
        Iterate :math:`u`, shape ``(k,)``.
    predictive_mean : TensorVariable
        Predictive mean :math:`\mu` of the parameters, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L` of the predictive covariance, shape ``(k, k)``.
    whitened_curvature : TensorVariable
        Positive definite curvature :math:`B`, shape ``(k, k)``.

    Returns
    -------
    mean : TensorVariable
        Posterior mean, shape ``(k,)``.
    covariance : TensorVariable
        Posterior covariance, shape ``(k, k)``.
    """
    mean = predictive_mean + predictive_factor @ whitened
    covariance = quad_form_sym(predictive_factor, pt.linalg.inv(whitened_curvature))
    return mean, covariance
