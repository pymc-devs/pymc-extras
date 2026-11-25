from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.assumptions import assume
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import (
    grad,
    hessian,
    pushforward,
)
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.scan.utils import until
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch, search_along

from pymc_extras.statespace.filters.utilities import quad_form_sym
from pymc_extras.statespace.nonlinear.cubature import cubature_points


def stop_gradient(variable: TensorVariable) -> TensorVariable:
    """
    Return ``variable`` with no gradient, in a form that rewrites cannot remove.

    ``disconnected_grad`` is an identity op that canonicalization deletes, so a gradient taken
    after rewriting, as PyMC takes model gradients, flows through it. This op is inlined only when
    a function is compiled.

    Parameters
    ----------
    variable : TensorVariable
        The value to pass through.

    Returns
    -------
    stopped : TensorVariable
        The value of ``variable``, disconnected from the gradient.
    """
    placeholder = variable.type()
    return OpFromGraph([placeholder], [placeholder], connection_pattern=[[False]], inline=True)(
        variable
    )


class NewtonSite(NamedTuple):
    r"""
    The Gaussian site on the parameters in whitened coordinates, and how it was found.

    Attributes
    ----------
    whitened : TensorVariable
        Final iterate :math:`\hat u`, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L = V \operatorname{diag}(\sqrt{\max(\lambda, 0)})` of the predictive
        covariance from its eigendecomposition, so its columns are orthogonal, shape ``(k, k)``.
    whitened_curvature : TensorVariable
        :math:`B` at :math:`\hat u`, shape ``(k, k)``, marked positive definite. The posterior of
        :math:`u` is :math:`N(\hat u, B^{-1})`.
    iterations : TensorVariable
        Number of loop iterations, each of which searches along one Newton direction.
    evaluations : TensorVariable
        Line-search trials summed over all iterations.
    search_failed : TensorVariable
        Whether any line search along a direction larger than the tolerance failed.
    invalid_direction : TensorVariable
        Whether a Newton direction larger than the tolerance was not finite or not a descent
        direction. A ``curvature`` that is not positive semidefinite can produce one, and then
        ``whitened_curvature`` may not be positive definite either.
    converged : TensorVariable
        Whether the Newton direction at :math:`\hat u` is within the tolerance, :math:`B` changed by
        at most the tolerance relative to its largest entry in its last update, and the last search
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


def newton_site(
    *,
    logp: TensorVariable,
    parameters: TensorVariable,
    curvature: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_covariance: TensorVariable,
    expectation: Literal["point", "cubature"],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> NewtonSite:
    r"""
    Find a Gaussian site on the parameters by damped Newton steps from the predictive mean.

    The parameters :math:`\theta` have the Gaussian predictive :math:`N(\mu, L L^\top)` and the
    observations contribute :math:`\ell(\theta)`. Writing :math:`\theta = \mu + L u`, the posterior
    of :math:`u` is approximated by :math:`N(m, B^{-1})`. With ``expectation="point"`` the
    iterations minimize :math:`J(m) = -\ell(\mu + L m) + \tfrac{1}{2} m^\top m`, whose minimizer is
    the mode at the center of the Laplace approximation [1]_. With ``expectation="cubature"`` they minimize
    :math:`J(m) = -E[\ell(\mu + L u)] + \tfrac{1}{2} m^\top m` over :math:`u \sim N(m, B^{-1})`
    with :math:`B` from the previous iterate, computed with :func:`cubature_points`, which is
    natural-gradient variational inference [2]_. In both, :math:`B = I + L^\top \bar\Lambda L`
    with :math:`\bar\Lambda` the curvature at the point or its expectation, at least the identity
    whenever :math:`\Lambda` is positive semidefinite, and each iteration searches along the Newton
    direction :math:`d = -B^{-1} \nabla J(m)`, starting from the full step. The cubature iterations
    start from :math:`B = I`, the predictive.

    The iterations run inside the graph as a ``scan`` that stops when the largest entry of
    :math:`d` at the current iterate is at most ``tolerance`` and the last update changed no entry
    of :math:`B` by more than ``tolerance`` times its largest entry, when a line search along a larger direction fails,
    when :math:`d` is not finite, or after ``max_iterations`` iterations. A direction within the
    tolerance is not taken. A ``scan`` that stops early cannot be compiled to JAX, so this runs on
    the numba and C backends. The predictive covariance is factored once, by an
    eigendecomposition, so it may be singular. The outputs carry no gradient:
    :func:`implicit_mode` and :func:`stationary_state_posterior` differentiate the Laplace mode
    without the iterations or the factorization.

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
    expectation : {"point", "cubature"}
        Evaluate ``logp``, its gradient and ``curvature`` at the iterate, or average them over the
        cubature points of the current Gaussian.
    line_search : LineSearch
        Search along each Newton direction, for example
        :func:`pytensor_ml.optim.zoom_line_search`.
    tolerance : float
        Largest absolute entry of the undamped whitened Newton direction, and of the last change in
        :math:`B` relative to its largest entry, at which the iterations stop.
    max_iterations : int
        Most loop iterations.

    Returns
    -------
    site : NewtonSite
        The final iterate, the curvature there, and the iteration diagnostics.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Section 3.4.3.
    .. [2] Wilkinson, W. J., S. Sarkka, and A. Solin. Bayes-Newton Methods for Approximate
           Bayesian Inference with PSD Guarantees. Journal of Machine Learning Research 24, 2023.
           Equations 14 and 15.
    """
    eigenvalues, eigenvectors = pt.linalg.eigh(predictive_covariance)
    predictive_factor = eigenvectors * pt.sqrt(pt.maximum(eigenvalues, 0.0))

    gradient = grad(logp, parameters)

    def expected_terms(whitened, whitened_curvature):
        mean = predictive_mean + predictive_factor @ whitened
        if expectation == "point":
            return graph_replace([logp, gradient, curvature], {parameters: mean})
        # With B = C C^T, the transpose of C^{-1} factors B^{-1}.
        whitened_factor = pt.linalg.solve_triangular(
            pt.linalg.cholesky(whitened_curvature), pt.eye(whitened.shape[0]), lower=True
        ).T
        points = cubature_points(mean=mean, factor=predictive_factor @ whitened_factor)
        terms = [logp, gradient, curvature]
        at_points = vectorize_graph(terms, {parameters: points})
        # A term that does not depend on the parameters gets no axis over the points.
        return [
            values.mean(axis=0) if values.ndim > term.ndim else values
            for term, values in zip(terms, at_points, strict=True)
        ]

    def newton_terms_at(whitened, whitened_curvature):
        logp_at, gradient_at, curvature_at = expected_terms(whitened, whitened_curvature)
        objective = -logp_at + 0.5 * whitened @ whitened
        objective_gradient = whitened - predictive_factor.T @ gradient_at
        next_curvature = assume(
            pt.eye(whitened.shape[0]) + predictive_factor.T @ curvature_at @ predictive_factor,
            positive_definite=True,
        )
        direction = pt.linalg.solve(next_curvature, -objective_gradient)
        return objective, objective_gradient, next_curvature, direction

    def iterate(whitened, direction, whitened_curvature, start_gradient):
        objective, objective_gradient, *_ = newton_terms_at(whitened, whitened_curvature)
        search = search_along(
            loss=objective,
            parameters=[whitened],
            gradients=[objective_gradient],
            directions=[direction],
            line_search=line_search,
            start_gradients=[start_gradient],
        )
        moving = pt.max(pt.abs(direction)) > tolerance
        next_whitened = pt.switch(moving, whitened + search.step_size * direction, whitened)
        failed = search.failed & moving

        _, _, next_curvature, _ = newton_terms_at(next_whitened, whitened_curvature)
        _, next_gradient, settled_curvature, next_direction = newton_terms_at(
            next_whitened, next_curvature
        )
        settled = pt.max(pt.abs(settled_curvature - next_curvature)) <= tolerance * pt.max(
            pt.abs(next_curvature)
        )
        negligible_direction = pt.max(pt.abs(next_direction)) <= tolerance
        ascending = (next_gradient @ next_direction >= 0) & ~negligible_direction
        invalid = ~pt.all(pt.isfinite(next_direction)) | ascending
        stop = (negligible_direction & settled) | failed | invalid
        outputs = (
            next_whitened,
            next_direction,
            next_curvature,
            next_gradient,
            search.evaluations,
            failed,
            invalid,
            settled,
        )
        return outputs, until(stop)

    initial_whitened = pt.zeros_like(predictive_mean)
    _, _, initial_curvature, _ = newton_terms_at(initial_whitened, pt.eye(predictive_mean.shape[0]))
    _, initial_gradient, _, initial_direction = newton_terms_at(initial_whitened, initial_curvature)

    (
        whitened_trace,
        direction_trace,
        curvature_trace,
        _,
        evaluations,
        failures,
        invalid_directions,
        settled_trace,
    ) = pytensor.scan(
        iterate,
        outputs_info=[
            initial_whitened,
            initial_direction,
            initial_curvature,
            initial_gradient,
            None,
            None,
            None,
            None,
        ],
        n_steps=max_iterations,
        name="newton_site",
        return_updates=False,
    )

    return NewtonSite(
        whitened=stop_gradient(whitened_trace[-1]),
        predictive_factor=stop_gradient(predictive_factor),
        whitened_curvature=stop_gradient(assume(curvature_trace[-1], positive_definite=True)),
        iterations=whitened_trace.shape[0],
        evaluations=evaluations.sum(),
        search_failed=failures.any(),
        invalid_direction=invalid_directions.any(),
        converged=(pt.max(pt.abs(direction_trace[-1])) <= tolerance)
        & settled_trace[-1]
        & ~failures[-1],
    )


def implicit_mode(
    *,
    mode: TensorVariable,
    logp: TensorVariable,
    parameters: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_covariance: TensorVariable,
) -> TensorVariable:
    r"""
    Return the mode of the parameters' log posterior with its gradient from the implicit function.

    The mode :math:`\hat\theta` solves

    .. math::

        F(\hat\theta) = \hat\theta - \mu - \Sigma \nabla \ell(\hat\theta) = 0,

    which holds for a singular predictive covariance :math:`\Sigma` as well. Its derivative with
    respect to :math:`\mu`, :math:`\Sigma` and the other inputs of :math:`\ell` is
    :math:`-(\partial F / \partial \theta)^{-1}` times the derivative of :math:`F` in them, with
    :math:`\partial F / \partial \theta = I - \Sigma \nabla^2 \ell(\hat\theta)` [1]_. The
    gradient does not pass through the iterations that found the mode or through the
    factorization of :math:`\Sigma`, and it is exact only where ``mode`` is the mode.

    Parameters
    ----------
    mode : TensorVariable
        The mode :math:`\hat\theta`, shape ``(k,)``.
    logp : TensorVariable
        Scalar log-likelihood of the observations, a graph of ``parameters``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``, that ``logp`` is built on.
    predictive_mean : TensorVariable
        Predictive mean :math:`\mu` of the parameters, shape ``(k,)``.
    predictive_covariance : TensorVariable
        Symmetric positive semidefinite predictive covariance :math:`\Sigma` of the parameters,
        shape ``(k, k)``.

    Returns
    -------
    mode : TensorVariable
        The value of ``mode``, with the gradient of the implicit function, shape ``(k,)``.

    References
    ----------
    .. [1] Blondel, M., Q. Berthet, M. Cuturi, R. Frostig, S. Hoyer, F. Llinares-Lopez,
           F. Pedregosa, and J.-P. Vert. Efficient and modular implicit differentiation.
           Advances in Neural Information Processing Systems 35, 2022.
    """
    fixed_mode = stop_gradient(mode)
    gradient_at, hessian_at = graph_replace(
        [grad(logp, parameters), hessian(cost=logp, wrt=parameters)], {parameters: fixed_mode}
    )
    residual = fixed_mode - predictive_mean - predictive_covariance @ gradient_at
    residual_jacobian = stop_gradient(pt.eye(mode.shape[0]) - predictive_covariance @ hessian_at)

    # The correction is zero in value, and its gradient is the implicit function's.
    correction = -pt.linalg.solve(residual_jacobian, residual)
    return fixed_mode + correction - stop_gradient(correction)


def laplace_log_likelihood(
    *,
    mode: TensorVariable,
    logp: TensorVariable,
    curvature: TensorVariable,
    parameters: TensorVariable,
    predictive_covariance: TensorVariable,
) -> TensorVariable:
    r"""
    Return the Laplace approximation to the log marginal likelihood of the observations.

    The parameters have the Gaussian predictive :math:`N(\mu, \Sigma)` and the observations
    contribute :math:`\ell(\theta)`. At the mode :math:`\hat\theta` of the log posterior,

    .. math::

        \log p(y) \approx \ell(\hat\theta)
        - \tfrac{1}{2} \nabla \ell(\hat\theta)^\top \Sigma \nabla \ell(\hat\theta)
        - \tfrac{1}{2} \log \det \left(I + \Sigma \Lambda(\hat\theta)\right),

    where the quadratic term equals :math:`(\hat\theta - \mu)^\top \Sigma^{-1} (\hat\theta - \mu)`
    at the mode and the determinant equals :math:`\det B` [1]_. It is exact when :math:`\ell` is
    quadratic and :math:`\Lambda` is its negative Hessian. It depends on :math:`\Sigma` and not on
    a factor of it, so with the mode from :func:`implicit_mode` its gradient does not pass through a
    factorization of :math:`\Sigma`.

    Parameters
    ----------
    mode : TensorVariable
        The mode :math:`\hat\theta`, shape ``(k,)``.
    logp : TensorVariable
        Scalar log-likelihood of the observations, a graph of ``parameters``.
    curvature : TensorVariable
        Symmetric approximation :math:`\Lambda` to the negative Hessian of ``logp``, a graph of
        ``parameters``, shape ``(k, k)``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``, that ``logp`` and ``curvature`` are built on.
    predictive_covariance : TensorVariable
        Symmetric positive semidefinite predictive covariance :math:`\Sigma` of the parameters,
        shape ``(k, k)``.

    Returns
    -------
    log_likelihood : TensorVariable
        Scalar approximate log marginal likelihood, :math:`-\infty` where
        :math:`\det(I + \Sigma \Lambda)` is not positive, which happens only away from a mode.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Equation 3.32.
    """
    logp_at, gradient_at, curvature_at = graph_replace(
        [logp, grad(logp, parameters), curvature], {parameters: mode}
    )
    sign, log_determinant = pt.linalg.slogdet(
        pt.eye(mode.shape[0]) + predictive_covariance @ curvature_at
    )
    log_likelihood = (
        logp_at - 0.5 * gradient_at @ predictive_covariance @ gradient_at - 0.5 * log_determinant
    )
    return pt.switch(sign > 0, log_likelihood, -np.inf)


def positive_semidefinite_part(matrix: TensorVariable) -> TensorVariable:
    """Return the symmetric part of ``matrix`` with its negative eigenvalues set to zero."""
    eigenvalues, eigenvectors = pt.linalg.eigh(0.5 * (matrix + matrix.mT))
    return (eigenvectors * pt.maximum(eigenvalues, 0.0)[..., None, :]) @ eigenvectors.mT


def fisher_curvature(
    *,
    information: TensorVariable,
    distribution_parameters: Sequence[TensorVariable],
    parameters: TensorVariable,
) -> TensorVariable:
    r"""
    Return the Fisher information of a log-likelihood in its parameters.

    Observation :math:`i` has distribution parameters :math:`\eta_i = \eta_i(\theta)` and Fisher
    information :math:`F_i` in them, the expected negative Hessian of its log-density. The
    curvature is

    .. math::

        \Lambda = \sum_i J_i^\top F_i J_i,

    where :math:`J_i` is the Jacobian of :math:`\eta_i` with respect to :math:`\theta`. It is the
    expected negative Hessian of the log-likelihood in :math:`\theta`, whatever parameterization
    :math:`\eta` the distribution uses, and the Hessian of Fisher scoring [1]_.

    Parameters
    ----------
    information : TensorVariable
        Positive semidefinite Fisher information :math:`F_i` of each observation in its
        ``distribution_parameters``, shape ``(n, p, p)``.
    distribution_parameters : sequence of TensorVariable
        The ``p`` distribution parameters :math:`\eta` that depend on ``parameters``, each of shape
        ``(n,)``. A parameter shared across observations has to be broadcast to ``(n,)``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``.

    Returns
    -------
    curvature : TensorVariable
        Positive semidefinite matrix of shape ``(k, k)``, a graph of ``parameters``.

    References
    ----------
    .. [1] McCullagh, P., and J. A. Nelder. Generalized Linear Models, 2nd ed. Chapman and Hall,
           1989. Section 2.5.
    """
    constant_positions = [
        position
        for position, eta in enumerate(distribution_parameters)
        if parameters not in ancestors([eta])
    ]
    if constant_positions:
        raise ValueError(
            f"distribution_parameters at positions {constant_positions} do not depend on "
            "parameters. Leave them out, along with their rows and columns of information."
        )

    # One pushforward per distribution parameter: a joint pushforward drops the dependence of one
    # output on another when one is computed from the other.
    tangent = parameters.type()
    tangent_derivatives = [
        pushforward(f=eta, wrt=parameters, tangents=tangent) for eta in distribution_parameters
    ]
    jacobian = vectorize_graph(
        pt.stack(tangent_derivatives), {tangent: pt.eye(parameters.shape[0])}
    )
    jacobian = jacobian.transpose(2, 1, 0)

    return (jacobian.mT @ information @ jacobian).sum(axis=0)


def parameter_posterior(
    *,
    whitened: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_factor: TensorVariable,
    whitened_curvature: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Return the Gaussian approximation to the parameters' posterior at a whitened iterate.

    The mean is :math:`\mu + L u` and the covariance is :math:`L B^{-1} L^\top`, with :math:`B`
    evaluated at :math:`u`, as :class:`NewtonSite` returns them.

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


def state_posterior(
    *,
    whitened: TensorVariable,
    predictive_factor: TensorVariable,
    whitened_curvature: TensorVariable,
    state_mean: TensorVariable,
    state_covariance: TensorVariable,
    design: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Return the Gaussian posterior of the state implied by a whitened parameter posterior.

    The state has the predictive :math:`N(a, P)` and the parameters are
    :math:`\theta = Z x + d + \eta` with :math:`\eta \sim N(0, H)`, so their predictive is
    :math:`N(Z a + d, Z P Z^\top + H)`, factored as :math:`L L^\top` by :func:`newton_site`. The
    observations depend on the state only through :math:`\theta`, so conditioning the joint
    Gaussian of :math:`(x, \theta)` on the parameter posterior gives

    .. math::

        a^+ = a + C \hat u, \qquad P^+ = P - C (I - B^{-1}) C^\top,
        \qquad C = P Z^\top (L^\top)^+,

    where :math:`(L^\top)^+` is the pseudo-inverse, so a singular parameter predictive is
    allowed. Observations that carry no information leave the state predictive unchanged. The
    covariance is a difference, so observations that shrink a variance by many orders of magnitude
    can leave :math:`P^+` slightly indefinite through cancellation.

    Parameters
    ----------
    whitened : TensorVariable
        Whitened mode :math:`\hat u` from :func:`newton_site`, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L` of the parameter predictive covariance from :func:`newton_site`, with
        orthogonal columns, shape ``(k, k)``.
    whitened_curvature : TensorVariable
        Positive definite :math:`B` at :math:`\hat u` from :func:`newton_site`, shape ``(k, k)``.
    state_mean : TensorVariable
        Predictive mean :math:`a` of the state, shape ``(m,)``.
    state_covariance : TensorVariable
        Predictive covariance :math:`P` of the state, shape ``(m, m)``.
    design : TensorVariable
        Design matrix :math:`Z` from the state to the parameters, shape ``(k, m)``.

    Returns
    -------
    mean : TensorVariable
        Posterior mean of the state, shape ``(m,)``.
    covariance : TensorVariable
        Posterior covariance of the state, shape ``(m, m)``.
    """
    # L has orthogonal columns, so the pseudo-inverse of its transpose divides each column by its
    # squared norm and zeroes the columns of a singular predictive.
    squared_norms = pt.sum(predictive_factor**2, axis=0)
    nonzero = squared_norms > 0
    inverse_squared_norms = pt.where(nonzero, 1 / pt.where(nonzero, squared_norms, 1), 0)
    gain = (state_covariance @ design.T @ predictive_factor) * inverse_squared_norms

    mean = state_mean + gain @ whitened
    whitened_variance_reduction = pt.eye(whitened.shape[0]) - pt.linalg.inv(whitened_curvature)
    covariance = state_covariance - quad_form_sym(gain, whitened_variance_reduction)
    return mean, covariance


def stationary_state_posterior(
    *,
    mode: TensorVariable,
    logp: TensorVariable,
    curvature: TensorVariable,
    parameters: TensorVariable,
    predictive_covariance: TensorVariable,
    state_mean: TensorVariable,
    state_covariance: TensorVariable,
    design: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Return the Gaussian posterior of the state from the stationarity of the parameters' mode.

    At the mode, :math:`\hat\theta - \mu = \Sigma g` with :math:`g = \nabla \ell(\hat\theta)`,
    so the posterior of :func:`state_posterior` is [1]_

    .. math::

        a^+ = a + P Z^\top g, \qquad
        P^+ = P - P Z^\top \Lambda (I + \Sigma \Lambda)^{-1} Z P,

    with :math:`\Lambda` the curvature at :math:`\hat\theta`. Neither needs a factor or an
    inverse of :math:`\Sigma`, so with the mode from :func:`implicit_mode` the gradient passes
    through neither. Both equal :func:`state_posterior` only where ``mode`` is the mode.

    Parameters
    ----------
    mode : TensorVariable
        The mode :math:`\hat\theta`, shape ``(k,)``.
    logp : TensorVariable
        Scalar log-likelihood of the observations, a graph of ``parameters``.
    curvature : TensorVariable
        Positive semidefinite approximation :math:`\Lambda` to the negative Hessian of ``logp``, a
        graph of ``parameters``, shape ``(k, k)``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``, that ``logp`` and ``curvature`` are built on.
    predictive_covariance : TensorVariable
        Predictive covariance :math:`\Sigma = Z P Z^\top + H` of the parameters, shape ``(k, k)``.
    state_mean : TensorVariable
        Predictive mean :math:`a` of the state, shape ``(m,)``.
    state_covariance : TensorVariable
        Predictive covariance :math:`P` of the state, shape ``(m, m)``.
    design : TensorVariable
        Design matrix :math:`Z` from the state to the parameters, shape ``(k, m)``.

    Returns
    -------
    mean : TensorVariable
        Posterior mean of the state, shape ``(m,)``.
    covariance : TensorVariable
        Posterior covariance of the state, shape ``(m, m)``.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Equations 3.21 and 3.24.
    """
    gradient_at, curvature_at = graph_replace(
        [grad(logp, parameters), curvature], {parameters: mode}
    )
    cross_covariance = state_covariance @ design.T
    # Lambda (I + Sigma Lambda)^{-1} equals (I + Lambda Sigma)^{-1} Lambda.
    precision_gain = pt.linalg.solve(
        pt.eye(mode.shape[0]) + curvature_at @ predictive_covariance, curvature_at
    )

    mean = state_mean + cross_covariance @ gradient_at
    covariance = state_covariance - quad_form_sym(cross_covariance, precision_gain)
    return mean, covariance
