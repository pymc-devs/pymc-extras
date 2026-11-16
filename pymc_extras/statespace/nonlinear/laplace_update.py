from collections.abc import Sequence
from typing import NamedTuple

import pytensor
import pytensor.tensor as pt

from pytensor.assumptions import assume
from pytensor.gradient import grad, hessian_vector_product, pushforward
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.scan.utils import until
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch, search_along

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


def gauss_newton_curvature(
    *,
    pointwise_logp: TensorVariable,
    distribution_parameters: Sequence[TensorVariable],
    parameters: TensorVariable,
) -> TensorVariable:
    r"""
    Return a generalized Gauss-Newton approximation to the negative Hessian of a log-likelihood.

    Observation :math:`i` has log-likelihood :math:`\ell_i(\eta_i)` in its distribution parameters
    :math:`\eta_i = \eta_i(\theta)`. The curvature is

    .. math::

        \Lambda = \sum_i J_i^\top \left[-\nabla^2_{\eta_i} \ell_i\right]_+ J_i,

    where :math:`J_i` is the Jacobian of :math:`\eta_i` with respect to :math:`\theta` and
    :math:`[\cdot]_+` clips each block's eigenvalues at zero, so :math:`\Lambda` is positive
    semidefinite. It drops the second derivatives of :math:`\eta` from the exact Hessian [1]_ [2]_.

    Parameters
    ----------
    pointwise_logp : TensorVariable
        Log-likelihood of each observation, shape ``(n,)``. Entry :math:`i` depends on the
        parameters only through entry :math:`i` of each distribution parameter.
    distribution_parameters : sequence of TensorVariable
        The distribution parameters :math:`\eta` that depend on ``parameters``, each of shape
        ``(n,)``. A parameter shared across observations has to be broadcast to ``(n,)``, and one
        that does not depend on ``parameters`` has to be left out.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``.

    Returns
    -------
    curvature : TensorVariable
        Positive semidefinite matrix of shape ``(k, k)``, a graph of ``parameters``.

    References
    ----------
    .. [1] Schraudolph, N. N. Fast curvature matrix-vector products for second-order gradient
           descent. Neural Computation 14(7), 2002.
    .. [2] Martens, J. New insights and perspectives on the natural gradient method. Journal of
           Machine Learning Research 21(146), 2020.
    """
    constant_positions = [
        position
        for position, eta in enumerate(distribution_parameters)
        if parameters not in ancestors([eta])
    ]
    if constant_positions:
        raise ValueError(
            f"distribution_parameters at positions {constant_positions} do not depend on "
            "parameters. "
            "Leave them out: they would distort each observation's curvature block without "
            "contributing to the Jacobian."
        )

    # Differentiating with respect to the distribution parameters themselves would give total
    # derivatives, which count a parameter computed from another one twice. Placeholders make the
    # Hessian partial in the distribution parameters, as the Gauss-Newton matrix needs.
    placeholders = [eta.type() for eta in distribution_parameters]
    to_placeholders = dict(zip(distribution_parameters, placeholders, strict=True))
    detached_logp = graph_replace(pointwise_logp.sum(), to_placeholders)
    if parameters in ancestors([detached_logp]):
        raise ValueError(
            "pointwise_logp depends on parameters other than through distribution_parameters."
        )

    # The Hessian in the stacked distribution parameters is block diagonal across observations, so
    # its product with the indicator of parameter j holds column j of every observation's block.
    hessian_columns = [
        pt.stack(
            hessian_vector_product(
                cost=detached_logp,
                wrt=placeholders,
                p=[
                    pt.ones_like(placeholder) if position == column else pt.zeros_like(placeholder)
                    for position, placeholder in enumerate(placeholders)
                ],
                disconnected_inputs="ignore",
            ),
            axis=-1,
        )
        for column in range(len(placeholders))
    ]
    negative_hessian = graph_replace(
        -pt.stack(hessian_columns, axis=-1),
        dict(zip(placeholders, distribution_parameters, strict=True)),
    )
    eigenvalues, eigenvectors = pt.linalg.eigh(0.5 * (negative_hessian + negative_hessian.mT))
    projected = (eigenvectors * pt.maximum(eigenvalues, 0.0)[..., None, :]) @ eigenvectors.mT

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

    return pt.einsum("npk,npq,nql->kl", jacobian, projected, jacobian)


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
    evaluated at :math:`u`, as :class:`LaplaceMode` returns them.

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
    :math:`N(Z a + d, Z P Z^\top + H)`, factored as :math:`L L^\top` by :func:`laplace_mode`. The
    observations depend on the state only through :math:`\theta`, so conditioning the joint
    Gaussian of :math:`(x, \theta)` on the parameter posterior gives

    .. math::

        a^+ = a + C \hat u, \qquad P^+ = P - C (I - B^{-1}) C^\top,
        \qquad C = P Z^\top (L^\top)^+,

    where :math:`(L^\top)^+` is the pseudo-inverse, so a singular parameter predictive is
    allowed. Observations that carry no information leave the state predictive unchanged.

    Parameters
    ----------
    whitened : TensorVariable
        Whitened mode :math:`\hat u` from :func:`laplace_mode`, shape ``(k,)``.
    predictive_factor : TensorVariable
        Factor :math:`L` of the parameter predictive covariance from :func:`laplace_mode`, with
        orthogonal columns, shape ``(k, k)``.
    whitened_curvature : TensorVariable
        Positive definite :math:`B` at :math:`\hat u` from :func:`laplace_mode`, shape ``(k, k)``.
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
