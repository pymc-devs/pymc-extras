import pytensor.tensor as pt

from pytensor.assumptions import assume
from pytensor.gradient import grad
from pytensor.graph.replace import graph_replace
from pytensor.tensor.variable import TensorVariable

from pymc_extras.statespace.filters.utilities import quad_form_sym


def newton_step(
    *,
    logp: TensorVariable,
    parameters: TensorVariable,
    curvature: TensorVariable,
    whitened: TensorVariable,
    predictive_mean: TensorVariable,
    predictive_factor: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Take one Newton step on the parameters' log posterior in whitened coordinates.

    The parameters :math:`\theta` have the Gaussian predictive :math:`N(\mu, L L^\top)` and the
    observations contribute :math:`\ell(\theta)`. Writing :math:`\theta = \mu + L u`, the step
    minimizes :math:`J(u) = -\ell(\mu + L u) + \tfrac{1}{2} u^\top u`, whose curvature
    :math:`B = I + L^\top \Lambda L` is at least the identity whenever :math:`\Lambda` is positive
    semidefinite [1]_. :math:`\Lambda` may be singular, and :math:`L` may have zero columns where the
    predictive is singular.

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

    Returns
    -------
    next_whitened : TensorVariable
        Next iterate, shape ``(k,)``.
    whitened_curvature : TensorVariable
        :math:`B` at the current iterate, shape ``(k, k)``, marked positive definite.

    References
    ----------
    .. [1] Rasmussen, C. E., and C. K. I. Williams. Gaussian Processes for Machine Learning.
           MIT Press, 2006. Section 3.4.3.
    """
    gradient = grad(logp, parameters)
    at_iterate = {parameters: predictive_mean + predictive_factor @ whitened}
    gradient, curvature = graph_replace([gradient, curvature], at_iterate)

    whitened_curvature = assume(
        pt.eye(whitened.shape[0]) + predictive_factor.T @ curvature @ predictive_factor,
        positive_definite=True,
    )
    step = pt.linalg.solve(whitened_curvature, predictive_factor.T @ gradient - whitened)

    return whitened + step, whitened_curvature


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
