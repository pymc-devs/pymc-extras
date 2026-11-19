from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch

from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.filters.utilities import quad_form_sym
from pymc_extras.statespace.nonlinear.laplace_update import (
    LaplaceMode,
    laplace_mode,
    state_posterior,
)


def filter_step(
    *,
    logp: TensorVariable,
    curvature: TensorVariable,
    parameters: TensorVariable,
    predicted_mean: TensorVariable,
    predicted_covariance: TensorVariable,
    c: TensorVariable,
    d: TensorVariable,
    T: TensorVariable,
    Z: TensorVariable,
    R: TensorVariable,
    H: TensorVariable,
    Q: TensorVariable,
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, LaplaceMode]:
    r"""
    Update the state on one date's observations by the Laplace approximation, then predict the next.

    The state predictive :math:`N(a, P)` implies the parameter predictive
    :math:`N(Z a + d, Z P Z^\top + H)`. :func:`laplace_mode` finds the mode of the parameters'
    posterior under ``logp``, and :func:`state_posterior` lifts it to the state. The filtered state
    is then predicted to the next date with :meth:`BaseFilter.predict`.

    Parameters
    ----------
    logp : TensorVariable
        Scalar log-likelihood of the date's observations, a graph of ``parameters``.
    curvature : TensorVariable
        Positive semidefinite approximation to the negative Hessian of ``logp``, a graph of
        ``parameters``, shape ``(k, k)``.
    parameters : TensorVariable
        Input vector of the parameters, shape ``(k,)``.
    predicted_mean : TensorVariable
        Predicted state mean :math:`a`, shape ``(m,)``.
    predicted_covariance : TensorVariable
        Predicted state covariance :math:`P`, shape ``(m, m)``.
    c, d, T, Z, R, H, Q : TensorVariable
        The statespace matrices at this date.
    line_search : LineSearch
        Search along each Newton direction.
    tolerance : float
        Largest absolute entry of the whitened Newton direction at which the iterations stop.
    max_iterations : int
        Most Newton iterations.

    Returns
    -------
    filtered_mean : TensorVariable
        Filtered state mean, shape ``(m,)``.
    filtered_covariance : TensorVariable
        Filtered state covariance, shape ``(m, m)``.
    next_predicted_mean : TensorVariable
        State mean predicted for the next date, shape ``(m,)``.
    next_predicted_covariance : TensorVariable
        State covariance predicted for the next date, shape ``(m, m)``.
    mode : LaplaceMode
        The parameters' mode and the iteration diagnostics.
    """
    mode = laplace_mode(
        logp=logp,
        parameters=parameters,
        curvature=curvature,
        predictive_mean=Z @ predicted_mean + d,
        predictive_covariance=quad_form_sym(Z, predicted_covariance) + H,
        line_search=line_search,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    filtered_mean, filtered_covariance = state_posterior(
        whitened=mode.whitened,
        predictive_factor=mode.predictive_factor,
        whitened_curvature=mode.whitened_curvature,
        state_mean=predicted_mean,
        state_covariance=predicted_covariance,
        design=Z,
    )
    next_predicted_mean, next_predicted_covariance = BaseFilter.predict(
        a=filtered_mean, P=filtered_covariance, c=c, T=T, R=R, Q=Q
    )
    return filtered_mean, filtered_covariance, next_predicted_mean, next_predicted_covariance, mode
