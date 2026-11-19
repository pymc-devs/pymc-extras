from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pytensor
import pytensor.tensor as pt

from numpy.typing import ArrayLike
from pytensor.gradient import hessian
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch

from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.filters.utilities import quad_form_sym
from pymc_extras.statespace.nonlinear.fusion import FusedStateSpace
from pymc_extras.statespace.nonlinear.laplace_update import (
    LaplaceMode,
    gauss_newton_curvature,
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


def run_filter(
    *,
    fused: FusedStateSpace,
    statespace_parameters: Mapping[str, ArrayLike],
    statespace_data: Mapping[str, ArrayLike],
    data: Mapping[str, Sequence[ArrayLike]],
    curvature: Literal["gauss_newton", "exact"],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> dict[str, np.ndarray]:
    r"""
    Filter the states of a fused model over dates whose numbers of observations differ.

    The step from :func:`filter_step` is compiled once and called on each date, starting from the
    statespace's initial state. A date whose observed values are empty is a pure prediction.

    Parameters
    ----------
    fused : FusedStateSpace
        The model and the statespace over its parameters.
    statespace_parameters : mapping of str to ArrayLike
        A value for each name in the statespace's ``param_names``.
    statespace_data : mapping of str to ArrayLike
        A value for each name in the statespace's ``data_names``. The values cannot make the
        statespace matrices vary over time.
    data : mapping of str to sequence of ArrayLike
        For each data variable of the model and each observed variable whose observations are
        constants, its value on each date.
    curvature : {"gauss_newton", "exact"}
        The curvature of the log-likelihood in the Newton iterations: the generalized Gauss-Newton
        matrix of each observed variable, or the exact negative Hessian.
    line_search : LineSearch
        Search along each Newton direction.
    tolerance : float
        Largest absolute entry of the whitened Newton direction at which the iterations stop.
    max_iterations : int
        Most Newton iterations on each date.

    Returns
    -------
    filtered : dict of str to np.ndarray
        Stacked over dates: ``predicted_mean`` and ``predicted_covariance`` before each date's
        observations, ``filtered_mean`` and ``filtered_covariance`` after them, and the Newton
        diagnostics ``iterations``, ``evaluations``, ``search_failed``, ``invalid_direction`` and
        ``converged``.
    """
    date_counts = {name: len(dates) for name, dates in data.items()}
    if len(set(date_counts.values())) != 1:
        raise ValueError(f"data has to give the same number of dates for each name: {date_counts}.")
    [n_dates] = set(date_counts.values())

    initial_mean, initial_covariance, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data=statespace_data
    )
    matrices = {"c": c, "d": d, "T": T, "Z": Z, "R": R, "H": H, "Q": Q}
    time_varying = [name for name, matrix in matrices.items() if matrix.ndim > _MATRIX_NDIMS[name]]
    if time_varying:
        raise ValueError(f"Statespace matrices {time_varying} vary over time.")

    parameters = pt.vector("parameters", shape=(len(fused.statespace.observed_states),))
    placeholders = {
        name: pt.tensor(
            name=name,
            shape=(None,) * fused.model[name].ndim,
            dtype=fused.model[name].dtype,
        )
        for name in data
    }
    observation_logps = fused.observation_logp(parameters=parameters, data=placeholders)
    logp = pt.sum([pointwise_logp.sum() for pointwise_logp, _ in observation_logps.values()])

    predicted_mean = initial_mean.type()
    predicted_covariance = initial_covariance.type()
    filtered_mean, filtered_covariance, next_mean, next_covariance, mode = filter_step(
        logp=logp,
        curvature=_curvature(
            kind=curvature,
            logp=logp,
            observation_logps=observation_logps,
            parameters=parameters,
        ),
        parameters=parameters,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        **matrices,
        line_search=line_search,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    recorded = {
        "filtered_mean": filtered_mean,
        "filtered_covariance": filtered_covariance,
        "iterations": mode.iterations,
        "evaluations": mode.evaluations,
        "search_failed": mode.search_failed,
        "invalid_direction": mode.invalid_direction,
        "converged": mode.converged,
    }
    step = pytensor.function(
        [predicted_mean, predicted_covariance, *placeholders.values()],
        [next_mean, next_covariance, *recorded.values()],
    )

    history = {name: [] for name in ["predicted_mean", "predicted_covariance", *recorded]}
    mean, covariance = pytensor.function([], [initial_mean, initial_covariance])()
    for date in range(n_dates):
        history["predicted_mean"].append(mean)
        history["predicted_covariance"].append(covariance)
        mean, covariance, *values = step(
            mean, covariance, *[dates[date] for dates in data.values()]
        )
        for name, value in zip(recorded, values, strict=True):
            history[name].append(value)

    return {name: np.stack(values) for name, values in history.items()}


_MATRIX_NDIMS = {"c": 1, "d": 1, "T": 2, "Z": 2, "R": 2, "H": 2, "Q": 2}


def _curvature(
    *,
    kind: Literal["gauss_newton", "exact"],
    logp: TensorVariable,
    observation_logps: Mapping[str, tuple[TensorVariable, list[TensorVariable] | None]],
    parameters: TensorVariable,
) -> TensorVariable:
    if kind == "exact":
        return -hessian(cost=logp, wrt=parameters)

    without_parameters = [
        name
        for name, (_, distribution_parameters) in observation_logps.items()
        if distribution_parameters is None
    ]
    if without_parameters:
        raise ValueError(
            f"Observed variables {without_parameters} have no distribution parameters for the "
            "Gauss-Newton curvature."
        )
    # An observed variable none of whose distribution parameters depends on the parameters has no
    # curvature in them.
    return sum(
        (
            gauss_newton_curvature(
                pointwise_logp=pointwise_logp,
                distribution_parameters=distribution_parameters,
                parameters=parameters,
            )
            for pointwise_logp, distribution_parameters in observation_logps.values()
            if distribution_parameters
        ),
        start=pt.zeros((parameters.shape[0], parameters.shape[0])),
    )
