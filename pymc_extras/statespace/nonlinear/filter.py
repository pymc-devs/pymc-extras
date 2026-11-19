from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pytensor
import pytensor.tensor as pt
import xarray as xr

from numpy.typing import ArrayLike
from pytensor.gradient import hessian
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch

from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.filters.kalman_smoother import RTSSmoother
from pymc_extras.statespace.filters.utilities import quad_form_sym
from pymc_extras.statespace.nonlinear.fusion import FusedStateSpace
from pymc_extras.statespace.nonlinear.laplace_update import (
    LaplaceMode,
    gauss_newton_curvature,
    laplace_mode,
    state_posterior,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    FILTER_OUTPUT_DIMS,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    TIME_DIM,
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
    dates: ArrayLike,
    curvature: Literal["gauss_newton", "exact"],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> xr.DataTree:
    r"""
    Filter and smooth the states of a fused model over dates whose numbers of observations differ.

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
    dates : ArrayLike
        Label of each date.
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
    filtered : xr.DataTree
        Group ``posterior_predictive`` holds the filter outputs under the names and dims of
        :meth:`PyMCStateSpace.sample_filter_outputs`, with one chain and one draw:
        ``predicted_states`` and ``predicted_covariances`` before each date's observations,
        ``filtered_states`` and ``filtered_covariances`` after them, ``smoothed_states`` and
        ``smoothed_covariances`` given all dates from :class:`RTSSmoother`, and the parameter
        predictive as ``predicted_observed_states`` and ``predicted_observed_covariances``. Group
        ``diagnostics`` holds each date's ``iterations``, ``evaluations``, ``search_failed``,
        ``invalid_direction`` and ``converged`` from :class:`LaplaceMode`.
    """
    date_counts = {name: len(values) for name, values in data.items()}
    if set(date_counts.values()) != {len(dates)}:
        raise ValueError(
            f"data has to give a value on each of the {len(dates)} dates for each name: "
            f"{date_counts}."
        )

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
        "filtered_states": filtered_mean,
        "filtered_covariances": filtered_covariance,
        "predicted_observed_states": Z @ predicted_mean + d,
        "predicted_observed_covariances": quad_form_sym(Z, predicted_covariance) + H,
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

    history = {name: [] for name in ["predicted_states", "predicted_covariances", *recorded]}
    mean, covariance = pytensor.function([], [initial_mean, initial_covariance])()
    for date in range(len(dates)):
        history["predicted_states"].append(mean)
        history["predicted_covariances"].append(covariance)
        mean, covariance, *values = step(
            mean, covariance, *[values_by_date[date] for values_by_date in data.values()]
        )
        for name, value in zip(recorded, values, strict=True):
            history[name].append(value)

    stacked = {name: np.stack(values) for name, values in history.items()}
    # The backward recursion reads only the filtered moments, at positions 0 and 3 of the filter
    # outputs, and the transition, so it needs no observations.
    smoothed_states, smoothed_covariances = RTSSmoother().build_graph(
        None,
        [initial_mean, initial_covariance, c, d, T, Z, R, H, Q],
        [
            pt.as_tensor(stacked["filtered_states"]),
            None,
            None,
            pt.as_tensor(stacked["filtered_covariances"]),
        ],
    )
    stacked["smoothed_states"], stacked["smoothed_covariances"] = pytensor.function(
        [], [smoothed_states, smoothed_covariances]
    )()

    state_names = list(fused.statespace.state_names)
    observed_states = list(fused.statespace.observed_states)
    filter_outputs = xr.Dataset(
        {
            name: (FILTER_OUTPUT_DIMS[name], values)
            for name, values in stacked.items()
            if name in FILTER_OUTPUT_DIMS
        },
        coords={
            TIME_DIM: dates,
            ALL_STATE_DIM: state_names,
            ALL_STATE_AUX_DIM: state_names,
            OBS_STATE_DIM: observed_states,
            OBS_STATE_AUX_DIM: observed_states,
        },
    ).expand_dims(chain=[0], draw=[0])
    diagnostics = xr.Dataset(
        {
            name: (TIME_DIM, values)
            for name, values in stacked.items()
            if name not in FILTER_OUTPUT_DIMS
        },
        coords={TIME_DIM: dates},
    )
    return xr.DataTree.from_dict(
        {"posterior_predictive": filter_outputs, "diagnostics": diagnostics}
    )


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
