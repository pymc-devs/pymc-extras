from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from numpy.typing import ArrayLike
from pymc.model import modelcontext
from pytensor.gradient import disconnected_grad, hessian
from pytensor.tensor import TensorLike
from pytensor.tensor.variable import TensorVariable
from pytensor_ml.optim.line_search import LineSearch

from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.filters.kalman_smoother import RTSSmoother
from pymc_extras.statespace.filters.utilities import quad_form_sym
from pymc_extras.statespace.nonlinear.fusion import FusedStateSpace
from pymc_extras.statespace.nonlinear.laplace_update import (
    NewtonSite,
    gauss_newton_curvature,
    implicit_mode,
    laplace_log_likelihood,
    newton_site,
    state_posterior,
    stationary_state_posterior,
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
) -> tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, NewtonSite, TensorVariable
]:
    r"""
    Update the state on one date's observations by the Laplace approximation, then predict the next.

    The state predictive :math:`N(a, P)` implies the parameter predictive
    :math:`N(Z a + d, Z P Z^\top + H)`. :func:`newton_site` finds the mode of the parameters'
    posterior under ``logp``. The filtered moments take their values from :func:`state_posterior`,
    which lifts the mode to the state, and their gradients from
    :func:`stationary_state_posterior` at the mode from :func:`implicit_mode`, so no gradient passes
    through the iterations or the factorization of the parameter predictive. The filtered state is
    then predicted to the next date with :meth:`BaseFilter.predict`.

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
    site : NewtonSite
        The parameters' Laplace site and the iteration diagnostics.
    parameter_mode : TensorVariable
        The parameters' mode from :func:`implicit_mode`, shape ``(k,)``.
    """
    parameter_mean = Z @ predicted_mean + d
    parameter_covariance = quad_form_sym(Z, predicted_covariance) + H
    site = newton_site(
        logp=logp,
        parameters=parameters,
        curvature=curvature,
        predictive_mean=parameter_mean,
        predictive_covariance=parameter_covariance,
        line_search=line_search,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    parameter_mode = implicit_mode(
        mode=parameter_mean + site.predictive_factor @ site.whitened,
        logp=logp,
        parameters=parameters,
        predictive_mean=parameter_mean,
        predictive_covariance=parameter_covariance,
    )

    filtered_mean, filtered_covariance = state_posterior(
        whitened=site.whitened,
        predictive_factor=site.predictive_factor,
        whitened_curvature=site.whitened_curvature,
        state_mean=predicted_mean,
        state_covariance=predicted_covariance,
        design=Z,
    )
    stationary_mean, stationary_covariance = stationary_state_posterior(
        mode=parameter_mode,
        logp=logp,
        curvature=curvature,
        parameters=parameters,
        predictive_covariance=parameter_covariance,
        state_mean=predicted_mean,
        state_covariance=predicted_covariance,
        design=Z,
    )
    filtered_mean = _value_with_gradient_of(value=filtered_mean, gradient=stationary_mean)
    filtered_covariance = _value_with_gradient_of(
        value=filtered_covariance, gradient=stationary_covariance
    )

    next_predicted_mean, next_predicted_covariance = BaseFilter.predict(
        a=filtered_mean, P=filtered_covariance, c=c, T=T, R=R, Q=Q
    )
    return (
        filtered_mean,
        filtered_covariance,
        next_predicted_mean,
        next_predicted_covariance,
        site,
        parameter_mode,
    )


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
        constants, its finite value on each date, cast to the variable's dtype. Missing
        observations are left out of a date's arrays.
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
        ``diagnostics`` holds each date's ``loglike_obs`` from :func:`laplace_log_likelihood`, and
        its ``iterations``, ``evaluations``, ``search_failed``, ``invalid_direction`` and
        ``converged`` from :class:`NewtonSite`. The moments of a date that did not converge are
        not at its mode, and every later date's moments start from them.
    """
    not_variables = sorted(set(data) - set(fused.model.named_vars))
    if not_variables:
        raise ValueError(f"data names {not_variables} are not variables of the model.")
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

    date_counts = {name: len(values) for name, values in data.items()}
    if set(date_counts.values()) != {len(dates)}:
        raise ValueError(
            f"data has to give a value on each of the {len(dates)} dates for each name: "
            f"{date_counts}."
        )
    non_finite = [
        (name, date)
        for name, values in data.items()
        for date, value in enumerate(values)
        if not np.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(
            f"data has non-finite values at (name, date) {non_finite}. Leave missing observations "
            "out of a date's arrays."
        )
    cast_data = {
        name: [np.asarray(value, dtype=placeholders[name].dtype) for value in values]
        for name, values in data.items()
    }
    inexact = [
        (name, date)
        for name, values in data.items()
        if np.dtype(placeholders[name].dtype).kind in "iub"
        for date, value in enumerate(values)
        if not np.array_equal(cast_data[name][date], value)
    ]
    if inexact:
        raise ValueError(
            f"data values at (name, date) {inexact} are not exactly representable as their "
            "variable's integer or boolean dtype."
        )

    initial_mean, initial_covariance, matrices = _initial_moments_and_matrices(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data=statespace_data,
    )

    predicted_mean = initial_mean.type()
    predicted_covariance = initial_covariance.type()
    next_mean, next_covariance, recorded = _date_update(
        observation_logps=observation_logps,
        parameters=parameters,
        curvature=curvature,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        matrices=matrices,
        line_search=line_search,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
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
            mean, covariance, *[values_by_date[date] for values_by_date in cast_data.values()]
        )
        for name, value in zip(recorded, values, strict=True):
            history[name].append(value)

    stacked = {name: np.stack(values) for name, values in history.items()}
    # The backward recursion reads only the filtered moments, at positions 0 and 3 of the filter
    # outputs, and the transition, so it needs no observations.
    smoothed_states, smoothed_covariances = RTSSmoother(cov_jitter=0.0).build_graph(
        None,
        [initial_mean, initial_covariance, *(matrices[name] for name in _MATRIX_NDIMS)],
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


def scan_filter(
    *,
    fused: FusedStateSpace,
    statespace_parameters: Mapping[str, TensorLike],
    statespace_data: Mapping[str, TensorLike],
    data: Mapping[str, TensorLike],
    observed_masks: Mapping[str, TensorLike],
    curvature: Literal["gauss_newton", "exact"],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> dict[str, TensorVariable]:
    r"""
    Build the filter over dates as a ``scan`` over the step from :func:`filter_step`.

    Each date's values are padded to a common size, and ``observed_masks`` marks the observations
    that are real. A padded observation contributes nothing to its date's log-likelihood, but its
    value has to have a finite log-likelihood. A date with no real observations is a pure
    prediction.

    Parameters
    ----------
    fused : FusedStateSpace
        The model and the statespace over its parameters.
    statespace_parameters : mapping of str to TensorLike
        A value, constant or symbolic, for each name in the statespace's ``param_names``.
    statespace_data : mapping of str to TensorLike
        A value, constant or symbolic, for each name in the statespace's ``data_names``. The values
        cannot make the statespace matrices vary over time.
    data : mapping of str to TensorLike
        For each data variable of the model and each observed variable whose observations are
        constants, its padded values with the dates on the first axis.
    observed_masks : mapping of str to TensorLike
        For each observed variable, a boolean array with the dates on the first axis and the
        variable's batch shape after it, True at its real observations.
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
    filtered : dict of str to TensorVariable
        Stacked over dates, the outputs :func:`run_filter` returns in ``posterior_predictive``
        except the smoothed ones, and the per-date diagnostics it returns in ``diagnostics``.
    """
    initial_mean, initial_covariance, matrices = _initial_moments_and_matrices(
        fused=fused,
        statespace_parameters=statespace_parameters,
        statespace_data=statespace_data,
    )
    data_names, mask_names = list(data), list(observed_masks)

    def step(*arguments):
        sequence_count = len(data_names) + len(mask_names)
        date_data = dict(zip(data_names, arguments[: len(data_names)], strict=True))
        date_masks = dict(zip(mask_names, arguments[len(data_names) : sequence_count], strict=True))
        predicted_mean, predicted_covariance, *date_matrices = arguments[sequence_count:]

        parameters = pt.vector("parameters", shape=(len(fused.statespace.observed_states),))
        observation_logps = fused.observation_logp(parameters=parameters, data=date_data)
        if set(date_masks) != set(observation_logps):
            raise ValueError(
                f"observed_masks has to give a mask for exactly {sorted(observation_logps)}."
            )
        masked_logps = {
            name: (pt.where(date_masks[name], pointwise_logp, 0.0), distribution_parameters)
            for name, (pointwise_logp, distribution_parameters) in observation_logps.items()
        }
        next_mean, next_covariance, recorded = _date_update(
            observation_logps=masked_logps,
            parameters=parameters,
            curvature=curvature,
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            matrices=dict(zip(matrices, date_matrices, strict=True)),
            line_search=line_search,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return [next_mean, next_covariance, *recorded.values()]

    next_means, next_covariances, *recorded = pytensor.scan(
        step,
        sequences=[
            *(pt.as_tensor_variable(values) for values in data.values()),
            *(pt.as_tensor_variable(masks) for masks in observed_masks.values()),
        ],
        outputs_info=[initial_mean, initial_covariance, *[None] * len(_RECORDED_NAMES)],
        non_sequences=list(matrices.values()),
        name="fused_filter",
        return_updates=False,
    )
    return {
        "predicted_states": pt.concatenate([initial_mean[None], next_means[:-1]], axis=0),
        "predicted_covariances": pt.concatenate(
            [initial_covariance[None], next_covariances[:-1]], axis=0
        ),
        **dict(zip(_RECORDED_NAMES, recorded, strict=True)),
    }


def register_fused_likelihood(
    *,
    name: str,
    fused: FusedStateSpace,
    data: Mapping[str, TensorLike],
    observed_masks: Mapping[str, TensorLike],
    curvature: Literal["gauss_newton", "exact"],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> TensorVariable:
    r"""
    Add the log-likelihood of the filter from :func:`scan_filter` to the active model.

    The statespace's parameters and data are the active model's variables with the names in its
    ``param_names`` and ``data_names``, as in :meth:`PyMCStateSpace.build_statespace_graph`.

    Parameters
    ----------
    name : str
        Name of the potential that holds the log-likelihood.
    fused : FusedStateSpace
        The model and the statespace over its parameters.
    data : mapping of str to TensorLike
        For each data variable of the model and each observed variable whose observations are
        constants, its padded values with the dates on the first axis.
    observed_masks : mapping of str to TensorLike
        For each observed variable, a boolean array with the dates on the first axis and the
        variable's batch shape after it, True at its real observations.
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
    loglike : TensorVariable
        The potential, the sum over dates of the ``loglike_obs`` from :func:`scan_filter`.
    """
    model = modelcontext(None)
    scanned = scan_filter(
        fused=fused,
        statespace_parameters={
            parameter: model[parameter] for parameter in fused.statespace.param_names
        },
        statespace_data={variable: model[variable] for variable in fused.statespace.data_names},
        data=data,
        observed_masks=observed_masks,
        curvature=curvature,
        line_search=line_search,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return pm.Potential(name, scanned["loglike_obs"].sum())


_MATRIX_NDIMS = {"c": 1, "d": 1, "T": 2, "Z": 2, "R": 2, "H": 2, "Q": 2}

_RECORDED_NAMES = (
    "filtered_states",
    "filtered_covariances",
    "predicted_observed_states",
    "predicted_observed_covariances",
    "loglike_obs",
    "iterations",
    "evaluations",
    "search_failed",
    "invalid_direction",
    "converged",
)


def _initial_moments_and_matrices(
    *,
    fused: FusedStateSpace,
    statespace_parameters: Mapping[str, TensorLike],
    statespace_data: Mapping[str, TensorLike],
) -> tuple[TensorVariable, TensorVariable, dict[str, TensorVariable]]:
    initial_mean, initial_covariance, *time_invariant = fused.statespace_matrices(
        statespace_parameters=statespace_parameters, statespace_data=statespace_data
    )
    matrices = dict(zip(_MATRIX_NDIMS, time_invariant, strict=True))
    time_varying = [name for name, matrix in matrices.items() if matrix.ndim > _MATRIX_NDIMS[name]]
    if time_varying:
        raise ValueError(f"Statespace matrices {time_varying} vary over time.")
    return initial_mean, initial_covariance, matrices


def _date_update(
    *,
    observation_logps: Mapping[str, tuple[TensorVariable, list[TensorVariable] | None]],
    parameters: TensorVariable,
    curvature: Literal["gauss_newton", "exact"],
    predicted_mean: TensorVariable,
    predicted_covariance: TensorVariable,
    matrices: Mapping[str, TensorVariable],
    line_search: LineSearch,
    tolerance: float,
    max_iterations: int,
) -> tuple[TensorVariable, TensorVariable, dict[str, TensorVariable]]:
    logp = pt.sum([pointwise_logp.sum() for pointwise_logp, _ in observation_logps.values()])
    curvature_matrix = _curvature(
        kind=curvature,
        logp=logp,
        observation_logps=observation_logps,
        parameters=parameters,
    )
    filtered_mean, filtered_covariance, next_mean, next_covariance, site, parameter_mode = (
        filter_step(
            logp=logp,
            curvature=curvature_matrix,
            parameters=parameters,
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            **matrices,
            line_search=line_search,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    parameter_mean = matrices["Z"] @ predicted_mean + matrices["d"]
    parameter_covariance = quad_form_sym(matrices["Z"], predicted_covariance) + matrices["H"]
    loglike = laplace_log_likelihood(
        mode=parameter_mode,
        logp=logp,
        curvature=curvature_matrix,
        parameters=parameters,
        predictive_covariance=parameter_covariance,
    )
    recorded = [
        filtered_mean,
        filtered_covariance,
        parameter_mean,
        parameter_covariance,
        loglike,
        site.iterations,
        site.evaluations,
        site.search_failed,
        site.invalid_direction,
        site.converged,
    ]
    return next_mean, next_covariance, dict(zip(_RECORDED_NAMES, recorded, strict=True))


def _value_with_gradient_of(*, value: TensorVariable, gradient: TensorVariable) -> TensorVariable:
    """Return the value of ``value`` with the gradient of ``gradient``."""
    return gradient + disconnected_grad(value - gradient)


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
