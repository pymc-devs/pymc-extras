import pytensor
import pytensor.tensor as pt

from pytensor.graph.replace import vectorize_graph
from pytensor.tensor.utils import _parse_gufunc_signature
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_extras.statespace.filters.kalman_filter import BaseFilter
from pymc_extras.statespace.filters.utilities import PARAM_NAMES
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_NAMES,
    MATRIX_DIMS,
    OBS_STATE_DIM,
    RESERVED_DIMS,
    TIME_DIM,
)

FILTER_INPUT_DIMS = {"data": (OBS_STATE_DIM,), **MATRIX_DIMS}

# m: states, p: observed states, r: shocks. Inputs follow (y, a, P, *PARAM_NAMES), outputs follow
# the filter's per-step outputs (a_filtered, a_hat, y_hat, P_filtered, P_hat, F, ll).
STEP_SIGNATURE = (
    "(p),(m),(m,m),(m),(p),(m,m),(p,m),(m,r),(p,p),(r,r)->(m),(m),(p),(m,m),(m,m),(p,p),()"
)


def find_batch_dims(inputs: dict[str, XTensorVariable]) -> tuple[str, ...]:
    """
    Return every dim that is neither ``time`` nor a core dim, in order of first appearance.

    Parameters
    ----------
    inputs : dict of str to XTensorVariable
        Filter inputs keyed by ``data`` and the short matrix names.

    Returns
    -------
    batch_dims : tuple of str
        Batch dims in the order the lowered tensors carry them.
    """
    batch_dims = {}
    for variable in inputs.values():
        batch_dims.update(dict.fromkeys(dim for dim in variable.dims if dim not in RESERVED_DIMS))
    return tuple(batch_dims)


def lower_to_tensor(
    variable: XTensorVariable, core_dims: tuple[str, ...], batch_sizes: dict[str, pt.TensorVariable]
) -> pt.TensorVariable:
    """
    Lower an xtensor to a tensor laid out as ``(time?, *batch, *core)``.

    Parameters
    ----------
    variable : XTensorVariable
        Filter input with named dims.
    core_dims : tuple of str
        Core dims of this input, in the order the filter step expects them.
    batch_sizes : dict of str to TensorVariable
        Batch dims in canonical order, mapped to the size each missing dim is created with.

    Returns
    -------
    tensor : TensorVariable
        The input with missing batch dims added at ``batch_sizes`` and all dims transposed.
    """
    missing = {dim: size for dim, size in batch_sizes.items() if dim not in variable.dims}
    if missing:
        variable = variable.expand_dims(missing)

    time_dims = (TIME_DIM,) if TIME_DIM in variable.dims else ()
    return variable.transpose(*time_dims, *batch_sizes, *core_dims).values


def build_batched_filter_graph(
    kalman_filter: BaseFilter, data, x0, P0, c, d, T, Z, R, H, Q
) -> dict[str, XTensorVariable]:
    """
    Run ``kalman_filter`` over every combination of batch dims carried by the inputs.

    A matrix that has a ``time`` dim is passed to ``scan`` as a sequence. Every dim outside
    ``time`` and the core dims in ``MATRIX_DIMS`` is a batch dim, and inputs that lack a batch dim
    are broadcast over it.

    Parameters
    ----------
    kalman_filter : BaseFilter
        Filter whose ``handle_missing_values``, ``update`` and ``predict`` define one step.
    data : XTensorVariable
        Observations with dims ``(time, *batch, observed_state)``.
    x0, P0, c, d, T, Z, R, H, Q : XTensorVariable
        Statespace matrices with the core dims given in ``MATRIX_DIMS``, plus optional ``time``
        and batch dims.

    Returns
    -------
    outputs : dict of str to XTensorVariable
        The filter outputs named in ``FILTER_OUTPUT_NAMES`` plus ``loglike_obs``, each with dims
        ``(time, *batch, *core)``.
    """
    inputs = {
        name: as_xtensor(variable)
        for name, variable in zip(
            ["data", "x0", "P0", *PARAM_NAMES], [data, x0, P0, c, d, T, Z, R, H, Q], strict=True
        )
    }
    if TIME_DIM not in inputs["data"].dims:
        raise ValueError(f"data must have a {TIME_DIM!r} dim, got {inputs['data'].dims}")

    batch_dims = find_batch_dims(inputs)
    batch_sizes = {
        dim: next(variable.sizes[dim] for variable in inputs.values() if dim in variable.dims)
        for dim in batch_dims
    }
    singleton_sizes = dict.fromkeys(batch_dims, 1)

    # The scan carry must have the full batch shape from the first step, so only the initial
    # state is broadcast to the true batch sizes. Everything else broadcasts inside the step.
    tensors = {
        name: lower_to_tensor(
            variable,
            core_dims=FILTER_INPUT_DIMS[name],
            batch_sizes=batch_sizes if name in ("x0", "P0") else singleton_sizes,
        )
        for name, variable in inputs.items()
    }

    sequence_names = ["data", *(name for name in PARAM_NAMES if TIME_DIM in inputs[name].dims)]
    non_sequence_names = [name for name in PARAM_NAMES if name not in sequence_names]
    batched_step = make_batched_step(
        kalman_filter, step_inputs=[tensors[name] for name in ["data", "x0", "P0", *PARAM_NAMES]]
    )

    def scan_step(*args):
        n_sequences = len(sequence_names)
        values = dict(zip(sequence_names, args[:n_sequences], strict=True))
        a, P = args[n_sequences : n_sequences + 2]
        values.update(zip(non_sequence_names, args[n_sequences + 2 :], strict=True))
        return batched_step(values["data"], a, P, *(values[name] for name in PARAM_NAMES))

    a0, P0 = tensors["x0"], tensors["P0"]
    results = pytensor.scan(
        scan_step,
        sequences=[tensors[name] for name in sequence_names],
        outputs_info=[None, a0, None, None, P0, None, None],
        non_sequences=[tensors[name] for name in non_sequence_names],
        name="batched_forward_kalman_pass",
        strict=False,
        return_updates=False,
    )

    a_filtered, a_hat, y_hat, P_filtered, P_hat, F, loglike_obs = results
    a_predicted = pt.concatenate([a0[None], a_hat[:-1]], axis=0)
    P_predicted = pt.concatenate([P0[None], P_hat[:-1]], axis=0)

    state_dim, state_aux_dim = MATRIX_DIMS["P0"]
    obs_dim, obs_aux_dim = MATRIX_DIMS["H"]
    leading_dims = (TIME_DIM, *batch_dims)
    dims_by_name = {
        "filtered_states": (*leading_dims, state_dim),
        "predicted_states": (*leading_dims, state_dim),
        "filtered_covariances": (*leading_dims, state_dim, state_aux_dim),
        "predicted_covariances": (*leading_dims, state_dim, state_aux_dim),
        "predicted_observed_states": (*leading_dims, obs_dim),
        "predicted_observed_covariances": (*leading_dims, obs_dim, obs_aux_dim),
        "loglike_obs": leading_dims,
    }
    tensors_by_name = dict(
        zip(
            [*FILTER_OUTPUT_NAMES, "loglike_obs"],
            [a_filtered, a_predicted, P_filtered, P_predicted, y_hat, F, loglike_obs],
            strict=True,
        )
    )

    return {
        name: as_xtensor(tensor, dims=dims_by_name[name], name=name)
        for name, tensor in tensors_by_name.items()
    }


def make_batched_step(kalman_filter: BaseFilter, step_inputs: list[pt.TensorVariable]):
    """
    Return one filter step that broadcasts over the batch dims of its inputs.

    The core step is built on inputs with the static core shapes of ``step_inputs``. Shapes known
    at that point fold to constants before vectorizing, so a core-sized identity stays a constant
    and is not repeated along the batch dims.

    Parameters
    ----------
    kalman_filter : BaseFilter
        Filter whose step is batched.
    step_inputs : list of TensorVariable
        Tensors laid out as ``(time?, *batch, *core)``, in the input order of ``STEP_SIGNATURE``.

    Returns
    -------
    batched_step : callable
        Takes the per-step batched inputs and returns the batched step outputs.
    """
    core_input_dims, _ = _parse_gufunc_signature(STEP_SIGNATURE)
    core_inputs = [
        pt.tensor(dtype=tensor.dtype, shape=tensor.type.shape[tensor.ndim - len(dims) :])
        for tensor, dims in zip(step_inputs, core_input_dims, strict=True)
    ]
    core_outputs = kalman_filter.filter_step(*core_inputs)

    def batched_step(*inputs):
        return vectorize_graph(
            list(core_outputs), replace=dict(zip(core_inputs, inputs, strict=True))
        )

    return batched_step
