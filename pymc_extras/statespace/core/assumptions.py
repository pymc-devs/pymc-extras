from pytensor.assumptions import (
    AssumptionKey,
    FactState,
    SpecifyAssumptions,
    check_assumption,
    register_assumption,
    specify_assumption_rule,
)
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.basic import Join
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.rewriting.assumptions import _KEY_BY_NAME
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
)

TIME_VARYING = AssumptionKey("time_varying", "tv")

# Lets the feature read a declaration back off the node ``declare_time_varying`` inserts.
register_assumption(TIME_VARYING, SpecifyAssumptions)(specify_assumption_rule)

# The rewrite that drains declarations into the assumption cache snapshots the key registry when
# pytensor imports, which is always before this module, so add ours to the snapshot by hand.
_KEY_BY_NAME[TIME_VARYING.name] = TIME_VARYING


@register_assumption(TIME_VARYING, Subtensor, AdvancedSubtensor)
def _subtensor_keeps_time_axis(key, op, feature, fgraph, node, input_states):
    """Slicing the leading axis shortens the sequence; indexing it removes the axis."""
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    index = op.idx_list
    keeps_time = not index or isinstance(index[0], slice)
    return [FactState.TRUE if keeps_time else FactState.FALSE]


@register_assumption(TIME_VARYING, IncSubtensor, AdvancedIncSubtensor)
def _set_subtensor_keeps_time_axis(key, op, feature, fgraph, node, input_states):
    """Writing into a matrix leaves its shape alone, so the base decides."""
    return [input_states[0]]


@register_assumption(TIME_VARYING, Elemwise)
def _elemwise_keeps_time_axis(key, op, feature, fgraph, node, input_states):
    """Elementwise work over a sequence is still a sequence."""
    if any(state is FactState.TRUE for state in input_states):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(TIME_VARYING, Join)
def _join_keeps_time_axis(key, op, feature, fgraph, node, input_states):
    """Concatenation keeps the time axis.

    Joining along it appends timesteps; joining along any other axis leaves it in place, since
    ``Join`` requires every input to agree on the axes it does not join.
    """
    if any(state is FactState.TRUE for state in input_states):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(TIME_VARYING, Blockwise)
def _blockwise_keeps_a_batched_time_axis(key, op, feature, fgraph, node, input_states):
    """A ``Blockwise`` broadcasts its batch dims onto every output, so a time axis to the left of
    an input's core dims survives the call.

    ``Component.__add__`` builds the combined transition, selection and state covariance with
    ``pt.linalg.block_diag``, and the filter takes a Cholesky of the covariances, so without this
    the declaration is lost exactly where components are combined. A declaration sitting on a core
    axis instead says nothing about the result, since the core op is free to reshape it.
    """
    n_outputs = len(op.outputs_sig)
    batched_time = any(
        state is FactState.TRUE and variable.type.ndim > len(core_dims)
        for state, core_dims, variable in zip(input_states, op.inputs_sig, node.inputs, strict=True)
    )
    return [FactState.TRUE if batched_time else FactState.UNKNOWN] * n_outputs


def declare_time_varying(tensor: Variable) -> Variable:
    """
    Return ``tensor`` carrying the time-varying assumption.

    Parameters
    ----------
    tensor : TensorVariable
        Matrix whose leading axis indexes time.

    Returns
    -------
    declared : TensorVariable
        A no-op view of ``tensor`` that downstream code can ask about.
    """
    return SpecifyAssumptions({TIME_VARYING.name: FactState.TRUE})(tensor)


def is_time_varying(*tensors: Variable) -> list[bool]:
    """
    Report which tensors still carry a declared time axis.

    Parameters
    ----------
    *tensors : TensorVariable
        Tensors to inspect.

    Returns
    -------
    flags : list of bool
        One entry per tensor, True when the time axis survives.
    """
    fgraph = FunctionGraph(outputs=list(tensors), clone=False)
    return [check_assumption(fgraph, tensor, TIME_VARYING) for tensor in tensors]
