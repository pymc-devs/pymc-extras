"""Support for marginalizing ``pymc.dims`` models.

A ``pymc.dims`` model is built out of ``XTensorVariable``s, whose dims are names rather than
positions. Rather than teach every marginalization strategy to reason about named dims, the
marginal subgraph is lowered to plain tensors before it is recognized, so the existing
strategies, their dim-connection analysis and their logps all apply unchanged. The resulting
``MarginalRV`` outputs are then cast back, so the model's variables keep their dims.
"""

from pytensor.compile import optdb
from pytensor.graph import FunctionGraph
from pytensor.graph.replace import graph_replace
from pytensor.tensor import tensor
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.type import XTensorType


def lower_marginal_subgraph(inputs, outputs):
    """Rewrite an xtensor subgraph into the equivalent tensor one.

    Returns ``(inner_inputs, outer_inputs, outputs)``. ``inner_inputs`` are fresh placeholders
    that cut the subgraph at its boundary, so the lowering stops there instead of descending
    into the rest of the model; they are what the MarginalRV's inner graph is written over.
    ``outer_inputs`` are the matching expressions in the model's own graph, which the resulting
    op is applied to. Variables that were already tensors are passed through untouched, so a
    subgraph that mixes dims and non-dims variables is fine.
    """
    inner_inputs = []
    outer_inputs = []
    replacements = {}
    for inp in inputs:
        if isinstance(inp.type, XTensorType):
            placeholder = tensor(dtype=inp.type.dtype, shape=inp.type.shape)
            replacements[inp] = xtensor_from_tensor(placeholder, dims=inp.type.dims)
            inner_inputs.append(placeholder)
            outer_inputs.append(tensor_from_xtensor(inp))
        else:
            inner_inputs.append(inp)
            outer_inputs.append(inp)

    # Casting the outputs to tensors leaves round-trips that the lowering rewrites cancel,
    # so what comes out is a graph of plain tensor Ops.
    cast_outputs = [
        tensor_from_xtensor(out) if isinstance(out.type, XTensorType) else out for out in outputs
    ]
    if replacements:
        cast_outputs = graph_replace(cast_outputs, replacements, strict=False)

    # Let the FunctionGraph collect its own inputs: the subgraph also reaches shared variables
    # (the RNGs the RVs draw from) that aren't part of the boundary we were handed.
    fgraph = FunctionGraph(outputs=cast_outputs, clone=False)
    optdb.query("+lower_xtensor").rewrite(fgraph)
    return inner_inputs, outer_inputs, list(fgraph.outputs)


def output_dims_of(node):
    """The dims of each output of a MarginalSubgraph node, or None for a plain tensor model.

    The marker's outputs are ``[marginalized_rv, *dependents]`` and carry the types of the model
    variables they stand for, so their dims are readable straight off them. An individual entry
    is None when that variable is a plain tensor, which happens in a model that mixes dims and
    non-dims variables.

    This is not the same as ``marginalized_dims``, which is the pymc model's dims metadata and
    exists for plain tensor models too. These are the dims the variables actually carry.
    """
    if not any(isinstance(out.type, XTensorType) for out in node.outputs):
        return None
    return tuple(
        out.type.dims if isinstance(out.type, XTensorType) else None for out in node.outputs
    )
