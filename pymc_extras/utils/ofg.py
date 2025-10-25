from collections.abc import Sequence

from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Variable
from pytensor.graph.replace import clone_replace


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> list[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    return clone_replace(
        op.inner_outputs,
        replace=tuple(zip(op.inner_inputs, inputs)),
    )
