from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pymc import SymbolicRandomVariable
from pymc.distributions.shape_utils import change_dist_size
from pytensor import tensor as pt
from pytensor.compile.ops import ViewOp
from pytensor.graph import ancestors, vectorize_graph
from pytensor.graph.replace import _vectorize_node, _vectorize_not_needed
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.random.op import RandomVariable

# TODO: Backport to pytensor. ViewOp (and its subclasses DisconnectedGrad, ZeroGrad,
# GradClip, ...) are rank-polymorphic identities: they pass their input through unchanged
# at any shape and only carry gradient metadata. Without a vectorize dispatch they hit the
# generic fallback that wraps them in a (zero-batch) Blockwise, which (a) is pure overhead
# and (b) hides the inner ViewOp from `local_view_op`, so the path-derivative
# `disconnected_grad` never gets stripped from the forward graph. Re-applying the same op to
# the batched input is exact for any identity op, so `_vectorize_not_needed` is correct here.
if ViewOp not in _vectorize_node.registry:
    _vectorize_node.register(ViewOp, _vectorize_not_needed)


def vectorize_random_graph(
    graph: Sequence[TensorVariable], batch_draws: TensorLike
) -> list[TensorVariable]:
    # Find the root random nodes
    rvs = tuple(
        var
        for var in ancestors(graph)
        if (
            var.owner is not None
            and isinstance(var.owner.op, RandomVariable | SymbolicRandomVariable)
        )
    )
    rvs_set = set(rvs)
    root_rvs = tuple(rv for rv in rvs if not (set(rv.owner.inputs) & rvs_set))

    # Vectorize graph by vectorizing root RVs
    batch_draws = pt.as_tensor(batch_draws, dtype=int)
    vectorized_replacements = {
        root_rv: change_dist_size(root_rv, new_size=batch_draws, expand=True)
        for root_rv in root_rvs
    }
    return cast(list[TensorVariable], vectorize_graph(graph, replace=vectorized_replacements))
