from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pymc import SymbolicRandomVariable
from pymc.distributions.shape_utils import change_dist_size
from pytensor import config
from pytensor import tensor as pt
from pytensor.graph import FunctionGraph, ancestors, vectorize_graph
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.basic import infer_shape_db
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.rewriting.shape import ShapeFeature


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
    return cast(
        list[TensorVariable], vectorize_graph(graph, replace=vectorized_replacements)
    )


def get_symbolic_rv_shapes(
    rvs: Sequence[TensorVariable], raise_if_rvs_in_graph: bool = True
) -> tuple[TensorVariable, ...]:
    # TODO: Move me to pymc.pytensorf, this is needed often

    rv_shapes = [rv.shape for rv in rvs]
    shape_fg = FunctionGraph(outputs=rv_shapes, features=[ShapeFeature()], clone=True)
    with config.change_flags(optdb__max_use_ratio=10, cxx=""):
        infer_shape_db.default_query.rewrite(shape_fg)
    rv_shapes = shape_fg.outputs

    if raise_if_rvs_in_graph and (overlap := (set(rvs) & set(ancestors(rv_shapes)))):
        raise ValueError(f"rv_shapes still depend the following rvs {overlap}")

    return cast(tuple[TensorVariable, ...], tuple(rv_shapes))
