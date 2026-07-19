from collections.abc import Sequence

from pymc.model.core import Model
from pymc.model.fgraph import fgraph_from_model
from pytensor.compile import SharedVariable
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Constant, Variable, equal_computations
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor.random.type import RandomType


def equal_computations_up_to_root(
    xs: Sequence[Variable], ys: Sequence[Variable], ignore_rng_values=True, strict_dtype=True
) -> bool:
    # Check if graphs are equivalent even if root variables have distinct identities

    x_graph_inputs = [var for var in graph_inputs(xs) if not isinstance(var, Constant)]
    y_graph_inputs = [var for var in graph_inputs(ys) if not isinstance(var, Constant)]
    if len(x_graph_inputs) != len(y_graph_inputs):
        return False
    for x, y in zip(x_graph_inputs, y_graph_inputs):
        if x.type != y.type:
            return False
        if x.name != y.name:
            return False
        if isinstance(x, SharedVariable):
            # if not isinstance(y, SharedVariable):
            #     return False
            if isinstance(x.type, RandomType) and ignore_rng_values:
                continue
            if not x.type.values_eq(x.get_value(), y.get_value()):
                return False

    return equal_computations(
        xs, ys, in_xs=x_graph_inputs, in_ys=y_graph_inputs, strict_dtype=strict_dtype
    )


def equivalent_models(
    model1: Model, model2: Model, *, strict_dtype: bool = True, canonicalize: bool = False
) -> bool:
    """Check whether two PyMC models are equivalent.

    With ``canonicalize=True`` both model graphs are canonicalized before the
    comparison, so models that differ only by canonicalization (e.g. one that
    went through a rewrite pass and one that did not) compare equal.

    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_extras.utils.model_equivalence import equivalent_models

        with pm.Model() as m1:
            x = pm.Normal("x")
            y = pm.Normal("y", x)

        with pm.Model() as m2:
            x = pm.Normal("x")
            y = pm.Normal("y", x + 1)

        with pm.Model() as m3:
            x = pm.Normal("x")
            y = pm.Normal("y", x)

        assert not equivalent_models(m1, m2)
        assert equivalent_models(m1, m3)

    """
    fgraph1, _ = fgraph_from_model(model1)
    fgraph2, _ = fgraph_from_model(model2)
    if canonicalize:
        canon = optdb.query("+canonicalize", "-local_eager_useless_unbatched_blockwise")
        canon.rewrite(fgraph1)
        canon.rewrite(fgraph2)
    # Model variable order is incidental (it follows graph construction
    # history); model variables are uniquely named, so compare by name.
    outputs1 = sorted(fgraph1.outputs, key=lambda var: var.name)
    outputs2 = sorted(fgraph2.outputs, key=lambda var: var.name)
    if [var.name for var in outputs1] != [var.name for var in outputs2]:
        return False
    return equal_computations_up_to_root(outputs1, outputs2, strict_dtype=strict_dtype)
