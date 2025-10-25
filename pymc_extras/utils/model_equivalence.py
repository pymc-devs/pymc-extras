from collections.abc import Sequence

from pymc.model.core import Model
from pymc.model.fgraph import fgraph_from_model
from pytensor import Variable
from pytensor.compile import SharedVariable
from pytensor.graph import Constant, graph_inputs
from pytensor.graph.basic import equal_computations
from pytensor.tensor.random.type import RandomType


def equal_computations_up_to_root(
    xs: Sequence[Variable], ys: Sequence[Variable], ignore_rng_values=True
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

    return equal_computations(xs, ys, in_xs=x_graph_inputs, in_ys=y_graph_inputs)


def equivalent_models(model1: Model, model2: Model) -> bool:
    """Check whether two PyMC models are equivalent.

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
    return equal_computations_up_to_root(fgraph1.outputs, fgraph2.outputs)
