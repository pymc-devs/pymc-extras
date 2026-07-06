from collections.abc import Sequence
from functools import singledispatch

from pymc.distributions.distribution import _support_point, support_point
from pymc.logprob.abstract import MeasurableOp
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.tensor.random.type import RandomType


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> list[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    outputs = op.fgraph.bind(list(inputs))
    for inner_out, out in zip(op.inner_outputs, outputs):
        out.name = inner_out.name
    return outputs


class MarginalRV(OpFromGraph, MeasurableOp):
    """Base class for supported MarginalRVs.

    The name and dims of the marginalized model variable, together with the
    number of dependent RVs, are stored explicitly (``marginalized_name``,
    ``marginalized_dims``, ``n_dependent_rvs``) because pytensor makes no
    guarantee that variable names/metadata survive cloning and rewrites.
    """

    def __init__(
        self,
        *args,
        marginalized_name: str,
        marginalized_dims,
        n_dependent_rvs: int,
        **kwargs,
    ) -> None:
        self.marginalized_name = marginalized_name
        self.marginalized_dims = marginalized_dims
        self.n_dependent_rvs = n_dependent_rvs
        super().__init__(*args, **kwargs)


@_support_point.register(MarginalRV)
def _support_point_marginal_rv(op, rv, *inputs):
    outputs = rv.owner.outputs

    inlined = inline_ofg_outputs(op, inputs)
    inlined_rv = inlined[outputs.index(rv)]
    marginalized_rv, *other_dependent_rvs = (
        out for out in inlined if out is not inlined_rv and not isinstance(out.type, RandomType)
    )

    # The support point of rv is its inner support point, with the marginalized
    # variable pinned to its own support point and the other dependents pinned
    # to the node's outer outputs.
    replacements = {marginalized_rv: support_point(marginalized_rv)}
    replacements.update(
        (other_rv, outputs[inlined.index(other_rv)]) for other_rv in other_dependent_rvs
    )
    return graph_replace(support_point(inlined_rv), replacements, strict=False)


@singledispatch
def marginalized_conditional(op, inputs, dep_rvs):
    """Build the conditional distribution of a marginalized variable given its dependents.

    Dispatches on the MarginalRV op type.

    The inner graph of a MarginalRV is generative: it draws the marginalized
    variable and then the dependents given it, factoring as
    ``p(marginalized | inputs) * p(dependents | marginalized, inputs)``.
    This function returns the reverse factor
    ``p(marginalized | dependents, inputs)``: a Categorical over the
    enumerated domain weighted by the joint logp for finite discrete
    marginals, the conjugate posterior Normal for Normal-Normal.

    Parameters
    ----------
    op : MarginalRV
        The marginal op whose marginalized variable is being conditioned.
    inputs : Sequence[Variable]
        Replacements for the node inputs, already expressed in the caller's
        target graph.
    dep_rvs : Sequence[Variable]
        The variables the dependents are conditioned on (model variables or
        observed data), one per dependent output.

    Returns
    -------
    Variable
        A random variable distributed as ``p(marginalized | dependents, inputs)``,
        expressed over ``inputs`` and ``dep_rvs``.
    """
    raise NotImplementedError(
        f"Cannot recover marginalized variable with distribution {type(op).__name__}"
    )
