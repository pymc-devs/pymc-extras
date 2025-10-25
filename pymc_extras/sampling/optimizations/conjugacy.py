from collections.abc import Sequence
from functools import partial

from pymc.distributions import Beta, Binomial
from pymc.model.fgraph import ModelFreeRV, ModelValuedVar, model_free_rv
from pymc.pytensorf import collect_default_updates
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.subtensor import _sum_grad_over_bcasted_dims as sum_bcasted_dims

from pymc_extras.sampling.optimizations.conjugate_sampler import (
    ConjugateRV,
)
from pymc_extras.sampling.optimizations.optimize import posterior_optimization_db


def register_conjugacy_rewrites_variants(rewrite_fn, tracks=(ModelFreeRV,)):
    """Register a rewrite function and its force variant in the posterior optimization DB."""
    name = rewrite_fn.__name__

    rewrite_fn_default = partial(rewrite_fn, eager=False)
    rewrite_fn_default.__name__ = name
    rewrite_default = node_rewriter(tracks=tracks)(rewrite_fn_default)

    rewrite_fn_eager = partial(rewrite_fn, eager=True)
    rewrite_fn_eager.__name__ = f"{name}_eager"
    rewrite_eager = node_rewriter(tracks=tracks)(rewrite_fn_eager)

    posterior_optimization_db.register(
        rewrite_default.__name__,
        rewrite_default,
        "default",
        "conjugacy",
    )

    posterior_optimization_db.register(
        rewrite_eager.__name__,
        rewrite_eager,
        "non-default",
        "conjugacy-eager",
    )

    return rewrite_default, rewrite_eager


def has_free_rv_ancestor(vars: Variable | Sequence[Variable]) -> bool:
    """Return True if any of the variables have a model variable as an ancestor."""
    if not isinstance(vars, Sequence):
        vars = (vars,)

    # TODO: It should stop at observed RVs, it doesn't matter if they have a free RV above
    #  Did not implement due to laziness and it being a rare case
    return any(
        var.owner is not None and isinstance(var.owner.op, ModelFreeRV) for var in ancestors(vars)
    )


def get_model_var_of_rv(fgraph: FunctionGraph, rv: Variable) -> Variable:
    """Return the Model dummy var that wraps the RV"""
    for client, _ in fgraph.clients[rv]:
        if isinstance(client.op, ModelValuedVar):
            return client.outputs[0]


def get_dist_params(rv: Variable) -> tuple[Variable]:
    return rv.owner.op.dist_params(rv.owner)


def get_size_param(rv: Variable) -> Variable:
    return rv.owner.op.size_param(rv.owner)


def rv_used_by(
    fgraph: FunctionGraph,
    rv: Variable,
    used_by_type: type,
    used_as_arg_idx: int | Sequence[int],
    arg_idx_offset: int = 2,  # Ignore the first two arguments (rng and size)
    strict: bool = True,
) -> list[Variable]:
    """Return the RVs that use `rv` as an argument in an operation of type `used_by_type`.

    RV may be used directly or broadcasted before being used.

    Parameters
    ----------
    fgraph : FunctionGraph
        The function graph containing the RVs
    rv : Variable
        The RV to check for uses.
    used_by_type : type
        The type of operation that may use the RV.
    used_as_arg_idx : int | Sequence[int]
        The index of the RV in the operation's inputs.
    strict : bool, default=True
        If True, return no results when the RV is used in an unrecognized way.

    """
    if isinstance(used_as_arg_idx, int):
        used_as_arg_idx = (used_as_arg_idx,)
    used_as_arg_idx = tuple(arg_idx + arg_idx_offset for arg_idx in used_as_arg_idx)

    clients = fgraph.clients
    used_by: list[Variable] = []
    for client, inp_idx in clients[rv]:
        if isinstance(client.op, Output):
            continue

        if isinstance(client.op, used_by_type) and inp_idx in used_as_arg_idx:
            # RV is directly used by the RV type
            used_by.append(client.default_output())

        elif isinstance(client.op, DimShuffle) and client.op.is_left_expand_dims:
            for sub_client, sub_inp_idx in clients[client.outputs[0]]:
                if isinstance(sub_client.op, used_by_type) and sub_inp_idx in used_as_arg_idx:
                    # RV is broadcasted and then used by the RV type
                    used_by.append(sub_client.default_output())
                elif strict:
                    # Some other unrecognized use, bail out
                    return []
        elif strict:
            # Some other unrecognized use, bail out
            return []

    return used_by


def wrap_rv_and_conjugate_rv(
    fgraph: FunctionGraph, rv: Variable, conjugate_rv: Variable, inputs: Sequence[Variable]
) -> Variable:
    """Wrap the RV and its conjugate posterior RV in a ConjugateRV node.

    Also takes care of handling the random number generators used in the conjugate posterior.
    """
    rngs, next_rngs = zip(*collect_default_updates(conjugate_rv, inputs=[rv, *inputs]).items())
    for rng in rngs:
        if rng not in fgraph.inputs:
            fgraph.add_input(rng)
    conjugate_op = ConjugateRV(inputs=[rv, *inputs, *rngs], outputs=[rv, conjugate_rv, *next_rngs])
    return conjugate_op(rv, *inputs, *rngs)[0]


def create_untransformed_free_rv(
    fgraph: FunctionGraph, rv: Variable, name: str, dims: Sequence[str | Variable]
) -> Variable:
    """Create a model FreeRV without transform."""
    transform = None
    value = rv.type(name=name)
    fgraph.add_input(value)
    free_rv = model_free_rv(rv, value, transform, *dims)
    free_rv.name = name
    return free_rv


def beta_binomial_conjugacy(fgraph: FunctionGraph, node, eager: bool = False):
    if not isinstance(node.op, ModelFreeRV):
        return None

    [beta_free_rv] = node.outputs
    beta_rv, _, *beta_dims = node.inputs

    if not isinstance(beta_rv.owner.op, Beta):
        return None

    a, b = get_dist_params(beta_rv)
    if not eager and has_free_rv_ancestor([a, b]):
        # Don't apply rewrite if a, b depend on other model variables as that will force a Gibbs sampling scheme
        return None

    p_arg_idx = 1  # Params to the Binomial are (n, p)
    binomial_rvs = rv_used_by(fgraph, beta_free_rv, Binomial, p_arg_idx)

    if len(binomial_rvs) != 1:
        # Question: Can we apply conjugacy when RV is used by more than one binomial?
        return None

    [binomial_rv] = binomial_rvs

    binomial_model_var = get_model_var_of_rv(fgraph, binomial_rv)
    if binomial_model_var is None:
        return None

    # We want to replace free_rv by ConjugateRV()->(free_rv, conjugate_posterior_rv)
    n, _ = get_dist_params(binomial_rv)

    # Use value of y in new graph to avoid circularity
    y = binomial_model_var.owner.inputs[1]

    conjugate_a = sum_bcasted_dims(beta_rv, a + y)
    conjugate_b = sum_bcasted_dims(beta_rv, b + (n - y))

    conjugate_beta_rv = Beta.dist(conjugate_a, conjugate_b, size=get_size_param(beta_rv))

    new_beta_rv = wrap_rv_and_conjugate_rv(fgraph, beta_rv, conjugate_beta_rv, [a, b, n, y])
    new_beta_free_rv = create_untransformed_free_rv(
        fgraph, new_beta_rv, beta_free_rv.name, beta_dims
    )
    return [new_beta_free_rv]


beta_binomial_conjugacy_default, beta_binomial_conjugacy_force = (
    register_conjugacy_rewrites_variants(beta_binomial_conjugacy)
)
