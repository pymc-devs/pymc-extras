from pymc import Normal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import get_symbolic_rv_shapes
from pytensor.graph import node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor import broadcast_to, constant, sqrt

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    marginal_rewrites_db,
)


class NormalNormalMarginalRV(MarginalRV):
    """Marginalized Normal-Normal conjugate pair.

    Inner graph: [marginalized_normal, dependent_normal, *rng_updates]
    """

    def __init__(self, *args, **kwargs):
        # Normal-Normal conjugacy always has exactly one dependent RV
        super().__init__(*args, n_dependent_rvs=1, **kwargs)


@_logprob.register(NormalNormalMarginalRV)
def normal_normal_marginal_rv_logp(op: NormalNormalMarginalRV, values, *inputs, **kwargs):
    [value] = values

    all_outputs = inline_ofg_outputs(op, inputs)
    marginalized_rv = all_outputs[0]
    dependent_rv = all_outputs[1]

    mu_m, sigma_m = marginalized_rv.owner.op.dist_params(marginalized_rv.owner)
    mu_d, sigma_d = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    # mu_m stands in for the marginalized RV below, so it must match its type
    # exactly; it may have a narrower dtype (e.g. an integer constant prior mu)
    # or size-1 broadcastable dims where the RV is sized.
    mu_m = mu_m.astype(marginalized_rv.type.dtype)
    if marginalized_rv.type.broadcastable != mu_m.type.broadcastable:
        mu_m = broadcast_to(mu_m, get_symbolic_rv_shapes([marginalized_rv])[0])

    new_mu = graph_replace(mu_d, {marginalized_rv: mu_m})
    new_sigma = sqrt(sigma_d**2 + sigma_m**2)
    return logp(Normal.dist(mu=new_mu, sigma=new_sigma), value)


@marginalized_conditional.register(NormalNormalMarginalRV)
def normal_normal_conditional(op, inputs, dep_rvs):
    marginalized, dependent = inline_ofg_outputs(op, inputs)[:2]
    [dep_rv] = dep_rvs

    mu_m, sigma_m = marginalized.owner.op.dist_params(marginalized.owner)
    mu_d, sigma_d = dependent.owner.op.dist_params(dependent.owner)

    offset = graph_replace(mu_d, {marginalized: constant(0, dtype=marginalized.type.dtype)})

    precision_m = 1 / sigma_m**2
    precision_d = 1 / sigma_d**2
    posterior_precision = precision_m + precision_d
    posterior_sigma = sqrt(1 / posterior_precision)
    posterior_mu = (mu_m * precision_m + (dep_rv - offset) * precision_d) / posterior_precision

    return Normal.dist(mu=posterior_mu, sigma=posterior_sigma)


@node_rewriter(tracks=[MarginalSubgraph])
def normal_normal_marginal_rewrite(fgraph, node):
    op = node.op

    if op.n_dependent_rvs != 1:
        return None

    inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv = outputs[0]
    dependent_rv = outputs[1]

    if not (
        isinstance(marginalized_rv.owner.op, Normal) and isinstance(dependent_rv.owner.op, Normal)
    ):
        return None

    mu_dep, sigma_dep = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    if marginalized_rv in ancestors([sigma_dep]):
        return None

    if mu_dep is not marginalized_rv:
        match mu_dep.owner_op_and_inputs:
            case (_, a, b):
                if a is marginalized_rv:
                    if marginalized_rv in ancestors([b]):
                        return None
                elif b is marginalized_rv:
                    if marginalized_rv in ancestors([a]):
                        return None
                else:
                    return None
            case _:
                return None

    typed_op = NormalNormalMarginalRV(
        inputs=inputs,
        outputs=outputs,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
    )

    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_rewrites_db.register("normal_normal_marginal", normal_normal_marginal_rewrite, "basic")
