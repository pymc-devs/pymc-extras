import pytensor.tensor as pt

from pymc.distributions import Gamma, Normal
from pymc.model.fgraph import ModelObservedRV, model_observed_rv
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter

from pymc_extras.sampling.optimizations.optimize import posterior_optimization_db


@node_rewriter(tracks=[ModelObservedRV])
def summary_stats_normal(fgraph: FunctionGraph, node):
    """Applies the equivalence (up to a normalizing constant) described in:

    https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#exploiting-sufficient-statistics
    """
    [observed_rv] = node.outputs
    [rv, data] = node.inputs

    if not isinstance(rv.owner.op, Normal):
        return None

    # Check the normal RV is not just a scalar
    if all(rv.type.broadcastable):
        return None

    # Check that the observed RV is not used anywhere else (like a Potential or Deterministic)
    # There should be only one use: as an "output"
    if len(fgraph.clients[observed_rv]) > 1:
        return None

    mu, sigma = rv.owner.op.dist_params(rv.owner)

    # Check if mu and sigma are scalar RVs
    if not all(mu.type.broadcastable) and not all(sigma.type.broadcastable):
        return None

    # Check that mu and sigma are not used anywhere else
    # Note: This is too restrictive, it's fine if they're used in Deterministics!
    # There should only be two uses: as an "output" and as the param of the `rv`
    if len(fgraph.clients[mu]) > 2 or len(fgraph.clients[sigma]) > 2:
        return None

    # Remove expand_dims
    mu = mu.squeeze()
    sigma = sigma.squeeze()

    # Apply the rewrite
    mean_data = pt.mean(data)
    mean_data.name = None
    var_data = pt.var(data, ddof=1)
    var_data.name = None
    N = data.size
    sqrt_N = pt.sqrt(N)
    nm1_over2 = (N - 1) / 2

    observed_mean = model_observed_rv(
        Normal.dist(mu=mu, sigma=sigma / sqrt_N),
        mean_data,
    )
    observed_mean.name = f"{rv.name}_mean"

    observed_var = model_observed_rv(
        Gamma.dist(alpha=nm1_over2, beta=nm1_over2 / (sigma**2)),
        var_data,
    )
    observed_var.name = f"{rv.name}_var"

    fgraph.add_output(observed_mean, import_missing=True)
    fgraph.add_output(observed_var, import_missing=True)
    fgraph.remove_node(node)
    # Just so it shows in the profile for verbose=True,
    # It won't do anything because node is not in the fgraph anymore
    return [node.out.copy()]


posterior_optimization_db.register(
    summary_stats_normal.__name__, summary_stats_normal, "default", "summary_stats"
)
