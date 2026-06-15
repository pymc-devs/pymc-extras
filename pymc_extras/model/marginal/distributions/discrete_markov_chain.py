import pytensor.tensor as pt

from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import constant_fold
from pytensor.graph import node_rewriter, vectorize_graph
from pytensor.graph.replace import clone_replace
from pytensor.scan import scan

from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.model.marginal.distributions.core import inline_ofg_outputs
from pymc_extras.model.marginal.distributions.enumerable import (
    DUMMY_ZERO,
    EnumerableMarginalRV,
    align_logp_dims,
    build_enumerable_marginal_rv,
    reduce_batch_dependent_logps,
    warn_non_separable_logp,
)
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    marginal_rewrites_db,
)


class MarginalDiscreteMarkovChainRV(EnumerableMarginalRV):
    """Base class for Marginalized Discrete Markov Chain RVs"""


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):
    all_outputs = inline_ofg_outputs(op, inputs)
    chain_rv = all_outputs[0]
    dependent_rvs = list(all_outputs[1 : 1 + op.n_dependent_rvs])

    P, n_steps_, init_dist_, rng = chain_rv.owner.inputs
    domain = pt.arange(P.shape[-1], dtype="int32")

    # Construct logp in two steps
    # Step 1: Compute the probability of the data ("emissions") under every possible state (vec_logp_emission)

    # First we need to vectorize the conditional logp graph of the data, in case there are batch dimensions floating
    # around. To do this, we need to break the dependency between chain and the init_dist_ random variable. Otherwise,
    # PyMC will detect a random variable in the logp graph (init_dist_), that isn't relevant at this step.
    chain_value = chain_rv.clone()
    dependent_rvs = clone_replace(dependent_rvs, {chain_rv: chain_value})
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs, values)))

    # Reduce and add the batch dims beyond the chain dimension
    reduced_logp_emissions = reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[dependent_rv.owner.op for dependent_rv in dependent_rvs],
        dependent_logps=[logp_emissions_dict[value] for value in values],
    )

    # Add a batch dimension for the domain of the chain
    chain_shape = constant_fold(tuple(chain_rv.shape))
    batch_chain_value = pt.moveaxis(pt.full((*chain_shape, domain.size), domain), -1, 0)
    batch_logp_emissions = vectorize_graph(reduced_logp_emissions, {chain_value: batch_chain_value})

    # Step 2: Compute the transition probabilities
    # This is the "forward algorithm", alpha_t = p(y | s_t) * sum_{s_{t-1}}(p(s_t | s_{t-1}) * alpha_{t-1})
    # We do it entirely in logs, though.

    # To compute the prior probabilities of each state, we evaluate the logp of the domain (all possible states)
    # under the initial distribution. This is robust to everything the user can throw at it.
    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    # Squeeze core dimension for n_lags=1 (only supported case)
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[..., :1]}
    ).squeeze(-1)
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None, ...] + log_P, axis=0)
        return logp_emission + step_log_prob

    # Add implicit dimensions of P, and place core dimensions at the front
    P = pt.atleast_Nd(P, n=len(chain_shape) + 1)
    P = pt.moveaxis(P, (-2, -1), (0, 1))
    log_P = pt.log(P)

    log_alpha_seq = scan(
        step_alpha,
        non_sequences=[log_P],
        outputs_info=[log_alpha_init],
        # Scan needs the time dimension first, and we already consumed the 1st logp computing the initial value
        sequences=pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0),
        return_updates=False,
    )
    # Final logp is just the sum of the last scan state
    joint_logp = pt.logsumexp(log_alpha_seq[-1], axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    remaining_dims_first_emission = list(op.dims_connections[0])
    # The last dim of chain_rv was removed when computing the logp
    remaining_dims_first_emission.remove(chain_rv.type.ndim - 1)
    joint_logp = align_logp_dims(remaining_dims_first_emission, joint_logp)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each emission stream.
    warn_non_separable_logp(values)
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps


@node_rewriter(tracks=[MarginalSubgraph])
def discrete_markov_chain_marginal(fgraph, node):
    inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv = outputs[0]
    marginalized_rv_op = marginalized_rv.owner.op
    if not isinstance(marginalized_rv_op, DiscreteMarkovChain):
        return None

    if marginalized_rv_op.n_lags > 1:
        raise NotImplementedError(
            "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
        )
    if marginalized_rv.owner.inputs[0].type.ndim > 2:
        raise NotImplementedError(
            "Marginalization for DiscreteMarkovChain with non-matrix transition probability "
            "is not supported"
        )

    return build_enumerable_marginal_rv(node, inputs, outputs, MarginalDiscreteMarkovChainRV)


marginal_rewrites_db.register(
    "discrete_markov_chain_marginal", discrete_markov_chain_marginal, "basic"
)
