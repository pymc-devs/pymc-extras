import pytensor.tensor as pt

from pymc.distributions import Categorical
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import resolve_shapes
from pytensor.graph import node_rewriter, vectorize_graph
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.scan import scan
from pytensor.tensor.special import log_softmax, softmax

from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.model.marginal.distributions.core import (
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.distributions.enumerable import (
    EnumerableMarginalRV,
    align_logp_dims,
    build_enumerable_marginal_rv,
    dummy_logps,
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


def _hmm_emission_logp(op, chain, dependent_rvs, values):
    """Emission log-probability of the data under every possible state at each step.

    This is the "forward algorithm" Step 1, shared by the marginal logp and the
    conditional (recovery) graph. We vectorize the conditional logp of the
    dependents over the chain domain, after breaking the dependency between the
    chain and its dependents (otherwise PyMC detects a leftover random variable
    in the logp graph).

    Returns
    -------
    batch_logp_emissions : TensorVariable
        Shape ``(n_states, *chain_batch, n_steps)``.
    batch_chain_value : TensorVariable
        The chain domain broadcast to the chain shape, ``(n_states, *chain_shape)``.
    chain_shape : tuple
        Symbolic shape of the chain, resolved to expressions over the chain's
        inputs (so it doesn't reference the RV itself).
    """
    P = chain.owner.inputs[0]
    domain = pt.arange(P.shape[-1], dtype="int32")

    chain_value = chain.clone()
    dependent_rvs = clone_replace(dependent_rvs, {chain: chain_value})
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs, values, strict=True)))

    # Reduce and add the batch dims beyond the chain dimension
    reduced_logp_emissions = reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[dependent_rv.owner.op for dependent_rv in dependent_rvs],
        dependent_logps=[logp_emissions_dict[value] for value in values],
    )

    # Add a batch dimension for the domain of the chain
    chain_shape = resolve_shapes(tuple(chain.shape))
    batch_chain_value = pt.moveaxis(pt.full((*chain_shape, domain.size), domain), -1, 0)
    batch_logp_emissions = vectorize_graph(reduced_logp_emissions, {chain_value: batch_chain_value})
    return batch_logp_emissions, batch_chain_value, chain_shape


def _hmm_log_init_and_transition(chain, chain_shape, batch_chain_value):
    """Log initial-state probabilities (per state) and the log transition matrix.

    Returns
    -------
    batch_logp_init_dist : TensorVariable
        Shape ``(n_states, *chain_batch)``.
    log_P : TensorVariable
        Log transition matrix. Homogeneous chains place the two core (from, to)
        axes at the front, ``(n_states, n_states, *batch)``. Time-varying chains
        keep one matrix per transition, ``(n_transitions, n_states, n_states)`` =
        ``(time, from, to)`` (unbatched), iterated as a scan sequence downstream.
    """
    P, n_steps_, init_dist_, rng = chain.owner.inputs
    time_varying = chain.owner.op.time_varying_P

    # To compute the prior probabilities of each state, we evaluate the logp of the domain (all
    # possible states) under the initial distribution. This is robust to everything the user can
    # throw at it.
    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    # Squeeze core dimension for n_lags=1 (only supported case)
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[..., :1]}
    ).squeeze(-1)

    if time_varying:
        # P is already (time, from, to); each transition uses its own matrix.
        log_P = pt.log(P)
    else:
        # Add implicit dimensions of P, and place core (from, to) dimensions at the front
        P = pt.atleast_Nd(P, n=len(chain_shape) + 1)
        P = pt.moveaxis(P, (-2, -1), (0, 1))
        log_P = pt.log(P)
    return batch_logp_init_dist, log_P


def _scan_messages(core, *, init, emissions_seq, log_P, time_varying):
    """Run a forward/backward HMM message pass and return the full message trace.

    ``core(emission, log_P_t, prev)`` is the per-step recurrence, where ``log_P_t``
    is the transition matrix to use at that step. A homogeneous chain passes its
    single matrix as a non-sequence; a time-varying chain iterates the per-step
    matrices as a scan sequence. ``emissions_seq`` (and ``log_P`` when time-varying)
    must already be in the desired scan order. Branching on ``time_varying`` lives
    here so the recurrences themselves stay free of scan plumbing.

    The returned trace includes the initial state: scan strips it from its output
    (``seq = full_trace[1:]``), but the parent ``Scan`` buffer still holds it, so we
    return that buffer directly instead of having callers concatenate ``init`` back on.
    """
    if time_varying:

        def step(emission, log_P_t, prev):
            return core(emission, log_P_t, prev)

        sequences = [emissions_seq, log_P]
        non_sequences = []
    else:

        def step(emission, prev, log_P_t):
            return core(emission, log_P_t, prev)

        sequences = [emissions_seq]
        non_sequences = [log_P]

    seq = scan(
        step,
        sequences=sequences,
        outputs_info=[init],
        non_sequences=non_sequences,
        return_updates=False,
    )
    # scan returns full_trace[1:]; the parent Scan output is the full trace (init included).
    return seq.owner.inputs[0]


def _hmm_forward_log_alphas(batch_logp_emissions, batch_logp_init_dist, log_P, time_varying=False):
    """Forward filter. Returns the full ``log_alpha`` trace, shape ``(T, n_states, *batch)``.

    ``alpha_t(s) = p(y_{1:t}, s_t=s)`` computed entirely in logs:
    ``alpha_t = p(y_t | s_t) * sum_{s_{t-1}}(p(s_t | s_{t-1}) * alpha_{t-1})``.
    The trace is time first and includes ``alpha_0`` as the initial state.
    """
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    # Scan needs the time dimension first, and we already consumed the 1st logp computing the initial value
    emissions_seq = pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0)

    def step_alpha(logp_emission, log_P_t, log_alpha):
        step_log_prob = pt.logsumexp(log_alpha[:, None, ...] + log_P_t, axis=0)
        return logp_emission + step_log_prob

    return _scan_messages(
        step_alpha,
        init=log_alpha_init,
        emissions_seq=emissions_seq,
        log_P=log_P,
        time_varying=time_varying,
    )


def _hmm_backward_log_betas(batch_logp_emissions, log_P, time_varying=False):
    """Backward messages ``log β_t(i) = log p(y_{t+1:} | s_t=i)``, shape ``(n_steps, n_states)``.

    ``β_{T-1} = 1`` (0 in logs); ``log β_t(i) = logsumexp_j[log A_{t+1}(i,j) + log b_{t+1}(j) +
    log β_{t+1}(j)]``. ``log_P`` is ``(n_states, n_states)`` for a homogeneous chain, or one
    matrix per transition ``(n_transitions, n_states, n_states)`` when ``time_varying`` (the
    transition into step ``t+1`` uses ``A_{t+1} = log_P[t]``). Unbatched chain.
    """
    n_states = batch_logp_emissions.shape[0]
    log_beta_last = pt.zeros((n_states,))
    # Emissions for the t+1 step, iterated backwards: b_{T-1}, ..., b_1
    rev_next_emissions = pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0)[::-1]

    def step_beta(logp_emission_next, log_P_t, log_beta_next):
        v = logp_emission_next + log_beta_next  # (n_states,) over j
        return pt.logsumexp(log_P_t + v[None, :], axis=1)  # (n_states,) over i

    # Full backward trace in scan (reverse) order: β_{T-1}, β_{T-2}, ..., β_0
    log_beta_seq = _scan_messages(
        step_beta,
        init=log_beta_last,
        emissions_seq=rev_next_emissions,
        # A_{t+1} in the same backwards order as the emissions: A_{T-1}, ..., A_1
        log_P=log_P[::-1] if time_varying else log_P,
        time_varying=time_varying,
    )
    return log_beta_seq[::-1]


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_discrete_markov_chain_logp(op, values, *inputs, **kwargs):
    all_outputs = inline_ofg_outputs(op, inputs)
    chain_rv = all_outputs[0]
    dependent_rvs = list(all_outputs[1 : 1 + op.n_dependent_rvs])

    # Step 1: Compute the probability of the data ("emissions") under every possible state
    batch_logp_emissions, batch_chain_value, chain_shape = _hmm_emission_logp(
        op, chain_rv, dependent_rvs, values
    )

    # Step 2: Run the forward algorithm over the transition probabilities
    batch_logp_init_dist, log_P = _hmm_log_init_and_transition(
        chain_rv, chain_shape, batch_chain_value
    )
    time_varying = chain_rv.owner.op.time_varying_P
    log_alphas = _hmm_forward_log_alphas(
        batch_logp_emissions, batch_logp_init_dist, log_P, time_varying
    )

    # Final logp is just the sum of the last scan state
    joint_logp = pt.logsumexp(log_alphas[-1], axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    remaining_dims_first_emission = list(op.dims_connections[0])
    # The last dim of chain_rv was removed when computing the logp
    remaining_dims_first_emission.remove(chain_rv.type.ndim - 1)
    joint_logp = align_logp_dims(remaining_dims_first_emission, joint_logp)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each emission stream.
    warn_non_separable_logp(values)
    return joint_logp, *dummy_logps(op, values)


@marginalized_conditional.register(MarginalDiscreteMarkovChainRV)
def discrete_markov_chain_marginalized_conditional(op, inputs, dep_rvs):
    """Conditional ``p(chain | emissions, inputs)`` of a marginalized DiscreteMarkovChain.

    The posterior over the latent path is itself a (time-inhomogeneous) Markov
    chain, so we return a :class:`DiscreteMarkovChain` with a time-varying
    transition matrix — which already has both an exact logp and a sampler, so
    we get a loggable *and* sampleable conditional without a bespoke Op.
    Forward-backward yields the smoothed initial distribution ``γ₀ ∝ α₀·β₀`` and
    the forward-smoothed transitions
    ``p(s_t | s_{t-1}, y) ∝ P(s_{t-1}, s_t)·b_t(s_t)·β_t(s_t)``.

    Mirrors :func:`finite_discrete_marginalized_conditional`: built on the inner
    (nominal) graph with value dummies for the dependents, then the real
    ``inputs`` / ``dep_rvs`` are substituted once at the end.
    """
    # inner_inputs/inner_outputs are frozen (immutable) views; the logp and
    # graph_replace below need mutable nodes, so work on an unfrozen copy.
    inner_graph = op.fgraph.unfreeze()
    inner_inputs = inner_graph.inputs
    chain = inner_graph.outputs[0]
    dependents = list(inner_graph.outputs[1 : 1 + op.n_dependent_rvs])

    if chain.type.ndim > 1:
        raise NotImplementedError(
            "Recovering a batched DiscreteMarkovChain (more than one chain) is not yet supported."
        )

    dep_dummies = [dep.type() for dep in dependents]

    batch_logp_emissions, batch_chain_value, chain_shape = _hmm_emission_logp(
        op, chain, dependents, dep_dummies
    )
    batch_logp_init_dist, log_P = _hmm_log_init_and_transition(
        chain, chain_shape, batch_chain_value
    )
    time_varying = chain.owner.op.time_varying_P
    log_alphas = _hmm_forward_log_alphas(
        batch_logp_emissions, batch_logp_init_dist, log_P, time_varying
    )  # (T, k)
    log_betas = _hmm_backward_log_betas(batch_logp_emissions, log_P, time_varying)  # (T, k)

    # Smoothed initial distribution γ₀ ∝ α₀·β₀
    log_gamma_0 = log_softmax(log_alphas[0] + log_betas[0], axis=-1)  # (k,)
    init_dist = Categorical.dist(logit_p=log_gamma_0)

    # Forward-smoothed time-varying transitions for steps t = 1 .. T-1:
    #   p(s_t=j | s_{t-1}=i, y) ∝ A_t(i,j) · b_t(j) · β_t(j)   (normalized over j)
    emissions_jt = pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0)  # (T-1, k) over (t, j)
    log_terms = emissions_jt + log_betas[1:]  # (T-1, k): b_t(j) + β_t(j)
    # log_A_t: a single matrix broadcast over steps (homogeneous), or the per-step prior
    # transition (already (T-1, k, k), where log_P[t] is the transition into state t+1).
    log_A_t = log_P[None, :, :] if not time_varying else log_P
    log_P_t = log_A_t + log_terms[:, None, :]  # (T-1, k, k) over (t, i, j)
    P_t = softmax(log_P_t, axis=-1)  # row-stochastic transition matrix per step

    steps = chain_shape[0] - 1
    cond_chain = DiscreteMarkovChain.dist(
        P=P_t, init_dist=init_dist, steps=steps, time_varying_P=True
    )

    replacements = dict(zip(inner_inputs, inputs, strict=True))
    replacements.update(zip(dep_dummies, dep_rvs, strict=True))
    [cond_chain] = graph_replace([cond_chain], replace=replacements, strict=False)
    return cond_chain


@node_rewriter(tracks=[MarginalSubgraph])
def discrete_markov_chain_marginal(fgraph, node):
    inputs, outer_inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv = outputs[0]
    marginalized_rv_op = marginalized_rv.owner.op
    if not isinstance(marginalized_rv_op, DiscreteMarkovChain):
        return None

    if marginalized_rv_op.n_lags > 1:
        raise NotImplementedError(
            "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
        )
    P_ndim = marginalized_rv.owner.inputs[0].type.ndim
    # A time-varying chain has an extra (time, k, k) core axis, so ndim == 3 is expected
    # there; the flag disambiguates it from a genuine batch dimension.
    if marginalized_rv_op.time_varying_P:
        if P_ndim > 3:
            raise NotImplementedError(
                "Marginalization for batched time-varying DiscreteMarkovChain is not supported"
            )
    elif P_ndim > 2:
        raise NotImplementedError(
            "Marginalization for DiscreteMarkovChain with non-matrix transition probability "
            "is not supported"
        )

    return build_enumerable_marginal_rv(
        node, inputs, outer_inputs, outputs, MarginalDiscreteMarkovChainRV
    )


marginal_rewrites_db.register(
    "discrete_markov_chain_marginal", discrete_markov_chain_marginal, "basic"
)
