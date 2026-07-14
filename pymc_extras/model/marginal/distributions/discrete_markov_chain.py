import pytensor.tensor as pt

from pymc.distributions import Categorical
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import resolve_shapes
from pytensor.graph import node_rewriter, vectorize_graph
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.scan import scan
from pytensor.tensor.special import log_softmax, softmax

from pymc_extras.distributions import DiscreteMarkovChain, JointCategorical
from pymc_extras.distributions.multivariate.joint_categorical import _unravel_states
from pymc_extras.model.marginal.distributions.core import (
    inline_ofg_outputs,
    marginalized_conditional,
)
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
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs, values)))

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
    """Joint log-prob of the initial states and the log transition matrix.

    Returns
    -------
    log_init_joint : TensorVariable
        Joint log-prob of the ``n_lags`` initial states, one leading axis per lag,
        ``((n_states,) * n_lags, *chain_batch)``.
    log_P : TensorVariable
        Log transition matrix. Homogeneous chains place the ``n_lags + 1`` core state
        axes at the front, ``((n_states,) * (n_lags + 1), *batch)``. Time-varying chains
        keep one matrix per transition, ``(time, (n_states,) * (n_lags + 1), *batch)``,
        iterated as a scan sequence downstream.
    """
    P, n_steps_, init_dist_, rng = chain.owner.inputs
    op = chain.owner.op
    n_lags = op.n_lags
    time_varying = op.time_varying_P
    batch_shape = tuple(chain_shape[:-1])
    batch_ndim = len(batch_shape)

    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    if init_dist_.owner.op.ndim_supp == 0:
        # Scalar init_dist: the n_lags initial states are independent. Evaluate the domain at
        # every lag position and outer-sum the per-position terms, one leading state axis per lag.
        logp_init_per_position = vectorize_graph(
            logp_init_dist, {init_dist_value: batch_chain_value[..., :n_lags]}
        )  # (n_states, *batch, n_lags)
        log_init_joint = 0
        for j in range(n_lags):
            g_j = logp_init_per_position[..., j]  # (n_states, *batch) over state j
            insert = tuple(i for i in range(n_lags) if i != j)
            log_init_joint = log_init_joint + (pt.expand_dims(g_j, insert) if insert else g_j)
    else:
        # Vector init_dist (e.g. the JointCategorical of a recovered chain): an arbitrary joint
        # over the n_lags initial states. Evaluate its logp on all n_states ** n_lags
        # configurations and unflatten the leading axis into one state axis per lag.
        n_states = P.shape[-1]
        combos = pt.stack(
            _unravel_states(pt.arange(n_states**n_lags), n_states, n_lags), axis=-1
        )  # (n_states ** n_lags, n_lags)
        if batch_ndim:
            combos = pt.broadcast_to(
                combos[(slice(None), *(None,) * batch_ndim)],
                (n_states**n_lags, *batch_shape, n_lags),
            )
        logp_init_flat = vectorize_graph(
            logp_init_dist, {init_dist_value: combos}
        )  # (n_states ** n_lags, *batch)
        log_init_joint = logp_init_flat.reshape((*(n_states,) * n_lags, *batch_shape))

    # Bring P's n_lags + 1 state axes (and the time axis, when time-varying) to the front so the
    # message pass indexes them directly; batch dims trail. Homogeneous P becomes
    # (n_states,) * (n_lags + 1) + batch; time-varying becomes (time, (n_states,) * (n_lags + 1),
    # batch), iterated as a scan sequence downstream.
    core_ndim = op.P_core_ndim
    P = pt.atleast_Nd(P, n=core_ndim + batch_ndim)
    state_axes = list(range(-(n_lags + 1), 0))
    source = [-(n_lags + 2), *state_axes] if time_varying else state_axes
    log_P = pt.log(pt.moveaxis(P, source, list(range(core_ndim))))
    return log_init_joint, log_P


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


def _hmm_forward_log_alphas(
    batch_logp_emissions, log_init_joint, log_P, n_lags, time_varying=False
):
    """Forward filter. Returns the full ``log_alpha`` trace, time first.

    ``alpha_t`` is a message over the last ``n_lags`` states,
    ``p(y_{0:t}, s_{t-n_lags+1}, ..., s_t)``, held with the ``n_lags`` state axes leading and batch
    trailing. The seed ``alpha_{n_lags-1}`` is the joint over the first ``n_lags`` initial states
    with their emissions; each transition then sums out the oldest state ``s_{t-n_lags}`` and
    folds in the newest emission ``p(y_t | s_t)``. For ``n_lags == 1`` this is the standard filter.
    """
    # Seed: joint init logp over the n_lags leading state axes, plus each position's emission
    # placed on its own axis.
    log_alpha_init = log_init_joint
    for j in range(n_lags):
        e_j = batch_logp_emissions[..., j]  # (n_states, *batch) over state j
        insert = tuple(i for i in range(n_lags) if i != j)
        log_alpha_init = log_alpha_init + (pt.expand_dims(e_j, insert) if insert else e_j)

    # Transitions consume the emissions for steps n_lags .. T-1 (the seed used 0 .. n_lags-1).
    emissions_seq = pt.moveaxis(batch_logp_emissions[..., n_lags:], -1, 0)

    def step_alpha(logp_emission, log_P_t, log_alpha):
        # log_alpha is (n_states,) * n_lags + batch over (s_{t-n_lags}, ..., s_{t-1}); add the s_t
        # axis, weight by the transition tensor, and sum out the oldest state s_{t-n_lags}.
        step_log_prob = pt.logsumexp(pt.expand_dims(log_alpha, n_lags) + log_P_t, axis=0)
        # Emission depends on s_t only -> broadcast onto the newest (last) state axis.
        emission = (
            pt.expand_dims(logp_emission, tuple(range(n_lags - 1))) if n_lags > 1 else logp_emission
        )
        return emission + step_log_prob

    return _scan_messages(
        step_alpha,
        init=log_alpha_init,
        emissions_seq=emissions_seq,
        log_P=log_P,
        time_varying=time_varying,
    )


def _hmm_backward_log_betas(batch_logp_emissions, log_P, n_lags, time_varying=False):
    """Backward messages over the last ``n_lags`` states, batch trailing.

    ``β_t(s_{t-n_lags+1}, ..., s_t) = log p(y_{t+1:} | those states)``; ``β_{T-1} = 0``. Each step
    sums out the newest future state ``s_{t+1}``, weighting by the transition into it and its
    emission and ``β_{t+1}``. Returns the trace ``β_{n_lags-1}, ..., β_{T-1}`` (chronological).
    For ``n_lags == 1`` this is the standard backward filter over a single state.
    """
    emissions_shape = tuple(batch_logp_emissions.shape)
    n_states, batch_shape = emissions_shape[0], emissions_shape[1:-1]
    log_beta_last = pt.zeros((*(n_states,) * n_lags, *batch_shape))
    # Emissions for the t+1 step, iterated backwards: b_{T-1}, ..., b_{n_lags}
    rev_next_emissions = pt.moveaxis(batch_logp_emissions[..., n_lags:], -1, 0)[::-1]

    def step_beta(logp_emission_next, log_A_next, log_beta_next):
        # log_A_next: (k,) * (n_lags + 1) over (s_{t-n_lags+1}, ..., s_t, s_{t+1}); log_beta_next is
        # over its last n_lags axes; emission over its last axis. Sum out the newest state s_{t+1}.
        v = (
            log_A_next
            + pt.expand_dims(log_beta_next, 0)
            + pt.expand_dims(logp_emission_next, tuple(range(n_lags)))
        )
        return pt.logsumexp(v, axis=n_lags)

    # Inputs are fed in reverse (latest step first), so the scan output is the backward trace in
    # reverse order too; it is flipped back to chronological order before being returned.
    log_beta_seq = _scan_messages(
        step_beta,
        init=log_beta_last,
        emissions_seq=rev_next_emissions,
        log_P=log_P[::-1] if time_varying else log_P,
        time_varying=time_varying,
    )
    return log_beta_seq[::-1]


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_discrete_markov_chain_logp(op, values, *inputs, **kwargs):
    """Marginal ``log p(emissions | inputs)`` of a marginalized DiscreteMarkovChain.

    Standard forward algorithm, generalized to ``n_lags`` and run in log space:

    1. Emissions. The conditional logp of the dependent RVs is vectorized over the chain domain,
       giving ``log p(y_t | s_t)`` for every state at every step.
    2. Forward filter. ``alpha_t`` is the joint ``log p(y_{0:t}, s_{t-n_lags+1}, ..., s_t)``, seeded
       from the init distribution and advanced one step at a time by a scan.
    3. Termination. The last message is logsumexp-ed over its ``n_lags`` state axes.

    A chain with ``n_lags > 1`` is *not* rewritten into an equivalent first-order chain over
    ``n_states ** n_lags`` compound states. The message keeps one axis per lag and the transition
    tensor is indexed directly, which avoids materializing the compound transition matrix.
    """
    all_outputs = inline_ofg_outputs(op, inputs)
    chain_rv = all_outputs[0]
    dependent_rvs = list(all_outputs[1 : 1 + op.n_dependent_rvs])

    # Step 1: Compute the probability of the data ("emissions") under every possible state
    batch_logp_emissions, batch_chain_value, chain_shape = _hmm_emission_logp(
        op, chain_rv, dependent_rvs, values
    )

    # Step 2: Run the forward algorithm over the transition probabilities
    log_init_joint, log_P = _hmm_log_init_and_transition(chain_rv, chain_shape, batch_chain_value)
    chain_op = chain_rv.owner.op
    log_alphas = _hmm_forward_log_alphas(
        batch_logp_emissions, log_init_joint, log_P, chain_op.n_lags, chain_op.time_varying_P
    )

    # Final logp sums the last message over all n_lags remaining (leading) state axes.
    joint_logp = pt.logsumexp(log_alphas[-1], axis=tuple(range(chain_op.n_lags)))

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


@marginalized_conditional.register(MarginalDiscreteMarkovChainRV)
def discrete_markov_chain_marginalized_conditional(op, inputs, dep_rvs):
    """Conditional ``p(chain | emissions, inputs)`` of a marginalized DiscreteMarkovChain.

    The posterior over the latent path is itself a Markov chain of the same order, but
    time-inhomogeneous: conditioning on the emissions makes each transition depend on the step. So
    we return a :class:`DiscreteMarkovChain` with ``time_varying_P=True``, which already has both an
    exact logp and a sampler, giving a loggable *and* sampleable conditional without a bespoke Op.

    Its two parameters come from a forward-backward pass (the same ``alpha`` filter used by the
    marginal logp, plus the backward ``beta`` messages):

    - ``init_dist``, the smoothed distribution over the first ``n_lags`` states,
      ``gamma ∝ alpha_{n_lags-1} · beta_{n_lags-1}``. For a single lag this is a
      :class:`~pymc.Categorical`; for ``n_lags > 1`` those states are correlated a posteriori, so
      it is a :class:`~pymc_extras.distributions.JointCategorical` over all ``n_states ** n_lags``
      configurations of them.
    - ``P``, the forward-smoothed transitions for each step ``t = n_lags .. T-1``,
      ``p(s_t | s_{t-n_lags}, ..., s_{t-1}, y) ∝ P(...) · b_t(s_t) · beta_t(...)``, normalized over
      ``s_t``. As in the logp, ``n_lags > 1`` keeps one axis per lag rather than reducing the chain
      to a first-order one over compound states.

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
    chain_op = chain.owner.op
    n_lags = chain_op.n_lags

    dep_dummies = [dep.type() for dep in dependents]

    batch_logp_emissions, batch_chain_value, chain_shape = _hmm_emission_logp(
        op, chain, dependents, dep_dummies
    )
    log_init_joint, log_P = _hmm_log_init_and_transition(chain, chain_shape, batch_chain_value)
    time_varying = chain_op.time_varying_P
    log_alphas = _hmm_forward_log_alphas(
        batch_logp_emissions, log_init_joint, log_P, n_lags, time_varying
    )  # trace (n_lags,) * n_lags leading state axes
    log_betas = _hmm_backward_log_betas(batch_logp_emissions, log_P, n_lags, time_varying)

    # Smoothed distribution over the first n_lags states, proportional to
    # alpha_{n_lags-1} * beta_{n_lags-1}. The message pass keeps state axes leading and batch
    # trailing; DiscreteMarkovChain expects the opposite, so move the states last. For a single
    # lag it is a plain Categorical (with an explicit length-1 lag axis so batch dims are read
    # correctly); for higher order it is the (correlated) joint over the n_lags states, which
    # JointCategorical takes with one axis per lag, exactly as the message pass produces it.
    log_gamma_init = log_alphas[0] + log_betas[0]  # ((k,) * n_lags, *batch)
    log_gamma_init = pt.moveaxis(
        log_gamma_init, tuple(range(n_lags)), tuple(range(-n_lags, 0))
    )  # (*batch, (k,) * n_lags)
    if n_lags == 1:
        init_dist = Categorical.dist(logit_p=log_softmax(log_gamma_init, axis=-1)[..., None, :])
    else:
        state_axes = tuple(range(-n_lags, 0))
        init_dist = JointCategorical.dist(
            # Normalize over the joint, i.e. all state axes at once.
            logit_p=log_gamma_init - pt.logsumexp(log_gamma_init, axis=state_axes, keepdims=True),
            n_lags=n_lags,
        )

    # Forward-smoothed time-varying order-n_lags transitions for steps t = n_lags .. T-1:
    #   p(s_t | s_{t-n_lags}..s_{t-1}, y) ∝ A_t(...) · b_t(s_t) · β_t(s_{t-n_lags+1}..s_t)
    # normalized over s_t. Everything below carries a leading time axis of length T - n_lags
    # and trailing batch axes.
    b_jt = pt.moveaxis(batch_logp_emissions[..., n_lags:], -1, 0)  # (T-L, k, *batch) over (t, s_t)
    beta_t = log_betas[1:]  # (T-L, (k,) * n_lags, *batch) over (t, s_{t-n_lags+1}..s_t)
    # log_A_t: single transition tensor broadcast over steps (homogeneous), or the per-step prior
    # (already (T-L, (k,) * (n_lags + 1), *batch), log_P[t] transitions into state t + n_lags).
    log_A_t = log_P[None] if not time_varying else log_P
    log_P_t = (
        log_A_t
        + pt.expand_dims(beta_t, 1)  # broadcast β_t over the oldest "from" state s_{t-n_lags}
        + pt.expand_dims(b_jt, tuple(range(1, n_lags + 1)))  # emission over the "to" state s_t
    )
    # DiscreteMarkovChain expects P as (*batch, time, states...), so move the leading
    # (time, states...) block behind the batch axes before normalizing over s_t.
    log_P_t = pt.moveaxis(log_P_t, tuple(range(n_lags + 2)), tuple(range(-(n_lags + 2), 0)))
    P_t = softmax(log_P_t, axis=-1)  # row-stochastic over s_t per step

    steps = chain_shape[-1] - n_lags
    cond_chain = DiscreteMarkovChain.dist(
        P=P_t, init_dist=init_dist, steps=steps, n_lags=n_lags, time_varying_P=True
    )

    replacements = dict(zip(inner_inputs, inputs))
    replacements.update(zip(dep_dummies, dep_rvs))
    [cond_chain] = graph_replace([cond_chain], replace=replacements, strict=False)
    return cond_chain


@node_rewriter(tracks=[MarginalSubgraph])
def discrete_markov_chain_marginal(fgraph, node):
    inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv = outputs[0]
    marginalized_rv_op = marginalized_rv.owner.op
    if not isinstance(marginalized_rv_op, DiscreteMarkovChain):
        return None

    return build_enumerable_marginal_rv(node, inputs, outputs, MarginalDiscreteMarkovChainRV)


marginal_rewrites_db.register(
    "discrete_markov_chain_marginal", discrete_markov_chain_marginal, "basic"
)
