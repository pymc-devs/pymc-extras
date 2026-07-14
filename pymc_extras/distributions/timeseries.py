import warnings

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
    support_point,
)
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape_1d,
)
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import intX, resolve_shapes
from pymc.step_methods import STEP_METHODS
from pymc.step_methods.arraystep import ArrayStep
from pymc.step_methods.compound import Competence
from pymc.step_methods.metropolis import CategoricalGibbsMetropolis
from pymc.util import check_dist_not_registered, get_value_vars_from_user_vars
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable
from pytensor.tensor.basic import ix_
from pytensor.tensor.random.op import RandomVariable


def _make_outputs_info(n_lags: int, init_dist: Distribution) -> list[Distribution | dict]:
    """
    Two cases are needed for outputs_info in the scans used by DiscreteMarkovRv. If n_lags = 1, we need to throw away
    the first dimension of init_dist_ or else markov_chain will have shape (steps, 1, *batch_size) instead of
    desired (steps, *batch_size)

    Parameters
    ----------
    n_lags: int
        Number of lags the Markov Chain considers when transitioning to the next state
    init_dist: RandomVariable
        Distribution over initial states

    Returns
    -------
    taps: list
        Lags to be fed into pytensor.scan when drawing a markov chain
    """

    if n_lags > 1:
        return [{"initial": init_dist, "taps": list(range(-n_lags, 0))}]
    else:
        return [init_dist[0]]


def _index_transition_probs(transition_probs: TensorVariable, states) -> TensorVariable:
    """Select the next-state distribution ``p(x_t | states)`` for each batch element.

    ``transition_probs`` is ``(*batch, k, ..., k)``: ``len(states)`` "from" state axes then the
    "to" axis (a single per-step slice, so no time axis here). ``states`` are the previous states,
    each ``(*batch,)``. The batch axes are indexed with an open mesh of aranges (:func:`ix_`) so
    they pair element-wise with ``states``; the trailing "to" axis is kept, giving ``(*batch, k)``.
    """
    batch_ndim = transition_probs.ndim - (len(states) + 1)
    tp_shape = tuple(transition_probs.shape)
    batch_index = ix_(*(pt.arange(tp_shape[axis]) for axis in range(batch_ndim)))
    return transition_probs[(*batch_index, *states)]


class DiscreteMarkovChainRV(SymbolicRandomVariable):
    n_lags: int
    time_varying_P: bool
    default_output = 1
    _print_name = ("DiscreteMC", "\\operatorname{DiscreteMC}")

    def __init__(self, *args, n_lags, time_varying_P=False, **kwargs):
        self.n_lags = n_lags
        self.time_varying_P = time_varying_P
        # Core (non-batch) axes of P: the n_lags + 1 state axes, plus a leading time axis
        # when the chain is time-varying.
        self.P_core_ndim = (n_lags + 1) + time_varying_P
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class DiscreteMarkovChain(Distribution):
    r"""
    A Discrete Markov Chain is a sequence of random variables

    .. math::

        \{x_t\}_{t=0}^T

    Where transition probability :math:`P(x_t | x_{t-1})` depends only on the state of the system at
    :math:`x_{t-1}`. With ``n_lags > 1`` the chain is of higher order, and the transition
    probability :math:`P(x_t | x_{t-1}, \dots, x_{t-n\_lags})` depends on the last ``n_lags``
    states.

    Parameters
    ----------
    P: tensor
        Matrix of transition probabilities between states. Rows must sum to 1.
        One of P or P_logits must be provided.

        When ``time_varying_P=False`` (default), ``P`` is a ``k x k`` matrix shared across
        all transitions, with optional leading batch dimensions: ``(*batch, k, k)``.

        When ``time_varying_P=True``, ``P`` carries an extra time axis just before the two
        state axes: ``(*batch, steps, k, k)``. ``P[..., t, :, :]`` is the transition matrix
        used to go from state ``t`` to state ``t + 1``, so the time axis must have length
        ``steps`` (one transition matrix per step). When ``steps`` is not given explicitly,
        it is inferred from this axis.

        With ``n_lags > 1`` the two state axes become ``n_lags + 1`` axes, all of length ``k``:
        ``P[..., x_{t-n_lags}, ..., x_{t-1}, x_t]``, so the shape is ``(*batch, k, ..., k)`` (or
        ``(*batch, steps, k, ..., k)`` when time-varying).
    P_logit: tensor, optional
        Matrix of transition logits. Converted to probabilities via Softmax activation.
        One of P or P_logits must be provided.
    steps: tensor, optional
        Length of the markov chain. Only needed if state is not provided.
    init_dist : unnamed distribution, optional
        Distribution over the ``n_lags`` initial states. Unnamed refers to distributions
        created with the ``.dist()`` API. A scalar-support distribution (e.g. ``Categorical``)
        is broadcast IID across the initial states; a vector-support distribution (e.g.
        :class:`JointCategorical`) provides their joint distribution and must have support
        length ``n_lags``.

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.
    n_lags : int, default 1
        Order of the chain: how many previous states the transition probability conditions on.
        ``P`` gains one state axis per extra lag (see ``P`` above) and the chain starts from
        ``n_lags`` initial states drawn from ``init_dist``, so it has ``n_lags + steps`` states in
        total.
    time_varying_P : bool, default False
        If ``True``, ``P`` is interpreted as a sequence of transition matrices, one per step,
        with shape ``(*batch, steps, k, k)`` (see ``P`` above). This disambiguates the time
        axis from a leading batch dimension.

    Notes
    -----
    The initial distribution will be cloned, rendering it distinct from the one passed as
    input.

    Examples
    --------
     Create a Markov Chain of length 100 with 3 states. The number of states is given by the shape of P,
     3 in this case.

    .. code-block:: python

        import numpy as np
        import pymc as pm
        import pymc_extras as pmx

        with pm.Model() as markov_chain:
            P = pm.Dirichlet("P", a=[1, 1, 1], size=(3,))
            init_dist = pm.Categorical.dist(p=np.full(3, 1 / 3))
            markov_chain = pmx.DiscreteMarkovChain(
                "markov_chain", P=P, init_dist=init_dist, shape=(100,)
            )

    Use a time-varying transition matrix. ``P`` carries a leading time axis of length
    ``steps`` (here 99 transitions for a chain of 100 states), so its shape is
    ``(steps, k, k)``. Below the chain switches regime partway through: a "sticky" kernel
    governs the first 40 transitions and a "mixing" kernel the remaining 59, assembled by
    repeating each kernel along the time axis.

    .. code-block:: python

        import numpy as np
        import pymc as pm
        import pymc_extras as pmx
        import pytensor.tensor as pt

        with pm.Model() as regime_chain:
            # Two 2 x 2 kernels: states persist under "sticky", shuffle under "mixing".
            P_sticky = pm.Dirichlet("P_sticky", a=np.eye(2) * 8 + 1, size=2)
            P_mixing = pm.Dirichlet("P_mixing", a=np.ones((2, 2)), size=2)

            # Stack one kernel per step: 40 sticky then 59 mixing -> shape (99, 2, 2)
            P = pt.concatenate(
                [pt.repeat(P_sticky[None], 40, axis=0), pt.repeat(P_mixing[None], 59, axis=0)],
                axis=0,
            )

            init_dist = pm.Categorical.dist(p=np.full(2, 0.5))
            markov_chain = pmx.DiscreteMarkovChain(
                "markov_chain",
                P=P,
                init_dist=init_dist,
                time_varying_P=True,
                shape=(100,),
            )

    Use a second-order chain, where each state depends on the previous two. ``P`` gains one
    state axis per lag, so it is ``(2, 2, 2)`` here: ``P[x_{t-2}, x_{t-1}, x_t]``. The chain starts
    from ``n_lags=2`` initial states, so ``shape=(100,)`` means 98 transitions.

    .. code-block:: python

        import numpy as np
        import pymc as pm
        import pymc_extras as pmx

        with pm.Model() as second_order_chain:
            P = pm.Dirichlet("P", a=np.ones((2, 2, 2)))
            init_dist = pm.Categorical.dist(p=np.full(2, 0.5))
            markov_chain = pmx.DiscreteMarkovChain(
                "markov_chain",
                P=P,
                init_dist=init_dist,
                n_lags=2,
                shape=(100,),
            )

    """

    rv_type = DiscreteMarkovChainRV

    def __new__(cls, *args, steps=None, n_lags=1, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=n_lags,
        )

        return super().__new__(cls, *args, steps=steps, n_lags=n_lags, **kwargs)

    @classmethod
    def dist(
        cls,
        P=None,
        logit_P=None,
        steps=None,
        init_dist=None,
        n_lags=1,
        time_varying_P=False,
        **kwargs,
    ):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=n_lags
        )

        if P is None and logit_P is None:
            raise ValueError("Must specify P or logit_P parameter")
        if P is not None and logit_P is not None:
            raise ValueError("Must specify only one of either P or logit_P parameter")

        if logit_P is not None:
            P = pm.math.softmax(logit_P, axis=-1)

        P = pt.as_tensor_variable(P)

        if time_varying_P:
            # time_varying_P disambiguates a time axis in P (*batch, time, k, ..., k) from a
            # leading batch dimension; without it that axis would be treated as batch. See
            # pymc-extras #392. The time axis sits just left of the (n_lags + 1) state axes and
            # is itself an encoding of `steps`. Reconcile it through the same helper (zero offset,
            # the time axis *is* the transition count): infers steps when only P is given, asserts
            # when both are.
            steps = get_support_shape_1d(
                support_shape=steps, shape=(P.shape[-(n_lags + 2)],), support_shape_offset=0
            )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(intX(steps))

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, RandomVariable | SymbolicRandomVariable
            ):
                raise ValueError(
                    f"Init dist must be a distribution created via the `.dist()` API, "
                    f"got {type(init_dist)}"
                )

            check_dist_not_registered(init_dist)
            if init_dist.owner.op.ndim_supp > 1:
                raise ValueError(
                    "Init distribution must have a scalar or vector support dimension, ",
                    f"got ndim_supp={init_dist.owner.op.ndim_supp}.",
                )
        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Categorical.dist(p=pt.full((k_states, ), 1/k_states), shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            k = P.shape[-1]
            init_dist = pm.Categorical.dist(p=pt.full((k,), 1 / k))

        return super().dist(
            [P, steps, init_dist], n_lags=n_lags, time_varying_P=time_varying_P, **kwargs
        )

    @classmethod
    def rv_op(cls, P, steps, init_dist, n_lags, size=None, time_varying_P=False):
        # Trailing core axes of P: the (n_lags + 1) state axes, plus a leading time axis
        # when P is time-varying. Everything to the left of those is batch.
        n_core_P = (n_lags + 1) + (1 if time_varying_P else 0)
        if size is not None:
            batch_size = size
        else:
            batch_size = pt.broadcast_shape(
                P[tuple([...] + [0] * n_core_P)], pt.atleast_1d(init_dist)[..., 0]
            )

        # A scalar init_dist is broadcast IID across the n_lags initial states; a vector
        # (multivariate) init_dist instead provides them jointly through its own support
        # dimension, which must cover n_lags. Mirrors pymc's AR.
        if init_dist.owner.op.ndim_supp == 0:
            init_dist = change_dist_size(init_dist, (*batch_size, n_lags))
        else:
            init_dist = change_dist_size(init_dist, batch_size)
        init_dist_ = init_dist.type()
        P_ = P.type()
        steps_ = steps.type()

        state_rng = pytensor.shared(np.random.default_rng())

        if time_varying_P:
            # Move the time core axis (at -(n_lags + 2), just left of the n_lags + 1 state axes)
            # to the front so scan iterates a distinct transition matrix per step. With a
            # sequence, scan passes the sequence element first, then the recurring outputs
            # (rng, state).
            def transition(transition_probs, old_rng, *states):
                p = _index_transition_probs(transition_probs, states)
                next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng, return_next_rng=True)
                return next_rng, next_state

            # Pass n_steps too: steps is reconciled with P's time axis at construction, so
            # this keeps steps live in the inner graph (its consistency Assert survives).
            scan_kwargs = dict(sequences=[pt.moveaxis(P_, -(n_lags + 2), 0)], n_steps=steps_)
        else:

            def transition(*args):
                old_rng, *states, transition_probs = args
                p = _index_transition_probs(transition_probs, states)
                next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng, return_next_rng=True)
                return next_rng, next_state

            scan_kwargs = dict(non_sequences=[P_], n_steps=steps_)

        state_next_rng, markov_chain = pytensor.scan(
            transition,
            outputs_info=[
                state_rng,
                # Move lags to the front for scan
                *_make_outputs_info(n_lags, pt.moveaxis(init_dist_, -1, 0)),
            ],
            strict=True,
            return_updates=False,
            **scan_kwargs,
        )

        # The scan's full buffer already carries the n_lags initial taps in front of the sampled
        # states (scan itself returns buffer[n_lags:]); grab it instead of concatenating init_dist_
        # back on. The buffer is time-first, so move time to the last axis.
        discrete_mc_ = pt.moveaxis(markov_chain.owner.inputs[0], 0, -1)

        # P is the full transition tensor P[s_{t-n}, ..., s_{t-1}, s_t] with n_lags + 1 state axes,
        # plus a leading time axis (u) when time-varying. Repeating "p" enforces all state axes are
        # k (the squareness constraint). The output chain (t) is n_lags longer than the u
        # transitions.
        core_states = ",".join(["p"] * (n_lags + 1))
        P_signature = f"(u,{core_states})" if time_varying_P else f"({core_states})"
        discrete_mc_op = DiscreteMarkovChainRV(
            inputs=[P_, steps_, init_dist_, state_rng],
            outputs=[state_next_rng, discrete_mc_],
            n_lags=n_lags,
            time_varying_P=time_varying_P,
            extended_signature=f"{P_signature},(),(p),[rng]->[rng],(t)",
        )

        discrete_mc = discrete_mc_op(P, steps, init_dist, state_rng)
        return discrete_mc


@_change_dist_size.register(DiscreteMarkovChainRV)
def change_mc_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return DiscreteMarkovChain.rv_op(
        *dist.owner.inputs[:-1],
        size=new_size,
        n_lags=op.n_lags,
        time_varying_P=op.time_varying_P,
    )


@_support_point.register(DiscreteMarkovChainRV)
def discrete_mc_moment(op, rv, P, steps, init_dist, state_rng):
    init_dist_moment = support_point(init_dist)
    n_lags = op.n_lags

    if op.time_varying_P:

        def greedy_transition(transition_probs, *states):
            p = _index_transition_probs(transition_probs, states)
            return pt.argmax(p, axis=-1)

        scan_kwargs = dict(sequences=[pt.moveaxis(P, -(n_lags + 2), 0)], n_steps=steps)
    else:

        def greedy_transition(*args):
            *states, transition_probs = args
            p = _index_transition_probs(transition_probs, states)
            return pt.argmax(p, axis=-1)

        scan_kwargs = dict(non_sequences=[P], n_steps=steps)

    chain_moment = pytensor.scan(
        greedy_transition,
        # Seed with the moment of the initial states (lags first for scan).
        outputs_info=_make_outputs_info(n_lags, pt.moveaxis(init_dist_moment, -1, 0)),
        strict=True,
        return_updates=False,
        **scan_kwargs,
    )
    # Full scan buffer (n_lags initial states + steps greedy states), time-first -> move time last.
    return pt.moveaxis(chain_moment.owner.inputs[0], 0, -1)


@_logprob.register(DiscreteMarkovChainRV)
def discrete_mc_logp(op, values, P, steps, init_dist, state_rng, **kwargs):
    value = values[0]
    n_lags = op.n_lags

    # The n_lags + 1 state indices per transition: x_{t-n_lags}, ..., x_{t-1}, x_t.
    indices = [value[..., i : -(n_lags - i) if n_lags != i else None] for i in range(n_lags + 1)]

    init_logp = logp(init_dist, value[..., :n_lags])
    # A scalar init_dist scores each of the n_lags initial states independently (per-position
    # logp, with the lag axis to sum over); a vector (multivariate) init_dist already returns
    # their joint logp. Mirrors pymc's AR. Decided by the logp dimensionality rather than the
    # op's ndim_supp, so inits rewritten into measurable graphs (e.g. an inlined
    # JointCategorical lookup) are handled too.
    if init_logp.type.ndim == value.type.ndim:
        init_logp = init_logp.sum(axis=-1)
    mc_logprob = init_logp

    # Gather the transition log-probabilities by advanced-indexing P's trailing state axes with
    # `indices`. P's leading batch axes (and the time axis, when time-varying) are indexed with
    # aranges so they pair element-wise with the value; leaving them to broadcast would instead
    # produce a spurious extra axis (P's batch crossed with the value's batch).
    log_P = pt.log(P)
    p_shape = tuple(P.shape)
    batch_ndim = log_P.ndim - op.P_core_ndim

    # Index the transition tensor per batch element: an open mesh of aranges over the batch axes
    # (each with a trailing axis to line up with the transition axis in `indices`), then the time
    # axis 1:1 with the transitions when time-varying, then the n_lags + 1 state coordinates.
    batch_index = [b[..., None] for b in ix_(*(pt.arange(p_shape[a]) for a in range(batch_ndim)))]
    time_index = [pt.arange(p_shape[batch_ndim])] if op.time_varying_P else []
    mc_logprob += log_P[(*batch_index, *time_index, *indices)].sum(axis=-1)

    # We cannot leave any RV in the logp graph, even if just for an assert
    # If this is a core_dim, it should be part of the signature
    [init_dist_core_dim] = resolve_shapes([pt.atleast_1d(init_dist).shape[-1]])

    return check_parameters(
        mc_logprob,
        pt.all(pt.eq(pt.stack(p_shape[-(n_lags + 1) :]), p_shape[-1])),
        pt.all(pt.allclose(P.sum(axis=-1), 1.0)),
        pt.eq(init_dist_core_dim, n_lags),
        msg="Last (n_lags + 1) dimensions of P must be square, "
        "P must sum to 1 along the last axis, "
        "First dimension of init_dist must be n_lags",
    )


class DiscreteMarkovChainGibbsMetropolis(CategoricalGibbsMetropolis):
    name = "discrete_markov_chain_gibbs_metropolis"

    def __init__(
        self,
        vars,
        proposal="uniform",
        order="random",
        model=None,
        initial_point=None,
        compile_kwargs: dict | None = None,
        **kwargs,
    ):
        model = pm.modelcontext(model)
        vars = get_value_vars_from_user_vars(vars, model)
        if initial_point is None:
            initial_point = model.initial_point()

        dimcats = []
        # The above variable is a list of pairs (aggregate dimension, number
        # of categories). For example, if vars = [x, y] with x being a 2-D
        # variable with M categories and y being a 3-D variable with N
        # categories, we will have dimcats = [(0, M), (1, M), (2, N), (3, N), (4, N)].
        for v in vars:
            v_init_val = initial_point[v.name]
            rv_var = model.values_to_rvs[v]
            rv_op = rv_var.owner.op

            if not isinstance(rv_op, DiscreteMarkovChainRV):
                raise TypeError("All variables must be DiscreteMarkovChainRV")

            k_graph = rv_var.owner.inputs[0].shape[-1]
            (k_graph,) = model.replace_rvs_by_values((k_graph,))
            k = model.compile_fn(
                k_graph,
                inputs=model.value_vars,
                on_unused_input="ignore",
                mode=Mode(linker="py", optimizer=None),
            )(initial_point)
            start = len(dimcats)
            dimcats += [(dim, k) for dim in range(start, start + v_init_val.size)]

        if order == "random":
            self.shuffle_dims = True
            self.dimcats = dimcats
        else:
            if sorted(order) != list(range(len(dimcats))):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.dimcats = [dimcats[j] for j in order]

        if proposal == "uniform":
            self.astep = self.astep_unif
        elif proposal == "proportional":
            # Use the optimized "Metropolized Gibbs Sampler" described in Liu96.
            self.astep = self.astep_prop
        else:
            raise ValueError("Argument 'proposal' should either be 'uniform' or 'proportional'")

        # Doesn't actually tune, but it's required to emit a sampler stat
        # that indicates whether a draw was done in a tuning phase.
        self.tune = True

        # We bypass CategoryGibbsMetropolis's __init__ to avoid it's specialiazed initialization logic
        if compile_kwargs is None:
            compile_kwargs = {}
        ArrayStep.__init__(self, vars, [model.compile_logp(**compile_kwargs)], **kwargs)

    @staticmethod
    def competence(var):
        if isinstance(var.owner.op, DiscreteMarkovChainRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


STEP_METHODS.append(DiscreteMarkovChainGibbsMetropolis)
