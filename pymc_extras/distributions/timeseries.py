import warnings

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (Distribution,
                                             SymbolicRandomVariable,
                                             _support_point, support_point)
from pymc.distributions.shape_utils import (_change_dist_size,
                                            change_dist_size,
                                            get_support_shape_1d)
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import constant_fold, intX
from pymc.step_methods import STEP_METHODS
from pymc.step_methods.arraystep import ArrayStep
from pymc.step_methods.compound import Competence
from pymc.step_methods.metropolis import CategoricalGibbsMetropolis
from pymc.util import check_dist_not_registered, get_value_vars_from_user_vars
from pytensor import Mode
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable


def _make_outputs_info(
    n_lags: int, init_dist: Distribution
) -> list[Distribution | dict]:
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


class DiscreteMarkovChainRV(SymbolicRandomVariable):
    n_lags: int
    default_output = 1
    _print_name = ("DiscreteMC", "\\operatorname{DiscreteMC}")

    def __init__(self, *args, n_lags, **kwargs):
        self.n_lags = n_lags
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class DiscreteMarkovChain(Distribution):
    r"""
    A Discrete Markov Chain is a sequence of random variables

    .. math::

        \{x_t\}_{t=0}^T

    Where transition probability :math:`P(x_t | x_{t-1})` depends only on the state of the system at :math:`x_{t-1}`.

    Parameters
    ----------
    P: tensor
        Matrix of transition probabilities between states. Rows must sum to 1.
        One of P or P_logits must be provided.
    P_logit: tensor, optional
        Matrix of transition logits. Converted to probabilities via Softmax activation.
        One of P or P_logits must be provided.
    steps: tensor, optional
        Length of the markov chain. Only needed if state is not provided.
    init_dist : unnamed distribution, optional
        Vector distribution for initial values. Unnamed refers to distributions
        created with the ``.dist()`` API. Distribution should have shape n_states.
        If not, it will be automatically resized.

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.

    Notes
    -----
    The initial distribution will be cloned, rendering it distinct from the one passed as
    input.

    Examples
    --------
     Create a Markov Chain of length 100 with 3 states. The number of states is given by the shape of P,
     3 in this case.

    .. code-block:: python

        import pymc as pm
        import pymc_extras as pmx

        with pm.Model() as markov_chain:
            P = pm.Dirichlet("P", a=[1, 1, 1], size=(3,))
            init_dist = pm.Categorical.dist(p=np.full(3, 1 / 3))
            markov_chain = pmx.DiscreteMarkovChain(
                "markov_chain", P=P, init_dist=init_dist, shape=(100,)
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
    def dist(cls, P=None, logit_P=None, steps=None, init_dist=None, n_lags=1, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=kwargs.get("shape", None),
            support_shape_offset=n_lags,
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        if P is None and logit_P is None:
            raise ValueError("Must specify P or logit_P parameter")
        if P is not None and logit_P is not None:
            raise ValueError("Must specify only one of either P or logit_P parameter")

        if logit_P is not None:
            P = pm.math.softmax(logit_P, axis=-1)

        P = pt.as_tensor_variable(P)
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

        return super().dist([P, steps, init_dist], n_lags=n_lags, **kwargs)

    @classmethod
    def rv_op(cls, P, steps, init_dist, n_lags, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = pt.broadcast_shape(
                P[tuple([...] + [0] * (n_lags + 1))], pt.atleast_1d(init_dist)[..., 0]
            )

        init_dist = change_dist_size(init_dist, (n_lags, *batch_size))
        init_dist_ = init_dist.type()
        P_ = P.type()
        steps_ = steps.type()

        state_rng = pytensor.shared(np.random.default_rng())

        def transition(*args):
            old_rng, *states, transition_probs = args
            p = transition_probs[tuple(states)]
            next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng).owner.outputs
            return next_rng, next_state

        state_next_rng, markov_chain = pytensor.scan(
            transition,
            outputs_info=[state_rng, *_make_outputs_info(n_lags, init_dist_)],
            non_sequences=[P_],
            n_steps=steps_,
            strict=True,
            return_updates=False,
        )

        discrete_mc_ = pt.moveaxis(
            pt.concatenate([init_dist_, markov_chain], axis=0), 0, -1
        )

        discrete_mc_op = DiscreteMarkovChainRV(
            inputs=[P_, steps_, init_dist_, state_rng],
            outputs=[state_next_rng, discrete_mc_],
            n_lags=n_lags,
            extended_signature="(p,p),(),(p),[rng]->[rng],(t)",
        )

        discrete_mc = discrete_mc_op(P, steps, init_dist, state_rng)
        return discrete_mc


@_change_dist_size.register(DiscreteMarkovChainRV)
def change_mc_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return DiscreteMarkovChain.rv_op(
        *dist.owner.inputs[:-1], size=new_size, n_lags=op.n_lags
    )


@_support_point.register(DiscreteMarkovChainRV)
def discrete_mc_moment(op, rv, P, steps, init_dist, state_rng):
    init_dist_moment = support_point(init_dist)
    n_lags = op.n_lags

    def greedy_transition(*args):
        *states, transition_probs = args
        p = transition_probs[tuple(states)]
        return pt.argmax(p)

    chain_moment = pytensor.scan(
        greedy_transition,
        non_sequences=[P],
        outputs_info=_make_outputs_info(n_lags, init_dist),
        n_steps=steps,
        strict=True,
        return_updates=False,
    )
    chain_moment = pt.concatenate([init_dist_moment, chain_moment])
    return chain_moment


@_logprob.register(DiscreteMarkovChainRV)
def discrete_mc_logp(op, values, P, steps, init_dist, state_rng, **kwargs):
    value = values[0]
    n_lags = op.n_lags

    indexes = [
        value[..., i : -(n_lags - i) if n_lags != i else None]
        for i in range(n_lags + 1)
    ]

    mc_logprob = logp(init_dist, value[..., :n_lags]).sum(axis=-1)
    mc_logprob += pt.log(P[tuple(indexes)]).sum(axis=-1)

    # We cannot leave any RV in the logp graph, even if just for an assert
    [init_dist_leading_dim] = constant_fold(
        [pt.atleast_1d(init_dist).shape[0]], raise_not_constant=False
    )

    return check_parameters(
        mc_logprob,
        pt.all(pt.eq(P.shape[-(n_lags + 1) :], P.shape[-1])),
        pt.all(pt.allclose(P.sum(axis=-1), 1.0)),
        pt.eq(init_dist_leading_dim, n_lags),
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
            raise ValueError(
                "Argument 'proposal' should either be 'uniform' or 'proportional'"
            )

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
