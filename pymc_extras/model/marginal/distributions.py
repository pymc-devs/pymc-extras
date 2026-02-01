import warnings

from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.distributions import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.distribution import _support_point, support_point
from pymc.distributions.multivariate import _logdet_from_cholesky
from pymc.logprob.abstract import MeasurableOp, ValuedRV, _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import constant_fold
from pytensor import Variable
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import Mode
from pytensor.graph import FunctionGraph, Op, vectorize_graph
from pytensor.graph.basic import equal_computations
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.scan import map as scan_map
from pytensor.scan import scan
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.optimize import minimize
from pytensor.tensor.random.type import RandomType

from pymc_extras.distributions import DiscreteMarkovChain


class GradientBlocker(Op):
    """
    An Op that forwards its input unchanged but blocks gradient computation.

    This is used to prevent gradient flow through operations like MinimizeOp
    that don't have properly defined gradients. Unlike disconnected_grad,
    this Op returns grad_undefined which ensures NullType gradients.
    """

    __props__ = ()
    itypes = None  # Will be set dynamically
    otypes = None  # Will be set dynamically

    def make_node(self, x):
        from pytensor.graph.basic import Apply

        # Ensure x is a tensor variable
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        # inputs[0] is a numpy array at runtime
        outputs[0][0] = np.asarray(inputs[0])

    def grad(self, inputs, output_grads):
        from pytensor.gradient import grad_undefined

        return [grad_undefined(self, 0, inputs[0])]

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


def block_gradient(x):
    """Block gradient computation through tensor x by returning grad_undefined."""
    return GradientBlocker()(x)


class MarginalRV(OpFromGraph, MeasurableOp):
    """Base class for Marginalized RVs"""

    def __init__(
        self,
        *args,
        dims_connections: tuple[tuple[int | None], ...],
        dims: tuple[Variable, ...],
        **kwargs,
    ) -> None:
        self.dims_connections = dims_connections
        self.dims = dims
        super().__init__(*args, **kwargs)

    @property
    def support_axes(self) -> tuple[tuple[int]]:
        """Dimensions of dependent RVs that belong to the core (non-batched) marginalized variable."""
        marginalized_ndim_supp = self.inner_outputs[0].owner.op.ndim_supp
        support_axes_vars = []
        for dims_connection in self.dims_connections:
            ndim = len(dims_connection)
            marginalized_supp_axes = ndim - marginalized_ndim_supp
            support_axes_vars.append(
                tuple(
                    -i
                    for i, dim in enumerate(reversed(dims_connection), start=1)
                    if (dim is None or dim > marginalized_supp_axes)
                )
            )
        return tuple(support_axes_vars)

    def __eq__(self, other):
        # Just to allow easy testing of equivalent models,
        # This can be removed once https://github.com/pymc-devs/pytensor/issues/1114 is fixed
        if type(self) is not type(other):
            return False

        return equal_computations(
            self.inner_outputs,
            other.inner_outputs,
            self.inner_inputs,
            other.inner_inputs,
        )

    def __hash__(self):
        # Just to allow easy testing of equivalent models,
        # This can be removed once https://github.com/pymc-devs/pytensor/issues/1114 is fixed
        return hash((type(self), len(self.inner_inputs), len(self.inner_outputs)))


@_support_point.register
def support_point_marginal_rv(op: MarginalRV, rv, *inputs):
    """Support point for a marginalized RV.

    The support point of a marginalized RV is the support point of the inner RV,
    conditioned on the marginalized RV taking its support point.
    """
    outputs = rv.owner.outputs

    inner_rv = op.inner_outputs[outputs.index(rv)]
    marginalized_inner_rv, *other_dependent_inner_rvs = (
        out
        for out in op.inner_outputs
        if out is not inner_rv and not isinstance(out.type, RandomType)
    )

    # Replace references to inner rvs by the dummy variables (including the marginalized RV)
    # This is necessary because the inner RVs may depend on each other
    marginalized_inner_rv_dummy = marginalized_inner_rv.clone()
    other_dependent_inner_rv_to_dummies = {
        inner_rv: inner_rv.clone() for inner_rv in other_dependent_inner_rvs
    }
    inner_rv = clone_replace(
        inner_rv,
        replace={marginalized_inner_rv: marginalized_inner_rv_dummy}
        | other_dependent_inner_rv_to_dummies,
    )

    # Get support point of inner RV and marginalized RV
    inner_rv_support_point = support_point(inner_rv)
    marginalized_inner_rv_support_point = support_point(marginalized_inner_rv)

    replacements = [
        # Replace the marginalized RV dummy by its support point
        (marginalized_inner_rv_dummy, marginalized_inner_rv_support_point),
        # Replace other dependent RVs dummies by the respective outer outputs.
        # PyMC will replace them by their support points later
        *(
            (v, outputs[op.inner_outputs.index(k)])
            for k, v in other_dependent_inner_rv_to_dummies.items()
        ),
        # Replace outer input RVs
        *zip(op.inner_inputs, inputs),
    ]
    fgraph = FunctionGraph(outputs=[inner_rv_support_point], clone=False)
    fgraph.replace_all(replacements, import_missing=True)
    [rv_support_point] = fgraph.outputs
    return rv_support_point


class MarginalFiniteDiscreteRV(MarginalRV):
    """Base class for Marginalized Finite Discrete RVs"""


class MarginalDiscreteMarkovChainRV(MarginalRV):
    """Base class for Marginalized Discrete Markov Chain RVs"""


class MarginalLaplaceRV(MarginalRV):
    """Base class for Marginalized Laplace-Approximated RVs.

    Estimates log likelihood using Laplace approximations.
    """

    def __init__(
        self,
        *args,
        Q: TensorVariable,
        minimizer_seed: int,
        minimizer_kwargs: dict = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}},
        **kwargs,
    ) -> None:
        self.Q = Q
        self.minimizer_seed = minimizer_seed
        self.minimizer_kwargs = minimizer_kwargs
        super().__init__(*args, **kwargs)


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    dist_params = rv.owner.op.dist_params(rv.owner)
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        [p_param] = dist_params
        [p_param_length] = constant_fold([p_param.shape[-1]])
        return tuple(range(p_param_length))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(dist_params)
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P, *_ = dist_params
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def reduce_batch_dependent_logps(
    dependent_dims_connections: Sequence[tuple[int | None, ...]],
    dependent_ops: Sequence[Op],
    dependent_logps: Sequence[TensorVariable],
) -> TensorVariable:
    """Combine the logps of dependent RVs and align them with the marginalized logp.

    This requires reducing extra batch dims and transposing when they are not aligned.

       idx = pm.Bernoulli(idx, shape=(3, 2))  # 0, 1
       pm.Normal("dep1", mu=idx.T[..., None] * 2, shape=(3, 2, 5))
       pm.Normal("dep2", mu=idx * 2, shape=(7, 2, 3))

       marginalize(idx)

       The marginalized op will have dims_connections = [(1, 0, None), (None, 0, 1)]
       which tells us we need to reduce the last axis of dep1 logp and the first of dep2 logp,
       as well as transpose the remaining axis of dep1 logp before adding the two element-wise.

    """
    from pymc_extras.model.marginal.graph_analysis import get_support_axes

    reduced_logps = []
    for dependent_op, dependent_logp, dependent_dims_connection in zip(
        dependent_ops, dependent_logps, dependent_dims_connections
    ):
        if dependent_logp.type.ndim > 0:
            # Find which support axis implied by the MarginalRV need to be reduced
            # Some may have already been reduced by the logp expression of the dependent RV (e.g., multivariate RVs)
            dep_supp_axes = get_support_axes(dependent_op)[0]

            # Dependent RV support axes are already collapsed in the logp, so we ignore them
            supp_axes = [
                -i
                for i, dim in enumerate(reversed(dependent_dims_connection), start=1)
                if (dim is None and -i not in dep_supp_axes)
            ]
            dependent_logp = dependent_logp.sum(supp_axes)

            # Finally, we need to align the dependent logp batch dimensions with the marginalized logp
            dims_alignment = [dim for dim in dependent_dims_connection if dim is not None]
            dependent_logp = dependent_logp.transpose(*dims_alignment)

        reduced_logps.append(dependent_logp)

    reduced_logp = pt.add(*reduced_logps)
    return reduced_logp


def align_logp_dims(dims: tuple[tuple[int, None]], logp: TensorVariable) -> TensorVariable:
    """Align the logp with the order specified in dims."""
    dims_alignment = [dim for dim in dims if dim is not None]
    return logp.transpose(*dims_alignment)


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> tuple[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    return graph_replace(
        op.inner_outputs,
        replace=tuple(zip(op.inner_inputs, inputs)),
        strict=False,
    )


class NonSeparableLogpWarning(UserWarning):
    pass


def warn_non_separable_logp(values):
    if len(values) > 1:
        warnings.warn(
            "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
            f"Their joint logp terms will be assigned to the first value: {values[0]}.",
            NonSeparableLogpWarning,
            stacklevel=2,
        )


DUMMY_ZERO = pt.constant(0, name="dummy_zero")


@_logprob.register(MarginalFiniteDiscreteRV)
def finite_discrete_marginal_rv_logp(op: MarginalFiniteDiscreteRV, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rv, *inner_rvs = inline_ofg_outputs(op, inputs)

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[inner_rv.owner.op for inner_rv in inner_rvs],
        dependent_logps=[logps_dict[value] for value in values],
    )

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it suffices to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        -1,
        0,
    )

    try:
        joint_logps = vectorize_graph(
            joint_logp, replace={marginalized_vv: marginalized_rv_domain_tensor}
        )
    except Exception:
        # Fallback to Scan
        def logp_fn(marginalized_rv_const, *non_sequences):
            return graph_replace(joint_logp, replace={marginalized_vv: marginalized_rv_const})

        joint_logps = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
            return_updates=False,
        )

    joint_logp = pt.logsumexp(joint_logps, axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    joint_logp = align_logp_dims(dims=op.dims_connections[0], logp=joint_logp)

    warn_non_separable_logp(values)
    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):
    chain_rv, *dependent_rvs = inline_ofg_outputs(op, inputs)

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
    # There is a degerate batch dim for lags=1 (the only supported case),
    # that we have to work around, by expanding the batch value and then squeezing it out of the logp
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[:, None, ..., 0]}
    ).squeeze(1)
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None] + log_P, axis=0)
        return logp_emission + step_log_prob

    P_bcast_dims = (len(chain_shape) - 1) - (P.type.ndim - 2)
    log_P = pt.shape_padright(pt.log(P), P_bcast_dims)
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


def _precision_mv_normal_logp(value: TensorLike, mean: TensorLike, tau: TensorLike):
    """
    Compute the log likelihood of a multivariate normal distribution in precision form. May be phased out - see https://github.com/pymc-devs/pymc/pull/7895

    Parameters
    ----------
    value: TensorLike
        Query point to compute the log prob at.
    mean: TensorLike
        Mean vector of the Gaussian,
    tau: TensorLike
        Precision matrix of the Gaussian (i.e. cov = inv(tau))

    Returns
    -------
    logp: TensorLike
        Log likelihood at value.
    posdef: TensorLike
        Boolean indicating whether the precision matrix is positive definite.
    """
    k = value.shape[-1].astype("floatX")

    delta = value - mean
    quadratic_form = delta.T @ tau @ delta
    logdet, posdef = _logdet_from_cholesky(pt.linalg.cholesky(tau, lower=True))
    logp = -0.5 * (k * pt.log(2 * np.pi) + quadratic_form) + logdet

    return logp, posdef


def get_laplace_approx(
    log_likelihood: TensorVariable,
    logp_objective: TensorVariable,
    x: TensorVariable,
    x0_init: TensorLike,
    Q: TensorLike,
    minimizer_kwargs: dict = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}},
):
    """
    Compute the laplace approximation logp_G(x | y, params) of some variable x.

    Parameters
    ----------
    log_likelihood: TensorVariable
        Model likelihood logp(y | x, params).
    logp_objective: TensorVariable
        Obective log likelihood to maximize, logp(x | y, params) (up to some constant in x).
    x: TensorVariable
        Variable to be laplace approximated.
    x0_init: TensorLike
        Initial guess for minimization.
    Q: TensorLike
        Precision matrix of x.
    minimizer_kwargs:
        Kwargs to pass to pytensor.optimize.minimize.

    Returns
    -------
    x0: TensorVariable
        x*, the maximizer of logp(x | y, params) in x.
    log_laplace_approx: TensorVariable
        Laplace approximation of logp(x | y, params) evaluated at x.
    """
    # Maximize log(p(x | y, params)) wrt x to find mode x0
    # This step is currently bottlenecking the logp calculation.
    #
    # IMPORTANT: We use clone_replace to create a copy of the logp_objective graph.
    # This prevents graph_replace (called later to substitute x with x0) from
    # modifying the MinimizeOp's inputs, which would create a new MinimizeOp node
    # in the final graph and cause gradient computation issues.
    from pytensor.graph.replace import clone_replace

    logp_objective_clone = clone_replace(logp_objective)

    x0, _ = minimize(
        objective=-logp_objective_clone,  # logp(x | y, params) = logp(y | x, params) + logp(x | params) + const (const omitted during minimization)
        x=x,
        use_vectorized_jac=True,
        **minimizer_kwargs,
    )

    # Set minimizer initialisation to be random
    x0 = pytensor.graph.replace.graph_replace(x0, {x: x0_init})

    # Block gradients through the minimize operation using our custom Op.
    # The optimization result x0 should be treated as a fixed point for gradient computation.
    # Without this, NUTS sampling fails because PyTensor cannot backpropagate through MinimizeOp.
    # We use block_gradient (which returns grad_undefined/NullType) instead of disconnected_grad
    # because MinimizeOp's gradient handling doesn't work well with DisconnectedType.
    x0 = block_gradient(x0)

    # This step is also expensive (but not as much as minimize). Could be made more efficient by recycling hessian from the minimizer step, however that requires a bespoke algorithm described in Rasmussen & Williams
    # since the general optimisation scheme maximises logp(x | y, params) rather than logp(y | x, params), and thus the hessian that comes out of methods
    # like L-BFGS-B is in fact not the hessian of logp(y | x, params)

    # Compute the Hessian as a function of x
    hess = pytensor.gradient.hessian(log_likelihood, x)

    # Evaluate the Hessian at x0 and block gradients through the MinimizeOp.
    # This is crucial: without blocking, gradient computation through the Hessian
    # would try to backpropagate through MinimizeOp, causing assertion errors.
    hess_at_x0 = pytensor.graph.replace.graph_replace(hess, {x: x0})
    hess_at_x0 = block_gradient(hess_at_x0)

    # Evaluate logp of Laplace approx of logp(x | y, params) at some point x
    # Note: we use hess_at_x0 (disconnected) instead of hess
    tau = Q - hess_at_x0
    mu = x0
    # Use x0 directly as the evaluation point, since the marginal likelihood formula is:
    # log p(y | params) = log p(y, x0 | params) - log p_G(x0 | y, params)
    log_laplace_approx, _ = _precision_mv_normal_logp(x0, mu, tau)

    return x0, log_laplace_approx


@_logprob.register(MarginalLaplaceRV)
def laplace_marginal_rv_logp(op: MarginalLaplaceRV, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    x, *inner_rvs = inline_ofg_outputs(op, inputs)

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))

    marginalized_vv = x.clone()
    rv_values = inner_rv_values | {x: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # logp(y | x, params)
    log_likelihood = pt.sum(
        [logp_term.sum() for value, logp_term in logps_dict.items() if value is not marginalized_vv]
    )

    # logp = logp(y | x, params) + logp(x | params) (i.e. logp(x | y, params) up to a constant in x)
    logp = pt.sum([pt.sum(logps_dict[k]) for k in logps_dict])

    # Set minimizer initialisation to be random
    # Assumes that the observed variable y is the only element in values, and that d is shape[-1] - if this is invalid it will simply crash rather than producing an invalid result.
    # A more robust method of obtaining d would be ideal.
    if len(values) > 1:
        warnings.warn(
            f"INLA assumes that the latent field {marginalized_vv.name} is of the same dimension as the observables and that there is only one observable, however more than one input value to the logp was provided."
        )
    d = values[0].data.shape[-1]
    rng = np.random.default_rng(op.minimizer_seed)
    x0_init = rng.random(d)

    # Get Q from the list of inputs
    Q = None
    if isinstance(op.Q, TensorVariable):
        for var in inputs:
            if var.owner is not None and isinstance(var.owner.op, ValuedRV):
                for inp in var.owner.inputs:
                    if (
                        inp.name is not None
                        and inp.name == op.Q.name
                        or inp.name == op.Q.name + "_log"
                    ):
                        Q = var
                        break

            if var.name is not None and var.name == op.Q.name or var.name == op.Q.name + "_log":
                Q = var
                break

        if Q is None:
            raise ValueError(f"No inputs could be matched to precision matrix {op.Q}: {inputs}.")

    # Q is an array
    else:
        Q = op.Q

    # Obtain laplace approx
    x0, log_laplace_approx = get_laplace_approx(
        log_likelihood, logp, marginalized_vv, x0_init, Q, op.minimizer_kwargs
    )

    # logp(y | params) = logp(y | x, params) + logp(x | params) - logp(x | y, params)
    marginal_likelihood = logp - log_laplace_approx

    # Block gradients through x0 AGAIN before the final graph_replace.
    # This ensures that when marginalized_vv is replaced with x0 throughout the graph,
    # no gradient paths through MinimizeOp are created.
    x0 = block_gradient(x0)

    return pytensor.graph.replace.graph_replace(marginal_likelihood, {marginalized_vv: x0})
