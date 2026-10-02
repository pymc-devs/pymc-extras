import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc import MvNormal
from pymc.distributions.multivariate import _logdet_from_cholesky
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp
from pymc.pytensorf import constant_fold
from pytensor.graph import node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.optimize import minimize

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.rewrites import (
    DEFAULT_MINIMIZER_KWARGS,
    LaplaceMarginalSubgraph,
    extract_marginal_subgraph,
    marginal_ir_rewrites_db,
)


class MarginalLaplaceRV(MarginalRV):
    """Base class for Marginalized Laplace-Approximated RVs.

    Estimates log likelihood using Laplace approximations.

    The precision matrix Q of the marginalized variable is passed as the
    last input of the node (a dummy input, unused by the inner graph).
    """

    is_approximate = True

    def __init__(
        self,
        *args,
        minimizer_kwargs: dict = DEFAULT_MINIMIZER_KWARGS,
        **kwargs,
    ) -> None:
        self.minimizer_kwargs = minimizer_kwargs
        super().__init__(*args, **kwargs)


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


def _laplace_mode_and_precision(log_likelihood, logp_objective, x, x0_init, Q, minimizer_kwargs):
    """Return the posterior mode and precision, with curvature evaluated at the mode."""
    mode, _ = minimize(
        objective=-logp_objective,
        x=x,
        use_vectorized_jac=True,
        **minimizer_kwargs,
    )
    mode = graph_replace(mode, {x: x0_init})
    likelihood_hessian = pytensor.gradient.hessian(log_likelihood, x)
    precision = graph_replace(Q - likelihood_hessian, {x: mode}, strict=False)
    return mode, precision


def get_laplace_approx(
    log_likelihood: TensorVariable,
    logp_objective: TensorVariable,
    x: TensorVariable,
    x0_init: TensorLike,
    Q: TensorLike,
    minimizer_kwargs: dict = DEFAULT_MINIMIZER_KWARGS,
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
    x0, tau = _laplace_mode_and_precision(
        log_likelihood, logp_objective, x, x0_init, Q, minimizer_kwargs
    )
    log_laplace_approx, _ = _precision_mv_normal_logp(x, x0, tau)

    return x0, log_laplace_approx


@_logprob.register(MarginalLaplaceRV)
def laplace_marginal_rv_logp(op: MarginalLaplaceRV, values, *inputs_and_Q, **kwargs):
    # Get Q and remove it from the graph (stored as a dummy input)
    *inputs, Q = inputs_and_Q

    # Clone the inner RV graph of the Marginalized RV
    all_outputs = inline_ofg_outputs(op, inputs_and_Q)
    x = all_outputs[0]
    inner_rvs = list(all_outputs[1 : 1 + op.n_dependent_rvs])

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))

    marginalized_vv = x.clone()
    rv_values = inner_rv_values | {x: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # logp(x | params)
    logp_x = logps_dict.pop(marginalized_vv).sum()

    # logp(y | x, params)
    logp_y = pt.sum([logp_term.sum() for value, logp_term in logps_dict.items()])

    # logp_total = logp(y | x, params) + logp(x | params) (i.e. logp(x | y, params) up to a constant in x)
    logp_total = logp_x + logp_y

    # Set minimizer initialisation to be random (TODO: Let pymc accept this one, maybe when rng is constant)
    # TODO: Use newer pytensor helper
    d = pt.prod(constant_fold(tuple(x.shape), raise_not_constant=True))
    x0_init = pt.ones(d)

    # Obtain laplace approx for logp(x | y, params)
    x0, log_laplace_approx = get_laplace_approx(
        logp_y,
        logp_total,
        x=marginalized_vv,
        x0_init=x0_init,
        Q=Q,
        minimizer_kwargs=op.minimizer_kwargs,
    )

    # logp(y | params) = logp(y | x, params) + logp(x | params) - logp(x | y, params)
    # TODO: Can we recover the elementwise logp?
    marginal_likelihood = logp_total - log_laplace_approx
    joint_logp = graph_replace(marginal_likelihood, {marginalized_vv: x0})
    # Assign the inseparable joint term once, with a factor for every dependent value.
    dummy_logps = (pt.constant(0),) * (len(values) - 1)
    return joint_logp, *dummy_logps


@marginalized_conditional.register(MarginalLaplaceRV)
def laplace_marginalized_conditional(op, inputs, dep_rvs):
    """Build the Gaussian approximation to the conditional posterior, not an exact draw."""
    # Derive logps over root placeholders, as in the enumerable conditional.
    # Otherwise conditional_logp clones upstream RVs, losing caller graph identity
    # and exposing generative samples to nested marginal logp implementations.
    inner_graph = op.fgraph.unfreeze()
    marginalized = inner_graph.outputs[0]
    dependents = inner_graph.outputs[1 : 1 + op.n_dependent_rvs]
    marginalized_value = marginalized.clone()
    dep_values = [dep.type() for dep in dependents]
    rv_values = {marginalized: marginalized_value} | dict(zip(dependents, dep_values))
    logps = conditional_logp(rv_values)
    prior_logp = logps.pop(marginalized_value).sum()
    likelihood_logp = pt.sum([term.sum() for term in logps.values()])

    d = pt.prod(constant_fold(tuple(marginalized.shape), raise_not_constant=True))
    mode, precision = _laplace_mode_and_precision(
        likelihood_logp,
        prior_logp + likelihood_logp,
        marginalized_value,
        pt.ones(d, dtype=marginalized.dtype),
        inner_graph.inputs[-1],
        op.minimizer_kwargs,
    )
    conditional_rv = MvNormal.dist(mu=mode, tau=precision)
    replacements = dict(zip(inner_graph.inputs, inputs))
    replacements.update(zip(dep_values, dep_rvs))
    return graph_replace(conditional_rv, replacements, strict=False)


@node_rewriter(tracks=[LaplaceMarginalSubgraph])
def laplace_marginal(fgraph, node):
    op = node.op

    # Q was appended as the last boundary input and is kept as a dummy input
    # of the OpFromGraph (popped again by the logp implementation)
    inputs, outputs = extract_marginal_subgraph(node)

    # Q may coincide with another boundary input (e.g. Q=tau where tau is also a
    # parameter of the marginalized RV). OpFromGraph requires distinct inputs, and
    # the inner graph never uses Q, so stand in a dummy variable for it.
    *rest_inputs, Q = inputs
    ofg_inputs = [*rest_inputs, Q.type()]

    typed_op = MarginalLaplaceRV(
        inputs=ofg_inputs,
        outputs=outputs,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
        n_dependent_rvs=op.n_dependent_rvs,
        minimizer_kwargs=op.minimizer_kwargs,
    )

    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_ir_rewrites_db.register("laplace_marginal", laplace_marginal, "basic")
