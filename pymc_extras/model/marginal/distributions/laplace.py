import numpy as np
import pytensor
import pytensor.tensor as pt

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
)
from pymc_extras.model.marginal.rewrites import (
    DEFAULT_MINIMIZER_KWARGS,
    LaplaceMarginalSubgraph,
    extract_marginal_subgraph,
    marginal_rewrites_db,
)


class MarginalLaplaceRV(MarginalRV):
    """Base class for Marginalized Laplace-Approximated RVs.

    Estimates log likelihood using Laplace approximations.

    The precision matrix Q of the marginalized variable is passed as the
    last input of the node (a dummy input, unused by the inner graph).
    """

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
    # Maximize log(p(x | y, params)) wrt x to find mode x0
    # This step is currently bottlenecking the logp calculation.
    x0, _ = minimize(
        objective=-logp_objective,  # logp(x | y, params) = logp(y | x, params) + logp(x | params) + const (const omitted during minimization)
        x=x,
        use_vectorized_jac=True,
        **minimizer_kwargs,
    )

    # Set minimizer initialisation to be random
    x0 = pytensor.graph.replace.graph_replace(x0, {x: x0_init})

    # This step is also expensive (but not as much as minimize). Could be made more efficient by recycling hessian from the minimizer step, however that requires a bespoke algorithm described in Rasmussen & Williams
    # since the general optimisation scheme maximises logp(x | y, params) rather than logp(y | x, params), and thus the hessian that comes out of methods
    # like L-BFGS-B is in fact not the hessian of logp(y | x, params)
    # TODO: Use vectorized hessian?
    hess = pytensor.gradient.hessian(log_likelihood, x)

    # Evaluate logp of Laplace approx of logp(x | y, params) at some point x
    tau = Q - hess
    mu = x0
    log_laplace_approx, _ = _precision_mv_normal_logp(x, mu, tau)

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
    return graph_replace(marginal_likelihood, {marginalized_vv: x0})


@node_rewriter(tracks=[LaplaceMarginalSubgraph])
def laplace_marginal(fgraph, node):
    op = node.op

    # Q was appended as the last boundary input and is kept as a dummy input
    # of the OpFromGraph (popped again by the logp implementation)
    inputs, outputs = extract_marginal_subgraph(node)

    typed_op = MarginalLaplaceRV(
        inputs=inputs,
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


marginal_rewrites_db.register("laplace_marginal", laplace_marginal, "basic")
