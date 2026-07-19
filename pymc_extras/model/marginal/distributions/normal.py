from pymc import Normal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import get_symbolic_rv_shapes
from pytensor.graph import node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor import broadcast_to, sqrt
from pytensor.tensor.math import add, mul, variadic_add, variadic_mul
from pytensor.tensor.rewriting.basic import broadcasted_by

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    marginal_ir_rewrites_db,
)


def affine_coefficients(mu, x):
    """Return ``(offset, slope)`` with ``mu == offset + slope * x``, or ``None`` if
    ``mu`` is not affine in ``x``. ``slope`` is ``None`` if ``mu`` lacks ``x``.

    Assumes Add/Mul are already flattened into variadic nodes by the pre-canonicalize
    pass, so ``mu`` is read as one flat sum of terms, each either constant in ``x`` or
    ``x`` scaled by constants (``x`` or ``Mul(*consts, x)``).

    Callers must gate on both failure modes before using the result: a ``None``
    return (not affine) and a ``None`` slope (``x`` absent). Only the rewrite that
    builds the NormalNormalMarginalRV does so; the logp and conditional run after
    it and may unpack unconditionally.
    """
    if x not in ancestors([mu]):
        # x absent: mu is pure offset
        return mu, None

    terms = mu.owner.inputs if mu.owner.op == add else [mu]

    offsets, slopes = [], []
    for term in terms:
        if x not in ancestors([term]):
            # constant term contributes to the offset
            offsets.append(term)
            continue
        # x-dependent term must be x scaled by constants (x or Mul(*consts, x))
        factors = term.owner.inputs if term.owner.op == mul else [term]
        const_factors = [f for f in factors if f is not x]
        if sum(f is x for f in factors) != 1 or any(x in ancestors([f]) for f in const_factors):
            # x*x, exp(x), or a non-flat op: not affine
            return None
        slopes.append(variadic_mul(*const_factors))

    return variadic_add(*offsets), variadic_add(*slopes)


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

    # mu_d is affine in the marginalized RV, mu_d = offset + slope * rv, so
    # marginalizing gives y ~ Normal(offset + slope * mu_m, sqrt(sigma_d**2 +
    # (slope * sigma_m)**2)). new_mu falls out of substituting mu_m for the rv.
    _, slope = affine_coefficients(mu_d, marginalized_rv)
    new_mu = graph_replace(mu_d, {marginalized_rv: mu_m})
    new_sigma = sqrt(sigma_d**2 + (slope * sigma_m) ** 2)
    return logp(Normal.dist(mu=new_mu, sigma=new_sigma), value)


@marginalized_conditional.register(NormalNormalMarginalRV)
def normal_normal_conditional(op, inputs, dep_rvs):
    marginalized, dependent = inline_ofg_outputs(op, inputs)[:2]
    [dep_rv] = dep_rvs

    mu_m, sigma_m = marginalized.owner.op.dist_params(marginalized.owner)
    mu_d, sigma_d = dependent.owner.op.dist_params(dependent.owner)

    # dep_rv ~ Normal(offset + slope * marginalized, sigma_d), so as a likelihood
    # for the marginalized variable it contributes precision slope**2 / sigma_d**2
    # with effective observation (dep_rv - offset) / slope.
    offset, slope = affine_coefficients(mu_d, marginalized)

    precision_m = 1 / sigma_m**2
    precision_d = 1 / sigma_d**2
    posterior_precision = precision_m + slope**2 * precision_d
    posterior_sigma = sqrt(1 / posterior_precision)
    posterior_mu = (
        mu_m * precision_m + slope * (dep_rv - offset) * precision_d
    ) / posterior_precision

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

    # Deny broadcasting of the marginalized RV into the dependent. The closed
    # form is elementwise: each dependent draw depends on its own marginalized
    # draw. If the marginalized RV is broadcast (stretched) to a wider dependent,
    # one latent is shared across several dependents and the true marginal is a
    # correlated MvNormal, not the elementwise Normal we emit. Unknown (None)
    # dependent dims count as broadcasting since they may be >1 at runtime.
    if broadcasted_by(marginalized_rv, dependent_rv):
        return None

    mu_dep, sigma_dep = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    if marginalized_rv in ancestors([sigma_dep]):
        return None

    # The dependent mean must be affine in the marginalized RV (mu_dep = a + b*rv);
    # otherwise the marginal is not Normal in closed form. A None slope means the
    # pre-canonicalize pass eliminated the rv from mu_dep (e.g. x - x), so the pair
    # the marker was built for no longer exists and there is nothing to resolve.
    coeffs = affine_coefficients(mu_dep, marginalized_rv)
    if coeffs is None or coeffs[1] is None:
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


marginal_ir_rewrites_db.register("normal_normal_marginal", normal_normal_marginal_rewrite, "basic")
