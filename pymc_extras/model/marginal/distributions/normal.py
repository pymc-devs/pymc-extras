from pymc import MvNormal, Normal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import get_symbolic_rv_shapes
from pytensor.graph import node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor import broadcast_to, eye, flatten, sqrt
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import add, mul, variadic_add, variadic_mul

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.graph_analysis import subgraph_batch_dim_connection
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    marginal_rewrites_db,
)


def affine_coefficients(mu, x):
    """Return ``(offset, slope)`` with ``mu == offset + slope * x``, or ``None`` if
    ``mu`` is not affine in ``x``. ``slope`` is ``None`` if ``mu`` lacks ``x``.

    Assumes Add/Mul are already flattened into variadic nodes by the pre-canonicalize
    pass, so ``mu`` is read as one flat sum of terms, each either constant in ``x`` or
    ``x`` scaled by constants (``x`` or ``Mul(*consts, x)``, ``x`` possibly reshaped).
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
        # x-dependent term must be x scaled by constants (x or Mul(*consts, x)). x may
        # be reshaped by a DimShuffle (broadcast/transpose); the dim analysis validates
        # the reshape, so any DimShuffle of x counts as the latent factor here.
        factors = term.owner.inputs if term.owner.op == mul else [term]
        is_latent = [
            f is x or (isinstance(f.owner_op, DimShuffle) and f.owner.inputs[0] is x)
            for f in factors
        ]
        const_factors = [f for f, latent in zip(factors, is_latent) if not latent]
        if sum(is_latent) != 1 or any(x in ancestors([f]) for f in const_factors):
            # x*x, exp(x), or a non-flat op: not affine
            return None
        slopes.append(variadic_mul(*const_factors))

    return variadic_add(*offsets), variadic_add(*slopes)


class NormalNormalMarginalRV(MarginalRV):
    """Marginalized Normal-Normal conjugate pair.

    Inner graph: [marginalized_normal, dependent_normal, *rng_updates]
    """

    def __init__(self, *args, dims_connections, **kwargs):
        # ``dims_connections[0]`` records, per dependent dim, the marginalized dim it
        # tracks (an int) or None if the latent is broadcast there (a shared dim).
        self.dims_connections = dims_connections
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

    # mu_d is affine in the marginalized RV, mu_d = offset + slope * rv. new_mu falls
    # out of substituting mu_m for the rv; slope scales the latent's contribution.
    _, slope = affine_coefficients(mu_d, marginalized_rv)
    new_mu = graph_replace(mu_d, {marginalized_rv: mu_m})

    # A dependent dim is shared (correlated) where the latent is broadcast into it:
    # either a new dim (dims_connection is None) or a size-1 latent dim stretched
    # wider. Dims that track a full marginalized dim one-to-one stay independent.
    (dims_connection,) = op.dims_connections
    marg_bcast = marginalized_rv.type.broadcastable
    dep_bcast = dependent_rv.type.broadcastable
    shared_axes = [
        i
        for i, d in enumerate(dims_connection)
        if d is None or (marg_bcast[d] and not dep_bcast[i])
    ]

    if not shared_axes:
        # No shared dims: each dependent draw has its own latent draw -> Normal, and
        # marginalizing gives y ~ Normal(new_mu, sqrt(sigma_d**2 + (slope*sigma_m)**2)).
        new_sigma = sqrt(sigma_d**2 + (slope * sigma_m) ** 2)
        return logp(Normal.dist(mu=new_mu, sigma=new_sigma), value)

    # Move the shared dims to the right and ravel them into a single MvNormal event;
    # the remaining (batch) dims stay as independent MvNormals. The event covariance
    # is diag(sigma_d**2) + (slope*sigma_m) outer product (one rank-1 per shared draw).
    batch_axes = [i for i in range(len(dims_connection)) if i not in shared_axes]
    perm = (*batch_axes, *shared_axes)
    dep_shape = get_symbolic_rv_shapes([dependent_rv])[0]

    def to_event(t):
        return flatten(broadcast_to(t, dep_shape).transpose(perm), ndim=len(batch_axes) + 1)

    mean = to_event(new_mu)
    u = to_event(slope * sigma_m)
    cov = u[..., :, None] * u[..., None, :] + to_event(sigma_d**2)[..., :, None] * eye(u.shape[-1])
    return logp(MvNormal.dist(mu=mean, cov=cov), to_event(value))


@marginalized_conditional.register(NormalNormalMarginalRV)
def normal_normal_conditional(op, inputs, dep_rvs):
    marginalized, dependent = inline_ofg_outputs(op, inputs)[:2]
    [dep_rv] = dep_rvs

    mu_m, sigma_m = marginalized.owner.op.dist_params(marginalized.owner)
    mu_d, sigma_d = dependent.owner.op.dist_params(dependent.owner)

    # dep_rv ~ Normal(offset + slope * marginalized, sigma_d), so as a likelihood
    # for the marginalized variable each dependent element contributes precision
    # slope**2 / sigma_d**2 with effective observation (dep_rv - offset) / slope.
    offset, slope = affine_coefficients(mu_d, marginalized)

    # Where the latent is broadcast into several dependents (the shared axes, the same
    # ones the marginal ravels into the MvNormal event) those observations all inform
    # one latent draw, so their evidence sums back onto it. to_latent reduces a
    # dependent-shaped term over the shared axes and lays the rest out as the latent
    # (dropping the summed broadcast dims, reordering matched dims to the latent).
    (dims_connection,) = op.dims_connections
    marg_bcast = marginalized.type.broadcastable
    dep_bcast = dependent.type.broadcastable
    shared_axes = tuple(
        i
        for i, d in enumerate(dims_connection)
        if d is None or (marg_bcast[d] and not dep_bcast[i])
    )
    marg_dim_axis = {d: i for i, d in enumerate(dims_connection) if d is not None}

    def to_latent(term):
        summed = broadcast_to(term, dep_rv.shape).sum(axis=shared_axes, keepdims=True)
        return summed.dimshuffle([marg_dim_axis[d] for d in range(marginalized.type.ndim)])

    precision_m = 1 / sigma_m**2
    precision_d = 1 / sigma_d**2
    posterior_precision = precision_m + to_latent(slope**2 * precision_d)
    posterior_sigma = sqrt(1 / posterior_precision)
    posterior_mu = (
        mu_m * precision_m + to_latent(slope * (dep_rv - offset) * precision_d)
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

    mu_dep, sigma_dep = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    if marginalized_rv in ancestors([sigma_dep]):
        return None

    # The dependent mean must be affine in the marginalized RV (mu_dep = a + b*rv);
    # otherwise the marginal is not Normal in closed form.
    if affine_coefficients(mu_dep, marginalized_rv) is None:
        return None

    # Map each dependent dim to the marginalized dim it tracks (or None where the
    # latent is broadcast/shared). This also rejects couplings the closed form can't
    # express, e.g. x[None, :] + x[:, None].
    try:
        dims_connections = subgraph_batch_dim_connection(marginalized_rv, [dependent_rv])
    except (ValueError, NotImplementedError):
        return None

    typed_op = NormalNormalMarginalRV(
        inputs=inputs,
        outputs=outputs,
        dims_connections=dims_connections,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
    )

    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_rewrites_db.register("normal_normal_marginal", normal_normal_marginal_rewrite, "basic")
