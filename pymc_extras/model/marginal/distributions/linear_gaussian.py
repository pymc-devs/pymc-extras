"""Linear-Gaussian conjugacy for `pymc_extras.model.marginal`.

Handles the pair

    f ~ MvNormal(m, K)
    y ~ Normal(g(f), s)   or   MvNormal(g(f), S)

where `g` is any *affine* function of `f`, i.e. `g(f) = A f + b` with `A`, `b`
free of `f`. Marginalizing `f` gives

    y ~ MvNormal(A m + b, A K A' + S)

and the conditional is the usual Gaussian update

    f | y ~ MvNormal(m + (A K)' G^-1 r,  K - (A K)' G^-1 (A K))
    G = A K A' + S,   r = y - (A m + b)

`A` is never materialized. The linear map is applied to a stack of vectors with
`vectorize_graph`, which preserves structure: when `g` selects a subset of `f`
(the GP case) the map stays a subtensor instead of becoming a dense selection
matrix.

This is the only GP-specific machinery the GP API needs. A GP prior is an
MvNormal over stacked inputs; "observed at a subset of those inputs" is an
affine map; `marginalize` and `conditional` then work unmodified.
"""

import pytensor.tensor as pt

from pymc import MvNormal, Normal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import get_symbolic_rv_shapes
from pytensor.graph import node_rewriter
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.scalar import Add, Mul, Neg, Sub, TrueDiv
from pytensor.tensor.basic import Join, MakeVector, Split
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.reshape import SplitDims
from pytensor.tensor.shape import Reshape, SpecifyShape
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor

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

JITTER = 1e-8


# ---------------------------------------------------------------- affineness


def _depends(var, f):
    return var is f or f in ancestors([var])


def is_affine_in(expr, f):
    """Whether `expr` is an affine function of `f`.

    Conservative whitelist walk over the ops between `f` and `expr`; anything
    not recognized is treated as non-affine so the rewrite declines cleanly.
    """
    if expr is f:
        return True
    if not _depends(expr, f):
        # f-free subgraphs are constants, which are affine
        return True

    node = expr.owner
    if node is None:
        return False

    op = node.op
    dep = [i for i in node.inputs if _depends(i, f)]

    if isinstance(op, Elemwise):
        scalar_op = op.scalar_op
        if isinstance(scalar_op, Add | Sub | Neg):
            return all(is_affine_in(i, f) for i in dep)
        if isinstance(scalar_op, Mul):
            # only one factor may carry f, and linearly
            return len(dep) == 1 and is_affine_in(dep[0], f)
        if isinstance(scalar_op, TrueDiv):
            # f may only appear in the numerator
            return len(dep) == 1 and dep[0] is node.inputs[0] and is_affine_in(dep[0], f)
        return False

    if isinstance(op, Dot) or (isinstance(op, Blockwise) and isinstance(op.core_op, Dot)):
        return len(dep) == 1 and is_affine_in(dep[0], f)

    if isinstance(op, Subtensor | AdvancedSubtensor | AdvancedSubtensor1 | Split):
        # f may only flow through the indexed tensor, never the indices/splits
        return dep == [node.inputs[0]] and is_affine_in(node.inputs[0], f)

    if isinstance(op, DimShuffle | Reshape | SpecifyShape | Join | MakeVector | SplitDims):
        return all(is_affine_in(i, f) for i in dep)

    if isinstance(op, CAReduce) and isinstance(op.scalar_op, Add):
        return all(is_affine_in(i, f) for i in dep)

    return False


# ------------------------------------------------------------------- pieces


def _noise_covariance(dependent_rv, n):
    op = dependent_rv.owner.op
    if isinstance(op, Normal):
        _, sigma = op.dist_params(dependent_rv.owner)
        return pt.diag(pt.broadcast_to(pt.atleast_1d(sigma**2), (n,)))
    if isinstance(op, MvNormal):
        _, cov = op.dist_params(dependent_rv.owner)
        return cov
    raise NotImplementedError(f"Unsupported dependent distribution {op}")


class LinearGaussianMarginalRV(MarginalRV):
    """Marginalized MvNormal latent under a linear-Gaussian observation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_dependent_rvs=1, **kwargs)


def _pieces(op, inputs):
    """Return (m, K, mean_y, AK, G) for the marginalized pair."""
    marginalized_rv, dependent_rv = inline_ofg_outputs(op, inputs)[:2]

    m, K = marginalized_rv.owner.op.dist_params(marginalized_rv.owner)
    # Take the shape from the RV's params, never from the RV itself: any
    # leftover reference would leave a RandomVariable in the logp graph.
    [rv_shape] = get_symbolic_rv_shapes([marginalized_rv])
    dtype = marginalized_rv.type.dtype
    m = pt.atleast_1d(pt.broadcast_to(m, rv_shape)).astype(dtype)

    mu_dep = pt.atleast_1d(dependent_rv.owner.op.dist_params(dependent_rv.owner)[0])

    # b = g(0);  mean_y = g(m) = A m + b
    zeros = pt.zeros(rv_shape, dtype=dtype)
    b = graph_replace(mu_dep, {marginalized_rv: zeros})
    mean_y = graph_replace(mu_dep, {marginalized_rv: m})

    # Apply A to a stack of vectors laid out along the leading axis:
    # apply_A(V)[i] == A @ V[i].  Never materializes A.
    def apply_A(V):
        return vectorize_graph(mu_dep, {marginalized_rv: V}) - b

    KAt = apply_A(K)  # (n_f, n_dep)   == K A'   (K symmetric)
    AK = KAt.T  # (n_dep, n_f)
    AKAt = apply_A(AK)  # (n_dep, n_dep) == A K A'

    n_dep = AKAt.shape[0]
    G = AKAt + _noise_covariance(dependent_rv, n_dep)
    G = G + JITTER * pt.eye(n_dep)

    return m, K, mean_y, AK, G


@_logprob.register(LinearGaussianMarginalRV)
def linear_gaussian_marginal_logp(op, values, *inputs, **kwargs):
    [value] = values
    _, _, mean_y, _, G = _pieces(op, inputs)
    return logp(MvNormal.dist(mu=mean_y, cov=G), pt.atleast_1d(value))


@marginalized_conditional.register(LinearGaussianMarginalRV)
def linear_gaussian_conditional(op, inputs, dep_rvs):
    [y] = dep_rvs
    m, K, mean_y, AK, G = _pieces(op, inputs)

    L = pt.linalg.cholesky(G)
    alpha = pt.linalg.solve_triangular(L, pt.atleast_1d(y) - mean_y, lower=True)
    V = pt.linalg.solve_triangular(L, AK, lower=True)  # (n_dep, n_f)

    post_mu = m + V.T @ alpha
    post_cov = K - V.T @ V
    post_cov = 0.5 * (post_cov + post_cov.T) + JITTER * pt.eye(post_cov.shape[0])

    return MvNormal.dist(mu=post_mu, cov=post_cov)


# ------------------------------------------------------------------ rewrite


@node_rewriter(tracks=[MarginalSubgraph])
def linear_gaussian_marginal_rewrite(fgraph, node):
    op = node.op
    if op.n_dependent_rvs != 1:
        return None

    inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv, dependent_rv = outputs[0], outputs[1]

    if not isinstance(marginalized_rv.owner.op, MvNormal):
        return None
    if not isinstance(dependent_rv.owner.op, Normal | MvNormal):
        return None

    mu_dep, *rest_params = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    # observation noise must not depend on the latent
    if any(_depends(p, marginalized_rv) for p in rest_params):
        return None

    if not is_affine_in(mu_dep, marginalized_rv):
        return None

    typed_op = LinearGaussianMarginalRV(
        inputs=inputs,
        outputs=outputs,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
    )
    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_ir_rewrites_db.register(
    "linear_gaussian_marginal", linear_gaussian_marginal_rewrite, "basic"
)
