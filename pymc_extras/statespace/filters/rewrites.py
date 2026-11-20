import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.random.basic import MvNormalRV
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorConstant


@register_canonicalize
@node_rewriter([MvNormalRV])
def local_mvnormal_zero_cov(fgraph, node):
    """Replace a multivariate normal draw with a constant zero covariance by its mean, leaving the
    rng unadvanced."""
    rng, size, mean, cov = node.inputs
    if not (isinstance(cov, TensorConstant) and cov.unique_value == 0):
        return None

    draw = pt.broadcast_arrays(mean, cov[..., 0])[0]
    if not isinstance(size.type, NoneTypeT):
        draw = pt.broadcast_to(draw, pt.concatenate([size, draw.shape[-1:]]))

    return [rng, draw.astype(node.outputs[1].dtype)]
