import pytensor.scalar as ps
import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import _matmul
from pytensor.tensor.random.basic import MvNormalRV
from pytensor.tensor.rewriting.basic import register_canonicalize, register_specialize
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorConstant, TensorVariable


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


def split_batched_scales(
    operand: TensorVariable, contracted_axis: int
) -> tuple[list[TensorVariable], list[TensorVariable]] | None:
    """
    Split the factors of an elementwise product into those kept in a matmul and those lifted out.

    A factor can be lifted when it is broadcastable along ``contracted_axis``, so it scales whole
    rows (or columns) of the product. It is lifted only when it carries a batch dim that every
    non-scale factor broadcasts over, because lifting it then removes that batch dim from the
    matmul operand.

    Parameters
    ----------
    operand : TensorVariable
        A matmul operand.
    contracted_axis : int
        The axis of ``operand`` that the matmul sums over, -1 for the left operand and -2 for the
        right.

    Returns
    -------
    split : tuple of (list of TensorVariable, list of TensorVariable) or None
        The kept factors and the lifted factors, or None when nothing can be lifted.
    """
    if not (
        operand.owner is not None
        and isinstance(operand.owner.op, Elemwise)
        and isinstance(operand.owner.op.scalar_op, ps.Mul)
    ):
        return None

    factors = operand.owner.inputs
    non_scales = [factor for factor in factors if not factor.type.broadcastable[contracted_axis]]
    if not non_scales:
        return None

    batch_ndim = operand.type.ndim - 2
    shared_batch_axes = [
        axis
        for axis in range(batch_ndim)
        if all(factor.type.broadcastable[axis] for factor in non_scales)
    ]

    def adds_batch_dim(factor):
        return factor.type.broadcastable[contracted_axis] and any(
            not factor.type.broadcastable[axis] for axis in shared_batch_axes
        )

    lifted = [factor for factor in factors if adds_batch_dim(factor)]
    if not lifted:
        return None

    kept = [factor for factor in factors if not adds_batch_dim(factor)]

    # The matmul must still run in the operand's dtype, or a narrower kept product would
    # overflow or round where the original did not.
    if ps.upcast(*(factor.type.dtype for factor in kept)) != operand.type.dtype:
        return None

    return kept, lifted


def product(factors: list[TensorVariable]) -> TensorVariable:
    """Multiply ``factors`` elementwise, without a unary ``Mul`` for a single factor."""
    return factors[0] if len(factors) == 1 else pt.mul(*factors)


@register_canonicalize
@register_specialize
@node_rewriter([_matmul])
def local_lift_batched_scale_out_of_matmul(fgraph, node):
    """
    Move row and column scalings that carry extra batch dims outside a matmul.

    ``(s * a) @ b`` becomes ``s * (a @ b)`` when ``s`` is broadcastable along the contracted
    axis and batched where ``a`` is not, and symmetrically for column scalings of ``b``. The
    matmul then no longer carries the batch dim of ``s``, which lets
    ``local_batched_matmul_to_core_matmul_with_reshape`` fold it into a single GEMM.
    """
    x, y = node.inputs
    [out] = node.outputs

    x_split = split_batched_scales(x, contracted_axis=-1)
    y_split = split_batched_scales(y, contracted_axis=-2)
    if x_split is None and y_split is None:
        return None

    new_x, x_lifted = (x, []) if x_split is None else (product(x_split[0]), x_split[1])
    new_y, y_lifted = (y, []) if y_split is None else (product(y_split[0]), y_split[1])
    new_out = pt.mul(_matmul(new_x, new_y), *x_lifted, *y_lifted)

    if not out.type.is_super(new_out.type):
        return None

    copy_stack_trace(out, new_out)
    return [new_out]
