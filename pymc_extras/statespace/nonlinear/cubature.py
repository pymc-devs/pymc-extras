import pytensor.tensor as pt

from pytensor.tensor import TensorLike
from pytensor.tensor.variable import TensorVariable


def cubature_points(*, mean: TensorLike, factor: TensorLike) -> TensorVariable:
    r"""
    Return the points of the third-degree spherical-radial cubature rule for a Gaussian.

    With :math:`n` columns in the factor :math:`L`, the rule has :math:`2n` points

    .. math::

        \mu \pm \sqrt{n} L e_i, \quad i = 1, \ldots, n,

    each with weight :math:`1 / (2n)`, and integrates polynomials up to degree three exactly
    (Arasaratnam and Haykin 2009).

    Parameters
    ----------
    mean : TensorLike
        Mean of the Gaussian, a vector of length :math:`k`.
    factor : TensorLike
        A :math:`k` by :math:`n` matrix whose product with its transpose is the covariance of the
        Gaussian.

    Returns
    -------
    points : TensorVariable
        The :math:`2n` points as rows, the :math:`n` points on the positive side first.
    """
    mean, factor = pt.as_tensor_variable(mean), pt.as_tensor_variable(factor)
    offsets = pt.sqrt(pt.cast(factor.shape[1], factor.dtype)) * factor.T
    return mean + pt.concatenate([offsets, -offsets], axis=0)
