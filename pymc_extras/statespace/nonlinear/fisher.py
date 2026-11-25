from collections.abc import Callable, Sequence

import pytensor.tensor as pt

from pytensor.tensor.random.basic import (
    BernoulliRV,
    BinomialRV,
    ExponentialRV,
    GammaRV,
    LogNormalRV,
    NormalRV,
    PoissonRV,
    StudentTRV,
)
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable


def _location_scale(location: TensorVariable, scale: TensorVariable) -> list[list[TensorVariable]]:
    return [[1 / scale**2, pt.zeros_like(scale)], [pt.zeros_like(scale), 2 / scale**2]]


def _student_t(
    nu: TensorVariable, location: TensorVariable, scale: TensorVariable
) -> list[list[TensorVariable]]:
    zero = pt.zeros_like(location)
    nu_scale = -2 / (scale * (nu + 1) * (nu + 3))
    nu_nu = 0.25 * (pt.tri_gamma(nu / 2) - pt.tri_gamma((nu + 1) / 2)) - (nu + 5) / (
        2 * nu * (nu + 1) * (nu + 3)
    )
    return [
        [nu_nu, zero, nu_scale],
        [zero, (nu + 1) / ((nu + 3) * scale**2), zero],
        [nu_scale, zero, 2 * nu / ((nu + 3) * scale**2)],
    ]


def _gamma(shape: TensorVariable, scale: TensorVariable) -> list[list[TensorVariable]]:
    return [[pt.tri_gamma(shape), 1 / scale], [1 / scale, shape / scale**2]]


def _binomial(trials: TensorVariable, probability: TensorVariable) -> list[list[TensorVariable]]:
    zero = pt.zeros_like(probability)
    return [[zero, zero], [zero, trials / (probability * (1 - probability))]]


# Each entry gives the expected negative Hessian of one observation's log-density in the
# distribution parameters, in the order of ``RandomVariable.dist_params``.
FISHER_INFORMATION: dict[type[RandomVariable], Callable[..., list[list[TensorVariable]]]] = {
    BernoulliRV: lambda probability: [[1 / (probability * (1 - probability))]],
    BinomialRV: _binomial,
    ExponentialRV: lambda scale: [[1 / scale**2]],
    GammaRV: _gamma,
    LogNormalRV: _location_scale,
    NormalRV: _location_scale,
    PoissonRV: lambda rate: [[1 / rate]],
    StudentTRV: _student_t,
}

# Distribution parameters that take integer values, so the log-density has no derivative in them.
DISCRETE_PARAMETERS: dict[type[RandomVariable], tuple[int, ...]] = {BinomialRV: (0,)}


def fisher_information(
    *,
    op: RandomVariable,
    distribution_parameters: Sequence[TensorVariable],
    positions: Sequence[int],
) -> TensorVariable | None:
    """
    Return each observation's Fisher information in some of its distribution parameters.

    Parameters
    ----------
    op : RandomVariable
        The distribution of the observations.
    distribution_parameters : sequence of TensorVariable
        All of the distribution's parameters, in the order of ``op.dist_params``, each broadcast
        to the observations' shape ``(n,)``.
    positions : sequence of int
        Positions in ``distribution_parameters`` of the parameters to give the information in.

    Returns
    -------
    information : TensorVariable or None
        Positive semidefinite information of shape ``(n, p, p)`` for ``p`` positions, None where
        ``op`` has no closed form here or a position is a parameter taking integer values.
    """
    if type(op) not in FISHER_INFORMATION:
        return None
    if set(positions) & set(DISCRETE_PARAMETERS.get(type(op), ())):
        return None
    full = FISHER_INFORMATION[type(op)](*distribution_parameters)
    return pt.stack(
        [pt.stack([full[row][column] for column in positions], axis=-1) for row in positions],
        axis=-2,
    )
