from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytensor
import pytensor.tensor as pt

TrendSpec = Literal["n", "c", "ct", "ctt"] | Sequence[int] | None

TREND_STRINGS = {"n": (), "c": (0,), "ct": (0, 1), "ctt": (0, 1, 2)}


def parse_trend(trend: TrendSpec) -> tuple[int, ...]:
    """
    Resolve a trend specification to the polynomial powers it includes.

    Strings follow statsmodels: ``"n"`` for none, ``"c"`` for a constant, ``"ct"`` for a
    constant and linear term, ``"ctt"`` for a quadratic on top. A sequence of ints is read as
    presence flags per power, so ``[1, 1, 0, 1]`` includes :math:`1, t, t^3`. ``None`` is ``"n"``.
    """
    if trend is None:
        return ()

    if isinstance(trend, str):
        if trend not in TREND_STRINGS:
            raise ValueError(
                f"trend must be one of {sorted(TREND_STRINGS)} or a sequence of presence flags, "
                f"got {trend!r}"
            )
        return TREND_STRINGS[trend]

    flags = np.asarray(trend)
    if flags.ndim != 1 or not np.isin(flags, [0, 1]).all():
        raise ValueError(f"A trend sequence must hold only 0 and 1 flags, got {trend!r}")

    return tuple(int(power) for power in np.flatnonzero(flags))


POWER_NAMES = ("constant", "linear", "quadratic", "cubic")


def trend_names(powers: Sequence[int]) -> tuple[str, ...]:
    """Coordinate labels for the polynomial powers, numbered past the cubic term."""
    return tuple(
        POWER_NAMES[power] if power < len(POWER_NAMES) else f"degree_{power}" for power in powers
    )


def trend_design(powers: Sequence[int], n_timesteps, offset: int):
    """
    Design matrix of the polynomial trend, of shape ``(n_timesteps, len(powers))``.

    Row ``i`` holds ``(offset + i) ** power`` for each power. ``n_timesteps`` may be symbolic.
    """
    time = pt.arange(n_timesteps).astype(pytensor.config.floatX) + offset

    return pt.stack([time**power for power in powers], axis=1)


def constant_as_regressor(mean, exog, exog_coefficients, n_timesteps):
    """
    Append a column of ones with ``mean`` as its coefficient to a set of regressors.

    A constant intercept in an autoregression is the same model as a level shift of the
    observations, which a closed form over the observations takes as one more regressor.

    Parameters
    ----------
    mean : TensorVariable
        Level shift per observed series, of shape ``(k_endog,)`` or a scalar.
    exog : TensorVariable or None
        Regressors of shape ``(n_timesteps, k_exog)``, or None when there are none.
    exog_coefficients : TensorVariable or None
        Coefficients of shape ``(k_endog, k_exog)``, or None when there are none.
    n_timesteps : TensorVariable or int
        Number of rows in the column of ones.

    Returns
    -------
    exog : TensorVariable
        Regressors with the column of ones appended.
    exog_coefficients : TensorVariable
        Coefficients with ``mean`` appended as the last column.
    """
    ones = pt.ones((n_timesteps, 1), dtype=pytensor.config.floatX)
    mean_column = pt.expand_dims(pt.atleast_1d(mean), -1)

    if exog is None:
        return ones, mean_column

    return (
        pt.concatenate([exog, ones], axis=1),
        pt.concatenate([exog_coefficients, mean_column], axis=1),
    )
