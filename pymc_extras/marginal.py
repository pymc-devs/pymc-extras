"""Public namespace for marginalization utilities.

The implementation lives in :mod:`pymc_extras.model.marginal`; this module
re-exports the public API under the shorter ``pymc_extras.marginal`` path.
"""

from pymc_extras.model.marginal.conditional import (
    approximate_conditional,
    approximate_recover,
    conditional,
    recover,
)
from pymc_extras.model.marginal.marginalize import (
    approximate_marginalize,
    marginalize,
    unmarginalize,
)

__all__ = [
    "approximate_conditional",
    "approximate_marginalize",
    "approximate_recover",
    "conditional",
    "marginalize",
    "recover",
    "unmarginalize",
]
