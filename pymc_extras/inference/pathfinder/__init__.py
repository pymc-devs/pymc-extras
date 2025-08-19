import importlib.util

from pymc_extras.inference.pathfinder.pathfinder import fit_pathfinder

# Optional Numba backend support
if importlib.util.find_spec("numba") is not None:
    try:
        from . import numba_dispatch  # noqa: F401 - needed for registering Numba dispatch functions

        NUMBA_AVAILABLE = True
    except ImportError:
        NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = False

__all__ = ["fit_pathfinder"]
