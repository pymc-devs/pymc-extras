import warnings

import pymc as pm


def find_MAP(*args, **kwargs):
    """Deprecated alias of :func:`pymc.find_MAP` returning a DataTree."""
    warnings.warn(
        "`pymc_extras.find_MAP` is deprecated, use `pymc.find_MAP` instead.",
        FutureWarning,
        stacklevel=2,
    )
    kwargs.setdefault("return_inferencedata", True)
    kwargs.setdefault("jitter", False)
    return pm.find_MAP(*args, **kwargs)
