from functools import wraps
from inspect import signature

import pytensor.tensor as pt

from pymc import Data, Model


def as_model(*model_args, **model_kwargs):
    R"""
    Decorator to provide context to PyMC models declared in a function.
    This removes all need to think about context managers and lets you separate creating a generative model from using the model.
    Additionally, a coords argument is added to the function so coords can be changed during function invocation

    All parameters are wrapped with a `pm.Data` object if the underlying type of the data supports it.

    Adapted from `Rob Zinkov's blog post <https://www.zinkov.com/posts/2023-alternative-frontends-pymc/>`_ and inspired by the `sampled <https://github.com/colcarroll/sampled>`_ decorator for PyMC3.

    Examples
    --------
    .. code:: python

        import pymc as pm
        import pymc_extras as pmx

        # The following are equivalent

        # standard PyMC API with context manager
        with pm.Model(coords={"obs": ["a", "b"]}) as model:
            x = pm.Normal("x", 0., 1., dims="obs")
            pm.sample()

        # functional API using decorator
        @pmx.as_model(coords={"obs": ["a", "b"]})
        def basic_model():
            pm.Normal("x", 0., 1., dims="obs")

        m = basic_model()
        pm.sample(model=m)

        # alternative way to use functional API
        @pmx.as_model()
        def basic_model():
            pm.Normal("x", 0., 1., dims="obs")

        m = basic_model(coords={"obs": ["a", "b"]})
        pm.sample(model=m)

    """

    def decorator(f):
        @wraps(f)
        def make_model(*args, **kwargs):
            coords = model_kwargs.pop("coords", {}) | kwargs.pop("coords", {})
            sig = signature(f)
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()

            with Model(*model_args, coords=coords, **model_kwargs) as m:
                for name, v in ba.arguments.items():
                    # Only wrap pm.Data around values pytensor can process
                    try:
                        _ = pt.as_tensor_variable(v)
                        ba.arguments[name] = Data(name, v)
                    except (NotImplementedError, TypeError, ValueError):
                        pass
                f(*ba.args, **ba.kwargs)
            return m

        return make_model

    return decorator
