Model building
==============

Tools for defining models. ``as_model`` turns a function with PyMC
statements into a reusable model factory, and ``ModelBuilder`` is a base
class for packaging a model behind a scikit-learn-like ``fit``/``predict``
interface, with saving and loading included.

.. currentmodule:: pymc_extras
.. autosummary::
   :toctree: ../generated/

   as_model
   model_builder.ModelBuilder

Float precision
---------------

``model_to_float32`` recreates a model with every float64 variable cast to
float32, so that it can be sampled in single precision.

.. currentmodule:: pymc_extras.model.transforms
.. autosummary::
   :toctree: ../generated/

   precision.model_to_float32
   precision.model_to_float64
