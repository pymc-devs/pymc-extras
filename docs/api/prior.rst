Prior specification
===================

A declarative way to define (hierarchical) prior distributions that can be
serialized to and from JSON. Useful when priors are part of a configuration
file rather than hardcoded in a model, as in
`pymc-marketing <https://www.pymc-marketing.io>`_.

.. currentmodule:: pymc_extras.prior
.. autosummary::
   :toctree: ../generated/

   Prior
   Censored
   Scaled
   sample_prior
   create_dim_handler
   handle_dims
   register_pymc_distribution
   register_tensor_transform
   VariableFactory

From a previous model
---------------------

Build a prior from the posterior of a previously fitted model, enabling
simple Bayesian updating workflows.

.. currentmodule:: pymc_extras.utils
.. autosummary::
   :toctree: ../generated/

   prior.prior_from_idata

Deserialization
---------------

Registry that maps JSON data back to Python objects, used to round-trip
``Prior`` definitions and extensible to arbitrary custom types.

.. currentmodule:: pymc_extras.deserialize
.. autosummary::
   :toctree: ../generated/

   deserialize
   register_deserialization
   Deserializer
