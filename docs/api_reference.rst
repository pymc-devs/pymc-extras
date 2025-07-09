API Reference
***************

Model
=====

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_extras
.. autosummary::
   :toctree: generated/

   as_model
   marginalize
   recover_marginals
   model_builder.ModelBuilder

Inference
=========

.. currentmodule:: pymc_extras.inference
.. autosummary::
   :toctree: generated/

   find_MAP
   fit
   fit_laplace
   fit_pathfinder


Distributions
=============

.. currentmodule:: pymc_extras.distributions
.. autosummary::
   :toctree: generated/

   Chi
   Maxwell
   DiscreteMarkovChain
   GeneralizedPoisson
   BetaNegativeBinomial
   GenExtreme
   R2D2M2CP
   Skellam
   histogram_approximation

Prior
=====

.. currentmodule:: pymc_extras.prior
.. autosummary::
   :toctree: generated/

   create_dim_handler
   handle_dims
   Prior
   VariableFactory
   sample_prior
   Censored
   Scaled

Deserialize
===========

.. currentmodule:: pymc_extras.deserialize
.. autosummary::
   :toctree: generated/

   deserialize
   register_deserialization
   Deserializer


Transforms
==========

.. currentmodule:: pymc_extras.distributions.transforms
.. autosummary::
   :toctree: generated/

   PartialOrder


Utils
=====

.. currentmodule:: pymc_extras.utils
.. autosummary::
   :toctree: generated/

   spline.bspline_interpolation
   prior.prior_from_idata
   model_equivalence.equivalent_models


Statespace Models
=================
.. automodule:: pymc_extras.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models


Model Transforms
================
.. automodule:: pymc_extras.model.transforms
.. autosummary::
   :toctree: generated/

   autoreparam.vip_reparametrize
   autoreparam.VIP


Printing
========
.. currentmodule:: pymc_extras.printing
.. autosummary::
   :toctree: generated/

   model_table
