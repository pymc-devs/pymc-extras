Composable model terms
=====================

Declarative, composable terms for building ``pymc.dims`` subgraphs from
xarray datasets. Terms compose with ``+``, ``*``, and ``-`` operators
and handle coordinates, data registration, tensor construction, and
prediction data updating.

.. currentmodule:: pymc_extras.terms
.. autosummary::
   :toctree: ../generated/

   ModelTerm
   Sum
   Product
   Intercept
   Dot
   Transform
   build_param
   get_coords
   register_data
   set_data
   collect_terms
   collect_coords
