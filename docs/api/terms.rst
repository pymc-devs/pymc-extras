Composable model terms
=====================

``pymc_extras.terms`` provides a small, expressive core of building blocks
for ``pymc.dims`` subgraphs from xarray data.  The built-in terms
(``Parameter`` / ``Intercept``, ``Dot``, ``Transform``) cover common
linear-predictor pieces, but **this module is not a full modeling toolkit** ---
it does not ship complete hierarchical, domain-specific, or
observation-model wrappers.  It is designed
to be **extended**: subclass ``ModelTerm`` for domain-specific terms
(gather/index, media transforms, group effects); they compose with
built-ins via ``+``, ``*``, and ``-`` and use the same helpers.  See the
``pymc_extras.terms`` module docstring **Extending** section for a full
custom-term recipe.

.. currentmodule:: pymc_extras.terms
.. autosummary::
   :toctree: ../generated/

   ModelTerm
   Parameter
   Intercept
   Sum
   Product
   Dot
   Transform
   build_param
   get_coords
   register_data
   set_data
   collect_terms
   collect_coords
