Inference
=========

Fitting methods beyond ``pm.sample``: optimization-based point estimates
(``find_MAP``), Gaussian approximations (Laplace, INLA), fast variational
methods (Pathfinder, DADVI), and embarrassingly parallel MCMC
(``fit_consensus_mc``). ``fit`` is a single entry point that dispatches to these
by name.

.. currentmodule:: pymc_extras.inference
.. autosummary::
   :toctree: ../generated/

   fit
   find_MAP
   fit_laplace
   fit_pathfinder
   fit_dadvi
   fit_INLA
   fit_consensus_mc
   merge_consensus
   estimate_parametric
   merge_parametric
