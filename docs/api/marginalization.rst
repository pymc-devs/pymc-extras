Marginalization
===============

Model transformations that integrate variables out of a model, and recover
them afterwards. Marginalizing discrete variables allows sampling with
gradient-based samplers like NUTS; marginalizing conjugate pairs or using the
Laplace approximation reduces the dimensionality of the posterior.

``marginalize``, ``conditional``, and ``recover`` support exact transformations
only. They reject models containing approximate marginalizations.
``unmarginalize`` restores the original generative variables.

Laplace approximations require explicit opt-in through
``approximate_marginalize``, ``approximate_conditional``, and
``approximate_recover``. The marginal likelihood and recovered posteriors are
approximate; Gaussian conditionals are a special case where recovery is exact.
For nonlinear likelihoods, recovery uses a local Gaussian approximation at the
posterior mode. These transformations alone do not implement full INLA.

.. currentmodule:: pymc_extras.marginal
.. autosummary::
   :toctree: ../generated/

   marginalize
   unmarginalize
   conditional
   recover
   approximate_marginalize
   approximate_conditional
   approximate_recover

The set of supported marginalizations is extensible; see
:doc:`../developer/extending_marginalization`.
