Marginalization
===============

Model transformations that integrate variables out of a model, and recover
them afterwards. Marginalizing discrete variables allows sampling with
gradient-based samplers like NUTS; marginalizing conjugate pairs or using the
Laplace approximation reduces the dimensionality of the posterior.

``marginalize`` integrates variables out; ``conditional`` reconstructs their
conditional distributions, and ``recover`` draws from those distributions.
``unmarginalize`` instead restores the original generative variables.

The Laplace support used by ``fit_INLA`` is also available through
``marginalize(..., laplace_approx={rv: Q})``. For these variables, the marginal
likelihood and the posteriors reconstructed by ``conditional`` and ``recover``
are approximate. Recovery uses a Gaussian centered at the conditional posterior
mode with curvature evaluated at that mode; it is exact when the conditional
posterior is Gaussian, but generally approximate for nonlinear likelihoods.
These transformations alone do not implement full INLA.

``fit_INLA(..., return_latent_posteriors=True)`` uses this recovery to add latent
field draws to the returned DataTree's ``posterior`` group, alongside the sampled
hyperparameters.

.. currentmodule:: pymc_extras.marginal
.. autosummary::
   :toctree: ../generated/

   marginalize
   unmarginalize
   conditional
   recover

The set of supported marginalizations is extensible; see
:doc:`../developer/extending_marginalization`.
