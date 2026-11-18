Distributions
=============

Distributions that are not (or not yet) part of PyMC itself. They behave
like regular PyMC distributions and can be used directly inside a model.

.. currentmodule:: pymc_extras.distributions
.. autosummary::
   :toctree: ../generated/

   Chi
   Maxwell
   DiscreteMarkovChain
   JointCategorical
   GeneralizedPoisson
   BetaNegativeBinomial
   GenExtreme
   GenPareto
   ExtGenPareto
   R2D2M2CP
   Skellam
   SparseDirichletMultinomial
   SparseNegativeBinomial
   SparseBetaBinomial
   histogram_approximation

Sparse count log probabilities
------------------------------

``SparseDirichletMultinomial``, ``SparseNegativeBinomial`` and
``SparseBetaBinomial`` use ordinary dense count tensors. They exploit zeros
in the observed values when evaluating log probabilities, while inheriting
PyMC's parameter conventions, support, sampling and support points. They are
ordinary observed random variables and need no ``Potential``.

For Dirichlet-multinomial counts, the usual log probability is

.. math::

   \log\Gamma(n+1) + \log\Gamma(A) - \log\Gamma(n+A)
   + \sum_i \left[\log\Gamma(x_i+a_i)-\log\Gamma(a_i)-\log\Gamma(x_i+1)\right],
   \qquad A=\sum_i a_i.

The specialized version restricts the sum to :math:`i:x_i>0`. Each omitted
term is zero. Every concentration still contributes to :math:`A`, its
derivative and parameter validation. Concentrations remain strictly positive.

For negative binomial counts, the zero-count contribution :math:`n\log p`
is retained for every observation. The positive-count correction, including
the gamma difference, uses sparse indexing. Both ``mu, alpha`` and
``n, p`` parameterizations are inherited from PyMC, including the Poisson-limit
fallback for large ``n``.

Beta-binomial is equivalent to a two-category DM over successes and failures.
Its alpha gamma difference is skipped when successes are zero, and its beta
gamma difference is skipped when failures are zero. The normalization terms
and parameter-dependent zero-count probability remain present.

Nonzero indices can be folded at compilation for constant observations.
``pm.Data`` observations can change their zero pattern without rebuilding the
model; their indices are recomputed during evaluation. These classes do not
accept SciPy sparse matrices or remove categories from the sampling law.
Dense parameter transformations, normalizations and gradients remain necessary.
The savings are most useful when shape parameters are inferred. Sparse indexing
can cost more than it saves for small or dense inputs.

To compare logp and gradient evaluation separately from compilation, run
``python scripts/benchmark_sparse_distributions.py`` from the repository root.
The default zero-percentage grid is ``0, 5, 25, 50, 75, 95, 100``. All
distributions use the same count array at each grid point; increasing sparsity
removes entries without changing the remaining positive counts. Timings include
gradients through learned probabilities, means and dispersion/concentration
parameters. Multinomial, binomial and Poisson are benchmark-only controls.

Counts are fixed observations, not draws during timing. Multinomial and DM
trial totals are row sums of the counts; binomial and beta-binomial use 20
trials. At 100% zeros the multinomial/DM trial totals are zero and the likelihoods
carry no information, so no speedup ratio is reported. These are kernel timings,
not end-to-end sampler speedups.

Transforms
----------

Value transforms for constrained sampling.

.. currentmodule:: pymc_extras.distributions.transforms
.. autosummary::
   :toctree: ../generated/

   PartialOrder
