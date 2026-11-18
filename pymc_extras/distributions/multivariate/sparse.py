#   Copyright 2026 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Count distributions that skip zero-observation terms in their log probabilities."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.multivariate import DirichletMultinomialRV


def _sparse_sum_last_axis(value, parameter, series):
    """Sum a category expression which is identically zero at zero observations."""
    value = pt.broadcast_arrays(value, parameter)[0]
    parameter = pt.shape_padleft(parameter, value.ndim - parameter.ndim)
    indices = pt.nonzero(value > 0)
    # Index the original parameter to avoid a dense broadcast allocation.
    parameter_indices = tuple(
        pt.where(pt.eq(length, 1), 0, index)
        for length, index in zip(parameter.shape, indices, strict=True)
    )
    terms = series(value[indices], parameter[parameter_indices])
    if value.ndim == 1:
        return terms.sum()
    result = pt.zeros(value.shape[:-1], dtype=terms.dtype)
    return pt.inc_subtensor(result[indices[:-1]], terms)


class SparseDirichletMultinomialRV(DirichletMultinomialRV):
    name = "sparse_dirichlet_multinomial"
    _print_name = ("SparseDirichletMultinomial", "\\operatorname{SparseDirichletMultinomial}")


class SparseDirichletMultinomial(pm.DirichletMultinomial):
    r"""Dirichlet-multinomial with a logp optimized for mostly zero counts.

    This is the same distribution and parameterization as
    :class:`pymc.DirichletMultinomial`. It inherits the native symbolic sampler
    and support point; only log-probability evaluation changes.

    For each category with observed count zero, the term
    :math:`\log\Gamma(x_i+a_i)-\log\Gamma(a_i)-\log\Gamma(x_i+1)`
    vanishes. The sparse logp evaluates this term only at positive counts.
    Every concentration still contributes to :math:`\sum_i a_i` and every
    category is included in the support and parameter checks.

    Parameters
    ----------
    n : tensor_like of int
        Total count in each replicate, with ``n >= 0``.
    a : tensor_like of float
        Strictly positive Dirichlet concentrations. The final axis indexes
        categories; leading batch dimensions broadcast as in PyMC.

    Notes
    -----
    Inputs and samples are ordinary dense tensors, not sparse matrix objects.
    The optimization exploits zeros in the observations, not zero
    concentrations. For fixed observed data the nonzero indices can be folded
    during compilation. Mutable observations are also supported, with indices
    recomputed when necessary. Indexing overhead can outweigh the savings for
    small or dense observations.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        import pymc_extras as pmx

        with pm.Model():
            logits = pm.Normal("logits", shape=5)
            p = pm.math.softmax(logits)
            concentration = pm.LogNormal("concentration")
            counts = pmx.SparseDirichletMultinomial(
                "counts", n=20, a=concentration * p, observed=[18, 0, 0, 2, 0]
            )
    """

    rv_type = SparseDirichletMultinomialRV
    rv_op = SparseDirichletMultinomialRV.rv_op

    def logp(value, n, a):
        sum_a = a.sum(axis=-1)
        const = pt.gammaln(n + 1) + pt.gammaln(sum_a) - pt.gammaln(n + sum_a)
        series = _sparse_sum_last_axis(
            value,
            a,
            lambda value, a: pt.gammaln(value + a) - pt.gammaln(value + 1) - pt.gammaln(a),
        )
        res = pt.switch(
            (value < 0).any(axis=-1) | pt.neq(value.sum(axis=-1), n), -np.inf, const + series
        )
        return check_parameters(res, a > 0, n >= 0, msg="a > 0, n >= 0")
