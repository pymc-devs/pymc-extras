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

"""Scalar count distributions with log probabilities that exploit observed zeros."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.dist_math import binomln, check_parameters, factln, logpow
from pytensor.tensor.random.basic import BetaBinomialRV, NegBinomialRV


def _sparse_correction(value, *parameters, term):
    """Evaluate a zero-at-zero correction only at positive observed values."""
    value = pt.broadcast_arrays(value, *parameters)[0]
    scalar = value.ndim == 0
    if scalar:
        value = pt.shape_padleft(value)
    indices = pt.nonzero(value > 0)
    selected = []
    for parameter in parameters:
        parameter = pt.shape_padleft(parameter, value.ndim - parameter.ndim)
        parameter_indices = tuple(
            pt.where(pt.eq(length, 1), 0, index)
            for length, index in zip(parameter.shape, indices, strict=True)
        )
        selected.append(parameter[parameter_indices])
    terms = term(value[indices], *selected)
    result = pt.set_subtensor(pt.zeros_like(value, dtype=terms.dtype)[indices], terms)
    return result.squeeze(0) if scalar else result


class SparseNegativeBinomialRV(NegBinomialRV):
    name = "sparse_negative_binomial"
    _print_name = ("SparseNegativeBinomial", "\\operatorname{SparseNegativeBinomial}")


class SparseNegativeBinomial(pm.NegativeBinomial):
    r"""Negative binomial with a logp optimized for mostly zero observations.

    Parameters, support, random sampling and support points are inherited from
    :class:`pymc.NegativeBinomial`. Both ``mu, alpha`` and ``n, p``
    parameterizations are supported. Only the logp changes.

    The zero-count contribution :math:`n\log p` is retained at every observation.
    The correction :math:`\log\Gamma(x+n)-\log\Gamma(n)-\log\Gamma(x+1)
    +x\log(1-p)` is evaluated only for positive counts. This saves gamma and
    digamma evaluations when dispersion is inferred. The dense zero-count
    contribution and PyMC's Poisson-limit fallback for large ``n`` are retained.

    Inputs are ordinary dense tensors; sparse matrix objects are not supported.
    Fixed observations allow the nonzero indices to be folded at compilation.
    Mutable observations are supported. Dense or small inputs may be faster
    with the native distribution because sparse indexing has its own cost.

    Parameters
    ----------
    mu : tensor_like of float, optional
        Mean of the count distribution. Specify ``mu`` or ``p``.
    alpha : tensor_like of float, optional
        Strictly positive dispersion parameter. Specify ``alpha`` or ``n``.
    p : tensor_like of float, optional
        Success probability in the alternative parameterization.
    n : tensor_like of float, optional
        Strictly positive shape in the alternative parameterization.
    """

    rv_op = SparseNegativeBinomialRV()

    def logp(value, n, p):
        # Keep logs outside indexing so sigmoid stabilization remains visible.
        log_p, log_complement = pt.log(p), pt.log(1 - p)
        mu = n * pt.exp(log_complement - log_p)
        correction = _sparse_correction(
            value,
            n,
            log_complement,
            term=lambda x, n, log_complement: binomln(x + n - 1, x) + x * log_complement,
        )
        res = pt.switch(
            n > 1e10,
            logpow(mu, value) - mu - factln(value),
            n * log_p + correction,
        )
        res = pt.switch(value < 0, -np.inf, res)
        return check_parameters(res, 0 <= p, p <= 1, n > 0, msg="0 <= p <= 1, n > 0")


class SparseBetaBinomialRV(BetaBinomialRV):
    name = "sparse_beta_binomial"
    _print_name = ("SparseBetaBinomial", "\\operatorname{SparseBetaBinomial}")


class SparseBetaBinomial(pm.BetaBinomial):
    r"""Beta-binomial with a logp optimized for boundary observations.

    Parameters, support, random sampling and support points are inherited from
    :class:`pymc.BetaBinomial`. Only log-probability evaluation changes.

    The gamma difference :math:`\log\Gamma(x+\alpha)-\log\Gamma(\alpha)`
    is evaluated only when ``x > 0``. The corresponding beta difference is
    evaluated only when ``n - x > 0``. The parameter-dependent zero-count
    probability and every parameter check are retained. This saves work when
    many observations have zero successes or zero failures and the shape
    parameters are inferred.

    Inputs are ordinary dense tensors; sparse matrix objects are not supported.
    Nonzero indices can be folded for fixed observations and are recomputed
    for mutable observations. Sparse indexing may cost more than it saves for
    small or dense inputs.

    Parameters
    ----------
    n : tensor_like of int
        Nonnegative number of trials.
    alpha : tensor_like of float
        Strictly positive beta shape parameter.
    beta : tensor_like of float
        Strictly positive beta shape parameter.
    """

    rv_op = SparseBetaBinomialRV()

    def logp(value, n, alpha, beta):
        alpha_part = _sparse_correction(
            value, alpha, term=lambda x, a: pt.gammaln(x + a) - pt.gammaln(a)
        )
        beta_part = _sparse_correction(
            n - value, beta, term=lambda x, b: pt.gammaln(x + b) - pt.gammaln(b)
        )
        total = alpha + beta
        res = binomln(n, value) + pt.gammaln(total) - pt.gammaln(n + total) + alpha_part + beta_part
        res = pt.switch((value < 0) | (value > n), -np.inf, res)
        return check_parameters(res, n >= 0, alpha > 0, beta > 0, msg="n >= 0, alpha > 0, beta > 0")
