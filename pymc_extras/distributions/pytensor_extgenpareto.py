#   Copyright 2022 The PyMC Developers
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

"""Extended Generalized Pareto distribution (Naveau et al. 2016 Type 1)."""

import numpy as np
import pytensor.tensor as pt

from pymc_extras.distributions.pytensor_distributions_helper import ppf_bounds_cont
from pymc_extras.distributions.pytensor_genpareto import (
    _gpd_log_H,
    _gpd_log_h,
    _gpd_log_S,
    _gpd_quantile_from_excess,
    _gpd_upper_bound,
    _in_gpd_support,
    _log1p_div,
)


def logpdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    # log g = log kappa + (kappa - 1) log H + log h. The carrier term vanishes
    # at kappa = 1; guarding it keeps the GPD reduction exact at the lower
    # endpoint (z = 0, log H = -inf), where (kappa - 1) * log H would be 0 * -inf.
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * _gpd_log_H(z, xi))
    logp = pt.log(kappa) + carrier + _gpd_log_h(z, sigma, xi)
    logp = pt.switch(_in_gpd_support(z, xi), logp, -np.inf)
    logp = pt.switch(pt.eq(z, np.inf), -np.inf, logp)
    return logp


def logcdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logcdf = pt.switch(above_upper, 0.0, kappa * _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    logcdf = pt.switch(pt.eq(z, np.inf), 0.0, logcdf)
    return logcdf


def logsf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    a = _gpd_log_S(z, xi)  # log(1 - H), accurate in the tail
    log_H = pt.log1mexp(a)
    generic = pt.log1mexp(kappa * log_H)
    s = pt.exp(a)  # S_gpd
    # r = kappa * S_gpd, formed via exp(log kappa + a) and capped at 1 so the unused
    # (generic-branch) tail expression cannot overflow.
    r = pt.exp(pt.minimum(pt.log(kappa) + a, 0.0))
    # 1 - H**kappa = kappa S [1 + (s - r)/2 + (r**2 - 3 r s + 2 s**2)/6 + ...].
    # This is a Taylor expansion in *both* S_gpd and kappa*S_gpd, so it is only
    # valid where both are small; writing it in r = kappa S and s = S keeps every
    # term bounded (no kappa**k powers, which overflow for huge kappa).
    series_m1 = (s - r) / 2.0 + (r * r - 3.0 * r * s + 2.0 * s * s) / 6.0
    tail = pt.log(kappa) + a + pt.log1p(series_m1)
    # The Taylor tail runs where S_gpd and kappa*S_gpd are both below machine epsilon
    # (a < log_eps and log(kappa) + a < log_eps); the generic log1mexp(kappa log H) runs
    # otherwise. Both conditions gate the switch: for tiny kappa the second holds in the body
    # (S_gpd ~ 1) too, where the Taylor does not apply.
    log_eps = float(np.log(np.finfo(a.dtype).eps))
    logsf = pt.switch(pt.and_(a < log_eps, pt.log(kappa) + a < log_eps), tail, generic)
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logsf = pt.switch(pt.or_(above_upper, pt.eq(z, np.inf)), -np.inf, logsf)
    logsf = pt.switch(z < 0, 0.0, logsf)
    return logsf


def _ext_gpd_excess_from_log_prob(log_q, kappa):
    """GPD excess ``m = -log(1 - F ** (1/kappa))`` from ``log_q = log F``.

    For the carrier ``F = H ** kappa``, the GPD CDF is ``H = exp(log_q / kappa)``
    and its survival ``1 - H``, so ``m = -log(1 - H) = -log1mexp(log_q / kappa)``
    (``pt.log1mexp(a) = log(1 - exp(a))`` for ``a <= 0``). This form stays accurate when ``H``
    rounds to ``1``: for small ``kappa`` the survival ``1 - H`` is tiny, and forming it directly
    would collapse the excess to ``0`` (and the quantile to ``mu``). Shared
    by the quantile, the sampler, ``support_point`` and the default transform so
    all four inverses agree. ``log_q`` must be ``<= 0`` (a log-probability).
    """
    return -pt.log1mexp(log_q / kappa)


def _ext_gpd_excess_from_logit(value, kappa):
    """GPD excess ``m = -log S(x)`` from ``value = logit(F(x))``.

    The ExtGPD CDF is ``F = G ** kappa = sigmoid(value)`` (``G`` the GPD CDF), so the excess
    depends only on ``value`` and ``kappa``:
        m = -log(1 - G) = -log(1 - sigmoid(value) ** (1/kappa)).
    Evaluated stably in two branches split at a large cutoff. Writing
    a := -log G = -log(sigmoid(value)) / kappa gives m = -log1mexp(-a):
      bulk (value < cutoff): m = -log1mexp(log_F / kappa), log_F = log sigmoid(value).
      tail (value >= cutoff, the extreme upper tail where F -> 1): a held in logs as log_a,
        m = -log_a where a < eps, else -log1mexp(-a).
    The unused branch's inputs are clamped (log_F via cutoff, a via log_max) so it stays finite.
    """
    finfo = np.finfo(value.dtype)
    log_max = float(np.log(finfo.max))  # a = exp(min(log_a, log_max)) stays finite
    log_eps = float(np.log(finfo.eps))  # at or below log_eps, m = -log_a
    # cutoff is the large value where log_F / kappa reaches finfo.tiny. Near the tail
    # log_F = -exp(-value), giving value = -log(tiny) - log(kappa); for kappa <= 1 it is -log(tiny).
    cutoff = np.asarray(-np.log(finfo.tiny), dtype=value.dtype) - pt.maximum(
        np.asarray(0.0, value.dtype), pt.log(kappa)
    )
    t = pt.softplus(value)
    log_F = -pt.softplus(-pt.minimum(value, cutoff))
    m_bulk = _ext_gpd_excess_from_log_prob(log_F, kappa)
    s = pt.exp(-pt.maximum(t, cutoff))  # S_ext, clamped so _log1p_div stays finite
    log_a = -t + pt.log(_log1p_div(-s)) - pt.log(kappa)
    a = pt.exp(pt.minimum(log_a, log_max))
    m_tail = pt.switch(log_a < log_eps, -log_a, -pt.log1mexp(-a))
    return pt.switch(value < cutoff, m_bulk, m_tail)


def ppf(q, mu, sigma, xi, kappa):
    q = pt.as_tensor_variable(q)
    # F = H ** kappa = q  ->  H = q ** (1/kappa); excess m = -log(1 - H), built
    # with log1mexp so a tiny 1 - H (small kappa) is not rounded away to 0.
    excess = _ext_gpd_excess_from_log_prob(pt.log(q), kappa)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi, kappa):
    return pt.exp(logcdf(x, mu, sigma, xi, kappa))


def pdf(x, mu, sigma, xi, kappa):
    return pt.exp(logpdf(x, mu, sigma, xi, kappa))


def sf(x, mu, sigma, xi, kappa):
    return pt.exp(logsf(x, mu, sigma, xi, kappa))


def isf(x, mu, sigma, xi, kappa):
    x = pt.as_tensor_variable(x)
    # log F = log1p(-x), accurate for tiny x; ppf(1 - x) forms 1 - x first and loses it.
    excess = _ext_gpd_excess_from_log_prob(pt.log1p(-x), kappa)
    quantile = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def rvs(mu, sigma, xi, kappa, size=None, random_state=None):
    # Inverse-CDF on a carrier draw u = F; excess = -log(1 - u ** (1/kappa)).
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)
