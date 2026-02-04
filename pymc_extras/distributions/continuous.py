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

# coding: utf-8
"""
Experimental probability distributions for stochastic nodes in PyMC.

The imports from pymc are not fully replicated here: add imports as necessary.
"""

import numpy as np
import pytensor.tensor as pt

from pymc import ChiSquared, CustomDist
from pymc.distributions import transforms
from pymc.distributions.dist_math import (
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
)
from pymc.distributions.distribution import Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable
from scipy import stats


class GenParetoRV(RandomVariable):
    name: str = "Generalized Pareto"
    signature = "(),(),()->()"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("Generalized Pareto", "\\operatorname{GPD}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState | np.random.Generator,
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: tuple[int, ...],
    ) -> np.ndarray:
        return stats.genpareto.rvs(c=xi, loc=mu, scale=sigma, random_state=rng, size=size)


gpd = GenParetoRV()


class GenExtremeRV(RandomVariable):
    name: str = "Generalized Extreme Value"
    signature = "(),(),()->()"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState | np.random.Generator,
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)


gev = GenExtremeRV()


class GenExtreme(Continuous):
    r"""
    Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001) [1]_, and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter
    scipy : bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [1] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gev

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Pytensor tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = pt.switch(
            pt.isclose(xi, 0),
            -pt.log(sigma) - scaled - pt.exp(-scaled),
            -pt.log(sigma)
            - ((xi + 1) / xi) * pt.log1p(xi * scaled)
            - pt.pow(1 + xi * scaled, -1 / xi),
        )

        logp = pt.switch(pt.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)

        return check_parameters(
            logp, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = pt.switch(
            pt.isclose(xi, 0), -pt.exp(-scaled), -pt.pow(1 + xi * scaled, -1 / xi)
        )

        logc = pt.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def support_point(rv, size, mu, sigma, xi):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = pt.switch(pt.isclose(xi, 0), mu, mu + sigma * (pt.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode


class GenPareto(Continuous):
    r"""
    Univariate Generalized Pareto log-likelihood.

    The Generalized Pareto Distribution (GPD) is used for modeling exceedances over
    a threshold in extreme value analysis. It is the natural distribution for
    peaks-over-threshold modeling, complementing the GEV distribution which is used
    for block maxima.

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = 1 - \left(1 + \xi \frac{x - \mu}{\sigma}\right)^{-\frac{1}{\xi}}

    for :math:`\xi \neq 0`, and

    .. math::

       G(x \mid \mu, \sigma, \xi) = 1 - \exp\left(-\frac{x - \mu}{\sigma}\right)

    for :math:`\xi = 0`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 10, 200)
        mus = [0., 0., 0.]
        sigmas = [1., 2., 1.]
        xis = [0.0, 0.5, -0.5]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genpareto.pdf(x, c=xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \geq \mu`, when :math:`\xi \geq 0`
              * :math:`\mu \leq x \leq \mu - \sigma/\xi`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma/(1 - \xi)`, when :math:`\xi < 1`
              * :math:`\infty`, when :math:`\xi \geq 1`
    Variance  * :math:`\sigma^2 / ((1-\xi)^2 (1-2\xi))`, when :math:`\xi < 0.5`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter (threshold).
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter. Controls tail behavior:

        * :math:`\xi > 0`: heavy tail (Pareto-like)
        * :math:`\xi = 0`: exponential tail
        * :math:`\xi < 0`: bounded upper tail

    References
    ----------
    .. [1] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    .. [2] Pickands, J. (1975).
        Statistical Inference Using Extreme Order Statistics.
        Annals of Statistics, 3(1), 119-131.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import GenPareto

        # Peaks-over-threshold model
        threshold = 10.0
        exceedances = data[data > threshold] - threshold

        with pm.Model():
            sigma = pm.HalfNormal("sigma", sigma=1)
            xi = pm.Normal("xi", mu=0, sigma=0.5)
            obs = GenPareto("obs", mu=0, sigma=sigma, xi=xi, observed=exceedances)
    """

    rv_op = gpd

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Pareto distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Pytensor tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = pt.switch(
            pt.isclose(xi, 0),
            -pt.log(sigma) - scaled,
            -pt.log(sigma) - (1 + 1 / xi) * pt.log1p(xi * scaled),
        )

        # Check support: scaled >= 0, and for xi < 0, also scaled <= -1/xi
        in_support = pt.switch(
            pt.ge(xi, 0),
            pt.ge(scaled, 0),
            pt.and_(pt.ge(scaled, 0), pt.le(scaled, -1 / xi)),
        )

        logp = pt.switch(
            pt.and_(in_support, pt.gt(1 + xi * scaled, 0)),
            logp_expression,
            -np.inf,
        )

        return check_parameters(logp, sigma > 0, msg="sigma > 0")

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Pareto
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logcdf_expression = pt.switch(
            pt.isclose(xi, 0),
            pt.log1p(-pt.exp(-scaled)),
            pt.log1p(-pt.pow(1 + xi * scaled, -1 / xi)),
        )

        # Handle bounds
        logcdf = pt.switch(
            pt.lt(scaled, 0),
            -np.inf,
            pt.switch(
                pt.and_(pt.lt(xi, 0), pt.gt(scaled, -1 / xi)),
                0.0,  # log(1) = 0 at upper bound for xi < 0
                logcdf_expression,
            ),
        )

        return check_parameters(logcdf, sigma > 0, msg="sigma > 0")

    def support_point(rv, size, mu, sigma, xi):
        r"""
        Using the median as the support point, since mean can be infinite when :math:`\xi \geq 1`.

        Median = mu + sigma * (2^xi - 1) / xi  for xi != 0
        Median = mu + sigma * log(2)           for xi = 0
        """
        median = pt.switch(
            pt.isclose(xi, 0),
            mu + sigma * pt.log(2),
            mu + sigma * (pt.pow(2, xi) - 1) / xi,
        )
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


class ExtGenParetoRV(RandomVariable):
    name: str = "Extended Generalized Pareto"
    signature = "(),(),(),()->()"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("Extended Generalized Pareto", "\\operatorname{ExtGPD}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, kappa=1.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, kappa, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState | np.random.Generator,
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        kappa: np.ndarray,
        size: tuple[int, ...],
    ) -> np.ndarray:
        # Use inverse transform sampling: X = H^{-1}(U^{1/kappa})
        # where H^{-1} is the GPD quantile function
        u = rng.uniform(size=size)
        # Transform uniform: v = u^{1/kappa} gives CDF v^kappa
        v = np.power(u, 1.0 / kappa)
        # Apply GPD quantile function
        return stats.genpareto.ppf(v, c=xi, loc=mu, scale=sigma)


ext_gpd = ExtGenParetoRV()


class ExtGenPareto(Continuous):
    r"""
    Univariate Extended Generalized Pareto log-likelihood.

    The Extended Generalized Pareto Distribution (ExtGPD) is a transformation of the
    GPD that provides a smooth connection between the upper tail and the main body
    of the function. This approach avoids the need for threshold specification and
    helps in sampling the entire time series for modeling extremes.

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi, \kappa) = \left[H\left(\frac{x - \mu}{\sigma}\right)\right]^\kappa

    where :math:`H` is the GPD cdf:

    .. math::

       H(z) = 1 - (1 + \xi z)^{-1/\xi}

    for :math:`\xi \neq 0`, and :math:`H(z) = 1 - e^{-z}` for :math:`\xi = 0`.

    The parameter :math:`\kappa > 0` controls the lower tail behavior, while
    :math:`\xi` controls the upper tail behavior. When :math:`\kappa = 1`,
    this reduces to the standard GPD.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0.01, 5, 200)
        sigma, xi = 1.0, 0.2
        for kappa in [0.5, 1.0, 2.0, 5.0]:
            # ExtGPD pdf: g(x) = kappa * H(x)^(kappa-1) * h(x)
            H = st.genpareto.cdf(x, c=xi, loc=0, scale=sigma)
            h = st.genpareto.pdf(x, c=xi, loc=0, scale=sigma)
            g = kappa * np.power(H, kappa - 1) * h
            plt.plot(x, g, label=rf'$\kappa$ = {kappa}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(rf'ExtGPD with $\sigma$={sigma}, $\xi$={xi}')
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \geq \mu`, when :math:`\xi \geq 0`
              * :math:`\mu \leq x \leq \mu - \sigma/\xi`, when :math:`\xi < 0`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter (threshold).
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter controlling upper tail behavior:

        * :math:`\xi > 0`: heavy tail (Pareto-like)
        * :math:`\xi = 0`: exponential tail
        * :math:`\xi < 0`: bounded upper tail
    kappa : float
        Lower tail parameter (kappa > 0). Controls the behavior near zero.
        When kappa = 1, reduces to standard GPD.

    References
    ----------
    .. [1] Naveau, P., Huser, R., Ribereau, P., and Hannart, A. (2016).
        Modeling jointly low, moderate, and heavy rainfall intensities without
        a threshold selection. Water Resources Research, 52(4), 2753-2769.

    .. [2] Papastathopoulos, I. and Tawn, J. A. (2013).
        Extended generalised Pareto models for tail estimation.
        Journal of Statistical Planning and Inference, 143(1), 131-143.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import ExtGenPareto

        # Model entire distribution without threshold selection
        with pm.Model():
            sigma = pm.HalfNormal("sigma", sigma=1)
            xi = pm.Normal("xi", mu=0, sigma=0.5)
            kappa = pm.HalfNormal("kappa", sigma=1)
            obs = ExtGenPareto("obs", mu=0, sigma=sigma, xi=xi, kappa=kappa, observed=data)
    """

    rv_op = ext_gpd

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, kappa=1, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))
        kappa = pt.as_tensor_variable(floatX(kappa))

        return super().dist([mu, sigma, xi, kappa], **kwargs)

    def logp(value, mu, sigma, xi, kappa):
        """
        Calculate log-probability of Extended Generalized Pareto distribution
        at specified value.

        The pdf is: g(x) = kappa * H(x)^(kappa-1) * h(x)
        where H is the GPD cdf and h is the GPD pdf.

        log g(x) = log(kappa) + (kappa-1)*log(H(x)) + log(h(x))

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        # GPD cdf: H(z) = 1 - (1 + xi*z)^(-1/xi) for xi != 0
        #          H(z) = 1 - exp(-z) for xi = 0
        H = pt.switch(
            pt.isclose(xi, 0),
            1 - pt.exp(-scaled),
            1 - pt.pow(1 + xi * scaled, -1 / xi),
        )

        # GPD log-pdf: log h(z) = -log(sigma) - (1 + 1/xi)*log(1 + xi*z) for xi != 0
        #              log h(z) = -log(sigma) - z for xi = 0
        log_h = pt.switch(
            pt.isclose(xi, 0),
            -pt.log(sigma) - scaled,
            -pt.log(sigma) - (1 + 1 / xi) * pt.log1p(xi * scaled),
        )

        # ExtGPD log-pdf: log(kappa) + (kappa-1)*log(H) + log(h)
        # Need to handle H near 0 carefully for numerical stability
        logp_expression = pt.log(kappa) + (kappa - 1) * pt.log(H) + log_h

        # Check support: scaled >= 0, and for xi < 0, also scaled <= -1/xi
        in_support = pt.switch(
            pt.ge(xi, 0),
            pt.ge(scaled, 0),
            pt.and_(pt.ge(scaled, 0), pt.le(scaled, -1 / xi)),
        )

        # Also need H > 0 and 1 + xi*scaled > 0
        logp = pt.switch(
            pt.and_(pt.and_(in_support, pt.gt(1 + xi * scaled, 0)), pt.gt(H, 0)),
            logp_expression,
            -np.inf,
        )

        return check_parameters(logp, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def logcdf(value, mu, sigma, xi, kappa):
        """
        Compute the log of the cumulative distribution function for Extended
        Generalized Pareto distribution at the specified value.

        G(x) = H(x)^kappa, so log G(x) = kappa * log(H(x))

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        # GPD cdf: H(z)
        H = pt.switch(
            pt.isclose(xi, 0),
            1 - pt.exp(-scaled),
            1 - pt.pow(1 + xi * scaled, -1 / xi),
        )

        # ExtGPD log-cdf: kappa * log(H)
        logcdf_expression = kappa * pt.log(H)

        # Handle bounds
        logcdf = pt.switch(
            pt.lt(scaled, 0),
            -np.inf,
            pt.switch(
                pt.and_(pt.lt(xi, 0), pt.gt(scaled, -1 / xi)),
                0.0,  # log(1) = 0 at upper bound for xi < 0
                logcdf_expression,
            ),
        )

        return check_parameters(logcdf, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def icdf(value, mu, sigma, xi, kappa):
        """
        Compute the inverse CDF (quantile function) for Extended Generalized Pareto distribution.

        For ExtGPD: G(x) = H(x)^kappa, so G^{-1}(p) = H^{-1}(p^{1/kappa})

        For GPD: H^{-1}(p) = mu + sigma * [(1-p)^(-xi) - 1] / xi  for xi != 0
                 H^{-1}(p) = mu - sigma * log(1-p)  for xi = 0

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Probability value(s) in [0, 1] for which to compute quantiles.

        Returns
        -------
        TensorVariable
        """
        # Transform p to get the GPD quantile input: p_gpd = p^(1/kappa)
        p_gpd = pt.pow(value, 1 / kappa)

        # GPD inverse CDF: H^{-1}(p) = mu + sigma * [(1-p)^(-xi) - 1] / xi
        res = pt.switch(
            pt.isclose(xi, 0),
            mu - sigma * pt.log(1 - p_gpd),
            mu + sigma * (pt.pow(1 - p_gpd, -xi) - 1) / xi,
        )

        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def support_point(rv, size, mu, sigma, xi, kappa):
        r"""
        Using the median as the support point.

        For ExtGPD, the median satisfies G(m) = 0.5, i.e., H(m)^kappa = 0.5
        So H(m) = 0.5^(1/kappa) and m = H^{-1}(0.5^{1/kappa})

        For GPD: H^{-1}(p) = sigma * [(1-p)^(-xi) - 1] / xi  for xi != 0
                 H^{-1}(p) = -sigma * log(1-p)  for xi = 0
        """
        p = pt.pow(0.5, 1 / kappa)  # H(median) = 0.5^(1/kappa)
        median = pt.switch(
            pt.isclose(xi, 0),
            mu - sigma * pt.log(1 - p),
            mu + sigma * (pt.pow(1 - p, -xi) - 1) / xi,
        )
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


class Chi:
    r"""
    :math:`\chi` log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{\nu - 1}e^{-x^2/2}}{2^{\nu/2 - 1}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 10, 200)
        for df in [1, 2, 3, 6, 9]:
            pdf = st.chi.pdf(x, df)
            plt.plot(x, pdf, label=r'$\nu$ = {}'.format(df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{2}\frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)}`
    Variance  :math:`\nu - 2\left(\frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)}\right)^2`
    ========  =========================================================================

    Parameters
    ----------
    nu : tensor_like of float
        Degrees of freedom (nu > 0).

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import Chi

        with pm.Model():
            x = Chi("x", nu=1)
    """

    @staticmethod
    def chi_dist(nu: TensorVariable, size: TensorVariable) -> TensorVariable:
        return pt.math.sqrt(ChiSquared.dist(nu=nu, size=size))

    def __new__(cls, name, nu, **kwargs):
        if "observed" not in kwargs:
            kwargs.setdefault("default_transform", transforms.log)
        return CustomDist(name, nu, dist=cls.chi_dist, class_name="Chi", **kwargs)

    @classmethod
    def dist(cls, nu, **kwargs):
        return CustomDist.dist(nu, dist=cls.chi_dist, class_name="Chi", **kwargs)


class Maxwell:
    R"""
    The Maxwell-Boltzmann distribution

    The pdf of this distribution is

    .. math::

       f(x \mid a) = {\displaystyle {\sqrt {\frac {2}{\pi }}}\,{\frac {x^{2}}{a^{3}}}\,\exp \left({\frac {-x^{2}}{2a^{2}}}\right)}

    Read more about it on `Wikipedia <https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution>`_

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 20, 200)
        for a in [1, 2, 5]:
            pdf = st.maxwell.pdf(x, scale=a)
            plt.plot(x, pdf, label=r'$a$ = {}'.format(a))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`2a \sqrt{\frac{2}{\pi}}`
    Variance  :math:`\frac{a^2(3 \pi - 8)}{\pi}`
    ========  =========================================================================

    Parameters
    ----------
    a : tensor_like of float
        Scale parameter (a > 0).

    """

    @staticmethod
    def maxwell_dist(a: TensorVariable, size: TensorVariable) -> TensorVariable:
        if rv_size_is_none(size):
            size = a.shape

        a = CheckParameterValue("a > 0")(a, pt.all(pt.gt(a, 0)))

        return Chi.dist(nu=3, size=size) * a

    def __new__(cls, name, a, **kwargs):
        return CustomDist(
            name,
            a,
            dist=cls.maxwell_dist,
            class_name="Maxwell",
            **kwargs,
        )

    @classmethod
    def dist(cls, a, **kwargs):
        return CustomDist.dist(
            a,
            dist=cls.maxwell_dist,
            class_name="Maxwell",
            **kwargs,
        )
