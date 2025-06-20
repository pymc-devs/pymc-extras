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
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable
from scipy import stats


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
            x = Chi('x', nu=1)
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
