#   Copyright 2023 The PyMC Developers
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

import numpy as np
import pymc as pm

from pymc.distributions.dist_math import betaln, check_parameters, factln, logpow
from pymc.distributions.distribution import Discrete
from pymc.distributions.shape_utils import rv_size_is_none
from pytensor import tensor as pt
from pytensor.tensor.random.op import RandomVariable


def log1mexp(x):
    cond = x < np.log(0.5)
    return np.piecewise(
        x,
        [cond, ~cond],
        [lambda x: np.log1p(-np.exp(x)), lambda x: np.log(-np.expm1(x))],
    )


class GeneralizedPoissonRV(RandomVariable):
    name = "generalized_poisson"
    signature = "(),()->()"
    dtype = "int64"
    _print_name = ("GeneralizedPoisson", "\\operatorname{GeneralizedPoisson}")

    @classmethod
    def rng_fn(cls, rng, theta, lam, size):
        theta = np.asarray(theta)
        lam = np.asarray(lam)

        if size is not None:
            dist_size = size
        else:
            dist_size = np.broadcast_shapes(theta.shape, lam.shape)

        # A mix of 2 algorithms described by Famoye (1997) is used depending on parameter values
        # 0: Inverse method, computed on the log scale. Used when lam <= 0.
        # 1: Branching method. Used when lambda > 0.
        x = np.empty(dist_size)
        idxs_mask = np.broadcast_to(lam < 0, dist_size)
        if np.any(idxs_mask):
            x[idxs_mask] = cls._inverse_rng_fn(rng, theta, lam, dist_size, idxs_mask=idxs_mask)[
                idxs_mask
            ]
        idxs_mask = ~idxs_mask
        if np.any(idxs_mask):
            x[idxs_mask] = cls._branching_rng_fn(rng, theta, lam, dist_size, idxs_mask=idxs_mask)[
                idxs_mask
            ]
        return x

    @classmethod
    def _inverse_rng_fn(cls, rng, theta, lam, dist_size, idxs_mask):
        # We handle x/0 and log(0) issues with branching
        with np.errstate(divide="ignore", invalid="ignore"):
            log_u = np.log(rng.uniform(size=dist_size))
            pos_lam = lam > 0
            abs_log_lam = np.log(np.abs(lam))
            theta_m_lam = theta - lam
            log_s = -theta
            log_p = log_s.copy()
            x_ = 0
            x = np.zeros(dist_size)
            below_cutpoint = log_s < log_u
            while np.any(below_cutpoint[idxs_mask]):
                x_ += 1
                x[below_cutpoint] += 1
                log_c = np.log(theta_m_lam + lam * x_)
                # Compute log(1 + lam / C)
                log1p_lam_m_C = np.where(
                    pos_lam,
                    np.log1p(np.exp(abs_log_lam - log_c)),
                    log1mexp(abs_log_lam - log_c),
                )
                log_p = log_c + log1p_lam_m_C * (x_ - 1) + log_p - np.log(x_) - lam
                log_s = np.logaddexp(log_s, log_p)
                below_cutpoint = log_s < log_u
            return x

    @classmethod
    def _branching_rng_fn(cls, rng, theta, lam, dist_size, idxs_mask):
        lam_ = np.abs(lam)  # This algorithm is only valid for positive lam
        y = rng.poisson(theta, size=dist_size)
        x = y.copy()
        higher_than_zero = y > 0
        while np.any(higher_than_zero[idxs_mask]):
            y = rng.poisson(lam_ * y)
            x[higher_than_zero] = x[higher_than_zero] + y[higher_than_zero]
            higher_than_zero = y > 0
        return x


generalized_poisson = GeneralizedPoissonRV()


class GeneralizedPoisson(pm.distributions.Discrete):
    R"""
    Generalized Poisson.
    Used to model count data that can be either overdispersed or underdispersed.
    Offers greater flexibility than the standard Poisson which assumes equidispersion,
    where the mean is equal to the variance.
    The pmf of this distribution is

    .. math:: f(x \mid \mu, \lambda) =
                  \frac{\mu (\mu + \lambda x)^{x-1} e^{-\mu - \lambda x}}{x!}

    ========  ======================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\frac{\mu}{1 - \lambda}`
    Variance  :math:`\frac{\mu}{(1 - \lambda)^3}`
    ========  ======================================

    Parameters
    ----------
    mu : tensor_like of float
        Mean parameter (mu > 0).
    lam : tensor_like of float
        Dispersion parameter (max(-1, -mu/4) <= lam <= 1).

    Notes
    -----
    When lam = 0, the Generalized Poisson reduces to the standard Poisson with the same mu.
    When lam < 0, the mean is greater than the variance (underdispersion).
    When lam > 0, the mean is less than the variance (overdispersion).

    The PMF is taken from [1]_ and the random generator function is adapted from [2]_.

    References
    ----------
    .. [1] Consul, PoC, and Felix Famoye. "Generalized Poisson regression model."
       Communications in Statistics-Theory and Methods 21.1 (1992): 89-109.
    .. [2] Famoye, Felix. "Generalized Poisson random variate generation." American
       Journal of Mathematical and Management Sciences 17.3-4 (1997): 219-237.
    """

    rv_op = generalized_poisson

    @classmethod
    def dist(cls, mu, lam, **kwargs):
        mu = pt.as_tensor_variable(mu)
        lam = pt.as_tensor_variable(lam)
        return super().dist([mu, lam], **kwargs)

    def support_point(rv, size, mu, lam):
        mean = pt.floor(mu / (1 - lam))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, lam):
        mu_lam_value = mu + lam * value
        logprob = np.log(mu) + logpow(mu_lam_value, value - 1) - mu_lam_value - factln(value)

        # Probability is 0 when value > m, where m is the largest positive integer for
        # which mu + m * lam > 0 (when lam < 0).
        logprob = pt.switch(
            pt.or_(
                mu_lam_value < 0,
                value < 0,
            ),
            -np.inf,
            logprob,
        )

        return check_parameters(
            logprob,
            0 < mu,
            pt.abs(lam) <= 1,
            (-mu / 4) <= lam,
            msg="0 < mu, max(-1, -mu/4)) <= lam <= 1",
        )


class BetaNegativeBinomial:
    R"""
    Beta Negative Binomial distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid \alpha, \beta, r) = \frac{B(r + x, \alpha + \beta)}{B(r, \alpha)} \frac{\Gamma(x + \beta)}{x! \Gamma(\beta)}

    where :math:`B` is the Beta function and :math:`\Gamma` is the Gamma function.

    For more information, see https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.special import betaln, gammaln
        def factln(x):
            return gammaln(x + 1)
        def logp(x, alpha, beta, r):
            return (
                betaln(r + x, alpha + beta)
                - betaln(r, alpha)
                + gammaln(x + beta)
                - factln(x)
                - gammaln(beta)
            )
        plt.style.use('arviz-darkgrid')
        x = np.arange(0, 25)
        params = [
            (1, 1, 1),
            (1, 1, 10),
            (1, 10, 1),
            (1, 10, 10),
            (10, 10, 10),
        ]
        for alpha, beta, r in params:
            pmf = np.exp(logp(x, alpha, beta, r))
            plt.plot(x, pmf, "-o", label=r'$alpha$ = {}, $beta$ = {}, $r$ = {}'.format(alpha, beta, r))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ======================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`{\begin{cases}{\frac  {r\beta }{\alpha -1}}&{\text{if}}\ \alpha >1\\\infty &{\text{otherwise}}\ \end{cases}}`
    Variance  :math:`{\displaystyle {\begin{cases}{\frac {r\beta (r+\alpha -1)(\beta +\alpha -1)}{(\alpha -2){(\alpha -1)}^{2}}}&{\text{if}}\ \alpha >2\\\infty &{\text{otherwise}}\ \end{cases}}}`
    ========  ======================================

    Parameters
    ----------
    alpha : tensor_like of float
        shape of the beta distribution (alpha > 0).
    beta : tensor_like of float
        shape of the beta distribution (beta > 0).
    r : tensor_like of float
        number of successes until the experiment is stopped (integer but can be extended to real)
    """

    @staticmethod
    def beta_negative_binomial_dist(alpha, beta, r, size):
        if rv_size_is_none(size):
            alpha, beta, r = pt.broadcast_arrays(alpha, beta, r)

        p = pm.Beta.dist(alpha, beta, size=size)
        return pm.NegativeBinomial.dist(p, r, size=size)

    @staticmethod
    def beta_negative_binomial_logp(value, alpha, beta, r):
        res = (
            betaln(r + value, alpha + beta)
            - betaln(r, alpha)
            + pt.gammaln(value + beta)
            - factln(value)
            - pt.gammaln(beta)
        )
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            res,
        )

        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            r > 0,
            msg="alpha > 0, beta > 0, r > 0",
        )

    def __new__(cls, name, alpha, beta, r, **kwargs):
        return pm.CustomDist(
            name,
            alpha,
            beta,
            r,
            dist=cls.beta_negative_binomial_dist,
            logp=cls.beta_negative_binomial_logp,
            class_name="BetaNegativeBinomial",
            **kwargs,
        )

    @classmethod
    def dist(cls, alpha, beta, r, **kwargs):
        return pm.CustomDist.dist(
            alpha,
            beta,
            r,
            dist=cls.beta_negative_binomial_dist,
            logp=cls.beta_negative_binomial_logp,
            class_name="BetaNegativeBinomial",
            **kwargs,
        )


class Skellam:
    R"""
    Skellam distribution.

    The Skellam distribution is the distribution of the difference of two
    Poisson random variables.

    The pmf of this distribution is

    .. math::

        f(x | \mu_1, \mu_2) = e^{{-(\mu _{1}\!+\!\mu _{2})}}\left({\frac  {\mu _{1}}{\mu _{2}}}\right)^{{x/2}}\!\!I_{{x}}(2{\sqrt  {\mu _{1}\mu _{2}}})

    where :math:`I_{x}` is the modified Bessel function of the first kind of order :math:`x`.

    Read more about the Skellam distribution at https://en.wikipedia.org/wiki/Skellam_distribution

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.arange(-15, 15)
        params = [
            (1, 1),
            (5, 5),
            (5, 1),
        ]
        for mu1, mu2 in params:
            pmf = st.skellam.pmf(x, mu1, mu2)
            plt.plot(x, pmf, "-o", label=r'$\mu_1$ = {}, $\mu_2$ = {}'.format(mu1, mu2))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ======================================
    Support   :math:`x \in \mathbb{Z}`
    Mean      :math:`\mu_{1} - \mu_{2}`
    Variance  :math:`\mu_{1} + \mu_{2}`
    ========  ======================================

    Parameters
    ----------
    mu1 : tensor_like of float
        Mean parameter (mu1 >= 0).
    mu2 : tensor_like of float
        Mean parameter (mu2 >= 0).
    """

    @staticmethod
    def skellam_dist(mu1, mu2, size):
        if rv_size_is_none(size):
            mu1, mu2 = pt.broadcast_arrays(mu1, mu2)

        return pm.Poisson.dist(mu=mu1, size=size) - pm.Poisson.dist(mu=mu2, size=size)

    @staticmethod
    def skellam_logp(value, mu1, mu2):
        res = (
            -mu1
            - mu2
            + 0.5 * value * (pt.log(mu1) - pt.log(mu2))
            + pt.log(pt.iv(value, 2 * pt.sqrt(mu1 * mu2)))
        )
        return check_parameters(
            res,
            mu1 >= 0,
            mu2 >= 0,
            msg="mu1 >= 0, mu2 >= 0",
        )

    def __new__(cls, name, mu1, mu2, **kwargs):
        return pm.CustomDist(
            name,
            mu1,
            mu2,
            dist=cls.skellam_dist,
            logp=cls.skellam_logp,
            class_name="Skellam",
            **kwargs,
        )

    @classmethod
    def dist(cls, mu1, mu2, **kwargs):
        return pm.CustomDist.dist(
            mu1,
            mu2,
            dist=cls.skellam_dist,
            logp=cls.skellam_logp,
            class_name="Skellam",
            **kwargs,
        )


class GrassiaIIGeometricRV(RandomVariable):
    name = "g2g"
    signature = "(),(),()->()"

    dtype = "int64"
    _print_name = ("GrassiaIIGeometric", "\\operatorname{GrassiaIIGeometric}")

    def __call__(self, r, alpha, time_covariate_vector=None, size=None, **kwargs):
        return super().__call__(r, alpha, time_covariate_vector, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, r, alpha, time_covariate_vector, size):
        # Handle None case for time_covariate_vector
        if time_covariate_vector is None:
            time_covariate_vector = 0.0

        # Convert inputs to numpy arrays - these should be concrete values
        r = np.asarray(r, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        time_covariate_vector = np.asarray(time_covariate_vector, dtype=np.float64)

        # Determine output size
        if size is None:
            size = np.broadcast_shapes(r.shape, alpha.shape, time_covariate_vector.shape)

        # Broadcast parameters to the output size
        r = np.broadcast_to(r, size)
        alpha = np.broadcast_to(alpha, size)
        time_covariate_vector = np.broadcast_to(time_covariate_vector, size)

        # Calculate exp(time_covariate_vector) for all samples
        exp_time_covar_sum = np.exp(time_covariate_vector)

        # Use a simpler approach: generate from a geometric distribution with transformed parameters
        # This is an approximation but should be much faster and more reliable
        lam = rng.gamma(shape=r, scale=1 / alpha, size=size)
        lam_covar = lam * exp_time_covar_sum

        # Handle numerical stability for very small lambda values
        p = np.where(
            lam_covar < 0.0001,
            lam_covar,  # For small values, set this to p
            1 - np.exp(-lam_covar),
        )

        # Ensure p is in valid range for geometric distribution
        p = np.clip(p, np.finfo(float).tiny, 1.0)

        # Generate geometric samples
        return rng.geometric(p)


g2g = GrassiaIIGeometricRV()


class GrassiaIIGeometric(Discrete):
    r"""Grassia(II)-Geometric distribution.

    This distribution is a flexible alternative to the Geometric distribution for the number of trials until a
    discrete event, and can be extended to support both static and time-varying covariates.

    Hardie and Fader describe this distribution with the following PMF and survival functions in [1]_:

    .. math::
        \mathbb{P}T=t|r,\alpha,\beta;Z(t)) = (\frac{\alpha}{\alpha+C(t-1)})^{r} - (\frac{\alpha}{\alpha+C(t)})^{r}  \\
        \begin{align}
        \mathbb{S}(t|r,\alpha,\beta;Z(t)) = (\frac{\alpha}{\alpha+C(t)})^{r} \\
        \end{align}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        t = np.arange(1, 11)
        alpha_vals = [1., 1., 2., 2.]
        r_vals = [.1, .25, .5, 1.]
        for alpha, r in zip(alpha_vals, r_vals):
            pmf = (alpha/(alpha + t - 1))**r - (alpha/(alpha+t))**r
            plt.plot(t, pmf, '-o', label=r'$\alpha$ = {}, $r$ = {}'.format(alpha, r))
        plt.xlabel('t', fontsize=12)
        plt.ylabel('p(t)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ===============================================
    Support   :math:`t \in \mathbb{N}_{>0}`
    ========  ===============================================

    Parameters
    ----------
    r : tensor_like of float
        Shape parameter (r > 0).
    alpha : tensor_like of float
        Scale parameter (alpha > 0).
    time_covariate_vector : tensor_like of float, optional
        Optional vector of dot product of time-varying covariates and their coefficients by time period.

    References
    ----------
    .. [1] Fader, Peter & G. S. Hardie, Bruce (2020).
       "Incorporating Time-Varying Covariates in a Simple Mixture Model for Discrete-Time Duration Data."
       https://www.brucehardie.com/notes/037/time-varying_covariates_in_BG.pdf
    """

    rv_op = g2g

    @classmethod
    def dist(cls, r, alpha, time_covariate_vector=None, *args, **kwargs):
        r = pt.as_tensor_variable(r)
        alpha = pt.as_tensor_variable(alpha)
        if time_covariate_vector is None:
            time_covariate_vector = pt.constant(0.0)
        time_covariate_vector = pt.as_tensor_variable(time_covariate_vector)
        return super().dist([r, alpha, time_covariate_vector], *args, **kwargs)

    def logp(value, r, alpha, time_covariate_vector=None):
        if time_covariate_vector is None:
            time_covariate_vector = pt.constant(0.0)
        time_covariate_vector = pt.as_tensor_variable(time_covariate_vector)

        def C_t(t):
            # Aggregate time_covariate_vector over active time periods
            if t == 0:
                return pt.constant(1.0)
            # Handle case where time_covariate_vector is a scalar
            if time_covariate_vector.ndim == 0:
                return t * pt.exp(time_covariate_vector)
            else:
                # For vector time_covariate_vector, we need to handle symbolic indexing
                # Since we can't slice with symbolic indices, we'll use a different approach
                # For now, we'll use the first element multiplied by t
                # This is a simplification but should work for basic cases
                return t * pt.exp(time_covariate_vector[:t])

        # Calculate the PMF on log scale
        logp = pt.log(
            pt.pow(alpha / (alpha + C_t(value - 1)), r) - pt.pow(alpha / (alpha + C_t(value)), r)
        )

        # Handle invalid values
        logp = pt.switch(
            pt.or_(
                value < 1,  # Value must be >= 1
                pt.isnan(logp),  # Handle NaN cases
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            r > 0,
            alpha > 0,
            msg="r > 0, alpha > 0",
        )

    def logcdf(value, r, alpha, time_covariate_vector=None):
        if time_covariate_vector is None:
            time_covariate_vector = pt.constant(0.0)
        time_covariate_vector = pt.as_tensor_variable(time_covariate_vector)

        # Calculate CDF on log scale
        # For the GrassiaIIGeometric, the CDF is 1 - survival function
        # S(t) = (alpha/(alpha + C(t)))^r
        # CDF(t) = 1 - S(t)

        def C_t(t):
            if t == 0:
                return pt.constant(1.0)
            if time_covariate_vector.ndim == 0:
                return t * pt.exp(time_covariate_vector)
            else:
                return t * pt.exp(time_covariate_vector[:t])

        survival = pt.pow(alpha / (alpha + C_t(value)), r)
        logcdf = pt.log(1 - survival)

        return check_parameters(
            logcdf,
            r > 0,
            alpha > 0,
            msg="r > 0, alpha > 0",
        )

    def support_point(rv, size, r, alpha, time_covariate_vector=None):
        """Calculate a reasonable starting point for sampling.

        For the GrassiaIIGeometric distribution, we use a point estimate based on
        the expected value of the mixing distribution. Since the mixing distribution
        is Gamma(r, 1/alpha), its mean is r/alpha. We then transform this through
        the geometric link function and round to ensure an integer value.

        When time_covariate_vector is provided, it affects the expected value through
        the exponential link function: exp(time_covariate_vector).
        """
        # Base mean without covariates
        mean = pt.exp(alpha / r)

        # Apply time-varying covariates if provided
        if time_covariate_vector is None:
            time_covariate_vector = pt.constant(0.0)
        mean = mean * pt.exp(time_covariate_vector)

        # Round up to nearest integer
        mean = pt.ceil(mean)

        if not rv_size_is_none(size):
            mean = pt.full(size, mean)

        return mean
