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

import numpy as np
import pytensor.tensor as pt

from pymc.distributions.dist_math import (
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
)
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.logprob.transforms import Transform
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.utils import normalize_size_param

from pymc_extras.distributions import pytensor_genpareto as genpareto


class GenParetoRV(SymbolicRandomVariable):
    name = "genpareto"
    extended_signature = "[rng],[size],(),(),()->[rng],()"
    _print_name = ("GenPareto", "\\operatorname{GenPareto}")

    @classmethod
    def rv_op(cls, mu, sigma, xi, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        sigma = pt.as_tensor(sigma)
        xi = pt.as_tensor(xi)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, sigma, xi, ndims_params=cls.ndims_params)
        # Draw the survival probability directly so excess = -log(v) avoids the
        # 1 - u cancellation that hurts the heavy upper tail.
        next_rng, v = rng.uniform(size=size)
        draws = genpareto._gpd_quantile_from_excess(-pt.log(v), mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi
        )


class GenPareto(Continuous):
    r"""Generalized Pareto distribution.

    The Generalized Pareto distribution (GPD) is the canonical model for
    exceedances over a threshold (peaks-over-threshold), the counterpart of the
    GEV used for block maxima. Its CDF is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \begin{cases}
           1 - \left(1 + \xi\,\frac{x - \mu}{\sigma}\right)^{-1/\xi}, & \xi \neq 0, \\[6pt]
           1 - \exp\!\left(-\dfrac{x - \mu}{\sigma}\right),           & \xi = 0.
       \end{cases}

    The shape :math:`\xi` is parametrized as in Scarrott and MacDonald (2012) [1]_,
    matching SciPy's ``genpareto`` ``c`` parameter.

    .. plot::
        :context: close-figs
        :include-source: False

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 4, 500)
        for xi in [-1.2, -1.0, -0.8, -0.5, 0.0, 2.0]:
            pdf = st.genpareto.pdf(x, c=xi, loc=0.0, scale=1.0)
            plt.plot(x, pdf, label=rf'$\xi$ = {xi:g}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('probability density', fontsize=12)
        plt.ylim(0, 2.5)
        plt.legend(loc='upper right', title=r'$\mu = 0,\ \sigma = 1$')
        plt.show()

    ========  =========================================================================
    Support   :math:`\begin{cases} x \geq \mu, & \xi \geq 0 \\ \mu \leq x < \mu - \sigma/\xi, & \xi < 0 \end{cases}`
    Mean      :math:`\begin{cases} \mu + \dfrac{\sigma}{1 - \xi}, & \xi < 1 \\ \infty, & \xi \geq 1 \end{cases}`
    Variance  :math:`\begin{cases} \dfrac{\sigma^2}{(1 - \xi)^2 (1 - 2\xi)}, & \xi < 1/2 \\ \infty, & \xi \geq 1/2 \end{cases}`
    Median    :math:`\begin{cases} \mu + \dfrac{\sigma (2^{\xi} - 1)}{\xi}, & \xi \neq 0 \\ \mu + \sigma \ln 2, & \xi = 0 \end{cases}`
    Mode      :math:`\begin{cases} \mu, & \xi \geq -1 \\ \mu - \sigma/\xi, & \xi \leq -1 \end{cases}`
    Entropy   :math:`\ln \sigma + \xi + 1`
    ========  =========================================================================

    Parameters
    ----------
    mu : tensor_like of float
        Location parameter (the threshold).
    sigma : tensor_like of float
        Scale parameter (sigma > 0).
    xi : tensor_like of float
        Shape parameter. :math:`\xi > 0` gives a heavy (Pareto) tail,
        :math:`\xi = 0` an exponential tail, and :math:`\xi < 0` a bounded
        upper tail.

    References
    ----------
    .. [1] Scarrott, C., & MacDonald, A. (2012). A review of extreme value
        threshold estimation and uncertainty quantification. REVSTAT -
        Statistical Journal, 10(1), 33-60.
        https://doi.org/10.57805/revstat.v10i1.110
    .. [2] Pickands, J. (1975). Statistical Inference Using Extreme Order
        Statistics. Annals of Statistics, 3(1), 119-131.
        https://www.jstor.org/stable/2958083

    Notes
    -----
    The support is :math:`[\mu, \infty)` for :math:`\xi \geq 0`. For
    :math:`\xi < 0`, the support has a finite right endpoint
    :math:`x_F := \mu - \sigma/\xi`; values at or above :math:`x_F` have density
    :math:`0`. Since :math:`x_F` depends on :math:`\sigma` and :math:`\xi`, a fit
    that leaves them free must keep every observation inside the support (see
    :ref:`Modeling recommendations <genpareto-modeling-recommendations>`).

    The :math:`r`-th moment is finite iff :math:`\xi < 1/r`. Thus the mean is
    finite for :math:`\xi < 1` and the variance for :math:`\xi < 1/2`. Every
    moment exists for :math:`\xi \leq 0`.

    Examples
    --------
    Fitting exceedances over a known threshold:

    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import GenPareto

        exceedances = data[data > threshold]
        xmax = exceedances.max()
        with pm.Model():
            # sigma carries the data's units; use a wide log-scale prior when the
            # data scale is unknown.
            pareto_sigma = pm.LogNormal("pareto_sigma", mu=0.0, sigma=10.0)
            # xi lower bound: see Modeling recommendations.
            pareto_xi = pm.TruncatedNormal(
                "pareto_xi",
                mu=0.0,
                sigma=2.0,
                lower=pm.math.maximum(-1.0, -pareto_sigma / (xmax - threshold)),
            )
            GenPareto("obs", mu=threshold, sigma=pareto_sigma, xi=pareto_xi, observed=exceedances)
            idata = pm.sample()

    .. _genpareto-modeling-recommendations:

    .. rubric:: Modeling recommendations

    - **Keep the data inside the support.** If :math:`\xi` is allowed to be
      negative, the support has a finite upper wall :math:`x = x_F`, where
      :math:`x_F := \mu - \sigma/\xi`,
      and the observations stay inside it only when :math:`\max(\text{data}) < x_F`,
      equivalently :math:`\xi > -\sigma/(\max(\text{data}) - \mu)`. Encoding this as
      the :math:`\xi` prior's lower bound keeps :math:`x_F` above the data; without
      it the sampler can reach :math:`(\sigma, \xi)` for which
      :math:`x_F < \max(\text{data})`, putting an observation out of support and
      causing divergences. (:math:`\xi \geq 0` has no upper wall and always
      contains the data.)

    - **Floor** :math:`\xi` **at** :math:`-1`. If :math:`\xi` is allowed below
      :math:`-1`, the density diverges at the wall :math:`x = x_F`: as :math:`x_F`
      decreases toward :math:`\max(\text{data})` the likelihood tends to
      :math:`+\infty`. Choose the :math:`\xi` prior to enforce :math:`\xi \geq -1` (where
      the wall density is finite); combined with the in-support bound, this gives the
      example prior's lower limit :math:`\max(-1,\ -\sigma/(\max(\text{data}) - \mu))`.

    - **Cap** :math:`\xi` **from above if moments must exist.** When the application needs a
      finite mean or variance, bound the :math:`\xi` prior above accordingly: :math:`\xi < 1`
      (mean) or :math:`\xi < 1/2` (variance).

    - **Keep** :math:`\mu` **fixed at the threshold.** In a peaks-over-threshold fit
      :math:`\mu` is chosen in advance, not inferred. Estimating it has two failure
      modes: proposals with :math:`\mu > \min(\text{data})` put observations out
      of support, and the corner :math:`\mu \to \min(\text{data})`,
      :math:`\sigma \to 0` has unbounded likelihood when :math:`\xi > n - 1`.

    .. rubric:: Implementation details

    **Numerical precision near the wall.** As :math:`x` nears the wall :math:`x = x_F`,
    the margin :math:`1 + \xi (x - \mu)/\sigma` cancels toward :math:`0`. The
    log-density value stays accurate, but its gradient w.r.t. :math:`\sigma` and
    :math:`\xi` has relative error of order the machine epsilon divided by that
    margin. It stems from the finite precision of :math:`x`, and no reformulation
    of ``logp`` avoids it.

    **Difference from SciPy.** For :math:`\xi < 0` the wall :math:`x = x_F` is treated
    as open: ``logp`` returns :math:`-\infty` there. There is a finite limiting
    density for :math:`-1 \leq \xi < 0` as :math:`x \to x_F^-`, and SciPy returns
    the limiting value. This endpoint carries zero probability mass, so the choice
    affects neither sampling nor integration.

    """

    rv_type = GenParetoRV
    rv_op = GenParetoRV.rv_op

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))
        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        return check_parameters(genpareto.logpdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def logcdf(value, mu, sigma, xi):
        return check_parameters(genpareto.logcdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def logccdf(value, mu, sigma, xi):
        return check_parameters(genpareto.logsf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def icdf(value, mu, sigma, xi):
        res = genpareto.ppf(value, mu, sigma, xi)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, msg="sigma > 0")

    def support_point(rv, size, mu, sigma, xi):
        # Median: the mean is infinite for xi >= 1.
        excess = np.log(2.0)  # -log(1 - 0.5)
        median = genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


class _AbstractGPDLogitCDFTransform(Transform):
    """Transform a GPD/ExtGPD from its constrained support to an unconstrained domain.

    The variate ``x`` has half-infinite support ``[mu, inf)`` for ``xi >= 0`` and bounded support
    ``[mu, mu - sigma/xi)`` for ``xi < 0``. The easy-to-compute logit-CDF ``y = logit(F(x))``
    handles both cases in a unified way: ``F`` maps any support onto ``(0, 1)`` and ``logit``
    onto the real line, so ``y`` is standard ``Logistic`` for every ``xi``. It is smooth across
    ``xi = 0``, so the sampler's target stays continuously differentiable in the parameters.

    Subclasses supply ``_logp`` / ``_logcdf`` / ``_logccdf`` / ``_excess_from_logit`` (the GPD
    excess ``m = -log S(x)`` from ``y = logit(F)``) / ``_quantile_from_excess``.
    """

    name = "gpd_logit_cdf"
    ndim_supp = 0

    # Set by subclasses (see the class docstring).
    _logp = _logcdf = _logccdf = _excess_from_logit = _quantile_from_excess = None

    def forward(self, value, rng, size, *distribution_parameters):
        """Quantile -> logit: ``value`` is the data point x; returns ``y = logit(F(x))``."""
        # logit(F) = log F - log(1 - F) = logcdf - logccdf, both stable.
        log_F = self._logcdf(value, *distribution_parameters)
        log_S = self._logccdf(value, *distribution_parameters)
        return log_F - log_S

    def backward(self, value, rng, size, *distribution_parameters):
        """Logit -> quantile: ``value`` is ``y = logit(F)``; returns the data point x."""
        mu = distribution_parameters[0]
        excess = self._excess_from_logit(value, *distribution_parameters)
        x = self._quantile_from_excess(excess, *distribution_parameters)
        # In finite precision the quantile can round onto mu (it is strictly above mu in
        # exact arithmetic), where a kappa < 1 density is +inf; floor it just above. The
        # offset scales with |mu| and the working precision.
        finfo = np.finfo(value.dtype)
        floor = pt.abs(mu) * (8.0 * finfo.eps) + finfo.tiny
        return pt.maximum(x, mu + floor)

    def log_jac_det(self, value, rng, size, *distribution_parameters):
        x = self.backward(value, rng, size, *distribution_parameters)
        # The transformed density is exactly Logistic(value); the framework adds
        # logp(x) back, so log|dx/dy| = logistic - logp(x).
        logistic = -pt.softplus(value) - pt.softplus(-value)
        return logistic - self._logp(x, *distribution_parameters)


class _GenParetoLogitCDF(_AbstractGPDLogitCDFTransform):
    """``y = logit(F(x))`` transform for :class:`GenPareto`."""

    _logp = staticmethod(genpareto.logpdf)
    _logcdf = staticmethod(genpareto.logcdf)
    _logccdf = staticmethod(genpareto.logsf)

    @staticmethod
    def _excess_from_logit(value, mu, sigma, xi):
        # u = sigmoid(y); m = -log(1 - u) = -log(sigmoid(-y)) = softplus(y). Stable for all y.
        return pt.softplus(value)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi):
        return genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)


@_default_transform.register(GenPareto)
def _genpareto_default_transform(op, rv):
    return _GenParetoLogitCDF()
