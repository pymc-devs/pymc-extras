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

import pytensor.tensor as pt

from pymc.distributions.dist_math import (
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
)
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.utils import normalize_size_param

from pymc_extras.distributions import pytensor_extgenpareto as extgenpareto
from pymc_extras.distributions import pytensor_genpareto as genpareto
from pymc_extras.distributions.pymc_genpareto import _AbstractGPDLogitCDFTransform


class ExtGenParetoRV(SymbolicRandomVariable):
    name = "extgenpareto"
    extended_signature = "[rng],[size],(),(),(),()->[rng],()"
    _print_name = ("ExtGenPareto", "\\operatorname{ExtGenPareto}")

    @classmethod
    def rv_op(cls, mu, sigma, xi, kappa, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        sigma = pt.as_tensor(sigma)
        xi = pt.as_tensor(xi)
        kappa = pt.as_tensor(kappa)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, sigma, xi, kappa, ndims_params=cls.ndims_params)
        next_rng, u = rng.uniform(size=size)
        # Carrier draw u = F; excess = -log(1 - u ** (1/kappa)), via log1mexp so
        # small-kappa draws do not collapse to the lower endpoint (1 - u**.. -> 1).
        excess = extgenpareto._ext_gpd_excess_from_log_prob(pt.log(u), kappa)
        draws = genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi, kappa], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi, kappa
        )


class ExtGenPareto(Continuous):
    r"""Extended Generalized Pareto distribution.

    The extended GPD (Type 1) of Naveau et al. (2016) [1]_ is equivalent to
    the EGP3 model of Papastathopoulos and Tawn (2013) [2]_. It transforms
    the GPD CDF :math:`G` through the carrier :math:`v \mapsto v^\kappa`:

    .. math::

       F(x \mid \mu, \sigma, \xi, \kappa) =
           \left[G\!\left(x \mid \mu, \sigma, \xi\right)\right]^{\kappa}.

    The GPD parameters :math:`\mu, \sigma, \xi` describe the tail (large :math:`x`),
    as in the GPD. :math:`\kappa > 0` is an additional shape parameter for the body
    (small :math:`x`); :math:`\kappa = 1` recovers the GPD. The extra parameter
    gives more flexibility to model the body while leaving the GPD tail unchanged.
    Typical applications include threshold exceedances [2]_ and modeling rainfall
    intensities [1]_.

    .. plot::
        :context: close-figs
        :include-source: False

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0.02, 4, 200)  # above 0: for kappa < 1 the density diverges at x = 0
        xi = 0.2
        for kappa in [0.8, 1.0, 1.2, 2.0, 3.0]:
            G = st.genpareto.cdf(x, c=xi)
            g = st.genpareto.pdf(x, c=xi)
            scaled_pdf = np.power(G, kappa - 1) * g  # ExtGPD density divided by kappa.
            plt.plot(x, scaled_pdf, label=rf'$\kappa$ = {kappa}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('tail-scaled density', fontsize=12)
        plt.ylim(0, 1.5)
        plt.legend(loc='upper right', title=rf'$\mu = 0,\ \sigma = 1,\ \xi = {xi}$')
        plt.show()

    ========  =========================================================================
    Support   :math:`\begin{cases} x \geq \mu, & \xi \geq 0 \\ \mu \leq x < \mu - \sigma/\xi, & \xi < 0 \end{cases}`
    Mean      :math:`\begin{cases} \mu + \dfrac{\sigma}{\xi}\left[\kappa B(\kappa, 1 - \xi) - 1\right], & \xi < 1,\ \xi \neq 0 \\ \mu + \sigma\left(\psi(\kappa + 1) + \gamma\right), & \xi = 0 \\ \infty, & \xi \geq 1 \end{cases}`
    Variance  :math:`\begin{cases} \left(\dfrac{\sigma}{\xi}\right)^2 \left[\kappa B(\kappa, 1 - 2\xi) - \kappa^2 B(\kappa, 1 - \xi)^2\right], & \xi < 1/2,\ \xi \neq 0 \\ \sigma^2\left(\pi^2/6 - \psi'(\kappa + 1)\right), & \xi = 0 \\ \infty, & \xi \geq 1/2 \end{cases}`
    Median    :math:`\begin{cases} \mu + \dfrac{\sigma}{\xi}\left[\left(1 - 2^{-1/\kappa}\right)^{-\xi} - 1\right], & \xi \neq 0 \\ \mu - \sigma\ln\!\left(1 - 2^{-1/\kappa}\right), & \xi = 0 \end{cases}`
    Mode      :math:`\begin{cases} \mu + \dfrac{\sigma}{\xi}\left(T^{-\xi} - 1\right), & \kappa > 1,\ \xi \geq -1,\ \xi \neq 0 \\ \mu + \sigma\ln\kappa, & \kappa > 1,\ \xi = 0 \\ \mu, & \kappa \leq 1,\ \xi \geq -1 \\ \mu - \sigma/\xi, & \kappa \geq 1,\ \xi < -1 \end{cases}` where :math:`T = \dfrac{1 + \xi}{\kappa + \xi}`
    ========  =========================================================================

    Here :math:`B(a, b) = \Gamma(a)\Gamma(b)/\Gamma(a + b)` is the Beta function,
    :math:`\psi` the digamma, :math:`\psi'` the trigamma, and :math:`\gamma` the
    Euler--Mascheroni constant.

    Parameters
    ----------
    mu : tensor_like of float
        Location parameter (the threshold).
    sigma : tensor_like of float
        Scale parameter (sigma > 0).
    xi : tensor_like of float
        Tail shape parameter (large :math:`x`), as in :class:`GenPareto`.
    kappa : tensor_like of float
        Body shape parameter (small :math:`x`, ``kappa > 0``). ``kappa = 1``
        reduces the distribution to :class:`GenPareto`.

    References
    ----------
    .. [1] Naveau, P., Huser, R., Ribereau, P., & Hannart, A. (2016). Modeling
        jointly low, moderate, and heavy rainfall intensities without a threshold
        selection. Water Resources Research, 52(4), 2753-2769.
        https://doi.org/10.1002/2015WR018552
    .. [2] Papastathopoulos, I., & Tawn, J. A. (2013). Extended generalised
        Pareto models for tail estimation. Journal of Statistical Planning and
        Inference, 143(1), 131-143.
        https://arxiv.org/abs/1111.6899

    Notes
    -----
    All of the Notes on :class:`GenPareto` apply here too. The :math:`r`-th moment
    is finite iff :math:`\xi < 1/r`, as for the GPD.

    For positive :math:`\xi`, tail-only data can make :math:`\sigma` and
    :math:`\kappa` hard to identify separately. In the small-:math:`\sigma`,
    large-:math:`\kappa` regime, observations above the threshold are informed
    mainly by a combined tail-survival scale rather than by the two parameters
    individually, making the fit sensitive to very broad :math:`\sigma` priors.

    .. plot::
        :context: close-figs
        :include-source: False

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        plt.style.use("arviz-darkgrid")

        def gpd_log_survival(x, sigma, xi):
            return -(1.0 / xi) * np.log1p(xi * x / sigma)

        def gpd_log_pdf(x, sigma, xi):
            return -np.log(sigma) - (1.0 / xi + 1.0) * np.log1p(xi * x / sigma)

        def extgpd_log_pdf(x, sigma, xi, kappa):
            log_s = gpd_log_survival(x, sigma, xi)
            log_g = np.log1p(-np.exp(log_s))
            return np.log(kappa) + (kappa - 1.0) * log_g + gpd_log_pdf(x, sigma, xi)

        def boundary_log_pdf(x, xi, lam):
            alpha = 1.0 / xi
            return (
                np.log(lam)
                - (alpha + 1.0) * np.log(xi)
                - (alpha + 1.0) * np.log(x)
                - lam * (xi * x) ** (-alpha)
            )

        x = np.linspace(1e-3, 8.0, 1200)
        xi = 1.5
        lam = 2.0
        alpha = 1.0 / xi
        sigmas = [0.7, 0.2, 0.05, 0.01]

        plt.plot(x, np.exp(boundary_log_pdf(x, xi, lam)), "k--", lw=2.5, label="limit")
        colors = plt.cm.plasma(np.linspace(0.08, 0.82, len(sigmas)))
        for sigma, color in zip(sigmas, colors):
            kappa = lam / sigma**alpha
            plt.plot(
                x,
                np.exp(extgpd_log_pdf(x, sigma, xi, kappa)),
                color=color,
                lw=1.9,
                label=rf"$\sigma={sigma:g}$, $\kappa={kappa:.1f}$",
            )
        plt.axvline(0.0, color="0.25", lw=1.2, ls=":")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("density", fontsize=12)
        plt.title(r"ExtGPD non-identifiability between $\sigma$ and $\kappa$")
        plt.xlim(0.0, 8.0)
        plt.ylim(bottom=0.0)
        plt.legend(
            fontsize=8,
            frameon=False,
            title=rf"$\mu=0$, $\xi={xi}$, $\kappa\sigma^{{1/\xi}}={lam:g}$",
            title_fontsize=9,
        )
        plt.show()

    Examples
    --------
    Fitting exceedances over a known threshold, with :math:`\kappa` as an
    additional body shape:

    .. code-block:: python

        import numpy as np
        import pymc as pm
        from pymc_extras.distributions import ExtGenPareto

        exceedances = data[data > threshold]
        xmax = exceedances.max()
        excess = exceedances - threshold
        s_hat = np.median(excess) / np.log(2.0)

        with pm.Model():
            # Regularize sigma to the observed excess scale; see recommendations.
            pareto_sigma = pm.LogNormal("pareto_sigma", mu=np.log(s_hat), sigma=1.0)
            # Center kappa on the GPD case while allowing broad body-shape variation.
            pareto_kappa = pm.LogNormal("pareto_kappa", mu=0.0, sigma=1.0)
            # xi lower bound: see modeling recommendations below.
            pareto_xi = pm.TruncatedNormal(
                "pareto_xi",
                mu=0.0,
                sigma=2.0,
                lower=pm.math.maximum(-1.0, -pareto_sigma / (xmax - threshold)),
            )
            ExtGenPareto(
                "obs",
                mu=threshold,
                sigma=pareto_sigma,
                xi=pareto_xi,
                kappa=pareto_kappa,
                observed=exceedances,
            )
            idata = pm.sample()

    .. _extgenpareto-modeling-recommendations:

    .. rubric:: Modeling recommendations

    For :math:`\xi`, use the same lower-bound rule as :ref:`GenPareto
    <genpareto-modeling-recommendations>`: combine the data-in-support constraint
    with the :math:`-1` floor.

    - **Regularize** :math:`\sigma` **to the excess scale.** :math:`\sigma` carries the
      data's units, so center its prior on the observed excess scale rather than on an
      arbitrary unit scale. A simple closed-form default is
      :math:`\hat{s}=\operatorname{median}(x-\mu)/\log 2`, the median match for the
      exponential GPD carrier (:math:`\xi=0`, :math:`\kappa=1`), with
      ``LogNormal(log(s_hat), 1)``. This gives a wide prior on the observed scale
      without encouraging the small-:math:`\sigma`, large-:math:`\kappa` trade
      described in the Notes.

    - **Center** :math:`\kappa` **on the GPD case.** ``LogNormal(0, 1)`` is symmetric in
      :math:`\log\kappa` about :math:`\kappa = 1` and allows a broad range of body shapes
      while enforcing :math:`\kappa` positive. :math:`\kappa` is mainly informed by the body of
      the exceedance distribution; tail-dominated data can leave it correlated with
      :math:`\sigma`. Inspect its posterior and prefer the GPD when :math:`\kappa`
      remains compatible with :math:`1`.

    - **Keep** :math:`\mu` **fixed at the threshold.** In a peaks-over-threshold
      fit, :math:`\mu` is chosen in advance. Estimating it is especially unstable
      for ExtGPD because a :math:`\kappa < 1` density diverges at the lower
      endpoint :math:`x = \mu`. If :math:`\mu` must be estimated, floor
      :math:`\kappa \geq 1` or put a prior on :math:`\min(\text{data}) - \mu`
      that vanishes at :math:`0`.
    """

    rv_type = ExtGenParetoRV
    rv_op = ExtGenParetoRV.rv_op

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, kappa=1, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))
        kappa = pt.as_tensor_variable(floatX(kappa))
        return super().dist([mu, sigma, xi, kappa], **kwargs)

    def logp(value, mu, sigma, xi, kappa):
        return check_parameters(
            extgenpareto.logpdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def logcdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            extgenpareto.logcdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def logccdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            extgenpareto.logsf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def icdf(value, mu, sigma, xi, kappa):
        res = extgenpareto.ppf(value, mu, sigma, xi, kappa)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def support_point(rv, size, mu, sigma, xi, kappa):
        # Support point = backward(0): the median, which the logit-CDF maps to y = 0 (F = 0.5).
        # backward floors it just above mu, so when the median rounds onto mu for small kappa,
        # forward stays finite (it would be -inf exactly at mu).
        point = _ExtGenParetoLogitCDF().backward(
            pt.zeros_like(mu), None, None, mu, sigma, xi, kappa
        )
        if not rv_size_is_none(size):
            point = pt.full(size, point)
        return point


class _ExtGenParetoLogitCDF(_AbstractGPDLogitCDFTransform):
    """``y = logit(F(x))`` transform for :class:`ExtGenPareto`."""

    _logp = staticmethod(extgenpareto.logpdf)
    _logcdf = staticmethod(extgenpareto.logcdf)
    _logccdf = staticmethod(extgenpareto.logsf)

    @staticmethod
    def _excess_from_logit(value, mu, sigma, xi, kappa):
        return extgenpareto._ext_gpd_excess_from_logit(value, kappa)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi, kappa):
        return genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)


@_default_transform.register(ExtGenPareto)
def _extgenpareto_default_transform(op, rv):
    return _ExtGenParetoLogitCDF()
