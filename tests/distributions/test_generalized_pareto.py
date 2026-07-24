#   Copyright 2020 The PyMC Developers
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
import pytensor
import pytensor.tensor as pt

# general imports
import pytest
import scipy.stats.distributions as sp


# test support imports from pymc
from pymc.distributions.distribution import support_point as _support_point
from pymc.logprob.utils import ParameterValueError
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_icdf,
    check_logccdf,
    check_logcdf,
    check_logp,
    check_selfconsistency_icdf,
    select_by_precision,
)
from scipy import stats

# the distributions to be tested
from pymc_extras.distributions import ExtGenPareto, GenPareto

# xi is unconstrained, so (None, None) edges skip the harness's invalid-edge probe.
# Bounded to (-1, 1]: xi <= -1 diverges at the wall (see TestGenParetoBoundaries) and
# xi > 1 pushes the q=0.99 quantile past check_icdf's absolute tolerance.
XI_DOMAIN = Domain([-0.9, -0.5, -0.1, 0, 0.1, 0.5, 1], dtype="float64", edges=(None, None))
# Leading 0 lets the harness probe the invalid kappa <= 0; trailing inf is unbounded.
KAPPA_DOMAIN = Domain([0, 0.25, 0.5, 1, 2, 5, np.inf], dtype="float64")
# sigma capped so check_icdf's absolute tolerance holds on the heavy-tail quantiles.
SIGMA_ICDF = Domain([0, 0.1, 0.5, 1.0, 2.0, np.inf], dtype="float64")


def ref_ext_logp(value, mu, sigma, xi, kappa):
    z = (value - mu) / sigma
    if z < 0 or (1 + xi * z) <= 0:
        return -np.inf
    log_H = sp.genpareto.logcdf(z, c=xi)
    log_h = sp.genpareto.logpdf(z, c=xi) - np.log(sigma)
    return np.log(kappa) + (kappa - 1) * log_H + log_h


def ref_ext_logcdf(value, mu, sigma, xi, kappa):
    z = (value - mu) / sigma
    if z < 0:
        return -np.inf
    if xi < 0 and (1 + xi * z) <= 0:
        return 0.0
    return kappa * sp.genpareto.logcdf(z, c=xi)


def ref_ext_logccdf(value, mu, sigma, xi, kappa):
    # S = 1 - H ** kappa, with H the GPD CDF. Compute via the GPD log-CDF so the
    # reference stays accurate deep in the tail (where H -> 1 and the naive
    # 1 - H**kappa underflows to 0) -- exactly the regime logccdf is for.
    z = (value - mu) / sigma
    if z < 0:
        return 0.0
    if xi < 0 and (1 + xi * z) <= 0:
        return -np.inf
    return np.log(-np.expm1(kappa * sp.genpareto.logcdf(z, c=xi)))


def ref_ext_icdf(q, mu, sigma, xi, kappa):
    # G^{-1}(q) = H^{-1}(q ** (1/kappa)).
    return sp.genpareto.ppf(q ** (1 / kappa), c=xi, loc=mu, scale=sigma)


class TestGenParetoClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logpdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logcdf(self):
        check_logcdf(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logcdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logccdf(self):
        # log survival function: exact and tail-stable (vs the lossy
        # log1mexp(logcdf) fallback). Compared against scipy genpareto.logsf.
        check_logccdf(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logsf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_icdf(self):
        check_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            lambda q, mu, sigma, xi: sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_icdf_selfconsistency(self):
        # cdf(icdf(q)) == q, no scipy reference needed.
        check_selfconsistency_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            decimal=select_by_precision(float64=5, float32=3),
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0.0, 1.0, 0.0, None, np.log(2.0)),
            (2.0, 3.0, 0.5, None, 2.0 + 3.0 * (2.0**0.5 - 1) / 0.5),
            (0.0, 1.0, 0.0, (3,), np.full(3, np.log(2.0))),
        ],
    )
    def test_genpareto_support_point(self, mu, sigma, xi, size, expected):
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi, size=size)
        assert_support_point_is_expected(model, expected)

    def test_rng_matches_scipy(self):
        # Inverse-CDF sampling cannot match SciPy's draws element-wise (different
        # uniform stream), so compare distributionally via a KS test.
        for xi in (-0.3, 0.0, 0.4):
            draws = pm.draw(GenPareto.dist(mu=0.0, sigma=2.0, xi=xi, size=20_000), random_seed=11)
            ks = stats.kstest(draws, lambda v, xi=xi: sp.genpareto.cdf(v, c=xi, scale=2.0))
            assert ks.pvalue > 0.01


class TestGenPareto(BaseTestDistributionRandom):
    pymc_dist = GenPareto
    pymc_dist_params = {"mu": 0.0, "sigma": 2.0, "xi": 0.3}
    expected_rv_op_params = {"mu": 0.0, "sigma": 2.0, "xi": 0.3}
    tests_to_run = ["check_pymc_params_match_rv_op", "check_rv_size"]


class TestExtGenParetoClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logcdf(self):
        check_logcdf(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logcdf,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logccdf(self):
        check_logccdf(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logccdf,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_icdf(self):
        check_icdf(
            ExtGenPareto,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": SIGMA_ICDF,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_icdf,
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_icdf_selfconsistency(self):
        check_selfconsistency_icdf(
            ExtGenPareto,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": SIGMA_ICDF,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            decimal=select_by_precision(float64=5, float32=3),
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, kappa, size",
        [
            (0.0, 1.0, 0.0, 2.0, None),
            (1.0, 2.0, 0.3, 0.5, None),
            (0.0, 1.0, -0.2, 3.0, (4,)),
        ],
    )
    def test_extgenpareto_support_point(self, mu, sigma, xi, kappa, size):
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa, size=size)
        expected = ref_ext_icdf(0.5, mu, sigma, xi, kappa)
        if size is not None:
            expected = np.full(size, expected)
        assert_support_point_is_expected(model, expected)

    @pytest.mark.parametrize("kappa", [0.5, 0.01])
    def test_small_kappa_inverses_share_the_stable_excess(self, kappa):
        # icdf, the transform's backward, and support_point share one log1mexp carrier
        # inverse, so for small kappa they agree and stay strictly above mu instead of
        # collapsing onto it (a -log(-expm1(.)) form would round the tiny survival to 1,
        # sending the excess to 0 and the initial logp to -inf).
        mu, sigma = 0.0, 1.0
        median = ref_ext_icdf(0.5, mu, sigma, 0.0, kappa)
        assert median > mu

        icdf_half = float(
            pm.icdf(ExtGenPareto.dist(mu=mu, sigma=sigma, xi=0.0, kappa=kappa), 0.5).eval()
        )
        np.testing.assert_allclose(icdf_half, median, rtol=1e-9)

        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=0.0, kappa=kappa)
        rv = model.free_RVs[0]
        tr = model.rvs_to_transforms[rv]
        backward0 = float(tr.backward(np.array(0.0), *rv.owner.inputs).eval())
        np.testing.assert_allclose(backward0, median, rtol=1e-9)  # y = logit(0.5) = 0

        assert_support_point_is_expected(model, np.array(median))
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    @pytest.mark.parametrize("kappa", [1e-4, 1e-300])
    def test_support_point_floors_when_median_collapses(self, kappa):
        # support_point is backward(0) (the median, at transformed y = 0). For small kappa the
        # median rounds onto mu, so the floor keeps it just above mu and the transformed initial
        # point stays finite (it would be -inf exactly at mu). mu = 2 makes even kappa = 1e-4
        # collapse (median excess below ULP(mu)).
        mu, sigma = 2.0, 1.5
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=0.0, kappa=kappa)
        rv = model.free_RVs[0]
        sp = float(_support_point(rv).eval())
        finfo = np.finfo(np.float64)
        floor = abs(mu) * (8.0 * finfo.eps) + finfo.tiny  # same floor as the backward transform
        np.testing.assert_allclose(sp, mu + floor, rtol=1e-12)
        assert sp > mu
        # finite transformed initial point and logp for every kappa > 0
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_support_point_is_transformed_median(self):
        # For ordinary kappa the median is interior; as backward(0) it maps to the transformed
        # Logistic's center y = 0.
        with pm.Model() as model:
            ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.3, kappa=2.0)
        (y_init,) = model.initial_point().values()
        np.testing.assert_allclose(y_init, 0.0, atol=1e-9)

    def test_rng_matches_distribution(self):
        # No SciPy equivalent: check that the empirical CDF (the model's own
        # logcdf) of the draws is Uniform(0, 1) via a KS test.
        for xi, kappa in ((-0.2, 0.5), (0.0, 0.01), (0.0, 2.0), (0.3, 3.0)):
            dist = ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=kappa, size=20_000)
            draws = pm.draw(dist, random_seed=7)
            u = np.exp(
                pm.logcdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=kappa), draws).eval()
            )
            assert stats.kstest(u, "uniform").pvalue > 0.01


class TestExtGenPareto(BaseTestDistributionRandom):
    pymc_dist = ExtGenPareto
    pymc_dist_params = {"mu": 0.0, "sigma": 1.5, "xi": 0.2, "kappa": 2.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": 1.5, "xi": 0.2, "kappa": 2.0}
    tests_to_run = ["check_pymc_params_match_rv_op", "check_rv_size"]


class TestGenParetoBoundaries:
    """Explicit boundary / invalid-input behaviour for both GPD classes.

    Covers ``x = inf``, ``q = 0``, ``q = 1`` (with ``xi < 0``), and ``sigma`` /
    ``kappa`` out of range.
    """

    def test_logp_logcdf_at_infinity(self):
        # density at +inf is 0 (logp -inf); CDF at +inf is 1 (logcdf 0). The
        # xi=0 path is the delicate one: 0*inf is nan, so the +inf tail is pinned
        # explicitly. Batch the three xi into one dist so each method compiles once.
        xi = np.array([-0.5, 0.0, 0.5])
        for dist in (
            GenPareto.dist(mu=0.0, sigma=1.0, xi=xi),
            ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=2.0),
        ):
            assert np.all(pm.logp(dist, np.inf).eval() == -np.inf)
            assert np.all(pm.logcdf(dist, np.inf).eval() == 0.0)

    def test_icdf_endpoints(self):
        # q=0 -> mu (lower endpoint); q=1 -> finite upper bound (xi<0) or +inf.
        # Batch the three xi into one dist so each endpoint compiles once, not per xi.
        xi = np.array([-0.5, 0.0, 0.5])
        with np.errstate(divide="ignore"):  # xi = 0 -> inf upper bound, not a warning
            expected_hi = np.where(xi < 0, 1.0 - 2.0 / xi, np.inf)
        # ExtGPD shares the same endpoints (carrier maps 0->0, 1->1).
        gpd = GenPareto.dist(mu=1.0, sigma=2.0, xi=xi)
        ext = ExtGenPareto.dist(mu=1.0, sigma=2.0, xi=xi, kappa=np.array([2.0, 0.5, 3.0]))
        for dist in (gpd, ext):
            assert np.all(pm.icdf(dist, 0.0).eval() == 1.0)
            np.testing.assert_allclose(pm.icdf(dist, 1.0).eval(), expected_hi)

    def test_icdf_outside_unit_interval_is_nan(self):
        for q in (-0.1, 1.1):
            assert np.isnan(pm.icdf(GenPareto.dist(mu=0.0, sigma=1.0, xi=0.2), q).eval())
            ext = ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.2, kappa=2.0)
            assert np.isnan(pm.icdf(ext, q).eval())

    def test_invalid_sigma_raises(self):
        for sigma in (0.0, -1.0):
            with pytest.raises(ParameterValueError):
                pm.logp(GenPareto.dist(mu=0.0, sigma=sigma, xi=0.1), 1.0).eval()
            with pytest.raises(ParameterValueError):
                pm.logp(ExtGenPareto.dist(mu=0.0, sigma=sigma, xi=0.1, kappa=2.0), 1.0).eval()

    def test_invalid_kappa_raises(self):
        # kappa <= 0 and kappa = nan both fail the kappa > 0 check and raise.
        for kappa in (0.0, -1.0, np.nan):
            with pytest.raises(ParameterValueError):
                pm.logp(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.1, kappa=kappa), 1.0).eval()


class TestGenParetoHeavyTail:
    """Heavy-tail (xi > 1) coverage with *relative* tolerance.

    The generic ``check_icdf`` harness uses an absolute tolerance, so it cannot
    exercise xi > 1 -- the quantiles there are ~1e10 and dwarf any absolute
    bound. But xi > 1 is precisely the infinite-mean regime where the median
    ``support_point`` matters, so test it directly against SciPy with rtol.
    """

    @pytest.mark.parametrize("xi", [1.5, 3.0])
    def test_support_point_is_median_with_infinite_mean(self, xi):
        # mean is infinite for xi >= 1, so support_point must fall back to the
        # median (not the mean) -- check it equals the GPD median.
        mu, sigma = 1.0, 2.0
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi)
        expected = sp.genpareto.ppf(0.5, c=xi, loc=mu, scale=sigma)
        assert_support_point_is_expected(model, expected)

    def test_ext_support_point_median_infinite_mean(self):
        mu, sigma, xi, kappa = 0.0, 1.0, 2.0, 3.0
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa)
        expected = ref_ext_icdf(0.5, mu, sigma, xi, kappa)
        assert_support_point_is_expected(model, expected)


class TestGenParetoTransforms:
    """Both distributions register a default probability-integral transform.

    Without a transform, an unobserved (latent) GPD variable would be sampled on
    all of R, where every proposal below mu has -inf logp -- breaking NUTS. The
    transform maps to the (parameter-dependent) support, so sampling stays valid
    for both the heavy (xi >= 0) and bounded (xi < 0) regimes. A naive Interval
    transform would do this too, but its log-Jacobian is discontinuous in xi at
    0; the probability-integral transform (``y = logit(F(x))``) is C1 in every
    parameter, so it does not inject a gradient kink when xi is random.
    """

    def test_default_transform_is_registered(self):
        with pm.Model() as model:
            x = GenPareto("x", mu=5.0, sigma=1.0, xi=0.3)
            e = ExtGenPareto("e", mu=2.0, sigma=1.0, xi=-0.4, kappa=2.0)
        assert model.rvs_to_transforms[x] is not None
        assert model.rvs_to_transforms[e] is not None

    @pytest.mark.parametrize(
        "dist_kwargs, builder",
        [
            ({"mu": 0.0, "sigma": 1.5, "xi": -0.5}, GenPareto),
            ({"mu": 0.0, "sigma": 1.5, "xi": 0.0}, GenPareto),
            ({"mu": 0.0, "sigma": 1.5, "xi": 0.5}, GenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": -0.3, "kappa": 2.0}, ExtGenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3, "kappa": 2.0}, ExtGenPareto),
        ],
    )
    def test_transformed_density_integrates_to_one(self, dist_kwargs, builder):
        # The transform's log-Jacobian must be correct: exp(transformed logp)
        # integrates to 1 over the unconstrained line.
        from scipy.integrate import trapezoid

        with pm.Model() as model:
            builder("x", **dist_kwargs)
        y = model.value_vars[0]
        logp = pytensor.function([y], model.logp(sum=True))
        # The integrand is the smooth, exactly-Logistic transformed density, so 2001
        # points over +-30 integrate to ~1e-13 (tail truncation dominates).
        ys = np.linspace(-30, 30, 2001)
        density = np.exp(np.array([float(logp(yi)) for yi in ys]))
        np.testing.assert_allclose(trapezoid(density, ys), 1.0, atol=1e-3)

    @pytest.mark.parametrize(
        "builder, kwargs, ys, roundtrip",
        [
            # bounded xi<0: the y~37 sigmoid-saturation point; round-trips to ~45.
            (GenPareto, {"mu": 0.0, "sigma": 1.5, "xi": -0.5}, (-45.0, -37.0, 37.0, 45.0), True),
            # unbounded xi=0: exact arbitrarily far out.
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.0}, (-30.0, 100.0, 1000.0), False),
            # heavy xi>0: below the y~709/xi quantile overflow.
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.3}, (37.0, 200.0, 400.0), False),
            # bounded ExtGPD: carrier on the moving xi<0 wall.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": -0.3, "kappa": 0.5},
                (-45.0, -37.0, 37.0, 45.0),
                True,
            ),
            # ExtGPD deep tail: the y~745 log-F underflow, recovered survival-side.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 2.0},
                (-30.0, 100.0, 1000.0),
                False,
            ),
            # ExtGPD large kappa: the inverse must depend on kappa.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e8},
                (-30.0, 0.0, 30.0),
                True,
            ),
            # ExtGPD kappa<1: the small-kappa inverse keeps a tiny survival off 0.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e-2},
                (-5.0, 0.0, 40.0),
                True,
            ),
            # ExtGPD collapse: quantile collapses onto mu, still exact finite Logistic.
            (
                ExtGenPareto,
                {"mu": 2.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e-300},
                (-30.0, 0.0, 80.0),
                False,
            ),
        ],
        ids=[
            "bounded-gpd",
            "unbounded-gpd",
            "heavy-gpd",
            "bounded-ext",
            "unbounded-ext-deep",
            "ext-large-kappa",
            "ext-small-kappa",
            "ext-collapse",
        ],
    )
    def test_transformed_logp_is_logistic_where_representable(self, builder, kwargs, ys, roundtrip):
        # Where the quantile is representable the transformed logp equals Logistic(y),
        # x stays in support (x >= mu), and the map round-trips. Covers the sigmoid
        # saturation (y ~ 37), the deep tail, and ten orders of magnitude in kappa.
        mu = kwargs["mu"]
        with pm.Model() as model:
            x = builder("x", **kwargs)
        yv = model.value_vars[0]
        inputs = x.owner.inputs
        tr = model.rvs_to_transforms[x]
        logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        forward_backward = (
            pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
            if roundtrip
            else None
        )
        for y in ys:
            lp = float(logp(y))
            assert np.isfinite(lp), (kwargs, y)
            np.testing.assert_allclose(lp, -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y), atol=1e-6)
            xb = float(backward(y))
            assert np.isfinite(xb) and xb >= mu, (kwargs, y)  # support is [mu, ...)
            if forward_backward is not None:
                np.testing.assert_allclose(float(forward_backward(y)), y, atol=1e-6)

    def test_transformed_logp_robust_in_unoptimized_mode(self):
        # The transformed logp must not rely on the optimizer cancelling logp(backward)
        # against log_jac_det: in fast_compile the numbers are evaluated, and over the
        # whole reachable range (incl. the small-kappa collapse) it stays finite Logistic.
        fast = pytensor.compile.mode.Mode(linker="py", optimizer="fast_compile")
        cases = [
            (GenPareto, {"mu": 2.0, "sigma": 1.5, "xi": -0.5}),
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.3}),
            (ExtGenPareto, {"mu": 2.0, "sigma": 1.0, "xi": 0.0, "kappa": 0.0067}),
            (ExtGenPareto, {"mu": 0.0, "sigma": 1e-50, "xi": 0.0, "kappa": 0.0067}),
        ]
        for builder, kw in cases:
            with pm.Model() as model:
                builder("x", **kw)
            fn = pytensor.function([model.value_vars[0]], model.logp(sum=True), mode=fast)
            for y in np.linspace(-25.0, 25.0, 51):
                lp = float(fn(y))
                assert not np.isnan(lp), (builder.__name__, kw, y)
                logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
                np.testing.assert_allclose(lp, logistic, atol=1e-3)

    def test_small_kappa_collapse_saturates_with_exact_density(self):
        # For kappa << 1 the ExtGPD median sits ~0.5 ** (1/kappa) below mu, under
        # ulp(mu), so the whole bulk is a numerical point mass: distinct y all map to
        # the same floored x. The map is therefore NOT injective here -- but the
        # sampled density stays exactly Logistic and the readout stays in support.
        mu = 2.0
        with pm.Model() as model:
            x = ExtGenPareto("x", mu=mu, sigma=1.0, xi=0.0, kappa=0.01)
        yv = model.value_vars[0]
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
        ys = (-10.0, -5.0, 0.0)
        xs = [float(backward(y)) for y in ys]
        rts = [float(roundtrip(y)) for y in ys]
        # saturation: distinct y collapse onto one floored x, just above mu (in support)
        assert xs[0] == xs[1] == xs[2] > mu
        # so it is not injective here -- forward(backward(.)) is constant, not identity
        assert rts[0] == rts[1] == rts[2]
        # yet the sampled density is still exactly Logistic at each y
        for y in ys:
            np.testing.assert_allclose(
                float(logp(y)), -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y), atol=1e-2
            )

    def test_transform_finite_under_float32(self):
        # dtype-aware floor: the small-kappa lower-tail collapse must not NaN under
        # float32, where a literal 1e-300 floor underflows to 0 (leaving logp = +inf).
        fast = pytensor.compile.mode.Mode(linker="py", optimizer="fast_compile")
        with pytensor.config.change_flags(floatX="float32"):
            with pm.Model() as model:
                ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.0, kappa=1e-20)
            fn = pytensor.function([model.value_vars[0]], model.logp(sum=True), mode=fast)
            for y in (-10.0, 0.0, 10.0):
                lp = float(fn(np.float32(y)))
                assert not np.isnan(lp)
                logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
                np.testing.assert_allclose(lp, logistic, atol=1e-2)

    def test_transform_roundtrip_resolves_subnormal_kappa_tail(self):
        # The PyTensor helper owns the exact excess value; this checks the PyMC
        # transform still wires that stable inverse through backward/forward.
        with pm.Model() as model:
            x = ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.0, kappa=4e-309)
        yv = model.value_vars[0]
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
        got = float(roundtrip(710.0))
        assert np.isfinite(got)
        np.testing.assert_allclose(got, 710.0, rtol=0, atol=1e-9)

    def test_jacobian_gradient_is_continuous_through_xi_zero(self):
        # Why y = logit(F(x)) and not an Interval transform: with random xi the
        # transformed logp must be C1 in xi across 0; an Interval transform's Jacobian
        # jumps by ~1e12 at xi = 0, driving divergences.
        with pm.Model() as model:
            xi = pm.Normal("xi", 0.0, 1.0)
            GenPareto("x", mu=0.0, sigma=1.0, xi=xi)
        val_xi = next(v for v in model.value_vars if v.name == "xi")
        val_x = next(v for v in model.value_vars if v.name != "xi")
        logp = model.logp(sum=True)
        fn = pytensor.function([val_xi, val_x], pt.grad(logp, val_xi), on_unused_input="ignore")
        grad_minus = float(fn(-1e-6, 0.5))
        grad_plus = float(fn(1e-6, 0.5))
        assert abs(grad_minus - grad_plus) < 1e-3
