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

# general imports
import pytest
import scipy.stats.distributions as sp


# test support imports from pymc
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    R,
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_logcdf,
    check_logp,
    seeded_scipy_distribution_builder,
    select_by_precision,
)

# the distributions to be tested
from pymc_extras.distributions import Chi, ExtGenPareto, GenExtreme, GenPareto, Maxwell

pytestmark = pytest.mark.filterwarnings(
    "ignore:Numba will use object mode to run Generalized Extreme Value:UserWarning"
)


class TestGenExtremeClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.

    pm.logp(GenExtreme.dist(mu=0.,sigma=1.,xi=0.5),value=-0.01)
    """

    def test_logp(self):
        def ref_logp(value, mu, sigma, xi):
            if 1 + xi * (value - mu) / sigma > 0:
                return sp.genextreme.logpdf(value, c=-xi, loc=mu, scale=sigma)
            else:
                return -np.inf

        check_logp(
            GenExtreme,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-1, -0.99, -0.5, 0, 0.5, 0.99, 1]),
            },
            ref_logp,
        )

    def test_logcdf(self):
        def ref_logcdf(value, mu, sigma, xi):
            if 1 + xi * (value - mu) / sigma > 0:
                return sp.genextreme.logcdf(value, c=-xi, loc=mu, scale=sigma)
            else:
                return -np.inf

        check_logcdf(
            GenExtreme,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-1, -0.99, -0.5, 0, 0.5, 0.99, 1]),
            },
            ref_logcdf,
            decimal=select_by_precision(float64=6, float32=2),
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0, 1, 0, None, 0),
            (1, np.arange(1, 4), 0.1, None, 1 + np.arange(1, 4) * (1.1**-0.1 - 1) / 0.1),
            (np.arange(5), 1, 0.1, None, np.arange(5) + (1.1**-0.1 - 1) / 0.1),
            (
                0,
                1,
                np.linspace(-0.2, 0.2, 6),
                None,
                ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
                / np.linspace(-0.2, 0.2, 6),
            ),
            (1, 2, 0.1, 5, np.full(5, 1 + 2 * (1.1**-0.1 - 1) / 0.1)),
            (
                np.arange(6),
                np.arange(1, 7),
                np.linspace(-0.2, 0.2, 6),
                (3, 6),
                np.full(
                    (3, 6),
                    np.arange(6)
                    + np.arange(1, 7)
                    * ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
                    / np.linspace(-0.2, 0.2, 6),
                ),
            ),
        ],
    )
    def test_genextreme_support_point(self, mu, sigma, xi, size, expected):
        with pm.Model() as model:
            GenExtreme("x", mu=mu, sigma=sigma, xi=xi, size=size)
        assert_support_point_is_expected(model, expected)

    def test_gen_extreme_scipy_kwarg(self):
        dist = GenExtreme.dist(xi=1, scipy=False)
        assert dist.owner.inputs[-1].eval() == 1

        dist = GenExtreme.dist(xi=1, scipy=True)
        assert dist.owner.inputs[-1].eval() == -1


class TestGenExtreme(BaseTestDistributionRandom):
    pymc_dist = GenExtreme
    pymc_dist_params = {"mu": 0, "sigma": 1, "xi": -0.1}
    expected_rv_op_params = {"mu": 0, "sigma": 1, "xi": -0.1}
    # Notice, using different parametrization of xi sign to scipy
    reference_dist_params = {"loc": 0, "scale": 1, "c": 0.1}
    reference_dist = seeded_scipy_distribution_builder("genextreme")
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestGenParetoClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        def ref_logp(value, mu, sigma, xi):
            scaled = (value - mu) / sigma
            if scaled < 0:
                return -np.inf
            if xi < 0 and scaled > -1 / xi:
                return -np.inf
            if 1 + xi * scaled <= 0:
                return -np.inf
            return sp.genpareto.logpdf(value, c=xi, loc=mu, scale=sigma)

        check_logp(
            GenPareto,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-0.5, -0.1, 0, 0.1, 0.5, 1]),
            },
            ref_logp,
            # GPD has no mathematical constraint on xi (unlike GEV), only sigma > 0
            skip_paramdomain_outside_edge_test=True,
        )

    def test_logcdf(self):
        def ref_logcdf(value, mu, sigma, xi):
            scaled = (value - mu) / sigma
            if scaled < 0:
                return -np.inf
            if xi < 0 and scaled > -1 / xi:
                return 0.0  # log(1) at upper bound
            return sp.genpareto.logcdf(value, c=xi, loc=mu, scale=sigma)

        check_logcdf(
            GenPareto,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-0.5, -0.1, 0, 0.1, 0.5, 1]),
            },
            ref_logcdf,
            decimal=select_by_precision(float64=6, float32=2),
            # GPD has no mathematical constraint on xi (unlike GEV), only sigma > 0
            skip_paramdomain_outside_edge_test=True,
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0, 1, 0, None, np.log(2)),  # Exponential case: median = log(2)
            (0, 1, 0.5, None, (2**0.5 - 1) / 0.5),  # median = (2^xi - 1) / xi
            (0, 1, -0.5, None, (2**-0.5 - 1) / -0.5),
            (1, 2, 0, None, 1 + 2 * np.log(2)),  # mu + sigma * log(2)
            (0, 1, 0.5, 5, np.full(5, (2**0.5 - 1) / 0.5)),
            (
                np.arange(3),
                np.arange(1, 4),
                0.5,
                (2, 3),
                np.full((2, 3), np.arange(3) + np.arange(1, 4) * (2**0.5 - 1) / 0.5),
            ),
        ],
    )
    def test_genpareto_support_point(self, mu, sigma, xi, size, expected):
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi, size=size)
        assert_support_point_is_expected(model, expected)


class TestGenPareto(BaseTestDistributionRandom):
    pymc_dist = GenPareto
    pymc_dist_params = {"mu": 0, "sigma": 1, "xi": 0.1}
    expected_rv_op_params = {"mu": 0, "sigma": 1, "xi": 0.1}
    # GenPareto uses same xi sign convention as scipy
    reference_dist_params = {"loc": 0, "scale": 1, "c": 0.1}
    reference_dist = seeded_scipy_distribution_builder("genpareto")
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestExtGenParetoClass:
    """
    Tests for the Extended Generalized Pareto Distribution (ExtGPD).
    ExtGPD has CDF G(x) = H(x)^kappa where H is the GPD CDF.
    """

    def test_logp(self):
        def ref_logp(value, mu, sigma, xi, kappa):
            # ExtGPD pdf: g(x) = kappa * H(x)^(kappa-1) * h(x)
            # where H is GPD CDF and h is GPD PDF
            scaled = (value - mu) / sigma
            if scaled < 0:
                return -np.inf
            if xi < 0 and scaled > -1 / xi:
                return -np.inf
            if 1 + xi * scaled <= 0:
                return -np.inf

            H = sp.genpareto.cdf(value, c=xi, loc=mu, scale=sigma)
            if H <= 0:
                return -np.inf
            log_h = sp.genpareto.logpdf(value, c=xi, loc=mu, scale=sigma)
            return np.log(kappa) + (kappa - 1) * np.log(H) + log_h

        check_logp(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0, 0, 0, 0], edges=(0, 0)),
                "sigma": Rplusbig,
                "xi": Domain([-0.3, 0, 0.1, 0.5]),
                "kappa": Domain([0.5, 1.0, 2.0, 5.0]),
            },
            ref_logp,
            skip_paramdomain_outside_edge_test=True,
        )

    def test_logcdf(self):
        def ref_logcdf(value, mu, sigma, xi, kappa):
            # ExtGPD CDF: G(x) = H(x)^kappa
            # log CDF = kappa * log(H(x))
            scaled = (value - mu) / sigma
            if scaled < 0:
                return -np.inf
            if xi < 0 and scaled > -1 / xi:
                return 0.0  # log(1) at upper bound

            H = sp.genpareto.cdf(value, c=xi, loc=mu, scale=sigma)
            if H <= 0:
                return -np.inf
            return kappa * np.log(H)

        check_logcdf(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0, 0, 0, 0], edges=(0, 0)),
                "sigma": Rplusbig,
                "xi": Domain([-0.3, 0, 0.1, 0.5]),
                "kappa": Domain([0.5, 1.0, 2.0, 5.0]),
            },
            ref_logcdf,
            decimal=select_by_precision(float64=6, float32=2),
            skip_paramdomain_outside_edge_test=True,
        )

    def test_kappa_one_equals_gpd(self):
        """When kappa=1, ExtGPD should equal GPD."""

        # Create ExtGPD with kappa=1
        ext_dist = ExtGenPareto.dist(mu=0, sigma=1, xi=0.2, kappa=1)
        gpd_dist = GenPareto.dist(mu=0, sigma=1, xi=0.2)

        test_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

        ext_logp = pm.logp(ext_dist, test_values).eval()
        gpd_logp = pm.logp(gpd_dist, test_values).eval()

        np.testing.assert_allclose(ext_logp, gpd_logp, rtol=1e-6)

    @pytest.mark.parametrize(
        "mu, sigma, xi, kappa, size, expected",
        [
            # kappa=1 should give GPD median: sigma * (2^xi - 1) / xi
            (0, 1, 0.5, 1.0, None, (2**0.5 - 1) / 0.5),
            # kappa=1, xi=0: exponential median = log(2)
            (0, 1, 0, 1.0, None, np.log(2)),
            # kappa=2: H(m) = 0.5^0.5, so m = sigma * [(1-0.5^0.5)^(-xi) - 1] / xi
            (0, 1, 0.5, 2.0, None, ((1 - 0.5**0.5) ** (-0.5) - 1) / 0.5),
            # With size
            (0, 1, 0.5, 1.0, 5, np.full(5, (2**0.5 - 1) / 0.5)),
        ],
    )
    def test_extgenpareto_support_point(self, mu, sigma, xi, kappa, size, expected):
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa, size=size)
        assert_support_point_is_expected(model, expected)


class TestExtGenPareto(BaseTestDistributionRandom):
    """Test random sampling for ExtGPD."""

    pymc_dist = ExtGenPareto
    pymc_dist_params = {"mu": 0, "sigma": 1, "xi": 0.1, "kappa": 2.0}
    expected_rv_op_params = {"mu": 0, "sigma": 1, "xi": 0.1, "kappa": 2.0}
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]

    def test_random_samples_follow_distribution(self):
        """Test that random samples follow the ExtGPD distribution using KS test."""
        rng = np.random.default_rng(42)
        mu, sigma, xi, kappa = 0, 1, 0.2, 2.0

        # Generate samples
        dist = ExtGenPareto.dist(mu=mu, sigma=sigma, xi=xi, kappa=kappa)
        samples = pm.draw(dist, draws=1000, random_seed=rng)

        # Define ExtGPD CDF for KS test
        def ext_gpd_cdf(x, mu, sigma, xi, kappa):
            H = sp.genpareto.cdf(x, c=xi, loc=mu, scale=sigma)
            return np.power(H, kappa)

        # KS test
        from scipy.stats import kstest

        stat, pvalue = kstest(samples, lambda x: ext_gpd_cdf(x, mu, sigma, xi, kappa))
        assert pvalue > 0.01, f"KS test failed with p-value {pvalue}"


class TestChiClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            Chi,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi.logpdf(value, df=nu),
        )

    def test_logcdf(self):
        check_logcdf(
            Chi,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi.logcdf(value, df=nu),
        )


class TestMaxwell:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            Maxwell,
            Rplus,
            {"a": Rplus},
            lambda value, a: sp.maxwell.logpdf(value, scale=a),
        )

    def test_logcdf(self):
        check_logcdf(
            Maxwell,
            Rplus,
            {"a": Rplus},
            lambda value, a: sp.maxwell.logcdf(value, scale=a),
        )
