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
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats

from pymc.logprob.utils import ParameterValueError
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    I,
    Rplus,
    assert_support_point_is_expected,
    check_logp,
    discrete_random_tester,
)
from pytensor import config

from pymc_extras.distributions import (
    BetaNegativeBinomial,
    GeneralizedPoisson,
    ShiftedBetaGeometric,
    Skellam,
)


class TestGeneralizedPoisson:
    class TestRandomVariable(BaseTestDistributionRandom):
        pymc_dist = GeneralizedPoisson
        pymc_dist_params = {"mu": 4.0, "lam": 1.0}
        expected_rv_op_params = {"mu": 4.0, "lam": 1.0}
        tests_to_run = [
            "check_pymc_params_match_rv_op",
            "check_rv_size",
        ]

        def test_random_matches_poisson(self):
            discrete_random_tester(
                dist=self.pymc_dist,
                paramdomains={"mu": Rplus, "lam": Domain([0], edges=(None, None))},
                ref_rand=lambda mu, lam, size: scipy.stats.poisson.rvs(mu, size=size),
            )

        @pytest.mark.parametrize("mu", (2.5, 20, 50))
        def test_random_lam_expected_moments(self, mu):
            lam = np.array([-0.9, -0.7, -0.2, 0, 0.2, 0.7, 0.9])
            dist = self.pymc_dist.dist(mu=mu, lam=lam, size=(10_000, len(lam)))
            draws = dist.eval()

            expected_mean = mu / (1 - lam)
            np.testing.assert_allclose(draws.mean(0), expected_mean, rtol=1e-1)

            expected_std = np.sqrt(mu / (1 - lam) ** 3)
            np.testing.assert_allclose(draws.std(0), expected_std, rtol=1e-1)

    def test_logp_matches_poisson(self):
        # We are only checking this distribution for lambda=0 where it's equivalent to Poisson.
        mu = pt.scalar("mu")
        lam = pt.scalar("lam")
        value = pt.vector("value", dtype="int64")

        logp = pm.logp(GeneralizedPoisson.dist(mu, lam), value)
        logp_fn = pytensor.function([value, mu, lam], logp)

        test_value = np.array([0, 1, 2, 30])
        for test_mu in (0.01, 0.1, 0.9, 1, 1.5, 20, 100):
            np.testing.assert_allclose(
                logp_fn(test_value, test_mu, lam=0),
                scipy.stats.poisson.logpmf(test_value, test_mu),
                rtol=1e-7 if config.floatX == "float64" else 1e-5,
            )

        # Check out-of-bounds values
        value = pt.scalar("value")
        logp = pm.logp(GeneralizedPoisson.dist(mu, lam), value)
        logp_fn = pytensor.function([value, mu, lam], logp)

        logp_fn(-1, mu=5, lam=0) == -np.inf
        logp_fn(9, mu=5, lam=-1) == -np.inf

        # Check mu/lam restrictions
        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=1, lam=2)

        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=0, lam=0)

        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=1, lam=-1)

    def test_logp_lam_expected_moments(self):
        mu = 30
        lam = np.array([-0.9, -0.7, -0.2, 0, 0.2, 0.7, 0.9])
        with pm.Model():
            x = GeneralizedPoisson("x", mu=mu, lam=lam)
            trace = pm.sample(chains=1, draws=10_000, random_seed=96).posterior

        expected_mean = mu / (1 - lam)
        np.testing.assert_allclose(trace["x"].mean(("chain", "draw")), expected_mean, rtol=1e-1)

        expected_std = np.sqrt(mu / (1 - lam) ** 3)
        np.testing.assert_allclose(trace["x"].std(("chain", "draw")), expected_std, rtol=1e-1)

    @pytest.mark.parametrize(
        "mu, lam, size, expected",
        [
            (50, [-0.6, 0, 0.6], None, np.floor(50 / (1 - np.array([-0.6, 0, 0.6])))),
            ([5, 50], -0.1, (4, 2), np.full((4, 2), np.floor(np.array([5, 50]) / 1.1))),
        ],
    )
    def test_moment(self, mu, lam, size, expected):
        with pm.Model() as model:
            GeneralizedPoisson("x", mu=mu, lam=lam, size=size)
        assert_support_point_is_expected(model, expected)


class TestBetaNegativeBinomial:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        """

        Beta Negative Binomial logp function test values taken from R package as
        there is currently no implementation in scipy.
        https://github.com/scipy/scipy/issues/17330

        The test values can be generated in R with the following code:

        .. code-block:: r

            library(extraDistr)

            create.test.rows <- function(alpha, beta, r, x) {
                logp <- dbnbinom(x, alpha, beta, r, log=TRUE)
                paste0("(", paste(alpha, beta, r, x, logp, sep=", "), ")")
            }

            x <- c(0, 1, 250, 5000)
            print(create.test.rows(1, 1, 1, x), quote=FALSE)
            print(create.test.rows(1, 1, 10, x), quote=FALSE)
            print(create.test.rows(1, 10, 1, x), quote=FALSE)
            print(create.test.rows(10, 1, 1, x), quote=FALSE)
            print(create.test.rows(10, 10, 10, x), quote=FALSE)

        """
        alpha, beta, r, value = pt.scalars("alpha", "beta", "r", "value")
        logp = pm.logp(BetaNegativeBinomial.dist(alpha, beta, r), value)
        logp_fn = pytensor.function([value, alpha, beta, r], logp)

        tests = [
            # 1, 1, 1
            (1, 1, 1, 0, -0.693147180559945),
            (1, 1, 1, 1, -1.79175946922805),
            (1, 1, 1, 250, -11.0548820266432),
            (1, 1, 1, 5000, -17.0349862828565),
            # 1, 1, 10
            (1, 1, 10, 0, -2.39789527279837),
            (1, 1, 10, 1, -2.58021682959232),
            (1, 1, 10, 250, -8.82261694534392),
            (1, 1, 10, 5000, -14.7359968760473),
            # 1, 10, 1
            (1, 10, 1, 0, -2.39789527279837),
            (1, 10, 1, 1, -2.58021682959232),
            (1, 10, 1, 250, -8.82261694534418),
            (1, 10, 1, 5000, -14.7359968760446),
            # 10, 1, 1
            (10, 1, 1, 0, -0.0953101798043248),
            (10, 1, 1, 1, -2.58021682959232),
            (10, 1, 1, 250, -43.5891148758123),
            (10, 1, 1, 5000, -76.2953173311091),
            # 10, 10, 10
            (10, 10, 10, 0, -5.37909807285049),
            (10, 10, 10, 1, -4.17512526852455),
            (10, 10, 10, 250, -21.781591505836),
            (10, 10, 10, 5000, -53.4836799634603),
        ]
        for test_alpha, test_beta, test_r, test_value, expected_logp in tests:
            np.testing.assert_allclose(
                logp_fn(test_value, test_alpha, test_beta, test_r), expected_logp
            )


class TestSkellam:
    def test_logp(self):
        # Scipy Skellam underflows to -inf earlier than PyMC
        Rplus_small = Domain([0, 0.01, 0.1, 0.9, 0.99, 1, 1.5, 2, 10, np.inf])
        # Suppress warnings coming from Scipy logpmf implementation
        with np.errstate(divide="ignore", invalid="ignore"):
            check_logp(
                Skellam,
                I,
                {"mu1": Rplus_small, "mu2": Rplus_small},
                lambda value, mu1, mu2: scipy.stats.skellam.logpmf(value, mu1, mu2),
            )


class TestShiftedBetaGeometric:
    class TestRandomVariable(BaseTestDistributionRandom):
        pymc_dist = ShiftedBetaGeometric
        pymc_dist_params = {"alpha": 1.0, "beta": 1.0}
        expected_rv_op_params = {"alpha": 1.0, "beta": 1.0}
        tests_to_run = [
            "check_pymc_params_match_rv_op",
            "check_rv_size",
        ]

        def test_random_basic_properties(self):
            """Test basic random sampling properties"""
            # Test with standard parameter values
            alpha_vals = [1.0, 0.5, 2.0]
            beta_vals = [0.5, 1.0, 2.0]

            for alpha in alpha_vals:
                for beta in beta_vals:
                    dist = self.pymc_dist.dist(alpha=alpha, beta=beta, size=1000)
                    draws = dist.eval()

                    # Check basic properties
                    assert np.all(draws > 0)
                    assert np.all(draws.astype(int) == draws)
                    assert np.mean(draws) > 0
                    assert np.var(draws) > 0

        def test_random_edge_cases(self):
            """Test edge cases with more reasonable parameter values"""
            # Test with small beta and large alpha values
            beta_vals = [0.1, 0.5]
            alpha_vals = [5.0, 10.0]

            for beta in beta_vals:
                for alpha in alpha_vals:
                    dist = self.pymc_dist.dist(alpha=alpha, beta=beta, size=1000)
                    draws = dist.eval()

                    # Check basic properties
                    assert np.all(draws > 0)
                    assert np.all(draws.astype(int) == draws)
                    assert np.mean(draws) > 0
                    assert np.var(draws) > 0

        @pytest.mark.parametrize(
            "alpha",
            [
                (0.5, 1.0, 10.0),
            ],
        )
        def test_random_moments(self, alpha):
            beta = np.array([0.5, 1.0, 10.0])
            dist = self.pymc_dist.dist(alpha=alpha, beta=beta, size=(10_000, len(beta)))
            draws = dist.eval()

            assert np.all(draws > 0)
            assert np.all(draws.astype(int) == draws)
            assert np.mean(draws) > 0
            assert np.var(draws) > 0

    def test_logp(self):
        # Create PyTensor variables with explicit values to ensure proper initialization
        alpha = pt.scalar("alpha")
        beta = pt.scalar("beta")
        value = pt.vector("value", dtype="int64")

        # Compile logp function for testing
        dist = ShiftedBetaGeometric.dist(alpha, beta)
        logp = pm.logp(dist, value)
        logp_fn = pytensor.function([value, alpha, beta], logp)

        # Test basic properties of logp
        test_value = np.array([1, 2, 3, 4, 5])
        logp_vals = logp_fn(test_value, 1.2, 3.4)
        assert not np.any(np.isnan(logp_vals))
        assert np.all(np.isfinite(logp_vals))

        assert logp_fn(-1, alpha=5, beta=1) == -np.inf

        # Check alpha/beta restrictions
        with pytest.raises(ParameterValueError):
            logp_fn(1, alpha=-1, beta=2)
        with pytest.raises(ParameterValueError):
            logp_fn(1, alpha=0, beta=0)
        with pytest.raises(ParameterValueError):
            logp_fn(1, alpha=1, beta=-1)

    def test_logp_matches_paper_alpha1_beta1(self):
        # For alpha=1, beta=1, P(T=t) = 1 / (t (t+1)) → logp = -log(t(t+1))
        # Derived from B(2, t) / B(1,1); see Appendix B (B3)
        # Reference: Fader & Hardie (2007), Appendix B [Figure B1, expression (B3)]
        # https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
        alpha = 1.0
        beta = 1.0
        t_vec = np.array([1, 2, 3, 4, 5], dtype="int64")
        expected = -np.log(t_vec * (t_vec + 1))

        alpha_sym = pt.scalar()
        beta_sym = pt.scalar()
        value = pt.vector(dtype="int64")
        dist = ShiftedBetaGeometric.dist(alpha_sym, beta_sym)
        logp = pm.logp(dist, value)
        fn = pytensor.function([value, alpha_sym, beta_sym], logp)
        np.testing.assert_allclose(fn(t_vec, alpha, beta), expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "alpha, beta, size, expected_shape",
        [
            (1.0, 1.0, None, ()),  # Scalar output
            ([1.0, 2.0], 1.0, None, (2,)),  # Vector output from alpha
            (1.0, [1.0, 2.0], None, (2,)),  # Vector output from beta
            ([1.0, 2.0], [1.0, 2.0], None, (2,)),  # Vector output from alpha and beta
            (1.0, [1.0, 2.0], (1, 2), (1, 2)),  # Explicit size with scalar alpha and vector beta
        ],
    )
    def test_support_point(self, alpha, beta, size, expected_shape):
        """Test that support_point returns reasonable values with correct shapes"""
        with pm.Model() as model:
            ShiftedBetaGeometric("x", alpha=alpha, beta=beta, size=size)

        init_point = model.initial_point()["x"]

        # Check shape
        assert init_point.shape == expected_shape

        # Check values are positive integers
        assert np.all(init_point > 0)
        # assert np.all(init_point.astype(int) == init_point)

        # Check values are finite and reasonable
        assert np.all(np.isfinite(init_point))
        assert np.all(init_point < 1e6)  # Should not be extremely large

        # TODO: expected values must be provided
        assert_support_point_is_expected(model, init_point)

    # TODO: Adapt this to ShiftedBetaGeometric and delete above test?
    @pytest.mark.parametrize(
        "mu, lam, size, expected",
        [
            (50, [-0.6, 0, 0.6], None, np.floor(50 / (1 - np.array([-0.6, 0, 0.6])))),
            ([5, 50], -0.1, (4, 2), np.full((4, 2), np.floor(np.array([5, 50]) / 1.1))),
        ],
    )
    def test_moment(self, mu, lam, size, expected):
        with pm.Model() as model:
            GeneralizedPoisson("x", mu=mu, lam=lam, size=size)
        assert_support_point_is_expected(model, expected)
