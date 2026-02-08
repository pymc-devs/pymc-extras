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
from pymc.testing import (BaseTestDistributionRandom, Domain, R, Rplus,
                          Rplusbig, assert_support_point_is_expected,
                          check_logcdf, check_logp,
                          seeded_scipy_distribution_builder,
                          select_by_precision)

# the distributions to be tested
from pymc_extras.distributions import Chi, GenExtreme, Maxwell

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
            (
                1,
                np.arange(1, 4),
                0.1,
                None,
                1 + np.arange(1, 4) * (1.1**-0.1 - 1) / 0.1,
            ),
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
                    * (
                        (1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6)
                        - 1
                    )
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
