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

from decimal import Decimal, getcontext

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pymc_extras.distributions import pytensor_genpareto as GenPareto


def _gpd_ref_logp(value, mu, sigma, xi):
    """100-digit GPD logp from the exact margin ``s = 1 + xi*z``."""
    getcontext().prec = 100
    z = (Decimal(value) - mu) / sigma
    log_s = (1 + xi * z).ln()
    return -sigma.ln() - (1 + 1 / xi) * log_s


def _gpd_ref_grad(value, params, which):
    """High-precision central difference d logp / d ``which``."""
    h = abs(params[which]) * Decimal("1e-22") or Decimal("1e-22")
    hi = dict(params, **{which: params[which] + h})
    lo = dict(params, **{which: params[which] - h})
    return (_gpd_ref_logp(value, **hi) - _gpd_ref_logp(value, **lo)) / (2 * h)


def test_functional_api_is_self_consistent():
    x = np.array([0.3, 1.0, 3.0])
    params = (0.0, 1.0, 0.2)

    np.testing.assert_allclose(
        GenPareto.cdf(x, *params).eval(), np.exp(GenPareto.logcdf(x, *params).eval())
    )
    np.testing.assert_allclose(
        GenPareto.pdf(x, *params).eval(), np.exp(GenPareto.logpdf(x, *params).eval())
    )
    np.testing.assert_allclose(
        GenPareto.sf(x, *params).eval(), np.exp(GenPareto.logsf(x, *params).eval())
    )
    np.testing.assert_allclose(
        GenPareto.cdf(x, *params).eval() + GenPareto.sf(x, *params).eval(), 1.0, atol=1e-9
    )

    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        GenPareto.isf(q, *params).eval(), GenPareto.ppf(1 - q, *params).eval()
    )
    assert np.all(np.isnan(GenPareto.ppf(np.array([-0.1, 1.1]), *params).eval()))
    np.testing.assert_allclose(float(GenPareto.ppf(0.0, *params).eval()), 0.0)


def test_rvs_shape_dtype_and_random_state():
    params = (0.0, 1.0, 0.2)
    draws = GenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
    )
    out = draws.eval()
    assert out.shape == (4, 3) and out.dtype == np.float64
    assert np.all(out >= 0.0)

    same = GenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
    ).eval()
    diff = GenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(9))
    ).eval()
    np.testing.assert_array_equal(out, same)
    assert not np.array_equal(out, diff)


def test_logp_logcdf_at_infinity():
    x = pt.constant(np.inf)
    xi = pt.constant(np.array([-0.5, 0.0, 0.5]))
    assert np.all(GenPareto.logpdf(x, 0.0, 1.0, xi).eval() == -np.inf)
    assert np.all(GenPareto.logcdf(x, 0.0, 1.0, xi).eval() == 0.0)


def test_ppf_endpoints():
    xi = np.array([-0.5, 0.0, 0.5])
    with np.errstate(divide="ignore"):
        expected_hi = np.where(xi < 0, 1.0 - 2.0 / xi, np.inf)
    assert np.all(GenPareto.ppf(0.0, 1.0, 2.0, xi).eval() == 1.0)
    np.testing.assert_allclose(GenPareto.ppf(1.0, 1.0, 2.0, xi).eval(), expected_hi)


def test_ppf_outside_unit_interval_is_nan():
    for q in (-0.1, 1.1):
        assert np.isnan(GenPareto.ppf(q, 0.0, 1.0, 0.2).eval())


@pytest.mark.parametrize("xi", [1.5, 5.0])
def test_heavy_tail_logpdf_logcdf_ppf_match_scipy(xi):
    mu, sigma = 0.0, 1.3
    x = np.array([0.5, 2.0, 10.0, 1e4])
    np.testing.assert_allclose(
        GenPareto.logpdf(x, mu, sigma, xi).eval(),
        sp.genpareto.logpdf(x, c=xi, loc=mu, scale=sigma),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        GenPareto.logcdf(x, mu, sigma, xi).eval(),
        sp.genpareto.logcdf(x, c=xi, loc=mu, scale=sigma),
        rtol=1e-10,
    )
    q = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
    np.testing.assert_allclose(
        GenPareto.ppf(q, mu, sigma, xi).eval(),
        sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
        rtol=1e-9,
    )


def test_logsf_tail_is_stable():
    x = np.array([1e2, 1e4, 1e6, 1e8])
    for xi in (0.1, 0.3, 0.7):
        got = GenPareto.logsf(x, 0.0, 1.0, xi).eval()
        ref = sp.genpareto.logsf(x, c=xi)
        np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_isf_keeps_the_upper_tail():
    x = np.array([1e-8, 1e-12, 1e-15])
    xi = pt.constant(0.0)
    isf_val = GenPareto.isf(x, 0.0, 1.0, xi).eval()
    np.testing.assert_allclose(isf_val, -np.log(x), rtol=1e-12)
    naive = GenPareto.ppf(1 - x, 0.0, 1.0, xi).eval()
    assert abs(isf_val[-1] - -np.log(x[-1])) < abs(naive[-1] - -np.log(x[-1]))


def test_boundary_logp_value_holds_but_gradient_tracks_the_margin():
    mu, sigma, xi = 0.4, 1.3, -0.3
    eps = np.finfo(np.float64).eps
    margins = [1e-2, 1e-6, 1e-12]

    v, sig, xs = (pt.dscalar(n) for n in ("v", "sig", "xs"))
    logp = GenPareto.logpdf(v, mu, sig, xs)
    fn = pytensor.function([v, sig, xs], [logp, pt.grad(logp, sig), pt.grad(logp, xs)])

    params = {"mu": Decimal(mu), "sigma": Decimal(sigma), "xi": Decimal(xi)}
    prev_logp = np.inf
    for s_target in margins:
        value = mu + sigma * ((s_target - 1) / xi)
        logp_f, *grads_f = (float(o) for o in fn(value, sigma, xi))
        logp_ref = _gpd_ref_logp(value, **params)

        assert np.isfinite(logp_f), s_target
        assert all(np.isfinite(g) for g in grads_f), s_target
        assert logp_f < prev_logp, s_target
        prev_logp = logp_f
        assert abs((Decimal(logp_f) - logp_ref) / logp_ref) < 1e-4, s_target

        grad_bound = 100 * eps / s_target
        for name, g in zip(["sigma", "xi"], grads_f):
            g_ref = _gpd_ref_grad(value, params, name)
            rel = abs((Decimal(g) - g_ref) / g_ref)
            assert np.sign(g) == np.sign(float(g_ref)), (name, s_target)
            assert rel < grad_bound, (name, s_target, float(rel))


def test_logp_gradient_is_continuous_through_xi_zero():
    xi = pt.dscalar("xi")
    logp = GenPareto.logpdf(pt.constant(2.3), 0.0, 1.0, xi)
    fn = pytensor.function([xi], [logp, pt.grad(logp, xi)], on_unused_input="ignore")

    _, grad0 = fn(0.0)
    assert np.isfinite(grad0)
    _, grad_minus = fn(-1e-7)
    _, grad_plus = fn(1e-7)
    assert abs(grad_minus - grad0) < 1e-5
    assert abs(grad_plus - grad0) < 1e-5
    h = 1e-5
    fd = (fn(h)[0] - fn(-h)[0]) / (2 * h)
    assert abs(grad0 - fd) < 1e-5


def test_value_continuous_through_xi_zero():
    value = np.linspace(0.05, 6.0, 50)
    lp0 = GenPareto.logpdf(value, 0.0, 2.0, 0.0).eval()
    for xi in (1e-9, -1e-9, 1e-6):
        lp = GenPareto.logpdf(value, 0.0, 2.0, xi).eval()
        np.testing.assert_allclose(lp, lp0, atol=1e-5)
    np.testing.assert_allclose(lp0, sp.expon.logpdf(value, scale=2.0), atol=1e-12)
