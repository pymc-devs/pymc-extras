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

from scipy import stats

from pymc_extras.distributions import pytensor_extgenpareto as ExtGenPareto
from pymc_extras.distributions import pytensor_genpareto as GenPareto


def ref_ext_logpdf(value, mu, sigma, xi, kappa):
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


def ref_ext_logsf(value, mu, sigma, xi, kappa):
    z = (value - mu) / sigma
    if z < 0:
        return 0.0
    if xi < 0 and (1 + xi * z) <= 0:
        return -np.inf
    return np.log(-np.expm1(kappa * sp.genpareto.logcdf(z, c=xi)))


def ref_ext_ppf(q, mu, sigma, xi, kappa):
    return sp.genpareto.ppf(q ** (1 / kappa), c=xi, loc=mu, scale=sigma)


def _gpd_ref_logp(value, mu, sigma, xi, kappa):
    """100-digit ExtGPD logp from the exact margin ``s = 1 + xi*z``."""
    getcontext().prec = 100
    z = (Decimal(value) - mu) / sigma
    log_s = (1 + xi * z).ln()
    logp = -sigma.ln() - (1 + 1 / xi) * log_s
    log_H = (1 - ((Decimal(-1) / xi) * log_s).exp()).ln()
    return logp + kappa.ln() + (kappa - 1) * log_H


def _gpd_ref_grad(value, params, which):
    """High-precision central difference d logp / d ``which``."""
    h = abs(params[which]) * Decimal("1e-22") or Decimal("1e-22")
    hi = dict(params, **{which: params[which] + h})
    lo = dict(params, **{which: params[which] - h})
    return (_gpd_ref_logp(value, **hi) - _gpd_ref_logp(value, **lo)) / (2 * h)


def _ext_gpd_excess_ref(y, kappa):
    """High-precision GPD excess m = -log(1 - sigmoid(y) ** (1/kappa)) at xi = 0."""
    getcontext().prec = 500
    y, kappa = Decimal(y), Decimal(kappa)
    log_sigmoid = -(1 + (-y).exp()).ln()
    return float(-(1 - (log_sigmoid / kappa).exp()).ln())


def test_functional_api_is_self_consistent():
    x = np.array([0.3, 1.0, 3.0])
    params = (0.0, 1.0, 0.2, 1.5)

    np.testing.assert_allclose(
        ExtGenPareto.cdf(x, *params).eval(), np.exp(ExtGenPareto.logcdf(x, *params).eval())
    )
    np.testing.assert_allclose(
        ExtGenPareto.pdf(x, *params).eval(), np.exp(ExtGenPareto.logpdf(x, *params).eval())
    )
    np.testing.assert_allclose(
        ExtGenPareto.sf(x, *params).eval(), np.exp(ExtGenPareto.logsf(x, *params).eval())
    )
    np.testing.assert_allclose(
        ExtGenPareto.cdf(x, *params).eval() + ExtGenPareto.sf(x, *params).eval(),
        1.0,
        atol=1e-9,
    )

    q = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        ExtGenPareto.isf(q, *params).eval(), ExtGenPareto.ppf(1 - q, *params).eval()
    )
    assert np.all(np.isnan(ExtGenPareto.ppf(np.array([-0.1, 1.1]), *params).eval()))
    np.testing.assert_allclose(float(ExtGenPareto.ppf(0.0, *params).eval()), 0.0)


def test_isf_keeps_the_upper_tail():
    # isf uses log1p(-x) (= log F) directly; ppf(1 - x) forms 1 - x first and loses the tiny
    # survival x in the heavy upper tail. For xi = 0 the deep-tail isf is -log(x) + log(kappa);
    # the direct route tracks it (rtol covers the O(x) asymptotic gap at x = 1e-8) while the naive
    # ppf(1 - x) drifts off (~8e-4 at x = 1e-15).
    x = np.array([1e-8, 1e-12, 1e-15])
    for kappa in (0.5, 2.0, 10.0):
        ref = -np.log(x) + np.log(kappa)
        isf_val = ExtGenPareto.isf(x, 0.0, 1.0, 0.0, kappa).eval()
        np.testing.assert_allclose(isf_val, ref, rtol=1e-9)
        naive = ExtGenPareto.ppf(1 - x, 0.0, 1.0, 0.0, kappa).eval()
        assert abs(isf_val[-1] - ref[-1]) < abs(naive[-1] - ref[-1])


def test_rvs_shape_dtype_and_random_state():
    params = (0.0, 1.0, 0.2, 1.5)
    draws = ExtGenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
    )
    out = draws.eval()
    assert out.shape == (4, 3) and out.dtype == np.float64
    assert np.all(out >= 0.0)

    same = ExtGenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
    ).eval()
    diff = ExtGenPareto.rvs(
        *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(9))
    ).eval()
    np.testing.assert_array_equal(out, same)
    assert not np.array_equal(out, diff)


@pytest.mark.parametrize(
    "mu, sigma, xi, kappa",
    [
        (0.0, 1.0, -0.3, 0.5),
        (0.0, 1.0, 0.0, 2.0),
        (1.0, 2.0, 0.4, 3.0),
    ],
)
def test_logpdf_logcdf_logsf_ppf_match_references(mu, sigma, xi, kappa):
    upper = mu - sigma / xi if xi < 0 else 10.0
    x = np.linspace(mu + 0.05, upper - 0.05 if xi < 0 else upper, 20)
    q = np.array([0.1, 0.5, 0.9])

    np.testing.assert_allclose(
        ExtGenPareto.logpdf(x, mu, sigma, xi, kappa).eval(),
        [ref_ext_logpdf(v, mu, sigma, xi, kappa) for v in x],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        ExtGenPareto.logcdf(x, mu, sigma, xi, kappa).eval(),
        [ref_ext_logcdf(v, mu, sigma, xi, kappa) for v in x],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        ExtGenPareto.logsf(x, mu, sigma, xi, kappa).eval(),
        [ref_ext_logsf(v, mu, sigma, xi, kappa) for v in x],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        ExtGenPareto.ppf(q, mu, sigma, xi, kappa).eval(),
        ref_ext_ppf(q, mu, sigma, xi, kappa),
        rtol=1e-10,
    )


def test_logpdf_logcdf_at_infinity():
    x = pt.constant(np.inf)
    xi = pt.constant(np.array([-0.5, 0.0, 0.5]))
    assert np.all(ExtGenPareto.logpdf(x, 0.0, 1.0, xi, 2.0).eval() == -np.inf)
    assert np.all(ExtGenPareto.logcdf(x, 0.0, 1.0, xi, 2.0).eval() == 0.0)


def test_ppf_endpoints():
    xi = np.array([-0.5, 0.0, 0.5])
    kappa = np.array([2.0, 0.5, 3.0])
    with np.errstate(divide="ignore"):
        expected_hi = np.where(xi < 0, 1.0 - 2.0 / xi, np.inf)
    assert np.all(ExtGenPareto.ppf(0.0, 1.0, 2.0, xi, kappa).eval() == 1.0)
    np.testing.assert_allclose(ExtGenPareto.ppf(1.0, 1.0, 2.0, xi, kappa).eval(), expected_hi)


def test_ppf_outside_unit_interval_is_nan():
    for q in (-0.1, 1.1):
        assert np.isnan(ExtGenPareto.ppf(q, 0.0, 1.0, 0.2, 2.0).eval())


def test_logsf_reduces_to_gpd_at_kappa_one():
    value = np.linspace(0.05, 8.0, 40)
    for xi in (-0.3, 0.0, 0.4):
        ext = ExtGenPareto.logsf(value, 0.0, 1.5, xi, 1.0).eval()
        gpd = GenPareto.logsf(value, 0.0, 1.5, xi).eval()
        np.testing.assert_allclose(ext, gpd, rtol=1e-12, atol=1e-12)


def test_kappa_one_equals_gpd():
    value = np.linspace(0.0, 8.0, 60)
    for xi in (-0.3, -1e-8, 0.0, 0.25, 0.8):
        ext = ExtGenPareto.logpdf(value, 0.0, 1.5, xi, 1.0).eval()
        gpd = GenPareto.logpdf(value, 0.0, 1.5, xi).eval()
        np.testing.assert_allclose(ext, gpd, rtol=1e-12, atol=1e-12)


def test_rvs_matches_distribution():
    for xi, kappa in ((-0.2, 0.5), (0.0, 2.0), (0.3, 3.0)):
        draws = ExtGenPareto.rvs(
            0.0, 1.0, xi, kappa, size=20_000, random_state=pytensor.shared(np.random.default_rng(7))
        ).eval()
        u = ExtGenPareto.cdf(draws, 0.0, 1.0, xi, kappa).eval()
        assert stats.kstest(u, "uniform").pvalue > 0.01


@pytest.mark.parametrize("kappa", [0.5, 0.01])
def test_small_kappa_ppf_uses_stable_excess(kappa):
    median = ref_ext_ppf(0.5, 0.0, 1.0, 0.0, kappa)
    assert median > 0.0

    got = float(ExtGenPareto.ppf(0.5, 0.0, 1.0, 0.0, kappa).eval())
    np.testing.assert_allclose(got, median, rtol=1e-6)


def test_small_kappa_draws_do_not_collapse_to_mu():
    draws = ExtGenPareto.rvs(
        0.0, 1.0, 0.0, 0.01, size=20_000, random_state=pytensor.shared(np.random.default_rng(7))
    ).eval()
    assert (draws >= 0.0).all()
    assert np.mean(draws == 0.0) < 0.01


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_excess_from_logit_across_cutoff_for_huge_kappa(dtype):
    y = pt.scalar("y", dtype=dtype)
    kappa = pt.scalar("kappa", dtype=dtype)
    m = ExtGenPareto._ext_gpd_excess_from_logit(y, kappa)
    f = pytensor.function([y, kappa], [m, *pt.grad(m, [y, kappa])])
    rtol = 1e-12 if dtype == "float64" else 1e-5
    ys = [0.0, 200.0, 500.0, 680.0, 700.0, 730.0] if dtype == "float64" else [0.0, 50.0, 70.0, 85.0]
    for yv in ys:
        for kv in (1.0, 1e3, 1e8, 1e15):
            m_val, dy, dk = f(np.asarray(yv, dtype), np.asarray(kv, dtype))
            assert np.all(np.isfinite([m_val, dy, dk])), f"non-finite at y={yv}, kappa={kv}"
            np.testing.assert_allclose(
                float(m_val),
                _ext_gpd_excess_ref(yv, kv),
                rtol=rtol,
                err_msg=f"excess wrong at y={yv}, kappa={kv}",
            )


def test_excess_from_logit_upper_tail_is_finite_under_float32():
    y = pt.scalar("y", dtype="float32")
    excess = ExtGenPareto._ext_gpd_excess_from_logit(y, np.float32(2.0))
    assert excess.dtype == "float32"
    fn = pytensor.function([y], excess)
    for yi in (90.0, 200.0, 700.0, 5000.0):
        m = float(fn(np.float32(yi)))
        assert np.isfinite(m)
        np.testing.assert_allclose(m, yi + np.log(2.0), rtol=1e-3)


def test_excess_from_logit_resolves_subnormal_kappa_tail():
    y = pt.dscalar("y")
    excess = float(
        pytensor.function([y], ExtGenPareto._ext_gpd_excess_from_logit(y, 4e-309))(710.0)
    )
    assert excess > 0.0
    np.testing.assert_allclose(excess, 0.395390331, rtol=1e-4)


def test_logsf_stable_in_far_tail():
    for kappa in (0.5, 1.0, 2.5, 5.0):
        x = np.array([100.0, 300.0, 1000.0])
        got = ExtGenPareto.logsf(x, 0.0, 1.0, 0.0, kappa).eval()
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, np.log(kappa) - x, rtol=1e-9)


@pytest.mark.parametrize("kappa", [10.0, 1e155, 1e300])
def test_logsf_is_a_valid_log_probability_for_large_kappa(kappa):
    x = np.array([40.0, 100.0, 1000.0])
    kappa_t = pt.constant(np.asarray(kappa, dtype="float64"))
    got = ExtGenPareto.logsf(x, 0.0, 1.0, 0.0, kappa_t).eval()
    assert np.all(got <= 0.0)
    small = np.log(kappa) - x < -30.0
    np.testing.assert_allclose(got[small], (np.log(kappa) - x)[small], rtol=1e-9)


@pytest.mark.parametrize("kappa", [1e-2, 1e-100])
def test_logsf_small_kappa_in_the_body_matches_reference(kappa):
    x = np.array([0.01, 0.1, 0.5, 2.0])
    got = ExtGenPareto.logsf(x, 0.0, 1.0, 0.0, kappa).eval()
    ref = np.log(-np.expm1(kappa * np.log1p(-np.exp(-x))))
    np.testing.assert_allclose(got, ref, rtol=1e-9)


def test_boundary_logp_value_holds_but_gradient_tracks_the_margin():
    mu, sigma, xi, kappa = 0.4, 1.3, -0.3, 2.5
    eps = np.finfo(np.float64).eps
    margins = [1e-2, 1e-6, 1e-12]

    v, sig, xs, ks = (pt.dscalar(n) for n in ("v", "sig", "xs", "ks"))
    logp = ExtGenPareto.logpdf(v, mu, sig, xs, ks)
    fn = pytensor.function(
        [v, sig, xs, ks],
        [logp, pt.grad(logp, sig), pt.grad(logp, xs), pt.grad(logp, ks)],
    )

    params = {
        "mu": Decimal(mu),
        "sigma": Decimal(sigma),
        "xi": Decimal(xi),
        "kappa": Decimal(kappa),
    }
    prev_logp = np.inf
    for s_target in margins:
        value = mu + sigma * ((s_target - 1) / xi)
        logp_f, *grads_f = (float(o) for o in fn(value, sigma, xi, kappa))
        logp_ref = _gpd_ref_logp(value, **params)

        assert np.isfinite(logp_f), s_target
        assert all(np.isfinite(g) for g in grads_f), s_target
        assert logp_f < prev_logp, s_target
        prev_logp = logp_f
        assert abs((Decimal(logp_f) - logp_ref) / logp_ref) < 1e-4, s_target

        grad_bound = 100 * eps / s_target
        for name, g in zip(["sigma", "xi", "kappa"], grads_f):
            g_ref = _gpd_ref_grad(value, params, name)
            rel = abs((Decimal(g) - g_ref) / g_ref)
            assert np.sign(g) == np.sign(float(g_ref)), (name, s_target)
            if name == "kappa":
                assert rel < 1e-10, (name, s_target, float(rel))
            else:
                assert rel < grad_bound, (name, s_target, float(rel))


def test_logp_gradient_is_continuous_through_xi_zero():
    xi = pt.dscalar("xi")
    logp = ExtGenPareto.logpdf(pt.constant(2.3), 0.0, 1.0, xi, 2.0)
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
