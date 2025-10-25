import numpy as np

from pymc.distributions import HalfNormal, Normal
from pymc.model.core import Model

from pymc_extras.sampling.optimizations import optimize_model_for_mcmc_sampling


def test_summary_stats_normal():
    rng = np.random.default_rng(3)
    y_data = rng.normal(loc=1, scale=0.5, size=(1000,))

    with Model() as m:
        mu = Normal("mu")
        sigma = HalfNormal("sigma")
        y = Normal("y", mu=mu, sigma=sigma, observed=y_data)

    assert len(m.free_RVs) == 2
    assert len(m.observed_RVs) == 1

    new_m, rewrite_counters = optimize_model_for_mcmc_sampling(m)
    assert "summary_stats_normal" in (r.name for rc in rewrite_counters for r in rc)

    assert len(new_m.free_RVs) == 2
    assert len(new_m.observed_RVs) == 2

    # Confirm equivalent (up to an additive normalization constant)
    m_logp = m.compile_logp()
    new_m_logp = new_m.compile_logp()

    ip = m.initial_point()
    first_logp_diff = m_logp(ip) - new_m_logp(ip)

    ip["mu"] += 0.5
    ip["sigma_log__"] += 1.5
    second_logp_diff = m_logp(ip) - new_m_logp(ip)

    np.testing.assert_allclose(first_logp_diff, second_logp_diff)

    # dlogp should be the same
    m_dlogp = m.compile_dlogp()
    new_m_dlogp = new_m.compile_dlogp()
    np.testing.assert_allclose(m_dlogp(ip), new_m_dlogp(ip))
