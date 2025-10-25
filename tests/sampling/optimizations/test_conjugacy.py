import numpy as np
import pytest

from pymc.distributions import Beta, Binomial, DiracDelta
from pymc.model.core import Model
from pymc.model.transform.conditioning import remove_value_transforms
from pymc.sampling import draw

from pymc_extras.sampling.optimizations import optimize_model_for_mcmc_sampling
from pymc_extras.sampling.optimizations.conjugate_sampler import ConjugateRV


@pytest.mark.parametrize("eager", [False, True])
def test_beta_binomial_conjugacy(eager):
    with Model() as m:
        if eager:
            a, b = DiracDelta("a,b", [1, 1])
        else:
            a, b = 1, 1
        p = Beta("p", a, b)
        y = Binomial("y", n=100, p=p, observed=99)

    assert m.rvs_to_transforms[p] is not None
    assert isinstance(p.owner.op, Beta)

    new_m, rewrite_counters = optimize_model_for_mcmc_sampling(m)
    rewrite_applied = "beta_binomial_conjugacy" in (r.name for rc in rewrite_counters for r in rc)
    if eager:
        assert not rewrite_applied
        new_m, rewrite_counters = optimize_model_for_mcmc_sampling(m, include="conjugacy-eager")
        assert "beta_binomial_conjugacy_eager" in (r.name for rc in rewrite_counters for r in rc)
    else:
        assert rewrite_applied

    new_p = new_m["p"]
    assert isinstance(new_p.owner.op, ConjugateRV)
    assert new_m.rvs_to_transforms[new_p] is None
    beta_rv, conjugate_beta_rv, *_ = new_p.owner.outputs

    # Check it behaves like a beta and its conjugate
    beta_draws, conjugate_beta_draws = draw(
        [beta_rv, conjugate_beta_rv], draws=1000, random_seed=25
    )
    np.testing.assert_allclose(beta_draws.mean(), 1 / 2, atol=1e-2)
    np.testing.assert_allclose(conjugate_beta_draws.mean(), 100 / 102, atol=1e-3)
    np.testing.assert_allclose(beta_draws.std(), np.sqrt(1 / 12), atol=1e-2)
    np.testing.assert_allclose(
        conjugate_beta_draws.std(), np.sqrt(100 * 2 / (102**2 * 103)), atol=1e-3
    )

    # Check if support point and logp is the same as the original model without transforms
    untransformed_m = remove_value_transforms(m)
    new_m_ip = new_m.initial_point()
    for key, value in untransformed_m.initial_point().items():
        np.testing.assert_allclose(new_m_ip[key], value)

    new_m_logp = new_m.compile_logp()(new_m_ip)
    untransformed_m_logp = untransformed_m.compile_logp()(new_m_ip)
    np.testing.assert_allclose(new_m_logp, untransformed_m_logp)
