import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_experimental.gp.pytensor_gp import GP, ExpQuad, conditional_gp


def build_latent_model():
    with pm.Model() as m:
        X = pm.Data("X", np.arange(3)[:, None])
        y = np.full(3, np.pi)
        ls = 1.0
        cov = ExpQuad(X, ls=ls)
        gp = GP("gp", cov=cov)

        sigma = 1.0
        obs = pm.Normal("obs", mu=gp, sigma=sigma, observed=y)

    return m


def build_latent_model_old_API():
    with pm.Model() as m:
        X = pm.Data("X", np.arange(3)[:, None])
        y = np.full(3, np.pi)
        ls = 1.0
        cov = pm.gp.cov.ExpQuad(1, ls)
        gp_class = pm.gp.Latent(cov_func=cov)
        gp = gp_class.prior("gp", X, reparameterize=False)

        sigma = 1.0
        obs = pm.Normal("obs", mu=gp, sigma=sigma, observed=y)

    return m, gp_class


def test_exp_quad():
    x = pt.arange(3)[:, None]
    ls = pt.ones(())
    cov = ExpQuad(x, ls=ls).eval()
    expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])

    np.testing.assert_allclose(cov, np.exp(-0.5 * expected_distance))


def test_latent_model_prior():
    m = build_latent_model()
    ref_m, _ = build_latent_model_old_API()

    prior = pm.draw(m["gp"], draws=1000)
    prior_ref = pm.draw(ref_m["gp"], draws=1000)

    np.testing.assert_allclose(
        prior.mean(),
        prior_ref.mean(),
        atol=0.1,
    )

    np.testing.assert_allclose(
        prior.std(),
        prior_ref.std(),
        rtol=0.1,
    )


def test_latent_model_logp():
    m = build_latent_model()
    ip = m.initial_point()

    ref_m, _ = build_latent_model_old_API()

    np.testing.assert_allclose(
        m.compile_logp()(ip),
        ref_m.compile_logp()(ip),
        rtol=1e-6,
    )


@pytest.mark.parametrize("inline", (False, True))
def test_latent_model_conditional(inline):
    rng = np.random.default_rng(0)
    posterior = az.from_dict(
        posterior={"gp": rng.normal(np.pi, 1e-3, size=(4, 1000, 3))},
        constant_data={"X": np.arange(3)[:, None]},
    )

    new_x = np.array([3, 4])[:, None]

    m = build_latent_model()
    with m:
        pm.Deterministic("gp_exp", m["gp"].exp())

    with conditional_gp(m, m["gp"], new_x, inline=inline) as cgp:
        pred = pm.sample_posterior_predictive(
            posterior,
            var_names=["gp_star", "gp_exp"],
            progressbar=False,
        ).posterior_predictive

    ref_m, ref_gp_class = build_latent_model_old_API()
    with ref_m:
        gp_star = ref_gp_class.conditional("gp_star", Xnew=new_x)
        pred_ref = pm.sample_posterior_predictive(
            posterior,
            var_names=["gp_star"],
            progressbar=False,
        ).posterior_predictive

    np.testing.assert_allclose(
        pred["gp_star"].mean(),
        pred_ref["gp_star"].mean(),
        atol=0.1,
    )

    np.testing.assert_allclose(
        pred["gp_star"].std(),
        pred_ref["gp_star"].std(),
        rtol=0.1,
    )

    if inline:
        assert np.testing.assert_allclose(
            pred["gp_exp"],
            np.exp(pred["gp_star"]),
        )
    else:
        np.testing.assert_allclose(
            pred["gp_exp"],
            np.exp(posterior.posterior["gp"]),
        )


#
# def test_marginal_sigma_rewrites_to_white_noise_cov(marginal_model, ):
#     obs = marginal_model["obs"]
#
#     # TODO: Bring these checks back after we implement marginalization of the GP RV
#     #
#     # assert sum(isinstance(var.owner.op, pm.Normal.rv_type)
#     #            for var in ancestors([obs])
#     #            if var.owner is not None) == 1
#     #
#     f = pm.compile_pymc([], obs)
#     #
#     # assert not any(isinstance(node.op, pm.Normal.rv_type) for node in f.maker.fgraph.apply_nodes)
#
#     draws = np.stack([f() for _ in range(10_000)])
#     empirical_cov = np.cov(draws.T)
#
#     expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
#
#     np.testing.assert_allclose(
#         empirical_cov, np.exp(-0.5 * expected_distance) + np.eye(3), atol=0.1, rtol=0.1
#     )
#
#
# def test_marginal_gp_logp(marginal_model):
#     expected_logps = {"obs": -8.8778}
#     point_logps = marginal_model.point_logps(round_vals=4)
#     for v1, v2 in zip(point_logps.values(), expected_logps.values()):
#         np.testing.assert_allclose(v1, v2, atol=1e-6)
