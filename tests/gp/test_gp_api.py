"""Validate the GP API against reference GP formulas."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import pymc_extras.gp as pgp

ETA, LS, SIGMA = 1.7, 0.3, 0.25
PRIOR_JITTER = 1e-6  # GP() adds this to the prior covariance
SOLVER_JITTER = 1e-8  # linear_gaussian adds this to the observation covariance


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = np.linspace(0, 1, 12)[:, None]
    X_new = np.linspace(-0.2, 1.2, 7)[:, None]
    y = np.sin(6 * X.ravel()) + 0.1 * rng.normal(size=12)
    return X, X_new, y


def ref_kernel():
    return ETA**2 * pm.gp.cov.Matern52(1, ls=LS)


def test_conjugate_gp_matches_reference(data):
    """Marginal likelihood and predictive moments match the textbook formulas."""
    X, X_new, y = data

    with pm.Model() as m:
        Xs, shapes = pt.pack(X, X_new, keep_axes=-1)
        k = ETA**2 * pgp.kernels.Matern52(ls=LS)
        gp = pgp.GP("gp", Xs, cov=k)
        f_train, _ = pt.unpack(gp, shapes)
        pm.Normal("y", mu=f_train, sigma=SIGMA, observed=y)

    marginal_m = pgp.marginalize(m, ["gp"])
    assert [v.name for v in marginal_m.free_RVs] == []

    kk = ref_kernel()
    nugget = PRIOR_JITTER + SOLVER_JITTER
    Koo = kk(X).eval() + (SIGMA**2 + nugget) * np.eye(len(X))
    ref_logp = pm.logp(pm.MvNormal.dist(np.zeros(len(X)), Koo), y).eval()
    np.testing.assert_allclose(marginal_m.compile_logp()({}), ref_logp, rtol=1e-6)

    # --- conditional: posterior over the joint latent, prediction block included
    cond_m = pgp.conditional(marginal_m)
    assert [v.name for v in cond_m.free_RVs] == ["gp"]

    f = cond_m["gp"]
    mu_post, cov_post = (p.eval() for p in f.owner.op.dist_params(f.owner))

    # the SAME unpack closure still applies after two model transforms
    _, mu_pred = (v.eval() for v in pt.unpack(mu_post, shapes))

    Kno = kk(X_new, X).eval()
    Knn = kk(X_new).eval() + PRIOR_JITTER * np.eye(len(X_new))
    np.testing.assert_allclose(mu_pred, Kno @ np.linalg.solve(Koo, y), rtol=1e-5, atol=1e-7)

    n_train = len(X)
    np.testing.assert_allclose(
        cov_post[n_train:, n_train:],
        Knn - Kno @ np.linalg.solve(Koo, Kno.T),
        rtol=1e-4,
        atol=1e-6,
    )


def test_hyperpriors_and_sampling(data):
    """Works with the kernel driven by free RVs, end to end."""
    X, X_new, y = data

    with pm.Model() as m:
        ls = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
        eta = pm.Exponential("eta", scale=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        Xs, shapes = pt.pack(X, X_new, keep_axes=-1)
        gp = pgp.GP("gp", Xs, cov=eta**2 * pgp.kernels.Matern52(ls=ls))
        f_train, _ = pt.unpack(gp, shapes)
        pm.Normal("y", mu=f_train, sigma=sigma, observed=y)

    marginal_m = pgp.marginalize(m, ["gp"])
    assert sorted(v.name for v in marginal_m.free_RVs) == ["eta", "ls", "sigma"]
    assert np.isfinite(marginal_m.compile_logp()(marginal_m.initial_point()))

    with marginal_m:
        idata = pm.sample(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)

    cond_m = pgp.conditional(marginal_m)
    with cond_m:
        pp = pm.sample_posterior_predictive(
            idata, var_names=["gp"], progressbar=False, random_seed=0
        )

    draws = pp.posterior_predictive["gp"].values
    assert draws.shape[-1] == len(X) + len(X_new)
    assert np.isfinite(draws).all()


def test_non_gaussian_likelihood_declines_then_laplace(data):
    """Bernoulli: exact marginalization declines, Laplace marginalizes."""
    X, X_new, y = data
    y_bin = (y > 0).astype(int)

    def build():
        with pm.Model() as m:
            Xs, shapes = pt.pack(X, X_new, keep_axes=-1)
            gp = pgp.GP("gp", Xs, cov=ETA**2 * pgp.kernels.Matern52(ls=LS))
            f_train, _ = pt.unpack(gp, shapes)
            pm.Bernoulli("y", logit_p=f_train, observed=y_bin)
        return m

    with pytest.raises(NotImplementedError, match="Cannot marginalize"):
        pgp.marginalize(build(), ["gp"])

    # The approximation is opt-in on the same entry point.
    n = len(X) + len(X_new)
    k = ETA**2 * pgp.kernels.Matern52(ls=LS)
    Q = np.linalg.inv((k(np.vstack([X, X_new])) + PRIOR_JITTER * pt.eye(n)).eval())

    laplace_m = pgp.marginalize(build(), laplace_approx={"gp": Q})
    assert [v.name for v in laplace_m.free_RVs] == []
    assert np.isfinite(laplace_m.compile_logp()({}))

    # GAP (upstream, not GP-specific): the conditional of a Laplace-marginalized
    # variable cannot be recovered, so this route has no predictive path.
    with pytest.raises(NotImplementedError, match="MarginalLaplaceRV"):
        pgp.conditional(laplace_m)

    # Sampling the latent directly remains the fallback.
    with build():
        idata = pm.sample(draws=25, tune=25, chains=1, progressbar=False, random_seed=0)
    assert np.isfinite(idata.posterior["gp"].values).all()


@pytest.mark.parametrize("fitc", [False, True], ids=["dtc", "fitc"])
def test_sparse_gp_needs_no_new_machinery(fitc):
    """DTC / FITC are just a linear map of a smaller GP."""
    rng = np.random.default_rng(0)
    n, n_ind = 40, 8
    X = np.linspace(0, 1, n)[:, None]
    Z = np.linspace(0, 1, n_ind)[:, None]
    y = np.sin(6 * X.ravel()) + 0.1 * rng.normal(size=n)

    with pm.Model() as m:
        k = ETA**2 * pgp.kernels.Matern52(ls=LS)
        u = pgp.GP("u", Z, cov=k)  # inducing values
        f = pgp.project(u, X)  # A @ u -- affine, so marginalizable
        sigma = SIGMA
        if fitc:
            sigma = pt.sqrt(SIGMA**2 + pgp.prior_variance_correction(u, X))
        pm.Normal("y", mu=f, sigma=sigma, observed=y)

    got = pgp.marginalize(m, ["u"]).compile_logp()({})

    kk = ref_kernel()
    Kzz = kk(Z).eval() + PRIOR_JITTER * np.eye(n_ind)
    Kxz = kk(X, Z).eval()
    Q = Kxz @ np.linalg.solve(Kzz, Kxz.T)
    C = Q + (SIGMA**2 + SOLVER_JITTER) * np.eye(n)
    if fitc:
        C = C + np.diag(np.clip(np.diag(kk(X).eval()) - np.diag(Q), 0, None))

    np.testing.assert_allclose(got, pm.logp(pm.MvNormal.dist(np.zeros(n), C), y).eval(), rtol=1e-5)


def test_fitc_with_inducing_at_data_recovers_exact_gp():
    """Z == X collapses FITC back to the exact GP, up to solver jitter."""
    rng = np.random.default_rng(0)
    n = 40
    X = np.linspace(0, 1, n)[:, None]
    y = np.sin(6 * X.ravel()) + 0.1 * rng.normal(size=n)

    with pm.Model() as m:
        k = ETA**2 * pgp.kernels.Matern52(ls=LS)
        u = pgp.GP("u", X, cov=k, jitter=1e-12)
        corr = pgp.prior_variance_correction(u, X, jitter=1e-12)
        pm.Normal(
            "y", mu=pgp.project(u, X, jitter=1e-12), sigma=pt.sqrt(SIGMA**2 + corr), observed=y
        )

    got = pgp.marginalize(m, ["u"]).compile_logp()({})
    exact = pm.logp(
        pm.MvNormal.dist(np.zeros(n), ref_kernel()(X).eval() + SIGMA**2 * np.eye(n)), y
    ).eval()

    # residual is the solver jitter on an n-diagonal, ~n*J/(2*sigma^2)
    assert abs(got - exact) < 5 * n * SOLVER_JITTER / SIGMA**2


def test_kernel_composition_and_ard():
    """Kernels compose and support ARD / active_dims."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    Xs = rng.normal(size=(4, 3))

    k = 2.0 * pgp.kernels.ExpQuad(ls=np.array([0.5, 1.0, 2.0])) + pgp.kernels.WhiteNoise(0.1)
    ref = 2.0 * pm.gp.cov.ExpQuad(3, ls=np.array([0.5, 1.0, 2.0])) + pm.gp.cov.WhiteNoise(0.1)

    K = k(X).eval()
    assert K.shape == (9, 9)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > 0
    np.testing.assert_allclose(K, ref(X).eval(), atol=1e-8)
    np.testing.assert_allclose(k(X, Xs).eval(), ref(X, Xs).eval(), atol=1e-8)

    k1 = pgp.kernels.Matern52(ls=0.7, active_dims=[0, 2])
    ref1 = pm.gp.cov.Matern52(3, ls=0.7, active_dims=[0, 2])
    np.testing.assert_allclose(k1(X).eval(), ref1(X).eval(), atol=1e-8)


def test_prediction_at_new_inputs_via_set_data(data):
    """X_pred is pm.Data, so the same model predicts at new locations."""
    X, X_new, y = data

    with pm.Model() as m:
        X_pred = pm.Data("X_pred", X_new)
        Xs, shapes = pt.pack(X, X_pred, keep_axes=-1)
        gp = pgp.GP("gp", Xs, cov=ETA**2 * pgp.kernels.Matern52(ls=LS))
        f_train, _ = pt.unpack(gp, shapes)
        pm.Normal("y", mu=f_train, sigma=SIGMA, observed=y)

    cond_m = pgp.conditional(pgp.marginalize(m, ["gp"]))
    f = cond_m["gp"]

    def pred_mean():
        _, shp = pt.pack(X, cond_m["X_pred"], keep_axes=-1)
        return pt.unpack(f.owner.op.dist_params(f.owner)[0], shp)[1].eval()

    mu_default = pred_mean()
    with cond_m:
        pm.set_data({"X_pred": X_new + 0.05})
    mu_shifted = pred_mean()

    assert not np.allclose(mu_default, mu_shifted)

    kk = ref_kernel()
    Koo = kk(X).eval() + (SIGMA**2 + PRIOR_JITTER + SOLVER_JITTER) * np.eye(len(X))
    Kno = kk(X_new + 0.05, X).eval()
    np.testing.assert_allclose(mu_shifted, Kno @ np.linalg.solve(Koo, y), rtol=1e-5, atol=1e-7)


def test_prediction_block_can_be_resized(data):
    """Symbolic pack boundaries: set_data may change the *number* of points."""
    X, X_new, y = data

    with pm.Model() as m:
        X_pred = pm.Data("X_pred", X_new)
        Xs, shapes = pt.pack(X, X_pred, keep_axes=-1)
        gp = pgp.GP("gp", Xs, cov=ETA**2 * pgp.kernels.Matern52(ls=LS))
        f_train, _ = pt.unpack(gp, shapes)
        pm.Normal("y", mu=f_train, sigma=SIGMA, observed=y)

    cond_m = pgp.conditional(pgp.marginalize(m, ["gp"]))
    f = cond_m["gp"]
    mu = f.owner.op.dist_params(f.owner)[0]

    _, shp = pt.pack(X, cond_m["X_pred"], keep_axes=-1)
    assert pt.unpack(mu, shp)[1].eval().shape == (len(X_new),)

    X_bigger = np.linspace(-0.5, 1.5, 31)[:, None]
    with cond_m:
        pm.set_data({"X_pred": X_bigger})

    got = pt.unpack(mu, shp)[1].eval()
    assert got.shape == (31,)

    kk = ref_kernel()
    Koo = kk(X).eval() + (SIGMA**2 + PRIOR_JITTER + SOLVER_JITTER) * np.eye(len(X))
    ref = kk(X_bigger, X).eval() @ np.linalg.solve(Koo, y)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)
