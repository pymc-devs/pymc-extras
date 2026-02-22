"""
Model factories for model-equivalence tests.

Each function returns a freshly constructed pm.Model with no data dependencies
other than constants embedded at construction time.  The same factory is used
both by generate_fixtures.py (which records LBFGS history + reference ELBO) and
by the test suite (which replays those fixtures through the current/refactored
ELBO implementation).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def make_ard_regression() -> pm.Model:
    """ARD linear regression: p=15 RBF features, N=40. Coupled (w,α) funnels
    produce many LBFGS steps (long path). Fast O(N·p) logp.
    """
    rng = np.random.default_rng(42)
    N, p = 40, 15
    centers = np.linspace(-3, 3, p)
    x = rng.uniform(-3, 3, N)
    Phi = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / 0.5) ** 2)
    w_true = np.zeros(p)
    w_true[[3, 7, 11]] = [1.5, -1.0, 0.8]
    y = Phi @ w_true + rng.normal(0, 0.3, N)

    Phi_t = pt.as_tensor_variable(Phi.astype("float64"))

    with pm.Model() as model:
        log_alpha = pm.Normal("log_alpha", mu=0, sigma=3, shape=(p,))
        alpha = pt.exp(log_alpha)
        w = pm.Normal("w", mu=0, sigma=1.0 / pt.sqrt(alpha), shape=(p,))
        log_sigma = pm.Normal("log_sigma", mu=0, sigma=1)
        mu = pt.dot(Phi_t, w)
        pm.Normal("y", mu=mu, sigma=pt.exp(log_sigma), observed=y)
    return model


def make_bpca_small() -> pm.Model:
    """Small Bayesian PCA: N=50, D=5, K=2 → 111 unconstrained params.
    Cheap per-obs logp; stress-tests streaming memory path.
    """
    rng = np.random.default_rng(42)
    N, D, K = 50, 5, 2
    W_true = rng.standard_normal((D, K)) * 0.8
    Z_true = rng.standard_normal((N, K))
    sigma_true = 0.5
    X_obs = Z_true @ W_true.T + rng.normal(0, sigma_true, size=(N, D))
    X_obs -= X_obs.mean(axis=0)

    with pm.Model(
        coords={"obs": np.arange(N), "feature": np.arange(D), "factor": np.arange(K)}
    ) as model:
        W = pm.Normal("W", mu=0, sigma=1, dims=("feature", "factor"))
        Z = pm.Normal("Z", mu=0, sigma=1, dims=("obs", "factor"))
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.Deterministic("mu", pm.math.dot(Z, W.T), dims=("obs", "feature"))
        pm.Normal("X", mu=mu, sigma=sigma, observed=X_obs, dims=("obs", "feature"))
    return model


def make_hd_gaussian() -> pm.Model:
    """High-dimensional diagonal Gaussian, N=50.  Exercises sparse routing (2J<N)."""
    scales = np.exp(np.linspace(-2.0, 2.0, 50))
    with pm.Model() as model:
        pm.Normal("y", mu=np.zeros(50), sigma=scales, shape=50)
    return model


def make_logistic_regression() -> pm.Model:
    """Bayesian logistic regression, N=11 (intercept + 10 coefficients).
    Data is generated deterministically from a fixed seed."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    true_beta = rng.normal(size=10)
    p_true = 1.0 / (1.0 + np.exp(-(X @ true_beta)))
    y_obs = rng.binomial(1, p_true).astype(np.float64)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=10)
        p = pm.math.sigmoid(alpha + X @ beta)
        pm.Bernoulli("obs", p=p, observed=y_obs)
    return model


MODEL_FACTORIES: dict = {
    "ard_regression": make_ard_regression,
    "bpca_small": make_bpca_small,
    "hd_gaussian": make_hd_gaussian,
    "logistic_regression": make_logistic_regression,
}
