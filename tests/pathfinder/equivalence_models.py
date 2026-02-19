"""
Model factories for T7 model-equivalence tests.

Each function returns a freshly constructed pm.Model with no data dependencies
other than constants embedded at construction time.  The same factory is used
both by generate_fixtures.py (which records LBFGS history + reference ELBO) and
by the test suite (which replays those fixtures through the current/refactored
ELBO implementation).
"""

import numpy as np
import pymc as pm


def make_iso_gaussian() -> pm.Model:
    """Isotropic Gaussian, N=4.  Trivial geometry; clean baseline."""
    with pm.Model() as model:
        pm.Normal("y", mu=0.0, sigma=1.0, shape=4)
    return model


def make_nealsfunnel() -> pm.Model:
    """Neal's funnel, N=2.  Highly curved; stresses alpha_recover numerics."""
    with pm.Model() as model:
        v = pm.Normal("v", mu=0.0, sigma=3.0)
        pm.Normal("x", mu=0.0, sigma=pm.math.exp(v / 2.0))
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
    "iso_gaussian": make_iso_gaussian,
    "nealsfunnel": make_nealsfunnel,
    "hd_gaussian": make_hd_gaussian,
    "logistic_regression": make_logistic_regression,
}
