import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc.blocking import DictToArrayBijection

from pymc_extras.inference.pathfinder.bfgs_sample import (
    alpha_step_numpy,
    get_logp_dlogp_of_ravel_inputs,
    get_neg_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
)


def test_alpha_step_numpy_known_values():
    """Pin the inverse-Hessian diagonal update to hand-derived values.

    For alpha_prev = [1, 1], s = [1, 2], z = [1, 1] the Zhang et al. (2022) update gives
    a = Σαz² = 2, b = Σzs = 3, c = Σs²/α = 5, so inv_alpha = [13/15, 7/15] and alpha = [15/13, 15/7].
    """
    out = alpha_step_numpy(np.array([1.0, 1.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(out, [15 / 13, 15 / 7])


@pytest.mark.parametrize("alpha_prev", [1.0, 0.3, 5.0])
def test_alpha_step_numpy_scalar_is_secant(alpha_prev):
    """In 1-D the update reduces to the secant s/z, independent of the previous alpha."""
    s, z = 0.7, 2.0
    out = alpha_step_numpy(np.array([alpha_prev]), np.array([s]), np.array([z]))
    np.testing.assert_allclose(out, [s / z])


@pytest.mark.parametrize(
    "s, z",
    [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([1.0, 1.0]), np.array([-1.0, -1.0])),
    ],
    ids=["zero_curvature", "zero_step", "negative_curvature"],
)
def test_alpha_step_numpy_rejects_degenerate_update(s, z):
    """Degenerate curvature returns a copy of the previous alpha rather than NaN/negative values.

    zero_curvature (s · z = 0) and zero_step (c = 0) trip the first guard; negative_curvature
    passes the first guard but yields alpha <= 0 and trips the second.
    """
    alpha_prev = np.array([2.0, 3.0])
    out = alpha_step_numpy(alpha_prev, s, z)

    np.testing.assert_array_equal(out, alpha_prev)
    assert out is not alpha_prev


def _standard_normal_logp(x: np.ndarray) -> np.ndarray:
    return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)


@pytest.fixture
def scalar_model():
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
    return model


def test_logp_dlogp_values(scalar_model):
    fn = get_logp_dlogp_of_ravel_inputs(scalar_model)
    logp, dlogp = fn(np.array([2.0]))

    np.testing.assert_allclose(logp, _standard_normal_logp(np.array(2.0)))
    np.testing.assert_allclose(dlogp, [-2.0])


def test_neg_logp_dlogp_negates(scalar_model):
    fn = get_neg_logp_dlogp_of_ravel_inputs(scalar_model)
    neg_logp, neg_dlogp = fn(np.array([2.0]))

    np.testing.assert_allclose(neg_logp, -_standard_normal_logp(np.array(2.0)))
    np.testing.assert_allclose(neg_dlogp, [2.0])


@pytest.fixture
def minibatch_model():
    """A minibatched model, whose logp carries a shared RNG for the batch draw."""
    y = np.random.default_rng(0).normal(size=100)
    with pm.Model() as model:
        observed = pm.Minibatch(pt.as_tensor_variable(y), batch_size=20)
        mu = pm.Normal("mu", 0.0, 1.0)
        pm.Normal("obs", mu=mu, sigma=1.0, observed=observed, total_size=100)
    return model


def _sample_fn_inputs(N, J, seed=1, M=8):
    """Well-formed (x, g, alpha, s_win, z_win, u) for a sample fn of size N, history J."""
    rng = np.random.default_rng(seed)
    s_win = rng.normal(size=(N, J)) * 0.1
    return (
        rng.normal(size=N),
        rng.normal(size=N),
        np.ones(N),
        s_win,
        s_win.copy(),  # z_win == s_win keeps S.T @ Z positive definite
        rng.normal(size=(M, N)),
    )


@pytest.mark.parametrize("vectorize", [False, True])
def test_sample_fn_supports_minibatch(minibatch_model, vectorize):
    """Both batching modes return a finite logP per draw for a minibatched model."""
    N = DictToArrayBijection.map(minibatch_model.initial_point()).data.shape[0]
    J = 6

    fn = make_pathfinder_sample_fn(minibatch_model, N, J, jacobian=True, vectorize=vectorize)
    phi, logQ, logP, inv_hessian_diag = fn(*_sample_fn_inputs(N, J))

    assert logP.shape == (phi.shape[0],)
    assert np.all(np.isfinite(logP))


def test_sample_fn_map_scores_all_draws_against_one_minibatch(minibatch_model):
    """The minibatch is drawn once per call and shared across the map, so two identical
    draws are scored against the same batch and get the same logP."""
    N = DictToArrayBijection.map(minibatch_model.initial_point()).data.shape[0]
    J = 6
    x, g, alpha, s_win, z_win, u = _sample_fn_inputs(N, J)
    u[1] = u[0]

    fn = make_pathfinder_sample_fn(minibatch_model, N, J, jacobian=True, vectorize=False)
    phi, _, logP, _ = fn(x, g, alpha, s_win, z_win, u)

    np.testing.assert_allclose(phi[0], phi[1])
    np.testing.assert_allclose(logP[0], logP[1])
