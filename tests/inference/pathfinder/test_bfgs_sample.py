import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.pathfinder.bfgs_sample import (
    alpha_step_numpy,
    get_logp_dlogp_of_ravel_inputs,
    get_neg_logp_dlogp_of_ravel_inputs,
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
