import numpy as np
import pymc as pm
import pytensor
import pytest

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


def test_neg_logp_dlogp_takes_model_dtype():
    with pytensor.config.change_flags(floatX="float32"), pm.Model() as model:
        pm.Normal("x", 0, 1)
    fn = get_neg_logp_dlogp_of_ravel_inputs(model)

    assert fn.maker.fgraph.inputs[0].type.dtype == "float32"


def _pathfinder_sample_fn(N, J, mode, floatX="float64"):
    with pytensor.config.change_flags(floatX=floatX):
        with pm.Model() as model:
            pm.Normal("x", 0, 1, shape=N)
        return make_pathfinder_sample_fn(
            model=model,
            N=N,
            J=J,
            jacobian=True,
            compile_kwargs={"mode": mode},
        )


def _pathfinder_sample_inputs(N, J, M):
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal(N),
        rng.standard_normal(N),
        np.abs(rng.standard_normal(N)) + 0.1,
        rng.standard_normal((N, J)),
        rng.standard_normal((N, J)),
        rng.standard_normal((M, N)),
    )


_DENSE = (3, 5)
_SPARSE = (8, 2)


@pytest.mark.parametrize("N, J", [_DENSE, _SPARSE], ids=["dense", "sparse"])
def test_make_pathfinder_sample_fn_preserves_float32(N, J):
    M = 7
    fn = _pathfinder_sample_fn(N, J, mode="FAST_RUN", floatX="float32")
    inputs = tuple(np.asarray(a, dtype="float32") for a in _pathfinder_sample_inputs(N, J, M))
    phi, logQ, logP, inv_hessian_diag = fn(*inputs)

    assert phi.shape == (M, N)
    assert logQ.shape == (M,)
    assert logP.shape == (M,)
    assert inv_hessian_diag.shape == (N,)
    for output in (phi, logQ, logP, inv_hessian_diag):
        assert np.asarray(output).dtype == np.float32


@pytest.mark.parametrize("N, J", [_DENSE, _SPARSE], ids=["dense", "sparse"])
def test_make_pathfinder_sample_fn_jax_matches_default_mode(N, J):
    pytest.importorskip("jax")

    M = 7
    inputs = _pathfinder_sample_inputs(N, J, M)
    default_outputs = _pathfinder_sample_fn(N, J, mode="FAST_RUN")(*inputs)
    jax_outputs = _pathfinder_sample_fn(N, J, mode="JAX")(*inputs)

    for default_out, jax_out in zip(default_outputs, jax_outputs):
        np.testing.assert_allclose(np.asarray(jax_out), default_out, rtol=1e-6, atol=1e-6)
