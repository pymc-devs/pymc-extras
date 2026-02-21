"""
T-B3 through T-B9: acceptance tests for the streaming LBFGS pathfinder (Section B).

Pre-refactor fixtures in tests/pathfinder/fixtures/ are NOT regenerated here.
Streaming results are compared against those fixtures to verify correctness.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pytensor.compile.mode import Mode

from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus, _CachedValueGrad
from pymc_extras.inference.pathfinder.pathfinder import (
    DEFAULT_LINKER,
    FAILED_PATH_STATUS,
    LBFGSStreamingCallback,
    PathInvalidLogP,
    PathStatus,
    alpha_recover,
    alpha_step_numpy,
    get_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
    make_single_pathfinder_fn,
)
from tests.pathfinder.equivalence_models import MODEL_FACTORIES, make_iso_gaussian

# ---------------------------------------------------------------------------
# Constants — kept in sync with generate_fixtures.py / test_model_equivalence.py
# ---------------------------------------------------------------------------
FIXTURES_DIR = __import__("os").path.join(__import__("os").path.dirname(__file__), "fixtures")
ELBO_SEED = 12345
MAXCOR = 6
MAXITER = 100
NUM_ELBO_DRAWS = 100
NUM_DRAWS = 50
JITTER = 2.0
EPSILON = 1e-8

COMPILE_KWARGS = {"mode": Mode(linker=DEFAULT_LINKER)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_fn(model, maxiter: int = MAXITER, maxcor: int = MAXCOR):
    return make_single_pathfinder_fn(
        model=model,
        num_draws=NUM_DRAWS,
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=1e-5,
        gtol=1e-8,
        maxls=1000,
        num_elbo_draws=NUM_ELBO_DRAWS,
        jitter=JITTER,
        epsilon=EPSILON,
        compile_kwargs=COMPILE_KWARGS,
    )


def _run_path(fn, seed: int = 42):
    return fn(seed)


def _load_fixture(name: str):
    import os

    path = os.path.join(FIXTURES_DIR, f"{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {path}. Run tests/pathfinder/generate_fixtures.py.")
    with np.load(path) as data:
        return data["x_full"], data["g_full"], data["elbo_ref"]


def _build_logp_and_neg(model, compile_kwargs=COMPILE_KWARGS):
    logp_dlogp = get_logp_dlogp_of_ravel_inputs(model, jacobian=True, **compile_kwargs)

    def logp_func(x):
        v, _ = logp_dlogp(x)
        return v

    def neg_logp_dlogp_func(x):
        v, dv = logp_dlogp(x)
        return -v, -dv

    return logp_func, neg_logp_dlogp_func


# ---------------------------------------------------------------------------
# T-B3: ELBO evaluated exactly once per accepted step
# ---------------------------------------------------------------------------


def test_tb3_elbo_call_count():
    """sample_logp_fn is called exactly once per accepted LBFGS step."""
    model = make_iso_gaussian()
    _, neg_logp_dlogp_func = _build_logp_and_neg(model)
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    sample_logp_fn, s_win_shared, z_win_shared = make_pathfinder_sample_fn(
        model, N, MAXCOR, jacobian=True, compile_kwargs=COMPILE_KWARGS
    )

    call_count = [0]

    def counting_fn(x, g, alpha, u):
        call_count[0] += 1
        return sample_logp_fn(x, g, alpha, u)

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    rng = np.random.default_rng(0)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    cached_fn = _CachedValueGrad(neg_logp_dlogp_func)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=counting_fn,
        s_win_shared=s_win_shared,
        z_win_shared=z_win_shared,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng,
        J=MAXCOR,
        epsilon=EPSILON,
    )

    from scipy.optimize import minimize as scipy_minimize

    scipy_minimize(
        cached_fn,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"maxcor": MAXCOR, "maxiter": MAXITER, "ftol": 1e-5, "gtol": 1e-8, "maxls": 1000},
    )

    assert (
        call_count[0] == cb.step_count
    ), f"sample_logp_fn called {call_count[0]} times but step_count={cb.step_count}"
    assert cb.step_count > 0, "No accepted steps — model may be degenerate"


# ---------------------------------------------------------------------------
# T-B5: RNG reproducibility within refactor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["iso_gaussian", "hd_gaussian"])
def test_tb5_rng_reproducibility(model_name):
    """Two streaming runs with the same seed produce bit-identical results."""
    model = MODEL_FACTORIES[model_name]()
    fn = _make_single_fn(model)

    r1 = _run_path(fn, seed=123)
    r2 = _run_path(fn, seed=123)

    assert r1.path_status not in FAILED_PATH_STATUS
    assert r2.path_status not in FAILED_PATH_STATUS

    np.testing.assert_array_equal(r1.samples, r2.samples)
    np.testing.assert_array_equal(r1.logP, r2.logP)
    np.testing.assert_array_equal(r1.logQ, r2.logQ)
    assert r1.best_step_idx == r2.best_step_idx if hasattr(r1, "best_step_idx") else True


# ---------------------------------------------------------------------------
# T-B6: Short-history fallback (maxiter << J)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["iso_gaussian", "nealsfunnel"])
def test_tb6_short_history_fallback(model_name):
    """Streaming handles partial windows (L < J) via zero-padding without crashing."""
    model = MODEL_FACTORIES[model_name]()

    for maxiter in (1, 2, 3):
        fn = _make_single_fn(model, maxiter=maxiter)
        result = _run_path(fn, seed=99)
        # Should either succeed or fail gracefully (not crash)
        assert result.path_status in list(
            PathStatus
        ), f"[{model_name}, maxiter={maxiter}] unexpected path_status {result.path_status}"
        # If it succeeded, shapes must be correct
        if result.path_status not in FAILED_PATH_STATUS and result.samples is not None:
            N = DictToArrayBijection.map(model.initial_point()).data.shape[0]
            assert result.samples.shape == (1, NUM_DRAWS, N)


# ---------------------------------------------------------------------------
# T-B7: Infinite-logP step tolerance
# ---------------------------------------------------------------------------


def test_tb7_infinite_logp_step_tolerance():
    """A step where sample_logp_fn raises is silently skipped; path still succeeds."""
    model = make_iso_gaussian()
    _, neg_logp_dlogp_func = _build_logp_and_neg(model)
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    sample_logp_fn, s_win_shared, z_win_shared = make_pathfinder_sample_fn(
        model, N, MAXCOR, jacobian=True, compile_kwargs=COMPILE_KWARGS
    )

    # Inject a failure at exactly one step (step index 1)
    fail_at_step = [1]
    call_count = [0]

    def failing_fn(x, g, alpha, u):
        idx = call_count[0]
        call_count[0] += 1
        if idx == fail_at_step[0]:
            raise PathInvalidLogP("synthetic failure for test")
        return sample_logp_fn(x, g, alpha, u)

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    rng = np.random.default_rng(77)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    cached_fn = _CachedValueGrad(neg_logp_dlogp_func)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=failing_fn,
        s_win_shared=s_win_shared,
        z_win_shared=z_win_shared,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng,
        J=MAXCOR,
        epsilon=EPSILON,
    )

    from scipy.optimize import minimize as scipy_minimize

    scipy_minimize(
        cached_fn,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"maxcor": MAXCOR, "maxiter": MAXITER, "ftol": 1e-5, "gtol": 1e-8, "maxls": 1000},
    )

    # Must have seen multiple steps (so the failure at step 1 is not the only one)
    assert cb.step_count >= 2, f"Only {cb.step_count} accepted steps — test not meaningful"

    # any_valid must be True (other steps produced finite ELBOs)
    assert cb.any_valid, "any_valid=False: no finite ELBO found despite other valid steps"

    # best_step_idx must not be the failed step
    assert (
        cb.best_step_idx != fail_at_step[0]
    ), f"best_step_idx={cb.best_step_idx} is the failed step {fail_at_step[0]}"

    # No PathInvalidLogP was raised (we got here without exception)
    # The callback continued normally after the failure


# ---------------------------------------------------------------------------
# T-B8: Both paths vs pre-refactor fixtures (statistical equivalence via LBFGS mock)
# ---------------------------------------------------------------------------


def _make_lbfgs_mock(x_full, g_full):
    """Return a context-manager that patches LBFGS in pathfinder.py.

    Replays x_full[1:] through the streaming callback, re-anchoring the
    callback's prev state to the fixture initial point so that all s/z diffs
    are computed consistently against x_full/g_full.
    """
    mock_inst = MagicMock()
    count = x_full.shape[0]

    def replay_streaming(callback, x0):
        _, g_init = callback.value_grad_fn(x_full[0])
        callback.x_prev = x_full[0].copy()
        callback.g_prev = np.array(g_init, dtype=np.float64)

        for x_k in x_full[1:]:
            callback.value_grad_fn(x_k)  # warm _CachedValueGrad cache
            callback(x_k)

        return count - 1, LBFGSStatus.CONVERGED

    mock_inst.minimize_streaming.side_effect = replay_streaming

    patcher = patch("pymc_extras.inference.pathfinder.pathfinder.LBFGS", return_value=mock_inst)
    return patcher


@pytest.mark.parametrize("model_name", list(MODEL_FACTORIES.keys()))
def test_tb8_fixture_match(model_name):
    """Streaming path selects an ELBO argmax consistent with the pre-refactor fixture.

    LBFGS is mocked so the streaming path processes the exact same trajectory
    stored in the fixture.  ELBO computations use fresh RNG draws so exact
    argmax match is not required; we check that it falls within a generous
    window of the fixture reference argmax and that outputs are finite.
    """
    x_full, g_full, elbo_ref = _load_fixture(model_name)
    ref_argmax = int(np.nanargmax(elbo_ref))
    L = x_full.shape[0] - 1  # number of accepted steps (excluding x0)

    model = MODEL_FACTORIES[model_name]()
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    with _make_lbfgs_mock(x_full, g_full):
        fn = _make_single_fn(model)
        result = _run_path(fn, seed=42)

    tag = f"[{model_name}]"

    assert result.path_status not in FAILED_PATH_STATUS, f"{tag} path failed: {result.path_status}"
    assert result.samples is not None
    assert result.samples.shape == (
        1,
        NUM_DRAWS,
        N,
    ), f"{tag} unexpected samples shape {result.samples.shape}"
    assert np.any(np.isfinite(result.logP)), f"{tag} all logP are non-finite"
    assert np.any(np.isfinite(result.logQ)), f"{tag} all logQ are non-finite"

    argmax = int(result.elbo_argmax)
    tolerance = max(3, L // 4)
    assert abs(argmax - ref_argmax) <= tolerance, (
        f"{tag} elbo_argmax={argmax} too far from fixture ref_argmax={ref_argmax} "
        f"(L={L}, tolerance={tolerance})"
    )


# ---------------------------------------------------------------------------
# T-B9: alpha_step_numpy vs compute_alpha_l parity
# ---------------------------------------------------------------------------


def _make_valid_trajectory(rng, L: int, N: int):
    """Construct (x_full, g_full) of shape (L+1, N) such that all s·z > 0.

    Uses a diagonal quadratic f(x) = 0.5 * x @ diag(w) @ x so that
    grad = diag(w) @ x and the curvature condition s·z = s @ diag(w) @ s > 0
    holds for any non-zero step.
    """
    w = rng.uniform(0.5, 2.0, size=N)  # positive diagonal Hessian
    x_full = np.zeros((L + 1, N))
    x_full[0] = rng.standard_normal(N)
    # small random steps to stay away from zero
    for i in range(1, L + 1):
        x_full[i] = x_full[i - 1] + rng.uniform(0.01, 0.1, size=N) * np.sign(rng.standard_normal(N))
    g_full = x_full * w[None, :]  # gradient of 0.5 * x @ diag(w) @ x
    return x_full, g_full


def test_tb9_alpha_step_numpy_parity():
    """alpha_step_numpy (numpy/streaming) matches alpha_recover (PyTensor/batch) exactly.

    Both implementations share the same formula; this test guards against drift.
    """
    rng = np.random.default_rng(42)
    N, L = 8, 10

    x_full, g_full = _make_valid_trajectory(rng, L, N)

    # --- Reference: PyTensor alpha_recover ---
    x_pt = pt.matrix("x", dtype="float64")
    g_pt = pt.matrix("g", dtype="float64")
    alpha_sym, _, _ = alpha_recover(x_pt, g_pt)
    alpha_fn = pytensor.function([x_pt, g_pt], alpha_sym)
    alpha_ref = alpha_fn(x_full, g_full)  # (L, N)

    # --- Streaming: alpha_step_numpy iterated ---
    s_all = np.diff(x_full, axis=0)  # (L, N)
    z_all = np.diff(g_full, axis=0)  # (L, N)
    alpha_prev = np.ones(N, dtype=np.float64)
    alpha_np = np.empty((L, N), dtype=np.float64)
    for step_idx in range(L):
        alpha_prev = alpha_step_numpy(alpha_prev, s_all[step_idx], z_all[step_idx])
        alpha_np[step_idx] = alpha_prev

    np.testing.assert_allclose(
        alpha_np,
        alpha_ref,
        rtol=1e-10,
        atol=1e-10,
        err_msg="alpha_step_numpy diverged from alpha_recover (PyTensor)",
    )
