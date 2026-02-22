#!/usr/bin/env python
"""
Generate LBFGS-history fixtures for model-equivalence tests.

Run once (from the repo root or any directory):

    python tests/pathfinder/generate_fixtures.py

Each fixture is saved as  tests/pathfinder/fixtures/<model_name>.npz  and
contains:
    x_full     – initial point + accepted step positions, shape (L+1, N)
    g_full     – gradients at each row of x_full, shape (L+1, N)
    elbo_ref   – per-step ELBO values, shape (L,), from LBFGSStreamingCallback

x_full[0] / g_full[0] is the starting point (x0); x_full[1:] / g_full[1:]
are the L accepted LBFGS steps. Alpha, s, z are computed internally by the
callback (full stack); tests replay the trajectory through the same callback.
"""

import os
import sys

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pytensor.compile.mode import Mode

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pymc_extras.inference.pathfinder.lbfgs import LBFGS, LBFGSStatus
from pymc_extras.inference.pathfinder.pathfinder import (
    DEFAULT_LINKER,
    LBFGSStreamingCallback,
    _CachedValueGrad,
    get_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
)
from tests.pathfinder.equivalence_models import MODEL_FACTORIES

# ---------------------------------------------------------------------------
# Parameters - must stay in sync with test_model_equivalence.py
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ELBO_SEED = 12345
LBFGS_JITTER_SEED = 42
JITTER = 5.0
MAXCOR = 6
MAXITER = 100
NUM_ELBO_DRAWS = 100


def generate_fixture(name: str, model_fn) -> None:
    print(f"[{name}] building model …")
    model = model_fn()

    compile_kwargs = {"mode": Mode(linker=DEFAULT_LINKER)}
    logp_dlogp_func = get_logp_dlogp_of_ravel_inputs(model, jacobian=True, **compile_kwargs)

    def neg_logp_dlogp_func(x):
        logp, dlogp = logp_dlogp_func(x)
        return -logp, -dlogp

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    N = x_base.shape[0]

    rng = np.random.default_rng(LBFGS_JITTER_SEED)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    # Compute gradient at x0 for the initial row of g_full
    _, neg_g0 = neg_logp_dlogp_func(x0)
    g0 = -neg_g0

    # Run LBFGS with full-stack callback (alpha, s, z computed internally).
    step_records = []  # (x, g, elbo_val)

    def _on_step(x, g, alpha, s_win, z_win, elbo):
        elbo_val = elbo if np.isfinite(elbo) else np.nan
        step_records.append((x.copy(), g.copy(), elbo_val))

    sample_logp_fn = make_pathfinder_sample_fn(
        model, N, MAXCOR, jacobian=True, compile_kwargs=compile_kwargs
    )
    cached_fn = _CachedValueGrad(neg_logp_dlogp_func)
    rng_elbo = np.random.default_rng(ELBO_SEED)
    lbfgs = LBFGS(neg_logp_dlogp_func, MAXCOR, MAXITER)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=sample_logp_fn,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng_elbo,
        J=MAXCOR,
        epsilon=1e-12,
        on_step_callback=_on_step,
    )
    niter, status = lbfgs.minimize_streaming(cb, x0)

    if status in {LBFGSStatus.INIT_FAILED, LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT}:
        raise RuntimeError(f"[{name}] LBFGS init failed: {status}")
    if not step_records:
        raise RuntimeError(f"[{name}] no accepted steps")

    L = len(step_records)
    # x_full/g_full: initial point prepended → shape (L+1, N)
    x_full = np.concatenate([[x0], [r[0] for r in step_records]], axis=0)
    g_full = np.concatenate([[g0], [r[1] for r in step_records]], axis=0)
    elbo_ref = np.array([r[2] for r in step_records])

    finite_pct = np.mean(np.isfinite(elbo_ref)) * 100
    print(f"[{name}] L={L}, N={x0.shape[0]}, status={status.name}")
    print(f"[{name}] ELBO finite: {finite_pct:.0f}%,  argmax={np.nanargmax(elbo_ref)}")

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    out_path = os.path.join(FIXTURES_DIR, f"{name}.npz")
    np.savez(out_path, x_full=x_full, g_full=g_full, elbo_ref=elbo_ref)
    print(f"[{name}] saved → {out_path}\n")


if __name__ == "__main__":
    for name, factory in MODEL_FACTORIES.items():
        generate_fixture(name, factory)
    print("All fixtures generated.")
