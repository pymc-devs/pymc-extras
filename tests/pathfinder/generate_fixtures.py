#!/usr/bin/env python
"""
Generate LBFGS-history fixtures for T7 model-equivalence tests.

Run once (from the repo root or any directory):

    python tests/pathfinder/generate_fixtures.py

Each fixture is saved as  tests/pathfinder/fixtures/<model_name>.npz  and
contains:
    x_full   – LBFGS position history, shape (L+1, N)
    g_full   – LBFGS gradient history, shape (L+1, N)
    elbo_ref – per-step ELBO values,  shape (L,), computed with ELBO_SEED
"""

import os
import sys

import numpy as np

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pymc.pytensorf import find_rng_nodes, reseed_rngs
from pytensor.compile.mode import Mode

# Make sure the repo root is on the path so we can import both the package
# and the test helpers regardless of where the script is invoked from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pymc_extras.inference.pathfinder.lbfgs import LBFGS, LBFGSStatus
from pymc_extras.inference.pathfinder.pathfinder import (
    DEFAULT_LINKER,
    get_logp_dlogp_of_ravel_inputs,
    make_elbo_fn,
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

    def logp_func(x):
        logp, _ = logp_dlogp_func(x)
        return logp

    def neg_logp_dlogp_func(x):
        logp, dlogp = logp_dlogp_func(x)
        return -logp, -dlogp

    # Deterministic initial point with small fixed jitter
    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data

    rng = np.random.default_rng(LBFGS_JITTER_SEED)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    # Run LBFGS
    lbfgs = LBFGS(neg_logp_dlogp_func, MAXCOR, MAXITER)
    x_full, g_full, niter, status = lbfgs.minimize(x0)

    if status in {LBFGSStatus.INIT_FAILED, LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT}:
        raise RuntimeError(f"[{name}] LBFGS init failed: {status}")

    print(f"[{name}] L+1={x_full.shape[0]}, N={x_full.shape[1]}, status={status.name}")

    # Compute reference ELBO with fixed seed
    elbo_fn = make_elbo_fn(logp_func, MAXCOR, NUM_ELBO_DRAWS, **compile_kwargs)
    rngs = find_rng_nodes(elbo_fn.maker.fgraph.outputs)
    reseed_rngs(rngs, [ELBO_SEED])
    (elbo_ref,) = elbo_fn(x_full, g_full)

    finite_pct = np.mean(np.isfinite(elbo_ref)) * 100
    print(f"[{name}] ELBO finite: {finite_pct:.0f}%,  argmax={np.nanargmax(elbo_ref)}")

    # Save
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    out_path = os.path.join(FIXTURES_DIR, f"{name}.npz")
    np.savez(out_path, x_full=x_full, g_full=g_full, elbo_ref=elbo_ref)
    print(f"[{name}] saved → {out_path}\n")


if __name__ == "__main__":
    for name, factory in MODEL_FACTORIES.items():
        generate_fixture(name, factory)
    print("All fixtures generated.")
