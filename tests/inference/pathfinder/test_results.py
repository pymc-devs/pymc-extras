from collections import Counter

import numpy as np

from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus
from pymc_extras.inference.pathfinder.results import (
    MultiPathfinderResult,
    PathfinderResult,
    PathStatus,
    _get_status_warning,
)

M, N = 5, 2


def _success(seed: int) -> PathfinderResult:
    rng = np.random.default_rng(seed)
    return PathfinderResult(
        samples=rng.normal(size=(1, M, N)),
        logP=rng.normal(size=(1, M)),
        logQ=rng.normal(size=(1, M)),
        lbfgs_niter=np.array([10]),
        elbo_argmax=np.array([3]),
        inv_hessian_diag=np.abs(rng.normal(size=(1, N))),
        lbfgs_status=LBFGSStatus.CONVERGED,
        path_status=PathStatus.SUCCESS,
    )


def _failed() -> PathfinderResult:
    return PathfinderResult(
        lbfgs_status=LBFGSStatus.LBFGS_FAILED, path_status=PathStatus.LBFGS_FAILED
    )


def test_from_path_results_aggregates_successful_paths():
    mpr = MultiPathfinderResult.from_path_results([_success(0), _success(1), _success(2)])

    assert not mpr.all_paths_failed
    assert mpr.samples.shape == (3, M, N)
    assert mpr.logP.shape == (3, M)
    assert mpr.lbfgs_niter.shape == (3,)
    assert mpr.path_status[PathStatus.SUCCESS] == 3
    assert mpr.lbfgs_status[LBFGSStatus.CONVERGED] == 3


def test_from_path_results_drops_failed_but_counts_them():
    mpr = MultiPathfinderResult.from_path_results([_success(0), _failed(), _success(1)])

    # Only the two successful paths contribute samples ...
    assert not mpr.all_paths_failed
    assert mpr.samples.shape == (2, M, N)
    # ... but every path is counted in the status tallies
    assert mpr.path_status[PathStatus.SUCCESS] == 2
    assert mpr.path_status[PathStatus.LBFGS_FAILED] == 1


def test_from_path_results_all_failed():
    mpr = MultiPathfinderResult.from_path_results([_failed(), _failed()])

    assert mpr.all_paths_failed
    assert mpr.samples is None
    assert mpr.path_status[PathStatus.LBFGS_FAILED] == 2


def test_get_status_warning_reports_problem_statuses():
    mpr = MultiPathfinderResult(
        lbfgs_status=Counter({LBFGSStatus.MAX_ITER_REACHED: 1}),
        path_status=Counter({PathStatus.SINGLE_STEP: 1}),
    )
    warnings = _get_status_warning(mpr)

    assert any(w.startswith("MAX_ITER_REACHED") for w in warnings)
    assert any(w.startswith("SINGLE_STEP") for w in warnings)


def test_get_status_warning_silent_on_success():
    mpr = MultiPathfinderResult(
        lbfgs_status=Counter({LBFGSStatus.CONVERGED: 3}),
        path_status=Counter({PathStatus.SUCCESS: 3}),
    )
    assert _get_status_warning(mpr) == []
