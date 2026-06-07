from collections import Counter
from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytest

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pytensor.tensor.optimize import LRUCache1

import pymc_extras as pmx

from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs
from pymc_extras.inference.pathfinder.lbfgs import LBFGSConfig, LBFGSStatus, LBFGSStreamingCallback
from pymc_extras.inference.pathfinder.results import PathfinderConfig, PathStatus
from pymc_extras.inference.pathfinder.single_path import make_single_pathfinder_fn

# ---------------------------------------------------------------------------
# Models and fixtures
# ---------------------------------------------------------------------------


def eight_schools() -> pm.Model:
    """Centred eight-schools model shared by the pathfinder end-to-end tests."""
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)
        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        pm.Normal("obs", mu=mu + tau * theta, sigma=sigma, shape=J, observed=y)
    return model


@pytest.fixture
def eight_schools_model() -> pm.Model:
    return eight_schools()


@pytest.fixture(scope="module")
def reference_idata():
    with eight_schools():
        idata = pmx.fit(
            method="pathfinder",
            num_paths=10,
            jitter=12.0,
            random_seed=41,
        )
    return idata


# ---------------------------------------------------------------------------
# Streaming single-path helpers (callbacks / single_path / model_equivalence)
# ---------------------------------------------------------------------------

MAXCOR = 6
MAXITER = 100
NUM_DRAWS = 50
NUM_ELBO_DRAWS = 100
JITTER = 2.0
STREAMING_EPSILON = 1e-8


def make_single_fn(model, maxiter=MAXITER, maxcor=MAXCOR, max_init_retries=10, vectorize=True):
    """Build a single-path pathfinder function with the shared test configuration."""
    return make_single_pathfinder_fn(
        model=model,
        num_draws=NUM_DRAWS,
        num_elbo_draws=NUM_ELBO_DRAWS,
        jitter=JITTER,
        lbfgs_config=LBFGSConfig(
            maxcor=maxcor,
            maxiter=maxiter,
            ftol=1e-5,
            gtol=1e-8,
            maxls=1000,
            epsilon=STREAMING_EPSILON,
        ),
        max_init_retries=max_init_retries,
        vectorize_logp=vectorize,
    )


def run_streaming_lbfgs(model, sample_logp_fn, *, seed):
    """Run one streaming L-BFGS optimization, returning (callback, rng, N)."""
    from scipy.optimize import minimize as scipy_minimize

    neg_logp_dlogp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    rng = np.random.default_rng(seed)
    x0 = x_base + rng.uniform(-JITTER, JITTER, size=x_base.shape)

    cached_fn = LRUCache1(neg_logp_dlogp, copy_x=True)
    cb = LBFGSStreamingCallback(
        value_grad_fn=cached_fn,
        x0=x0,
        sample_logp_fn=sample_logp_fn,
        num_elbo_draws=NUM_ELBO_DRAWS,
        rng=rng,
        J=MAXCOR,
        epsilon=STREAMING_EPSILON,
    )

    scipy_minimize(
        cached_fn,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"maxcor": MAXCOR, "maxiter": MAXITER, "ftol": 1e-5, "gtol": 1e-8, "maxls": 1000},
    )

    return cb, rng, N


# ---------------------------------------------------------------------------
# MultiPathfinderResult mock and builders (idata / report tests)
# ---------------------------------------------------------------------------


@dataclass
class MockMultiPathfinderResult:
    """Lightweight stand-in for MultiPathfinderResult used by idata/report tests.

    Kept structurally in sync with the real dataclass by
    ``test_idata.test_mock_result_is_subset_of_real_schema``.
    """

    samples: np.ndarray = None
    logP: np.ndarray = None
    logQ: np.ndarray = None
    lbfgs_niter: np.ndarray = None
    elbo_argmax: np.ndarray = None
    inv_hessian_diag: np.ndarray = None
    lbfgs_status: Counter = None
    path_status: Counter = None
    importance_sampling: str = "psis"
    warnings: list = None
    pareto_k: float = None
    num_paths: int = None
    num_draws: int = None
    pathfinder_config: PathfinderConfig = None
    compile_time: float = None
    compute_time: float = None
    all_paths_failed: bool = False

    def __post_init__(self):
        if self.lbfgs_status is None:
            self.lbfgs_status = Counter()
        if self.path_status is None:
            self.path_status = Counter()
        if self.warnings is None:
            self.warnings = []


def make_config(**overrides) -> PathfinderConfig:
    base = dict(
        num_draws=100,
        maxcor=5,
        maxiter=1000,
        ftol=1e-5,
        gtol=1e-8,
        maxls=1000,
        jitter=2.0,
        epsilon=1e-8,
        num_elbo_draws=10,
    )
    base.update(overrides)
    return PathfinderConfig(**base)


def make_result(**overrides) -> MockMultiPathfinderResult:
    rng = np.random.default_rng(0)
    base = dict(
        samples=rng.normal(0, 1, (3, 100, 2)),
        logP=rng.normal(-10, 1, (3, 100)),
        logQ=rng.normal(-11, 1, (3, 100)),
        lbfgs_niter=np.array([50, 45, 55]),
        elbo_argmax=np.array([25, 30, 20]),
        inv_hessian_diag=np.abs(rng.normal(1.0, 0.1, (3, 2))),
        lbfgs_status=Counter({LBFGSStatus.CONVERGED: 3}),
        path_status=Counter({PathStatus.SUCCESS: 3}),
        num_paths=3,
        num_draws=300,
        compile_time=1.5,
        compute_time=10.2,
        pareto_k=0.5,
        pathfinder_config=make_config(),
    )
    base.update(overrides)
    return MockMultiPathfinderResult(**base)
