from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Literal, Self

import numpy as np

from numpy.typing import NDArray

from pymc_extras.inference.pathfinder.importance_sampling import (
    importance_sampling as _importance_sampling,
)
from pymc_extras.inference.pathfinder.lbfgs import LBFGSStatus


class PathStatus(Enum):
    """
    Statuses of a single-path pathfinder.
    """

    SUCCESS = auto()
    ELBO_ARGMAX_AT_ZERO = auto()
    # Statuses that lead to Exceptions:
    INVALID_LOGP = auto()
    INVALID_LOGQ = auto()
    LBFGS_FAILED = auto()
    PATH_FAILED = auto()
    SINGLE_STEP = auto()


FAILED_PATH_STATUS = [
    PathStatus.INVALID_LOGP,
    PathStatus.INVALID_LOGQ,
    PathStatus.LBFGS_FAILED,
    PathStatus.PATH_FAILED,
    PathStatus.SINGLE_STEP,
]


class PathException(Exception):
    """
    raises a PathException if the path failed.
    """

    DEFAULT_MESSAGE = "Path failed."

    def __init__(self, message=None, status: PathStatus = PathStatus.PATH_FAILED) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)
        self.status = status


class PathInvalidLogP(PathException):
    """
    raises a PathException if all the logP values in a path are not finite.
    """

    DEFAULT_MESSAGE = "Path failed because all the logP values in a path are not finite."

    def __init__(self, message=None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE, PathStatus.INVALID_LOGP)


class PathInvalidLogQ(PathException):
    """
    raises a PathException if all the logQ values in a path are not finite.
    """

    DEFAULT_MESSAGE = "Path failed because all the logQ values in a path are not finite."

    def __init__(self, message=None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE, PathStatus.INVALID_LOGQ)


class SingleStepPathException(PathException):
    """
    raises when the path has only one LBFGS step (insufficient for valid approximation).
    """

    DEFAULT_MESSAGE = "Path failed because only a single step was performed."

    def __init__(self, message=None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE, PathStatus.SINGLE_STEP)


@dataclass(slots=True, frozen=True)
class PathfinderResult:
    """
    container for storing results from a single pathfinder run.

    Attributes
    ----------
        samples: posterior samples (1, M, N)
        logP: log probability of model (1, M)
        logQ: log probability of approximation (1, M)
        lbfgs_niter: number of lbfgs iterations (1,)
        elbo_argmax: elbo values at convergence (1,)
        lbfgs_status: LBFGS status
        path_status: path status

    where:
        M: number of samples
        N: number of parameters
    """

    samples: NDArray | None = None
    logP: NDArray | None = None
    logQ: NDArray | None = None
    lbfgs_niter: NDArray | None = None
    elbo_argmax: NDArray | None = None
    inv_hessian_diag: NDArray | None = None
    lbfgs_status: LBFGSStatus = LBFGSStatus.LBFGS_FAILED
    path_status: PathStatus = PathStatus.PATH_FAILED


@dataclass(frozen=True)
class PathfinderConfig:
    """configuration parameters for a single pathfinder"""

    num_draws: int  # same as num_draws_per_path
    maxcor: int
    maxiter: int
    ftol: float
    gtol: float
    maxls: int
    jitter: float
    epsilon: float
    num_elbo_draws: int


@dataclass(slots=True, frozen=True)
class MultiPathfinderResult:
    """
    container for aggregating results from multiple paths.

    Attributes
    ----------
        samples: posterior samples (S, M, N)
        logP: log probability of model (S, M)
        logQ: log probability of approximation (S, M)
        lbfgs_niter: number of lbfgs iterations (S,)
        elbo_argmax: elbo values at convergence (S,)
        lbfgs_status: counter for LBFGS status occurrences
        path_status: counter for path status occurrences
        importance_sampling: importance sampling method used
        warnings: list of warnings
        pareto_k
        pathfinder_config: pathfinder configuration
        compile_time
        compute_time
    where:
        S: number of successful paths, where S <= num_paths
        M: number of samples per path
        N: number of parameters
    """

    samples: NDArray | None = None
    logP: NDArray | None = None
    logQ: NDArray | None = None
    lbfgs_niter: NDArray | None = None
    elbo_argmax: NDArray | None = None
    inv_hessian_diag: NDArray | None = None
    lbfgs_status: Counter = field(default_factory=Counter)
    path_status: Counter = field(default_factory=Counter)
    importance_sampling: str | None = "psis"
    warnings: list[str] = field(default_factory=list)
    pareto_k: float | None = None

    # config
    num_paths: int | None = None
    num_draws: int | None = None
    pathfinder_config: PathfinderConfig | None = None

    # timing
    compile_time: float | None = None
    compute_time: float | None = None

    all_paths_failed: bool = False  # raises ValueError if all paths failed

    @classmethod
    def from_path_results(cls, path_results: list[PathfinderResult]) -> "MultiPathfinderResult":
        """Aggregate successful path results and tally PathStatus/LBFGSStatus occurrences."""

        NUMERIC_ATTRIBUTES = [
            "samples",
            "logP",
            "logQ",
            "lbfgs_niter",
            "elbo_argmax",
            "inv_hessian_diag",
        ]

        success_results = []
        mpr = cls()

        for pr in path_results:
            if pr.path_status not in FAILED_PATH_STATUS:
                success_results.append(tuple(getattr(pr, attr) for attr in NUMERIC_ATTRIBUTES))

            mpr.lbfgs_status[pr.lbfgs_status] += 1
            mpr.path_status[pr.path_status] += 1

        warnings = _get_status_warning(mpr)

        if success_results:
            results_arr = [np.asarray(x) for x in zip(*success_results)]
            numeric_fields = {
                name: (np.concatenate(arr) if arr.ndim > 1 else arr)
                for name, arr in zip(NUMERIC_ATTRIBUTES, results_arr)
            }
            return cls(
                **numeric_fields,
                lbfgs_status=mpr.lbfgs_status,
                path_status=mpr.path_status,
                warnings=warnings,
            )
        else:
            return cls(
                lbfgs_status=mpr.lbfgs_status,
                path_status=mpr.path_status,
                warnings=warnings,
                all_paths_failed=True,  # raises ValueError later
            )

    def with_timing(self, compile_time: float, compute_time: float) -> Self:
        """Add timing information."""
        return replace(self, compile_time=compile_time, compute_time=compute_time)

    def with_pathfinder_config(self, config: PathfinderConfig) -> Self:
        """Add pathfinder configuration."""
        return replace(self, pathfinder_config=config)

    def with_importance_sampling(
        self,
        num_draws: int,
        method: Literal["psis", "psir", "identity"] | None,
        random_seed: int | None = None,
    ) -> Self:
        """Perform importance sampling."""
        if not self.all_paths_failed:
            isres = _importance_sampling(
                samples=self.samples,
                logP=self.logP,
                logQ=self.logQ,
                num_draws=num_draws,
                method=method,
                random_seed=random_seed,
            )
            return replace(
                self,
                samples=isres.samples,
                importance_sampling=method,
                warnings=[*self.warnings, *isres.warnings],
                pareto_k=isres.pareto_k,
            )
        else:
            return self


def _get_status_warning(mpr: MultiPathfinderResult) -> list[str]:
    """get list of relevant LBFGSStatus and PathStatus warnings given a MultiPathfinderResult"""
    warnings = []

    lbfgs_status_message = {
        LBFGSStatus.MAX_ITER_REACHED: (
            "MAX_ITER_REACHED: LBFGS maximum number of iterations reached. Consider increasing "
            "maxiter if this occurrence is high relative to the number of paths."
        ),
        LBFGSStatus.INIT_FAILED: (
            "INIT_FAILED: LBFGS failed to initialize. Consider reparameterizing the model or "
            "reducing jitter if this occurrence is high relative to the number of paths."
        ),
        LBFGSStatus.NON_FINITE: (
            "NON_FINITE: LBFGS objective function produced inf or nan at the last iteration. "
            "Consider reparameterizing the model or adjusting the pathfinder arguments if this "
            "occurrence is high relative to the number of paths."
        ),
        LBFGSStatus.LOW_UPDATE_PCT: (
            "LOW_UPDATE_PCT: Majority of LBFGS iterations were not accepted due to the either: "
            "(1) LBFGS function or gradient values containing too many inf or nan values or "
            "(2) gradient changes being significantly large, set by epsilon. Consider "
            "reparameterizing the model, adjusting initvals or jitter or other pathfinder "
            "arguments if this occurrence is high relative to the number of paths."
        ),
        LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT: (
            "INIT_FAILED_LOW_UPDATE_PCT: LBFGS failed to initialize due to the either: "
            "(1) LBFGS function or gradient values containing too many inf or nan values or "
            "(2) gradient changes being significantly large, set by epsilon. Consider "
            "reparameterizing the model, adjusting initvals or jitter or other pathfinder "
            "arguments if this occurrence is high relative to the number of paths."
        ),
    }

    path_status_message = {
        PathStatus.ELBO_ARGMAX_AT_ZERO: (
            "ELBO_ARGMAX_AT_ZERO: ELBO argmax at zero refers to the first iteration during "
            "LBFGS. A high occurrence suggests the model's default initial point + jitter "
            "values are concentrated in high-density regions in the target distribution and "
            "may result in poor exploration of the parameter space. Consider increasing "
            "jitter if this occurrence is high relative to the number of paths."
        ),
        PathStatus.INVALID_LOGQ: (
            "INVALID_LOGQ: Invalid logQ values occur when a path's logQ values are not "
            "finite. The failed path is not included in samples when importance sampling is "
            "used. Consider reparameterizing the model or adjusting the pathfinder arguments "
            "if this occurrence is high relative to the number of paths."
        ),
        PathStatus.SINGLE_STEP: (
            "SINGLE_STEP: Pathfinder requires at least two LBFGS steps on a path. A path with "
            "only one step produces an invalid result. Consider adjusting initvals/jitter if "
            "this occurs."
        ),
    }

    for lbfgs_status in mpr.lbfgs_status:
        if lbfgs_status in lbfgs_status_message:
            warnings.append(lbfgs_status_message.get(lbfgs_status))

    for path_status in mpr.path_status:
        if path_status in path_status_message:
            warnings.append(path_status_message.get(path_status))

    return warnings
