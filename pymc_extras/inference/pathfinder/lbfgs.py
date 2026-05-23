import logging
import time

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Self

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import minimize

from pymc_extras.inference.pathfinder.bfgs_sample import alpha_step_numpy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LBFGSConfig:
    r"""Configuration for Pathfinder's L-BFGS optimizer.

    Parameters
    ----------
    maxcor : int, optional
        L-BFGS history size (number of variable-metric corrections). If None, Pathfinder
        derives a default from the number of model parameters :math:`N`,
        :math:`\max(\lceil 3 \ln N \rceil, 5)`. Default None.
    maxiter : int, optional
        Maximum L-BFGS iterations. Default 1000.
    ftol : float, optional
        Tolerance for the decrease in the objective function. Default 1e-5.
    gtol : float, optional
        Tolerance for the norm of the gradient. Default 1e-8.
    maxls : int, optional
        Maximum line-search steps per iteration. Default 1000.
    epsilon : float, optional
        Curvature-condition threshold for accepting an L-BFGS step,
        :math:`s \cdot z \ge \epsilon \lVert z \rVert^2` (Zhang et al. 2022, Alg. 3).
        Default 1e-12.
    """

    maxcor: int | None = None
    maxiter: int = 1000
    ftol: float = 1e-5
    gtol: float = 1e-8
    maxls: int = 1000
    epsilon: float = 1e-12

    def set_default_maxcor(self, N: int) -> Self:
        """Set a default for ``maxcor`` based on N, if it was None."""
        if self.maxcor is not None:
            return self

        # Set heuristically after testing; higher values rarely helped and slow the algorithm.
        return replace(self, maxcor=max(int(np.ceil(3 * np.log(N))), 5))


def _check_lbfgs_curvature_condition(s: NDArray, z: NDArray, epsilon: float) -> bool:
    r"""Check the L-BFGS curvature condition :math:`s \cdot z \ge \epsilon \lVert z \rVert^2`.

    Implements the step-acceptance test from Zhang et al. (2022), Algorithm 3.
    """
    sz = float((s * z).sum())
    return sz >= epsilon * float(np.sum(z**2))


class LBFGSStatus(Enum):
    CONVERGED = auto()
    MAX_ITER_REACHED = auto()
    NON_FINITE = auto()
    LOW_UPDATE_PCT = auto()
    # Statuses that lead to Exceptions:
    INIT_FAILED = auto()
    INIT_FAILED_LOW_UPDATE_PCT = auto()
    LBFGS_FAILED = auto()


class LBFGSException(Exception):
    DEFAULT_MESSAGE = "LBFGS failed."

    def __init__(self, message=None, status: LBFGSStatus = LBFGSStatus.LBFGS_FAILED):
        super().__init__(message or self.DEFAULT_MESSAGE)
        self.status = status


class LBFGSInitFailed(LBFGSException):
    DEFAULT_MESSAGE = "LBFGS failed to initialize."

    def __init__(self, status: LBFGSStatus, message=None):
        super().__init__(message or self.DEFAULT_MESSAGE, status)


class LBFGS:
    """L-BFGS optimizer wrapper around scipy's implementation.

    Parameters
    ----------
    value_grad_fn : Callable
        function that returns tuple of (value, gradient) given input x
    maxcor : int
        maximum number of variable metric corrections
    maxiter : int
        maximum number of iterations, defaults to 1000
    ftol : float
        function tolerance for convergence, defaults to 1e-5
    gtol : float
        gradient tolerance for convergence, defaults to 1e-8
    maxls : int
        maximum number of line search steps, defaults to 1000
    epsilon : float
        tolerance for lbfgs update, defaults to 1e-8
    """

    def __init__(
        self,
        value_grad_fn,
        maxcor: int,
        maxiter: int = 1000,
        ftol: float = 1e-5,
        gtol: float = 1e-8,
        maxls: int = 1000,
        epsilon: float = 1e-12,
    ) -> None:
        self.value_grad_fn = value_grad_fn
        self.maxcor = maxcor
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls
        self.epsilon = epsilon

    @property
    def _scipy_options(self) -> dict:
        return {
            "maxcor": self.maxcor,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "maxls": self.maxls,
        }

    def _classify_status(self, result, update_count: int) -> LBFGSStatus:
        """Classify the LBFGS termination status.

        Parameters
        ----------
        result : OptimizeResult
            scipy result object.
        update_count : int
            Number of accepted history entries **including the initial point**.
            For non-streaming this is ``history.count``; for streaming it is
            ``step_count + 1``.
        """
        low_update_threshold = 3
        if update_count <= 1:  # triggers LBFGSInitFailed
            return (
                LBFGSStatus.INIT_FAILED
                if result.nit < low_update_threshold
                else LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT
            )
        elif result.status == 1:
            # (result.nit > maxiter) or (result.nit > maxls)
            return LBFGSStatus.MAX_ITER_REACHED
        elif result.status == 2:
            # precision loss resulting to inf or nan
            return LBFGSStatus.NON_FINITE
        elif update_count * low_update_threshold < result.nit:
            return LBFGSStatus.LOW_UPDATE_PCT
        else:
            return LBFGSStatus.CONVERGED

    def minimize_streaming(self, callback, x0) -> tuple[int, LBFGSStatus]:
        """Minimize objective using a streaming callback that processes each step.

        Unlike :meth:`minimize`, no position/gradient history is accumulated.
        The ``callback`` is responsible for maintaining whatever per-step state
        it needs (e.g. ring buffers, best-ELBO tracking).

        ``callback.value_grad_fn`` is used as the scipy objective so that a
        single-entry cache (e.g. :class:`pytensor.tensor.optimize.LRUCache1`) eliminates the
        duplicate evaluation that would otherwise occur on each accepted step.

        Parameters
        ----------
        callback : object
            Must expose:
            - ``value_grad_fn``: callable ``(x) -> (value, grad)`` passed to scipy
              as the objective.  Wrap with :class:`pytensor.tensor.optimize.LRUCache1`
              before constructing the callback to avoid duplicate evaluations.
            - ``step_count``: int, updated by ``__call__`` for each accepted step.
        x0 : array_like
            Initial position.

        Returns
        -------
        step_count : int
            Number of accepted callback steps (does not count the initial point).
        lbfgs_status : LBFGSStatus
        """
        x0 = np.array(x0, dtype=np.float64)
        result = minimize(
            callback.value_grad_fn,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options=self._scipy_options,
        )
        step_count = callback.step_count
        return step_count, self._classify_status(result, step_count + 1)


class LBFGSStreamingCallback:
    """Streaming LBFGS callback: computes ELBO at each accepted step, O(J*N + M*N) peak memory.

    Instead of collecting the full (L+1, N) history, it processes each accepted step
    immediately and tracks only the best state seen so far.

    Parameters
    ----------
    value_grad_fn : Callable
        Single-entry cached value/gradient function (e.g. LRUCache1 from pytensor.tensor.optimize).
    x0 : NDArray
        Initial position, shape (N,).
    sample_logp_fn : Callable
        Compiled PyTensor function (x, g, alpha, s_win, z_win, u) → (phi, logQ, logP).
        Built by make_pathfinder_sample_fn.
    num_elbo_draws : int
        Number of draws per step for ELBO estimation.
    rng : np.random.Generator
        Random number generator for draw generation.
    J : int
        L-BFGS history size (maxcor).
    epsilon : float
        Tolerance for the LBFGS update condition.
    progress_callback : Callable | None
        Optional progress reporting.
    on_step_callback : Callable | None
        If set, called after each accepted step with (x, g, alpha, s_win, z_win, elbo).
        Used by fixture generation to record per-step state.
    """

    def __init__(
        self,
        value_grad_fn: Callable,
        x0: NDArray,
        sample_logp_fn: Callable,
        num_elbo_draws: int,
        rng: np.random.Generator,
        J: int,
        epsilon: float,
        progress_callback: Callable | None = None,
        on_step_callback: Callable | None = None,
    ) -> None:
        self.value_grad_fn = value_grad_fn
        self.sample_logp_fn = sample_logp_fn
        self.num_elbo_draws = num_elbo_draws
        self._rng = rng
        self.J = J
        self.epsilon = epsilon
        self.progress_callback = progress_callback
        self.on_step_callback = on_step_callback

        N = x0.shape[0]
        self._N = N
        _, g0 = value_grad_fn(x0)

        self.x_prev: NDArray = x0.copy()
        self.g_prev: NDArray = np.array(g0, dtype=np.float64)
        self.alpha_prev: NDArray = np.ones(N, dtype=np.float64)

        # Ring buffer: numpy arrays passed as inputs to sample_logp_fn each call.
        # Thread-safe: no shared mutable state across concurrent invocations.
        self.s_win: NDArray = np.zeros((N, J), dtype=np.float64)
        self.z_win: NDArray = np.zeros((N, J), dtype=np.float64)
        self.win_idx: int = -1
        self.best_elbo: float = -np.inf
        self.best_state: dict = {}
        self.best_step_idx: int = 0
        self.step_count: int = 0
        self.any_valid: bool = False
        self.current_elbo: float | None = None
        self._start_time: float = time.time()

    def __call__(self, x: NDArray) -> float | None:
        """Process one accepted LBFGS step. Returns current_elbo for testability."""
        value, g = self.value_grad_fn(x)

        s = x - self.x_prev
        z = g - self.g_prev

        if not (np.all(np.isfinite(g)) and np.isfinite(value)):
            self.current_elbo = None
            return None
        if not _check_lbfgs_curvature_condition(s, z, self.epsilon):
            self.current_elbo = None
            return None

        alpha = alpha_step_numpy(self.alpha_prev, s, z)

        # Ring-buffer update (numpy, O(N))
        self.win_idx = (self.win_idx + 1) % self.J
        self.s_win[:, self.win_idx] = s
        self.z_win[:, self.win_idx] = z

        # Sample + logP in a single compiled call. Pass s_win/z_win as inputs.
        u = self._rng.standard_normal((self.num_elbo_draws, self._N))
        try:
            sample_out = self.sample_logp_fn(x, g, alpha, self.s_win, self.z_win, u)
            _, logQ, logP = sample_out[:3]
            logP = np.asarray(logP)
            logQ = np.asarray(logQ)
            finite = np.isfinite(logP)
            if not np.any(finite):
                elbo = -np.inf
            else:
                logP_safe = np.where(finite, logP, -np.inf)
                elbo = float(np.mean(logP_safe - logQ))
                if not np.isfinite(elbo):
                    elbo = -np.inf
        except Exception:
            # A bad step (e.g. singular sample covariance or non-finite draws) must not abort
            # the path; record it as worthless so optimization continues.
            elbo = -np.inf

        if np.isfinite(elbo):
            self.any_valid = True

        if self.on_step_callback is not None:
            self.on_step_callback(x, g, alpha, self.s_win.copy(), self.z_win.copy(), elbo)

        if elbo > self.best_elbo:
            self.best_elbo = elbo
            self.best_state = {
                "alpha": alpha.copy(),
                "s_win": self.s_win.copy(),
                "z_win": self.z_win.copy(),
                "win_idx": self.win_idx,
                "x": x.copy(),
                "g": g.copy(),
            }
            self.best_step_idx = self.step_count

        self.alpha_prev = alpha
        self.x_prev = x.copy()
        self.g_prev = g.copy()
        self.step_count += 1

        self.current_elbo = elbo if np.isfinite(elbo) else None

        if self.progress_callback is not None:
            best_elbo = self.best_elbo if np.isfinite(self.best_elbo) else None
            current_elbo = self.current_elbo
            elapsed = time.time() - self._start_time
            steps_per_sec = self.step_count / elapsed if elapsed > 0 else None
            step_size = float(np.linalg.norm(s))
            self.progress_callback(
                {
                    "lbfgs_steps": self.step_count,
                    "best_elbo": best_elbo,
                    "best_ind": self.best_step_idx,
                    "current_elbo": current_elbo,
                    "step_size": step_size,
                    "steps_per_sec": steps_per_sec,
                }
            )

        return elbo
