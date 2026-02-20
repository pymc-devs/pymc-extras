import logging

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class _CachedValueGrad:
    """Single-entry cache for value/gradient evaluation.

    SciPy L-BFGS-B evaluates the objective at ``x_k`` before invoking
    ``callback(x_k)``.  Wrapping the objective in this cache means the
    callback's ``value_grad_fn(x_k)`` call is a free hit rather than a
    duplicate evaluation.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._last_x: NDArray | None = None
        self._last_val: float | None = None
        self._last_grad: NDArray | None = None

    def __call__(self, x: NDArray) -> tuple[float, NDArray]:
        if self._last_x is None or not np.array_equal(x, self._last_x):
            val, grad = self._fn(x)
            self._last_x = x.copy()
            self._last_val = float(val)
            self._last_grad = np.array(grad, dtype=np.float64)
        return self._last_val, self._last_grad


def _check_lbfgs_curvature_condition(s: NDArray, z: NDArray, epsilon: float) -> bool:
    """Check the L-BFGS curvature condition: s·z > epsilon * ||z||.

    Shared by LBFGSHistoryManager (batch path) and LBFGSStreamingCallback
    (streaming path) to ensure the acceptance criterion stays identical.
    """
    sz = float((s * z).sum())
    return sz > epsilon * float(np.sqrt(np.sum(z**2)))


@dataclass(slots=True)
class LBFGSHistory:
    """History of LBFGS iterations."""

    x: NDArray[np.float64]
    g: NDArray[np.float64]
    count: int

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.g = np.ascontiguousarray(self.g, dtype=np.float64)


@dataclass(slots=True)
class LBFGSHistoryManager:
    """manages and stores the history of lbfgs optimisation iterations.

    Parameters
    ----------
    value_grad_fn : Callable
        function that returns tuple of (value, gradient) given input x
    x0 : NDArray
        initial position
    maxiter : int
        maximum number of iterations to store
    epsilon : float
        tolerance for lbfgs update
    """

    value_grad_fn: Callable[[NDArray[np.float64]], tuple[np.float64, NDArray[np.float64]]]
    x0: NDArray[np.float64]
    maxiter: int
    epsilon: float
    x_history: NDArray[np.float64] = field(init=False)
    g_history: NDArray[np.float64] = field(init=False)
    count: int = field(init=False)

    def __post_init__(self) -> None:
        self.x_history = np.empty((self.maxiter + 1, self.x0.shape[0]), dtype=np.float64)
        self.g_history = np.empty((self.maxiter + 1, self.x0.shape[0]), dtype=np.float64)
        self.count = 0

        value, grad = self.value_grad_fn(self.x0)
        if self.entry_condition_met(self.x0, value, grad):
            self.add_entry(self.x0, grad)

    def add_entry(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> None:
        """adds new position and gradient to history.

        Parameters
        ----------
        x : NDArray
            position vector
        g : NDArray
            gradient vector
        """
        self.x_history[self.count] = x
        self.g_history[self.count] = g
        self.count += 1

    def get_history(self) -> LBFGSHistory:
        """returns history of optimisation iterations."""
        return LBFGSHistory(
            x=self.x_history[: self.count], g=self.g_history[: self.count], count=self.count
        )

    def entry_condition_met(self, x, value, grad) -> bool:
        """Checks if the LBFGS iteration should continue."""

        if np.all(np.isfinite(grad)) and np.isfinite(value) and (self.count < self.maxiter + 1):
            if self.count == 0:
                return True
            else:
                s = x - self.x_history[self.count - 1]
                z = grad - self.g_history[self.count - 1]
                return _check_lbfgs_curvature_condition(s, z, self.epsilon)
        else:
            return False

    def __call__(self, x: NDArray[np.float64]) -> None:
        value, grad = self.value_grad_fn(x)
        if self.entry_condition_met(x, value, grad):
            self.add_entry(x, grad)


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
    DEFAULT_MESSAGE = "LBFGS failed to initialise."

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
    maxiter : int, optional
        maximum number of iterations, defaults to 1000
    ftol : float, optional
        function tolerance for convergence, defaults to 1e-5
    gtol : float, optional
        gradient tolerance for convergence, defaults to 1e-8
    maxls : int, optional
        maximum number of line search steps, defaults to 1000
    epsilon : float, optional
        tolerance for lbfgs update, defaults to 1e-8
    """

    def __init__(
        self, value_grad_fn, maxcor, maxiter=1000, ftol=1e-5, gtol=1e-8, maxls=1000, epsilon=1e-8
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

    def minimize(self, x0) -> tuple[NDArray, NDArray, int, LBFGSStatus]:
        """Minimise objective, collecting the full position/gradient history.

        Parameters
        ----------
        x0 : array_like
            initial position

        Returns
        -------
        x : NDArray
            history of positions, shape (count, N)
        g : NDArray
            history of gradients, shape (count, N)
        count : int
            number of accepted history entries (including initial point)
        status : LBFGSStatus
            final status of optimisation
        """
        x0 = np.array(x0, dtype=np.float64)
        history_manager = LBFGSHistoryManager(
            value_grad_fn=self.value_grad_fn, x0=x0, maxiter=self.maxiter, epsilon=self.epsilon
        )
        result = minimize(
            self.value_grad_fn,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=history_manager,
            options=self._scipy_options,
        )
        history = history_manager.get_history()
        return history.x, history.g, history.count, self._classify_status(result, history.count)

    def minimize_streaming(self, callback, x0) -> tuple[int, LBFGSStatus]:
        """Minimise objective using a streaming callback that processes each step.

        Unlike :meth:`minimize`, no position/gradient history is accumulated.
        The ``callback`` is responsible for maintaining whatever per-step state
        it needs (e.g. ring buffers, best-ELBO tracking).

        ``callback.value_grad_fn`` is used as the scipy objective so that a
        single-entry cache (e.g. :class:`_CachedValueGrad`) eliminates the
        duplicate evaluation that would otherwise occur on each accepted step.

        Parameters
        ----------
        callback : object
            Must expose:
            - ``value_grad_fn``: callable ``(x) -> (value, grad)`` passed to scipy
              as the objective.  Wrap with :class:`_CachedValueGrad` before
              constructing the callback to avoid duplicate evaluations.
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
