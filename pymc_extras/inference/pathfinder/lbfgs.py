import logging

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


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
                sz = (s * z).sum(axis=-1)
                update = sz > self.epsilon * np.sqrt(np.sum(z**2, axis=-1))

                if update:
                    return True
                else:
                    return False
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

    def minimize(self, x0) -> tuple[NDArray, NDArray, int, LBFGSStatus]:
        """minimizes objective function starting from initial position.

        Parameters
        ----------
        x0 : array_like
            initial position

        Returns
        -------
        x : NDArray
            history of positions
        g : NDArray
            history of gradients
        count : int
            number of iterations
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
            options={
                "maxcor": self.maxcor,
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxls": self.maxls,
            },
        )
        history = history_manager.get_history()

        # warnings and suggestions for LBFGSStatus are displayed at the end
        # threshold determining if the number of lbfgs updates is low compared to iterations
        low_update_threshold = 3

        if history.count <= 1:  # triggers LBFGSInitFailed
            if result.nit < low_update_threshold:
                lbfgs_status = LBFGSStatus.INIT_FAILED
            else:
                lbfgs_status = LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT
        elif result.status == 1:
            # (result.nit > maxiter) or (result.nit > maxls)
            lbfgs_status = LBFGSStatus.MAX_ITER_REACHED
        elif result.status == 2:
            # precision loss resulting to inf or nan
            lbfgs_status = LBFGSStatus.NON_FINITE
        elif history.count * low_update_threshold < result.nit:
            lbfgs_status = LBFGSStatus.LOW_UPDATE_PCT
        else:
            lbfgs_status = LBFGSStatus.CONVERGED

        return history.x, history.g, history.count, lbfgs_status
