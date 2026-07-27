import logging

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Self

import numpy as np

from better_optimize import minimize
from numpy.typing import NDArray

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


def _as_numpy_value_grad(value_grad_fn: Callable) -> Callable:
    """Wrap a fused ``(value, grad)`` objective so its outputs are numpy arrays.

    scipy's L-BFGS-B needs numpy for the value and gradient. Array backends that return
    their own type (e.g. MLX returns ``mx.array``) otherwise flow through unconverted and
    the optimizer degenerates to zero steps. The computation's own dtype is preserved
    (float32 backends stay float32); JAX already returns numpy, so this is a no-op there.
    """

    def wrapped(x):
        value, grad = value_grad_fn(x)
        return np.asarray(value), np.asarray(grad)

    return wrapped


class LBFGS:
    """L-BFGS optimizer wrapping ``scipy.optimize.minimize(method="L-BFGS-B")``.

    Parameters
    ----------
    value_grad_fn : callable
        Returns ``(value, gradient)`` for a position.
    maxcor : int
        Number of variable-metric corrections.
    maxiter : int, optional
        Maximum number of iterations. Default 1000.
    ftol : float, optional
        Function-value tolerance for convergence. Default 1e-5.
    gtol : float, optional
        Gradient-norm tolerance for convergence. Default 1e-8.
    maxls : int, optional
        Maximum line-search steps per iteration. Default 1000.
    epsilon : float, optional
        Curvature-condition threshold for accepting a step. Default 1e-12.
    dtype : str or numpy.dtype, optional
        Working precision of the L-BFGS objective, matching the compiled model (e.g. float32
        for MLX/float32 models). Default ``"float64"``.
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
        dtype: str | np.dtype = "float64",
    ) -> None:
        self.value_grad_fn = _as_numpy_value_grad(value_grad_fn)
        self.maxcor = maxcor
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls
        self.epsilon = epsilon
        self.dtype = np.dtype(dtype)

    def _classify_status(self, result, update_count: int) -> LBFGSStatus:
        """Classify the LBFGS termination status.

        Parameters
        ----------
        result : OptimizeResult
            scipy result object.
        update_count : int
            Number of accepted history entries, including the initial point (``step_count + 1``).
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
        """Minimize the objective, invoking ``callback`` after each accepted step.

        Parameters
        ----------
        callback : object
            Exposes ``value_grad_fn``, the fused ``(x) -> (value, grad)`` objective, and a
            ``step_count`` it updates on each accepted step.
        x0 : array_like
            Initial position.

        Returns
        -------
        step_count : int
            Number of accepted steps, excluding the initial point.
        lbfgs_status : LBFGSStatus
            The classified termination status.
        """
        # scipy calls the objective at x0's dtype, so this sets the working precision of the
        # L-BFGS objective (float32 for MLX/float32 models); scipy still runs its own state in
        # double regardless.
        x0 = np.asarray(x0, dtype=self.dtype)
        # better_optimize detects the fused (value, grad) objective and promotes maxcor/maxiter/...
        # to scipy's options dict itself, so they pass straight through as keyword arguments.
        result = minimize(
            callback.value_grad_fn,
            x0,
            method="L-BFGS-B",
            callback=callback,
            progressbar=False,
            maxcor=self.maxcor,
            maxiter=self.maxiter,
            ftol=self.ftol,
            gtol=self.gtol,
            maxls=self.maxls,
        )
        step_count = callback.step_count
        return step_count, self._classify_status(result, step_count + 1)


class LBFGSStreamingCallback:
    """Scipy L-BFGS callback that scores the Gaussian approximation at each accepted step.

    On each accepted step it updates the inverse-Hessian estimate, draws from the resulting
    Gaussian to estimate its ELBO, and keeps the best-scoring state for downstream sampling.

    Parameters
    ----------
    value_grad_fn : callable
        Single-entry cached ``(x) -> (value, gradient)`` function.
    x0 : ndarray
        Initial position, shape (N,).
    sample_logp_fn : callable
        Function ``(x, g, alpha, s_win, z_win, u) -> (phi, logQ, logP, inv_hessian_diag)`` from
        :func:`make_pathfinder_sample_fn`.
    num_elbo_draws : int
        Number of draws per step for ELBO estimation.
    rng : numpy Generator
        Random generator for draws.
    J : int
        L-BFGS history size (``maxcor``).
    epsilon : float
        Curvature-condition threshold for accepting a step.
    progress_callback : callable, optional
        Called with a progress dict after each step. Default None.
    on_step_callback : callable, optional
        Called after each accepted step with ``(x, g, alpha, s_win, z_win, elbo)``. Default None.
    dtype : str or numpy.dtype, optional
        Working precision for the initial point, smoothing buffers, and ELBO draws passed to
        ``sample_logp_fn``, matching the compiled model. scipy's float64 step positions are cast
        down to it. Default ``"float64"``.
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
        dtype: str | np.dtype = "float64",
    ) -> None:
        value_grad_fn = _as_numpy_value_grad(value_grad_fn)
        self.value_grad_fn = value_grad_fn
        self.sample_logp_fn = sample_logp_fn
        self.num_elbo_draws = num_elbo_draws
        self._rng = rng
        self.J = J
        self.epsilon = epsilon
        self.progress_callback = progress_callback
        self.on_step_callback = on_step_callback
        # Working precision for everything fed to the (model-dtype) sample_logp_fn: the initial
        # point, the smoothing buffers, and the ELBO draws. scipy hands back float64 positions,
        # which __call__ casts down to this.
        self.dtype = np.dtype(dtype)

        N = x0.shape[0]
        self._N = N
        x0 = np.asarray(x0, dtype=self.dtype)
        _, g0 = value_grad_fn(x0)

        self.x_prev: NDArray = x0.copy()
        self.g_prev: NDArray = np.asarray(g0, dtype=self.dtype)
        self.alpha_prev: NDArray = np.ones(N, dtype=self.dtype)

        # Ring buffer: numpy arrays passed as inputs to sample_logp_fn each call.
        # Thread-safe: no shared mutable state across concurrent invocations.
        self.s_win: NDArray = np.zeros((N, J), dtype=self.dtype)
        self.z_win: NDArray = np.zeros((N, J), dtype=self.dtype)
        self.win_idx: int = -1
        self.best_elbo: float = -np.inf
        self.best_state: dict = {}
        self.best_step_idx: int = 0
        self.step_count: int = 0
        self.any_valid: bool = False

    def __call__(self, intermediate) -> float | None:
        """Process one accepted L-BFGS step and return its ELBO, or None if the step is skipped
        (non-finite value/gradient, or curvature condition not met).

        ``intermediate`` is a scipy/better_optimize ``OptimizeResult`` (carrying ``.x``, ``.fun``,
        and ``.jac``) or, for a classic scipy callback, the raw position array.
        """
        # scipy iterates in float64, but sample_logp_fn is compiled at self.dtype with
        # trust_input=True. Positions and gradients must be cast down to self.dtype here; widening
        # back to float64 would reintroduce a float64 op into the (possibly Metal) sample graph.
        if hasattr(intermediate, "x"):
            x = np.asarray(intermediate.x, dtype=self.dtype)
            value = float(intermediate.fun)
            g = getattr(intermediate, "jac", None)
            g = self.value_grad_fn(x)[1] if g is None else np.asarray(g, dtype=self.dtype)
        else:
            x = np.asarray(intermediate, dtype=self.dtype)
            value, g = self.value_grad_fn(x)

        s = x - self.x_prev
        z = g - self.g_prev

        if not (np.all(np.isfinite(g)) and np.isfinite(value)):
            return None
        if not _check_lbfgs_curvature_condition(s, z, self.epsilon):
            return None

        alpha = alpha_step_numpy(self.alpha_prev, s, z)

        # Ring-buffer update (numpy, O(N))
        self.win_idx = (self.win_idx + 1) % self.J
        self.s_win[:, self.win_idx] = s
        self.z_win[:, self.win_idx] = z

        # Sample + logP in a single compiled call. Pass s_win/z_win as inputs.
        u = self._rng.standard_normal((self.num_elbo_draws, self._N)).astype(self.dtype)
        sample_out = self.sample_logp_fn(x, g, alpha, self.s_win, self.z_win, u)
        _, logQ, logP, _ = sample_out
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
                "x": x.copy(),
                "g": g.copy(),
            }
            self.best_step_idx = self.step_count

        self.alpha_prev = alpha
        self.x_prev = x.copy()
        self.g_prev = g.copy()
        self.step_count += 1

        if self.progress_callback is not None:
            self.progress_callback({"iteration": self.step_count, "best_elbo": self.best_elbo})

        return elbo
