"""A stochastic L-BFGS trajectory for streaming (minibatch) Pathfinder.

Pathfinder builds a Gaussian approximation at every L-BFGS iterate from the recent
``(s, y)`` curvature pairs, keeping the best by ELBO. The deterministic Pathfinder
(``pymc_extras.inference.pathfinder.lbfgs``) delegates the optimization to SciPy's
``L-BFGS-B``, whose Wolfe line search and convergence tests assume a deterministic
objective and break under minibatch-noisy gradients.

This module replaces just that optimizer with a stochastic quasi-Newton loop that
keeps the objective deterministic *within* each step so a plain backtracking line
search remains valid, and forms each curvature pair from a **single minibatch**
(Schraudolph, Yu & Günter, 2007):

    y_k = grad_{B_k}(x_{k+1}) - grad_{B_k}(x_k)

Both gradients are evaluated on the same batch, so the same noise enters both and cancels in
the difference: the secant condition ``y ~= H s`` needs both gradients to be of the *same*
objective. Differencing across two batches -- Schraudolph et al.'s eq. (13), for which they
report divergence below batch 1000 on their quadratic model -- leaves the noise in. Measured
over logistic and Gaussian trajectories at batches 4 to 512, same-batch pairing produced no
negative ``s . y`` on any cell and cross-batch produced 0.3% to 12% of steps; the cross-batch
rate rises to about one in two once ``||s||`` is below 1e-2, its noise term scaling with
``||s||`` against the curvature term's ``||s||^2``. Pairs still failing the curvature condition
are skipped, not forced into the history. Each accepted step records the
``(x, g, alpha, s_win, z_win)`` that ``make_pathfinder_sample_fn`` consumes, with the window
ordered oldest-to-newest rather than in physical ring order.
"""

from dataclasses import dataclass, field

import numpy as np

from pymc_extras.inference.pathfinder.bfgs_sample import alpha_step_numpy

__all__ = ["StochasticLBFGSConfig", "Trajectory", "run_stochastic_lbfgs"]


@dataclass(frozen=True)
class StochasticLBFGSConfig:
    """Hyperparameters for :func:`run_stochastic_lbfgs`.

    Parameters
    ----------
    maxcor : int
        L-BFGS history size ``J`` (number of curvature pairs retained).
    init_step : float
        Initial trial step length for the backtracking line search.
    backtrack : float
        Line-search step-shrink factor ``rho`` in ``(0, 1)``.
    armijo_c1 : float
        Armijo sufficient-decrease constant.
    maxls : int
        Maximum backtracking iterations before the step is declared a failure.
    epsilon : float
        A pair is accepted iff ``s . y >= epsilon * (s . s)``.
    """

    maxcor: int = 6
    init_step: float = 1.0
    backtrack: float = 0.5
    armijo_c1: float = 1e-4
    maxls: int = 20
    epsilon: float = 1e-8

    def __post_init__(self):
        # A backtrack outside (0, 1) flips the step direction, which makes the Armijo
        # bound trivially true: the run then climbs with every counter reading healthy.
        if not 0.0 < self.backtrack < 1.0:
            raise ValueError(f"backtrack must be in (0, 1), got {self.backtrack}")
        if self.maxcor < 1:
            raise ValueError(f"maxcor must be >= 1, got {self.maxcor}")
        if self.maxls < 1:
            raise ValueError(f"maxls must be >= 1, got {self.maxls}")


@dataclass
class Trajectory:
    """Recorded output of :func:`run_stochastic_lbfgs`.

    ``iterates`` holds one dict ``{x, g, alpha, s_win, z_win}`` per accepted step —
    the exact inputs (besides the noise draws ``u``) that
    ``make_pathfinder_sample_fn`` expects. Counters summarize optimizer health.
    """

    iterates: list = field(default_factory=list)
    n_accepted: int = 0
    n_curvature_violations: int = 0
    n_null: int = 0
    n_ls_failures: int = 0

    @property
    def violation_rate(self):
        """Curvature-rejection rate among steps that moved. Null steps are excluded:
        they signal convergence, not a curvature failure."""
        moved = self.n_accepted + self.n_curvature_violations
        return self.n_curvature_violations / moved if moved else 0.0


def _two_loop_direction(g, alpha, s_win, z_win, order):
    """L-BFGS two-loop recursion with a diagonal initial inverse-Hessian ``diag(alpha)``.

    ``order`` lists the ring-buffer column indices newest-first.
    """
    q = g.copy()
    coeffs = []
    for c in order:
        s_c, y_c = s_win[:, c], z_win[:, c]
        sy = s_c @ y_c
        # s_win columns are strided views, so sy can differ in the last ulp from the
        # value that passed the acceptance test; rho = 1 / sy must not see a zero.
        if not np.isfinite(sy) or sy < 1e-16:
            continue
        rho = 1.0 / sy
        a = rho * (s_c @ q)
        q = q - a * y_c
        coeffs.append((c, rho, a))
    r = alpha * q  # H0 = diag(alpha)
    for c, rho, a in reversed(coeffs):
        s_c, y_c = s_win[:, c], z_win[:, c]
        b = rho * (y_c @ r)
        r = r + s_c * (a - b)
    return -r


def run_stochastic_lbfgs(value_grad_fn, on_batch_advance, x0, num_iters, config=None, callbacks=()):
    """Run stochastic L-BFGS, recording a Gaussian-ready iterate at each accepted step.

    Parameters
    ----------
    value_grad_fn : callable
        ``x -> (value, gradient)`` on the *currently active* minibatch. The caller
        owns which batch is active; this loop never changes it except through
        ``on_batch_advance``.
    on_batch_advance : callable
        Zero-argument hook called once at the end of each step, after the step's
        curvature pair has been formed, to advance the active minibatch. All
        gradients within a step are therefore on one batch (Schraudolph pairing).
    x0 : ndarray, shape (N,)
        Starting position.
    num_iters : int
        Number of optimization steps.
    config : StochasticLBFGSConfig, optional
    callbacks : iterable of callable, optional
        Called after each step as ``(approx, losses, i)``, ``pm.fit``'s contract, with ``i``
        counting from 1. One raising ``StopIteration`` ends the run. There is no
        ``Approximation`` here, so ``approx`` is ``None`` and a callback that inspects it, such
        as ``pymc.variational.callbacks.CheckParametersConvergence``, cannot be used.
        ``losses[i - 1]`` is measured on the batch installed *after* step ``i``, not on the
        value that step's Armijo test accepted: over the 200 steps of the operating
        configuration the gap averaged +393 with an sd of 1661, so it is not a steady offset.

    Returns
    -------
    Trajectory
    """
    config = config or StochasticLBFGSConfig()
    J = config.maxcor
    x = np.asarray(x0, dtype=np.float64).copy()
    N = x.shape[0]

    s_win = np.zeros((N, J))
    z_win = np.zeros((N, J))
    win_idx = -1
    n_valid = 0
    alpha = np.ones(N)

    traj = Trajectory()
    losses = []
    f, g = value_grad_fn(x)

    for i in range(num_iters):
        if n_valid == 0:
            d = -alpha * g
        else:
            order = [(win_idx - k) % J for k in range(n_valid)]
            d = _two_loop_direction(g, alpha, s_win, z_win, order)

        gd = g @ d
        if not np.isfinite(gd) or gd >= 0:  # not a descent direction
            d = -g
            gd = g @ d

        # Backtracking Armijo line search on the *fixed* current batch.
        t = config.init_step
        x_new = g_new = None
        for _ in range(config.maxls):
            x_trial = x + t * d
            f_trial, g_trial = value_grad_fn(x_trial)
            # pymc's Bernoulli(logit_p) gradient divides by 1 - sigmoid(z), and
            # sigmoid(37.0) == 1.0 in float64, so a finite f_trial can carry a NaN gradient.
            if (
                np.isfinite(f_trial)
                and np.all(np.isfinite(g_trial))
                and f_trial <= f + config.armijo_c1 * t * gd
            ):
                x_new, g_new = x_trial, g_trial
                break
            t *= config.backtrack

        if x_new is None:
            # A failed line search gives no trusted step: forcing an untested move into
            # the history pollutes the L-BFGS memory and lets violation_rate read 0% on
            # a stuck run. Hold x and retry on fresh data.
            traj.n_ls_failures += 1
        else:
            s = x_new - x
            y = g_new - g
            s2 = s @ s
            sy = s @ y
            if s2 < 1e-16:
                traj.n_null += 1  # stopped moving, which is not a curvature failure
            # Absolute 1e-16 decides short steps (epsilon * s2 drops below it once ||s|| < 1e-4),
            # relative epsilon * s2 the rest; _two_loop_direction re-applies the same floor.
            elif np.isfinite(sy) and sy > 1e-16 and sy >= config.epsilon * s2:
                alpha = alpha_step_numpy(alpha, s, y)
                win_idx = (win_idx + 1) % J
                s_win[:, win_idx] = s
                z_win[:, win_idx] = y
                n_valid = min(n_valid + 1, J)
                traj.n_accepted += 1
                # bfgs_sample reads E = triu(S.T @ Z), which is the textbook recursion
                # only when the columns run oldest-to-newest; the roll puts the oldest
                # resident pair first. An unfilled ring already reads that way.
                shift = -(win_idx + 1) % J if n_valid == J else 0
                traj.iterates.append(
                    {
                        "x": x_new.copy(),
                        "g": g_new.copy(),
                        "alpha": alpha.copy(),
                        "s_win": np.roll(s_win, shift, axis=1),
                        "z_win": np.roll(z_win, shift, axis=1),
                    }
                )
            else:
                traj.n_curvature_violations += 1
            x = x_new

        on_batch_advance()
        f, g = value_grad_fn(x)  # f must be on the batch the next Armijo test uses

        losses.append(f)
        try:
            for cb in callbacks:
                cb(None, losses, i + 1)
        except StopIteration:
            break

    return traj
