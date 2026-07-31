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

Both gradients are evaluated on the same batch, so the minibatch noise cancels in
the difference (unlike differencing across two batches, which leaves the noise and
routinely produces ``s . y < 0``). Pairs still failing the curvature condition are
skipped, not forced into the history.

The trajectory records, at every accepted step, exactly the state the pathfinder
sampler consumes — ``x, g, alpha, s_win, z_win`` — with the ``(N, J)`` ring-buffer
layout of ``pymc_extras.inference.pathfinder.lbfgs.LBFGSStreamingCallback`` and the
diagonal update ``alpha_step_numpy`` from ``bfgs_sample``, so ``streaming_pathfinder``
can feed each iterate straight into ``make_pathfinder_sample_fn``.

The optimizer loop is pure numpy and testable on analytic objectives; only the
diagonal update is imported from ``bfgs_sample``.
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


@dataclass
class Trajectory:
    """Recorded output of :func:`run_stochastic_lbfgs`.

    ``iterates`` holds one dict ``{x, g, alpha, s_win, z_win}`` per accepted step —
    the exact inputs (besides the noise draws ``u``) that
    ``make_pathfinder_sample_fn`` expects. Counters summarize optimizer health.
    """

    iterates: list = field(default_factory=list)
    n_steps: int = 0
    n_accepted: int = 0
    n_curvature_violations: int = 0
    n_null: int = 0
    n_ls_failures: int = 0

    @property
    def violation_rate(self):
        """Curvature-rejection rate among steps that actually moved (target < 20%).

        Null steps (the optimizer has effectively stopped, so ``s`` is below the
        floor) are excluded — they signal convergence, not a curvature failure.
        """
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
        if not np.isfinite(sy) or sy < 1e-16:  # skip a numerically degenerate pair
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


def run_stochastic_lbfgs(value_grad_fn, on_batch_advance, x0, num_iters, config=None):
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
    f, g = value_grad_fn(x)

    for _ in range(num_iters):
        traj.n_steps += 1

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
        f_new, x_new, g_new = f, x, g
        ls_ok = False
        for _ls in range(config.maxls):
            x_trial = x + t * d
            f_trial = value_grad_fn(x_trial)[0]
            if np.isfinite(f_trial) and f_trial <= f + config.armijo_c1 * t * gd:
                f_new, x_new = f_trial, x_trial
                g_new = value_grad_fn(x_new)[1]
                ls_ok = True
                break
            t *= config.backtrack
        if not ls_ok:
            traj.n_ls_failures += 1
            # A failed line search gives no trusted step: never force an untested
            # move into the curvature history (that both pollutes the L-BFGS memory
            # and lets violation_rate read 0% on a stuck run). Hold x, advance the
            # batch, and try again on fresh data.
            on_batch_advance()
            f, g = value_grad_fn(x)
            continue

        s = x_new - x
        y = g_new - g
        s2 = s @ s
        sy = s @ y

        if s2 < 1e-16:
            # Null step: the optimizer has effectively stopped moving. Not a
            # curvature failure; just don't extend the history.
            traj.n_null += 1
        elif np.isfinite(sy) and sy > 1e-16 and sy >= config.epsilon * s2:
            alpha = alpha_step_numpy(alpha, s, y)
            win_idx = (win_idx + 1) % J
            s_win[:, win_idx] = s
            z_win[:, win_idx] = y
            n_valid = min(n_valid + 1, J)
            traj.n_accepted += 1
            # The sampler reads the window as a chronological sequence and has no
            # ring index, so hand it oldest-to-newest. Before the ring wraps the
            # physical layout already is that, with the unused columns last.
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

        x, f, g = x_new, f_new, g_new
        on_batch_advance()
        f, g = value_grad_fn(x)  # re-evaluate on the freshly advanced batch

    return traj
