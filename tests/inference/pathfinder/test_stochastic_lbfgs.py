"""Analytic-objective tests for the stochastic L-BFGS optimizer."""

import itertools

import numpy as np

from pymc_extras.inference.pathfinder.stochastic_lbfgs import (
    StochasticLBFGSConfig,
    run_stochastic_lbfgs,
)


def quadratic(A, b):
    """Return value_grad_fn for f(x) = 0.5 x'Ax - b'x (grad = Ax - b, min at A^-1 b)."""

    def vg(x):
        return 0.5 * x @ A @ x - b @ x, A @ x - b

    return vg


def noop():
    return None


def test_quadratic_full_batch_converges_to_optimum():
    """On a deterministic strongly-convex quadratic the iterate reaches the mode."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    A = M @ M.T + 5 * np.eye(5)  # SPD, well-conditioned
    b = rng.normal(size=5)
    x_star = np.linalg.solve(A, b)
    traj = run_stochastic_lbfgs(quadratic(A, b), noop, np.zeros(5), num_iters=60)
    x_final = traj.iterates[-1]["x"]
    assert np.allclose(x_final, x_star, atol=1e-6)
    assert traj.violation_rate == 0.0  # a quadratic never violates curvature


def test_two_loop_direction_matches_dense_bfgs():
    """After one pair, the two-loop direction equals -(diag-init BFGS update) . g."""
    from pymc_extras.inference.pathfinder.stochastic_lbfgs import _two_loop_direction

    rng = np.random.default_rng(1)
    N = 4
    s = rng.normal(size=N)
    y = s * np.array([2.0, 3.0, 1.5, 4.0]) + 0.01  # ensure s.y > 0
    alpha = np.abs(rng.normal(size=N)) + 0.5
    g = rng.normal(size=N)

    s_win = s[:, None]
    z_win = y[:, None]
    d = _two_loop_direction(g, alpha, s_win, z_win, order=[0])

    # Dense reference: H0 = diag(alpha); one BFGS inverse-Hessian update.
    rho = 1.0 / (s @ y)
    H0 = np.diag(alpha)
    Ident = np.eye(N)
    V = Ident - rho * np.outer(s, y)
    H = V @ H0 @ V.T + rho * np.outer(s, s)
    assert np.allclose(d, -H @ g, atol=1e-10)


def test_pair_rejected_when_curvature_violated():
    """A step crossing a negative-curvature region gives s.y < 0 and is rejected."""

    def vg(x):
        # 1-D double well f = 0.25 x^4 - x^2: concave (negative curvature) for
        # |x| < sqrt(2/3), so a descent step out of x0=0.3 yields s.y < 0.
        xv = x[0]
        return 0.25 * xv**4 - xv**2, np.array([xv**3 - 2 * xv])

    traj = run_stochastic_lbfgs(vg, noop, np.array([0.3]), num_iters=8)
    assert traj.n_curvature_violations >= 1
    assert traj.n_accepted + traj.n_curvature_violations + traj.n_null == traj.n_steps


def test_line_search_decreases_objective_each_step():
    """Every accepted iterate's value is no worse than the previous one on a fixed objective."""
    rng = np.random.default_rng(2)
    A = np.diag([1.0, 3.0, 10.0])
    b = rng.normal(size=3)
    vg = quadratic(A, b)
    traj = run_stochastic_lbfgs(vg, noop, np.array([5.0, 5.0, 5.0]), num_iters=40)
    values = [vg(it["x"])[0] for it in traj.iterates]
    assert all(v2 <= v1 + 1e-9 for v1, v2 in itertools.pairwise(values))


def test_line_search_exhaustion_handled():
    """When Armijo can never be satisfied the step is flagged, not crashed."""

    def vg(x):
        # constant value with a nonzero gradient: no step ever decreases f
        return 1.0, np.array([1.0, 1.0])

    traj = run_stochastic_lbfgs(
        vg, noop, np.zeros(2), num_iters=3, config=StochasticLBFGSConfig(maxls=5)
    )
    assert traj.n_ls_failures == 3
    assert traj.n_steps == 3


def test_ring_buffer_wraps_after_maxcor_pairs():
    """The stored history never exceeds maxcor columns and stays two-loop usable."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    A = A @ A.T + 6 * np.eye(6)
    b = rng.normal(size=6)
    cfg = StochasticLBFGSConfig(maxcor=3)
    traj = run_stochastic_lbfgs(quadratic(A, b), noop, np.zeros(6), num_iters=20, config=cfg)
    last = traj.iterates[-1]
    assert last["s_win"].shape == (6, 3)
    assert last["z_win"].shape == (6, 3)
    # still converges with a short history
    assert np.allclose(last["x"], np.linalg.solve(A, b), atol=1e-5)


def test_trajectory_records_expected_shapes():
    """Every recorded iterate carries the sampler-ready arrays with matching shapes."""
    rng = np.random.default_rng(4)
    A = np.diag(np.abs(rng.normal(size=4)) + 1.0)
    vg = quadratic(A, rng.normal(size=4))
    cfg = StochasticLBFGSConfig(maxcor=5)
    traj = run_stochastic_lbfgs(vg, noop, np.zeros(4), num_iters=15, config=cfg)
    for it in traj.iterates:
        assert it["x"].shape == (4,)
        assert it["g"].shape == (4,)
        assert it["alpha"].shape == (4,)
        assert it["s_win"].shape == (4, 5)


def test_reproducible_given_same_inputs():
    """A deterministic objective yields a bit-identical trajectory across runs."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(4, 4))
    A = A @ A.T + 4 * np.eye(4)
    b = rng.normal(size=4)
    x0 = rng.normal(size=4)
    a = run_stochastic_lbfgs(quadratic(A, b), noop, x0, num_iters=25)
    c = run_stochastic_lbfgs(quadratic(A, b), noop, x0, num_iters=25)
    assert np.array_equal(a.iterates[-1]["x"], c.iterates[-1]["x"])


def test_violation_rate_low_on_streaming_quadratic():
    """Same-batch pairing keeps the violation rate near zero even with per-step noise.

    Each 'batch' shifts the linear term by fresh noise; because s and y use the
    same shifted objective within a step, curvature stays clean.
    """
    rng = np.random.default_rng(6)
    A = np.diag([1.0, 2.0, 4.0])
    state = {"b": np.zeros(3)}

    def vg(x):
        return 0.5 * x @ A @ x - state["b"] @ x, A @ x - state["b"]

    def advance():
        state["b"] = rng.normal(0, 0.05, size=3)  # small minibatch-like perturbation

    traj = run_stochastic_lbfgs(vg, advance, np.array([3.0, -3.0, 3.0]), num_iters=60)
    assert traj.violation_rate < 0.20


def test_stored_window_is_chronological_after_the_ring_wraps():
    """The sampler has no ring index, so the window it receives has to read
    oldest-to-newest. Physical ring order silently computes the Gaussian for a
    different update sequence once the buffer wraps."""
    A = np.array([[3.0, 0.4], [0.4, 1.0]])
    J = 2
    traj = run_stochastic_lbfgs(
        quadratic(A, np.zeros(2)),
        noop,
        np.array([2.0, -1.5]),
        num_iters=12,
        config=StochasticLBFGSConfig(maxcor=J),
    )
    assert traj.n_accepted > J, "the ring never wrapped, so the test proves nothing"
    xs = [it["x"] for it in traj.iterates]
    for k in range(J, len(traj.iterates)):
        window = traj.iterates[k]["s_win"]
        for age in range(J):  # age 0 is the newest pair, in the last column
            np.testing.assert_allclose(
                window[:, J - 1 - age], xs[k - age] - xs[k - age - 1], atol=1e-12
            )
