"""Analytic-objective tests for the stochastic L-BFGS optimizer."""

import itertools

import numpy as np
import pytest

from pymc_extras.inference.pathfinder.stochastic_lbfgs import (
    StochasticLBFGSConfig,
    _two_loop_direction,
    run_stochastic_lbfgs,
)


def quadratic(A, b):
    """Return value_grad_fn for f(x) = 0.5 x'Ax - b'x (grad = Ax - b, min at A^-1 b)."""

    def vg(x):
        return 0.5 * x @ A @ x - b @ x, A @ x - b

    return vg


def double_well(x):
    """value_grad_fn for f(x) = sum(0.25 x^4 - x^2).

    The Hessian is negative definite inside ``|x| < sqrt(2/3)``, so descent steps taken
    from there routinely produce ``s . y < 0``.
    """
    return float(np.sum(0.25 * x**4 - x**2)), x**3 - 2 * x


def dense_inverse_hessian(alpha, pairs):
    """Textbook BFGS inverse-Hessian recursion from H0 = diag(alpha), oldest pair first."""
    ident = np.eye(alpha.size)
    H = np.diag(alpha)
    for s, y in pairs:
        rho = 1.0 / (s @ y)
        V = ident - rho * np.outer(s, y)
        H = V @ H @ V.T + rho * np.outer(s, s)
    return H


def fill_ring(J, n_pairs, rng, N=5):
    """Write n_pairs curvature pairs into an (N, J) ring exactly as the optimizer does.

    Returns ``(s_win, z_win, order)`` with ``order`` newest-first over the resident pairs.
    """
    M = rng.normal(size=(N, N))
    P = M @ M.T + N * np.eye(N)  # SPD, so y = P s guarantees s . y > 0
    s_win = np.zeros((N, J))
    z_win = np.zeros((N, J))
    win_idx = -1
    for _ in range(n_pairs):
        s = rng.normal(size=N)
        win_idx = (win_idx + 1) % J
        s_win[:, win_idx] = s
        z_win[:, win_idx] = P @ s
    order = [(win_idx - k) % J for k in range(min(n_pairs, J))]
    return s_win, z_win, order


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


def test_a_pair_with_zero_curvature_drops_out_of_the_recursion():
    """The recursion divides by ``s . y``, so an exactly orthogonal pair has to be
    skipped rather than contribute an infinite coefficient."""
    s_win = np.zeros((3, 2))
    z_win = np.zeros((3, 2))
    s_win[:, 0], z_win[:, 0] = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    s_win[:, 1], z_win[:, 1] = [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]
    g, alpha = np.ones(3), np.ones(3)

    d = _two_loop_direction(g, alpha, s_win, z_win, order=[1, 0])
    assert np.all(np.isfinite(d))
    np.testing.assert_allclose(
        d, _two_loop_direction(g, alpha, s_win[:, [1]], z_win[:, [1]], order=[0])
    )


def test_pair_rejected_when_curvature_violated():
    """A step crossing a negative-curvature region gives s.y < 0 and is rejected."""
    traj = run_stochastic_lbfgs(double_well, noop, np.array([0.3]), num_iters=8)
    assert traj.n_curvature_violations >= 1
    assert traj.n_accepted + traj.n_curvature_violations + traj.n_null == 8


def test_line_search_decreases_objective_each_step():
    """Every accepted iterate's value is no worse than the previous one on a fixed objective."""
    rng = np.random.default_rng(2)
    A = np.diag([1.0, 3.0, 10.0])
    b = rng.normal(size=3)
    vg = quadratic(A, b)
    traj = run_stochastic_lbfgs(vg, noop, np.array([5.0, 5.0, 5.0]), num_iters=40)
    values = [vg(it["x"])[0] for it in traj.iterates]
    assert all(v2 <= v1 + 1e-9 for v1, v2 in itertools.pairwise(values))


def test_no_point_is_evaluated_twice_within_a_step():
    """The accepted trial's gradient comes from the call that tested it.

    Re-evaluating the accepted point to fetch a gradient the joint value_grad_fn already
    returned costs a full logp+grad pass per step, and no counter reports it.
    """
    rng = np.random.default_rng(30)
    A = np.diag([1.0, 2.0, 4.0])
    vg = quadratic(A, rng.normal(size=3))
    state = {"batch": 0}
    seen = []

    def tagged(x):
        seen.append((state["batch"], tuple(x)))
        return vg(x)

    def advance():
        state["batch"] += 1

    run_stochastic_lbfgs(tagged, advance, np.array([3.0, -3.0, 3.0]), num_iters=25)
    assert len(seen) == len(set(seen)), "a point was evaluated twice on the same batch"


@pytest.mark.parametrize("kw", [{"backtrack": -0.5}, {"maxcor": 0}])
def test_config_rejects_settings_that_make_a_wrong_fit_look_healthy(kw):
    """backtrack outside (0, 1) is the dangerous one: a negative step length satisfies
    the Armijo bound trivially, so the run climbs while every counter reads clean."""
    with pytest.raises(ValueError):
        StochasticLBFGSConfig(**kw)


def test_line_search_exhaustion_handled():
    """When Armijo can never be satisfied the step is flagged, not crashed, and the loop
    still moves on to fresh data — holding both x and the batch replays the identical
    failing step forever, and only n_ls_failures would ever show it."""

    def vg(x):
        # constant value with a nonzero gradient: no step ever decreases f
        return 1.0, np.array([1.0, 1.0])

    advances = []
    traj = run_stochastic_lbfgs(
        vg,
        lambda: advances.append(1),
        np.zeros(2),
        num_iters=3,
        config=StochasticLBFGSConfig(maxls=5),
    )
    assert traj.n_ls_failures == 3
    assert len(advances) == 3


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


def test_both_gradients_of_every_accepted_pair_come_from_one_batch():
    """s and y are differences of two evaluations made on the same batch.

    Schraudolph pairing is what cancels the minibatch noise in y; differencing across
    two batches leaves the noise in, and no counter the Trajectory reports can tell the
    two apart. Each value_grad_fn call is tagged with the batch active when it ran, so
    the pair stored at each accepted step can be traced back to its two evaluations.
    """
    rng = np.random.default_rng(20)
    A = np.diag([1.0, 2.0, 5.0])
    J = 4
    state = {"batch": 0, "b": rng.normal(0, 0.3, size=3)}
    log = []

    def vg(x):
        g = A @ x - state["b"]
        log.append((state["batch"], np.array(x, dtype=float), g.copy()))
        return 0.5 * x @ A @ x - state["b"] @ x, g

    def advance():
        state["batch"] += 1
        state["b"] = rng.normal(0, 0.3, size=3)

    traj = run_stochastic_lbfgs(
        vg,
        advance,
        np.array([4.0, -4.0, 4.0]),
        num_iters=40,
        config=StochasticLBFGSConfig(maxcor=J),
    )
    assert traj.n_accepted > J

    for k, it in enumerate(traj.iterates):
        newest = min(k + 1, J) - 1
        owners = {
            b for b, xx, gg in log if np.array_equal(xx, it["x"]) and np.array_equal(gg, it["g"])
        }
        assert len(owners) == 1, f"iterate {k} gradient traced to batches {owners}"
        (batch,) = owners
        x_first, g_first = next((xx, gg) for b, xx, gg in log if b == batch)
        np.testing.assert_array_equal(it["s_win"][:, newest], it["x"] - x_first)
        np.testing.assert_array_equal(it["z_win"][:, newest], it["g"] - g_first)

    assert sorted({b for b, _, _ in log}) == list(range(41))


@pytest.mark.parametrize("J, n_pairs", [(1, 1), (2, 5), (3, 2), (3, 3), (3, 7), (6, 4), (6, 13)])
def test_two_loop_direction_matches_dense_recursion_over_the_ring(J, n_pairs):
    """The two-loop recursion returns -H g for the dense BFGS H built from the resident
    pairs in ring order, before the buffer wraps (n_pairs <= J) and after it has wrapped
    several times. Any drift in the ordering or in the H0 = diag(alpha) seed shows up as
    a different direction once more than one pair is resident.
    """
    rng = np.random.default_rng(100 + 31 * J + n_pairs)
    s_win, z_win, order = fill_ring(J, n_pairs, rng)
    N = s_win.shape[0]
    alpha = np.abs(rng.normal(size=N)) + 0.5
    g = rng.normal(size=N)

    d = _two_loop_direction(g, alpha, s_win, z_win, order)
    H = dense_inverse_hessian(alpha, [(s_win[:, c], z_win[:, c]) for c in reversed(order)])
    np.testing.assert_allclose(d, -H @ g, rtol=1e-9, atol=1e-9)


def test_history_holds_only_pairs_that_passed_the_curvature_test():
    """Every column ever handed to the sampler satisfies s.y >= epsilon * s.s, and each
    step lands in exactly one counter.

    A rejected pair silently entering the ring makes the L-BFGS memory indefinite while
    violation_rate keeps reading zero, so the counters alone cannot detect it.
    """
    cfg = StochasticLBFGSConfig(maxcor=4)
    traj = run_stochastic_lbfgs(
        double_well, noop, np.array([0.05, -0.05, 0.6]), num_iters=30, config=cfg
    )
    for it in traj.iterates:
        for s, y in zip(it["s_win"].T, it["z_win"].T):
            if not np.any(s):
                continue
            assert s @ y > 1e-16
            assert s @ y >= cfg.epsilon * (s @ s)
    assert traj.n_accepted == len(traj.iterates)
    assert traj.n_accepted + traj.n_curvature_violations + traj.n_null + traj.n_ls_failures == 30
    assert traj.n_curvature_violations >= 1, "the objective produced no rejections to check"


def test_window_layout_holds_before_the_ring_wraps():
    """A partially filled ring is handed over unrolled: filled columns oldest-to-newest
    starting at column 0, unused columns zero and trailing.

    Rolling a not-yet-full ring would put the zero columns in the newest slots, which the
    sampler cannot distinguish from real pairs.
    """
    rng = np.random.default_rng(21)
    M = rng.normal(size=(5, 5))
    A = M @ M.T + 0.2 * np.eye(5)  # deliberately ill-conditioned: no early convergence
    J = 4
    x0 = np.full(5, 3.0)
    traj = run_stochastic_lbfgs(
        quadratic(A, rng.normal(size=5)),
        noop,
        x0,
        num_iters=9,
        config=StochasticLBFGSConfig(maxcor=J),
    )
    assert traj.n_accepted == 9 > J, "every step must be accepted for xs to line up"
    xs = [x0, *[it["x"] for it in traj.iterates]]
    for k, it in enumerate(traj.iterates):
        n_valid = min(k + 1, J)
        for age in range(n_valid):
            np.testing.assert_allclose(
                it["s_win"][:, n_valid - 1 - age], xs[k + 1 - age] - xs[k - age], atol=1e-12
            )
        np.testing.assert_array_equal(it["s_win"][:, n_valid:], 0.0)
        np.testing.assert_array_equal(it["z_win"][:, n_valid:], 0.0)
