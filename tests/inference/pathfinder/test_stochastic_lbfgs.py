"""Analytic-objective tests for the stochastic L-BFGS optimizer."""

import collections
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


def spd(n, rng, ridge):
    """A random (n, n) SPD matrix ``M M' + ridge * I``; the ridge sets the conditioning."""
    M = rng.normal(size=(n, n))
    return M @ M.T + ridge * np.eye(n)


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
    P = spd(N, rng, N)  # SPD, so y = P s guarantees s . y > 0
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
    A = spd(5, rng, 5)  # well-conditioned
    b = rng.normal(size=5)
    x_star = np.linalg.solve(A, b)
    traj = run_stochastic_lbfgs(quadratic(A, b), noop, np.zeros(5), num_iters=60)
    x_final = traj.iterates[-1]["x"]
    assert np.allclose(x_final, x_star, atol=1e-6)
    assert traj.violation_rate == 0.0  # a quadratic never violates curvature


def test_zero_curvature_pair_skipped():
    """The recursion divides by s.y, so an orthogonal pair is skipped, not divided by."""
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


def test_line_search_decreases_objective_each_step():
    """Every accepted iterate's value is no worse than the previous one on a fixed objective."""
    rng = np.random.default_rng(2)
    A = np.diag([1.0, 3.0, 10.0])
    b = rng.normal(size=3)
    vg = quadratic(A, b)
    traj = run_stochastic_lbfgs(vg, noop, np.array([5.0, 5.0, 5.0]), num_iters=40)
    values = [vg(it["x"])[0] for it in traj.iterates]
    assert all(v2 <= v1 + 1e-9 for v1, v2 in itertools.pairwise(values))


def test_no_duplicate_gradient_evaluations():
    """The accepted trial's gradient comes from the call that tested it, not a re-evaluation.

    On a well-conditioned quadratic started near the optimum every step accepts its first
    trial, so a step costs exactly two evaluations: the trial, and the re-measurement on the
    batch installed after it. Re-fetching the accepted gradient would make it three.
    """
    vg = quadratic(np.eye(3), np.zeros(3))
    state = {"batch": 0}
    per_batch = collections.Counter()

    def tagged(x):
        per_batch[state["batch"]] += 1
        return vg(x)

    def advance():
        state["batch"] += 1

    run_stochastic_lbfgs(tagged, advance, np.array([0.3, -0.2, 0.1]), num_iters=6)
    assert all(c == 2 for c in list(per_batch.values())[:-1]), dict(per_batch)


@pytest.mark.parametrize("kw", [{"backtrack": -0.5}, {"maxcor": 0}])
def test_config_rejects_invalid_settings(kw):
    """Settings that would silently break the line search or the history are refused."""
    with pytest.raises(ValueError):
        StochasticLBFGSConfig(**kw)


def test_line_search_exhaustion_handled():
    """When Armijo can never be satisfied the step is flagged, not crashed, and the batch
    still advances so the next step does not replay the same failure."""

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
    A = spd(6, rng, 6)
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
    A = spd(4, rng, 4)
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
    num_iters = 40
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
        num_iters=num_iters,
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

    assert sorted({b for b, _, _ in log}) == list(range(num_iters + 1))


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
    A = spd(5, rng, 0.2)  # deliberately ill-conditioned: no early convergence
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


def test_recorded_loss_is_measured_after_the_batch_advance():
    """``losses[i]`` is measured on the batch installed after step ``i + 1``, not the value
    that step's Armijo test accepted. The per-batch offset is what makes the two
    distinguishable; it cancels inside Armijo, which only compares values on one batch."""
    offset = 1000.0
    centre = np.array([1.0, -2.0, 0.5])
    curv = np.array([1.0, 3.0, 9.0])  # ill-conditioned: no single step reaches the minimum
    state = {"b": 0}

    def vg(x):
        d = x - centre
        return 0.5 * (d * curv) @ d + state["b"] * offset, curv * d

    def advance():
        state["b"] += 1

    losses = []
    traj = run_stochastic_lbfgs(
        vg, advance, np.zeros(3), num_iters=5, callbacks=[lambda a, L, i: losses.append(L[-1])]
    )

    assert traj.n_accepted == 5, "every step must be accepted for the positions to line up"
    assert len(losses) == 5
    for i, it in enumerate(traj.iterates):
        d = it["x"] - centre
        quad = 0.5 * (d * curv) @ d
        # batch (i + 1), not batch i: the advance has already run when the loss is recorded.
        assert losses[i] == pytest.approx(quad + (i + 1) * offset, rel=1e-12)
