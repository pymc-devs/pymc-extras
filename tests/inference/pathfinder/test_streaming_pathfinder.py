"""Tests for the streaming Pathfinder driver (compiles PyTensor; small models)."""

import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.pathfinder import fit_streaming_pathfinder
from pymc_extras.inference.pathfinder.stochastic_lbfgs import StochasticLBFGSConfig

CFG = StochasticLBFGSConfig(maxcor=6)


class ArrayLoader:
    """Minimal sized re-iterable batch source over a dense (N, cols) array.

    ``total_size`` is the dataset row count N (what the model passes as
    ``total_size``), and every ``iter()`` is a fresh complete pass over all rows
    (no dropped tail), so the loader can serve as its own ``full_pass``.
    """

    def __init__(self, data, batch_size, shuffle=False, seed=0):
        self.data = np.asarray(data, dtype=np.float64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    @property
    def total_size(self):
        return self.data.shape[0]

    def __iter__(self):
        idx = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, idx.size, self.batch_size):
            yield self.data[idx[start : start + self.batch_size]]


def gaussian_regression(X, y, sigma, prior_sd=10.0):
    """Linear regression with fixed noise; all-Normal so unconstrained == constrained.

    Returns ``(model, packed_data, analytic_mean, analytic_cov)``. The packed data
    is ``hstack([X, y])`` for the loader; the analytic Gaussian posterior of ``beta``
    is the ground truth the streaming fit must recover.
    """
    n, k = X.shape
    packed = np.hstack([X, y[:, None]]).astype(np.float64)
    with pm.Model() as model:
        batch = pm.Data("batch", packed)
        beta = pm.Normal("beta", 0.0, prior_sd, shape=k)
        mu = pm.math.dot(batch[:, :k], beta)
        pm.Normal("y", mu, sigma, observed=batch[:, k], total_size=n)

    prec = X.T @ X / sigma**2 + np.eye(k) / prior_sd**2
    cov = np.linalg.inv(prec)
    mean = cov @ (X.T @ y / sigma**2)
    return model, packed, mean, cov


def gaussian_case(seed=0, n=60, beta=(1.0, -0.5)):
    """Seeded Gaussian-regression model, its packed data, and the rng that drew them."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    y = X @ np.asarray(beta) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    return model, packed, rng


def logistic_regression(X, y, prior_sd=2.0):
    """Bayesian logistic regression with a pm.Data batch placeholder.

    Returns ``(model, packed_data)`` where packed is ``hstack([X, y])``.
    """
    n, k = X.shape
    packed = np.hstack([X, y[:, None]]).astype(np.float64)
    with pm.Model() as model:
        batch = pm.Data("batch", packed)
        beta = pm.Normal("beta", 0.0, prior_sd, shape=k)
        logit = pm.math.dot(batch[:, :k], beta)
        pm.Bernoulli("y", logit_p=logit, observed=batch[:, k], total_size=n)
    return model, packed


def sample_logistic(k, n, rng, beta_true=None):
    """Draw a logistic-regression dataset (X, y, beta_true)."""
    beta_true = rng.normal(size=k) if beta_true is None else beta_true
    X = rng.normal(size=(n, k))
    p = 1.0 / (1.0 + np.exp(-(X @ beta_true)))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y, beta_true


def logistic_case(seed=0, k=2, n=200):
    """Seeded logistic-regression model and its packed data (the Gaussian case's twin)."""
    rng = np.random.default_rng(seed)
    X, y, _ = sample_logistic(k, n, rng)
    return logistic_regression(X, y)


def record_trajectory(monkeypatch, edit=lambda traj: None):
    """Return the list the driver's Trajectories land in, one per fit run after this call.

    ``edit`` runs on each Trajectory before the fit reads it, for tests that need counters
    the optimizer would not produce on its own.
    """
    import pymc_extras.inference.pathfinder.streaming_pathfinder as sp

    run, recorded = sp.run_stochastic_lbfgs, []

    def wrapper(*args, **kwargs):
        traj = run(*args, **kwargs)
        edit(traj)
        recorded.append(traj)
        return traj

    monkeypatch.setattr(sp, "run_stochastic_lbfgs", wrapper)
    return recorded


def test_set_data_changes_compiled_objective():
    """The compiled gradient closes over the pm.Data batch: set_data changes it.

    This is the seam the whole streaming design rests on — one compiled graph,
    different minibatches fed between calls.
    """
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs

    model, packed = logistic_case(seed=0, n=200)
    vg = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)

    x = np.zeros(2)
    _, g_all = vg(x)
    model.set_data("batch", packed[:50])  # a sub-batch
    _, g_sub = vg(x)
    assert not np.allclose(g_all, g_sub), "gradient did not respond to set_data"


def test_saturated_trial_point_rejected():
    """A saturated logit has a finite logp and a NaN gradient, so Armijo on the value alone
    would accept it and every later step would inherit the NaN direction."""
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs
    from pymc_extras.inference.pathfinder.stochastic_lbfgs import run_stochastic_lbfgs

    k = 5
    model, packed = logistic_case(seed=1, k=k, n=1000)
    model.set_data("batch", packed[:64])
    neg_logp_dlogp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)

    def value_grad(x):
        value, grad = neg_logp_dlogp(np.asarray(x, dtype=np.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    x0 = np.random.default_rng(1).uniform(-6.0, 6.0, size=k)
    assert np.all(np.isfinite(value_grad(x0)[1])), "the start itself must be usable"
    traj = run_stochastic_lbfgs(value_grad, lambda: None, x0, 30, CFG)

    assert traj.n_accepted > 0, "every line search failed; the trajectory is empty"
    # mu = x - H_inv @ g, so an iterate carrying a NaN g is a NaN Gaussian mean.
    for it in traj.iterates:
        assert np.all(np.isfinite(it["g"]))


def test_smoke_streaming_logistic():
    """A short streaming fit on logistic data returns finite, correctly shaped draws."""
    model, packed = logistic_case(seed=1, n=400)
    loader = ArrayLoader(packed, batch_size=64)
    res = fit_streaming_pathfinder(
        model,
        loader,
        num_iters=15,
        num_draws=500,
        random_seed=2,
        lbfgs_config=CFG,
    )
    assert res.samples.shape == (500, 2)
    assert np.isfinite(res.samples).all()
    assert 0.0 <= res.violation_rate <= 1.0


def test_crn_determinism():
    """Same seed gives an identical fit (loader, optimizer, and draws are all seeded)."""
    model, packed = logistic_case(seed=3, n=300)

    def run():
        loader = ArrayLoader(packed, batch_size=64)
        return fit_streaming_pathfinder(
            model,
            loader,
            num_iters=20,
            num_draws=400,
            random_seed=7,
            lbfgs_config=CFG,
        )

    a, b = run(), run()
    np.testing.assert_array_equal(a.logQ, b.logQ)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_gaussian_is_centred_on_tail_averaged_position(monkeypatch):
    """The proposal centre is the mean of the last 75% of iterate positions, not one iterate,
    because each step's accepted point sits near its own minibatch's MAP."""
    import pymc_extras.inference.pathfinder.streaming_pathfinder as sp

    from pymc_extras.inference.pathfinder.bfgs_sample import make_pathfinder_sample_fn

    model, packed = logistic_case(seed=4, n=300)

    # Record the Gaussian the final draw is taken from, and the iterates it came from.
    gaussian = {}
    sample_fn = make_pathfinder_sample_fn(model, N=2, J=CFG.maxcor, jacobian=True)

    def record_gaussian(x, g, alpha, s_win, z_win, u):
        gaussian.update(x=x, g=g, alpha=alpha, s_win=s_win, z_win=z_win)
        return sample_fn(x, g, alpha, s_win, z_win, u)

    monkeypatch.setattr(sp, "make_pathfinder_sample_fn", lambda *a, **k: record_gaussian)
    trajectories = record_trajectory(monkeypatch)
    fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, batch_size=64),
        num_iters=25,
        num_draws=300,
        importance_sampling=None,
        random_seed=5,
        lbfgs_config=CFG,
    )

    (traj,) = trajectories  # the fit walks one path, so it builds one trajectory
    m = max(1, round(sp._TAIL_AVERAGE_FRAC * len(traj.iterates)))
    assert 1 < m < len(traj.iterates), "the average has to span several iterates to mean anything"
    x = gaussian["x"]
    np.testing.assert_allclose(x, np.mean([it["x"] for it in traj.iterates[-m:]], axis=0))
    assert not np.allclose(x, traj.iterates[-1]["x"]), "averaging collapsed to the last iterate"
    # mu = x - H_inv @ g, so only g = 0 centres the Gaussian on the averaged position.
    np.testing.assert_array_equal(gaussian["g"], np.zeros_like(x))
    # Curvature is the last iterate's: averaged (s, z) pairs are not a valid L-BFGS memory.
    for key in ("alpha", "s_win", "z_win"):
        np.testing.assert_array_equal(gaussian[key], traj.iterates[-1][key])


def test_health_counters_reach_the_result(monkeypatch):
    """violation_rate and n_ls_failures are read off the trajectory, not defaulted."""

    def struggle(traj):
        traj.n_curvature_violations = traj.n_accepted
        traj.n_ls_failures = 7

    record_trajectory(monkeypatch, struggle)
    model, packed, _ = gaussian_case()
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, batch_size=20),
        num_iters=6,
        num_draws=10,
        importance_sampling=None,
        random_seed=0,
    )
    assert result.violation_rate == 0.5
    assert result.n_ls_failures == 7


def test_jitter_varies_the_start_by_seed(monkeypatch):
    """Without jitter every seed starts at the same prior point and retraces one trajectory."""
    model, packed, _ = gaussian_case()
    trajectories = record_trajectory(monkeypatch)
    for seed in (0, 1):
        fit_streaming_pathfinder(
            model,
            ArrayLoader(packed, 20),  # unshuffled: the seed's only entry point is the jitter
            num_iters=4,
            num_draws=10,
            importance_sampling=None,
            random_seed=seed,
        )
    first, second = (traj.iterates[0]["x"] for traj in trajectories)
    assert not np.allclose(first, second)


def test_callbacks_get_pm_fit_arguments():
    """Callbacks see ``(None, losses_so_far, 1-based step)`` and one can stop the fit early."""
    model, packed, _ = gaussian_case()
    seen = []

    def stop_at_3(approx, losses, i):
        seen.append((approx, len(losses), i))
        if i == 3:
            raise StopIteration

    res = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, 20),
        num_iters=30,
        num_draws=10,
        importance_sampling=None,
        callbacks=[stop_at_3],
        random_seed=0,
    )
    assert seen == [(None, 1, 1), (None, 2, 2), (None, 3, 3)]
    assert res.samples.shape == (10, 2)


def test_missing_batch_var_raises():
    """A model without the named placeholder gives an actionable error."""
    model, packed = logistic_case(seed=6, n=100)
    loader = ArrayLoader(packed, batch_size=32)
    with pytest.raises(KeyError, match=r"pm\.Data"):
        fit_streaming_pathfinder(model, loader, batch_var="nope", num_iters=3)


@pytest.mark.slow
def test_gaussian_equivalence_to_analytic():
    """On a Gaussian posterior the streaming fit recovers the analytic mean and sd.

    Full-batch loader = deterministic Pathfinder; the target is exactly Gaussian,
    which Pathfinder is built to represent, so mean/sd must match the closed-form
    posterior within Monte-Carlo error across seeds.
    """
    rng = np.random.default_rng(10)
    k, n, sigma = 3, 500, 1.0
    beta_true = np.array([1.5, -2.0, 0.5])
    X = rng.normal(size=(n, k))
    y = X @ beta_true + rng.normal(0, sigma, size=n)
    model, packed, amean, acov = gaussian_regression(X, y, sigma)
    asd = np.sqrt(np.diag(acov))

    for seed in range(3):
        loader = ArrayLoader(packed, batch_size=n)  # full batch
        res = fit_streaming_pathfinder(
            model,
            loader,
            num_iters=60,
            num_draws=4000,
            random_seed=seed,
            lbfgs_config=CFG,
        )
        post_mean = res.samples.mean(0)
        post_sd = res.samples.std(0)
        se = asd / np.sqrt(res.samples.shape[0])
        assert np.all(np.abs(post_mean - amean) < 6 * se + 0.01), (
            f"seed {seed}: mean {post_mean} vs analytic {amean}"
        )
        assert np.all(np.abs(post_sd - asd) < 0.3 * asd), (
            f"seed {seed}: sd {post_sd} vs analytic {asd}"
        )


@pytest.mark.slow
def test_batch_size_robustness():
    """Posterior means agree across batch sizes, down to small minibatches."""
    model, packed = logistic_case(seed=11, k=3, n=2000)

    means = {}
    for bs in (2000, 256):
        loader = ArrayLoader(packed, batch_size=bs, shuffle=(bs < 2000), seed=0)
        res = fit_streaming_pathfinder(
            model,
            loader,
            num_iters=150,
            num_draws=3000,
            random_seed=0,
            lbfgs_config=CFG,
        )
        means[bs] = res.samples.mean(0)
    # minibatch mean tracks the full-batch mean to within a modest tolerance
    assert np.all(np.abs(means[256] - means[2000]) < 0.25 + 0.15 * np.abs(means[2000]))


def full_data_logp_parts(model, k, rng, n_phi=4):
    """Return ``(_full_data_logp, prior_fn, obs_fn, exact_logp, phi)`` for ``model``.

    ``exact_logp`` evaluates the target on whatever is in the placeholder; call it with the
    whole dataset set to get the value ``_full_data_logp`` reproduces from a streamed pass.
    """
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs
    from pymc_extras.inference.pathfinder.streaming_pathfinder import (
        _compile_batched_logp,
        _full_data_logp,
    )

    prior_fn = _compile_batched_logp(model, model.free_RVs, jacobian=True)
    obs_fn = _compile_batched_logp(model, model.observed_RVs, jacobian=False)
    nlp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    phi = rng.normal(size=(n_phi, k))

    def exact_logp():
        return np.array([-nlp(row.astype(np.float64))[0] for row in phi])

    return _full_data_logp, prior_fn, obs_fn, exact_logp, phi


def test_full_data_logp_invariant_to_batch_order_and_size():
    """logP is the exact full-data density, so any partition of the rows into batches, in
    any order, gives the same value as evaluating the target on the whole dataset.

    Every batch's contribution is un-rescaled by its own row count, so a pass of uneven
    or single-row batches, or one that visits the rows shuffled, must not shift the total.
    """
    k, n = 2, 60
    model, packed, rng = gaussian_case(seed=21, beta=(0.8, -1.2))
    full_data_logp, prior_fn, obs_fn, exact_logp, phi = full_data_logp_parts(model, k, rng)

    shuffled = packed[rng.permutation(n)]
    passes = {
        "one-batch": [packed],
        "uneven-tail": [packed[i : i + 7] for i in range(0, n, 7)],
        "single-rows": [packed[i : i + 1] for i in range(n)],
        "shuffled": [shuffled[i : i + 13] for i in range(0, n, 13)],
    }
    got = {
        name: full_data_logp(phi, p, model, "batch", prior_fn, obs_fn, n)
        for name, p in passes.items()
    }
    model.set_data("batch", packed)
    truth = exact_logp()
    for name, value in got.items():
        np.testing.assert_allclose(value, truth, atol=1e-8, err_msg=name)


def test_drop_last_loader_is_refused():
    """A drop-last epoch cannot give an exact logP; the error names full_pass, which works."""
    n = 350
    model, packed, _ = gaussian_case(seed=24, n=n)

    class DropLastLoader(ArrayLoader):
        def __iter__(self):
            for start in range(0, self.total_size - self.batch_size + 1, self.batch_size):
                yield self.data[start : start + self.batch_size]

    loader = DropLastLoader(packed, batch_size=128)
    assert sum(b.shape[0] for b in loader) == 256 < loader.total_size

    with pytest.raises(RuntimeError, match=r"pass full_pass=<the loader's dataset source>"):
        fit_streaming_pathfinder(model, loader, num_iters=5, num_draws=50, random_seed=0)


def test_incomplete_full_pass_is_refused():
    """A pass that does not visit every row is rejected rather than returning a
    subset-rescaled density that silently looks like the full-data one."""
    k, n = 2, 40
    model, packed, rng = gaussian_case(seed=22, n=n, beta=(0.5, 0.25))
    full_data_logp, prior_fn, obs_fn, _, phi = full_data_logp_parts(model, k, rng)

    with pytest.raises(RuntimeError, match=r"visited 25 rows but total_size=40"):
        full_data_logp(phi, [packed[:25]], model, "batch", prior_fn, obs_fn, n)


def _restore_case(seed):
    """A model whose batch placeholder holds all n rows, plus a loader that does not
    divide n, so any block the fit installs differs from the incoming value."""
    n = 300
    model, packed, _ = gaussian_case(seed=seed, n=n)
    loader = ArrayLoader(packed, batch_size=64)  # 300 = 4 x 64 + 44
    assert [b.shape[0] for b in loader] == [64, 64, 64, 64, 44]
    return model, packed, loader


def test_fit_restores_the_batch_placeholder():
    """The fit hands the model back with the placeholder the caller set up.

    Without the restore it keeps whatever block the final logP pass installed last, so a
    model.logp() or pm.sample() straight after a fit silently scores that block rescaled by
    total_size instead of the dataset.
    """
    model, _, loader = _restore_case(31)
    before = np.array(model["batch"].get_value(), copy=True)
    assert before.shape[0] == loader.total_size

    fit_streaming_pathfinder(
        model, loader, num_iters=5, num_draws=50, random_seed=0, lbfgs_config=CFG
    )

    np.testing.assert_array_equal(model["batch"].get_value(), before)


def test_batch_placeholder_is_restored_when_the_fit_raises():
    """The restore is on the exception path too, so a caller that catches the error and
    goes on using the model is not handed a mutated one."""
    model, packed, loader = _restore_case(32)
    before = np.array(model["batch"].get_value(), copy=True)

    with pytest.raises(RuntimeError, match=r"visited 50 rows but total_size=300"):
        fit_streaming_pathfinder(
            model,
            loader,
            num_iters=5,
            num_draws=50,
            full_pass=[packed[:50]],
            random_seed=0,
            lbfgs_config=CFG,
        )

    np.testing.assert_array_equal(model["batch"].get_value(), before)


@pytest.mark.parametrize(
    "importance_sampling, num_proposal_draws, n_prop, has_pareto_k",
    [
        (None, None, 50, False),
        (None, 137, 137, False),
        ("identity", None, 200, False),
        ("identity", 137, 137, False),
        ("psis", None, 200, True),
        ("psis", 137, 137, True),
    ],
    ids=["none", "none-pool", "identity", "identity-pool", "psis", "psis-pool"],
)
def test_returned_draws_have_requested_shape(
    importance_sampling, num_proposal_draws, n_prop, has_pareto_k
):
    """samples is (num_draws, N) for every weighting method and proposal-pool size, while
    logP and logQ stay the length of the proposal pool they were computed on.

    The pool defaults to 4x num_draws when resampling, matching fit_pathfinder's
    num_paths x num_draws_per_path, and to num_draws when not; an explicit
    num_proposal_draws overrides both, and none of that may leak into the returned count.
    """
    num_draws = 50
    model, packed, _ = gaussian_case(seed=23)
    res = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, 20),
        num_iters=8,
        num_draws=num_draws,
        num_proposal_draws=num_proposal_draws,
        importance_sampling=importance_sampling,
        random_seed=0,
        lbfgs_config=CFG,
    )
    assert res.samples.shape == (num_draws, 2)
    assert res.logP.shape == (n_prop,)
    assert res.logQ.shape == (n_prop,)
    assert np.isfinite(res.samples).all()
    assert (res.pareto_k is not None) == has_pareto_k


def test_final_target_includes_potentials():
    """model.logp(vars=[...]) drops pm.Potential terms, so the returned weights would
    target a different posterior than the one the optimizer walked."""
    n = 10
    y = np.zeros(n)
    with pm.Model() as model:  # a potential that pulls theta away from the data
        theta = pm.Normal("theta", 0.0, 10.0)
        batch = pm.Data("batch", y)
        pm.Potential("penalty", -100.0 * (theta - 3.0) ** 2)
        pm.Normal("obs", theta, 1.0, observed=batch, total_size=n)
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(y, 5),
        num_iters=10,
        num_draws=25,
        importance_sampling=None,  # keeps logP row-aligned with samples
        random_seed=0,
    )
    model.set_data("batch", y)
    logp = model.compile_logp()
    truth = np.array([logp({"theta": float(t)}) for t in result.samples[:, 0]])
    np.testing.assert_allclose(result.logP, truth, atol=1e-8)


def test_batch_dependent_potential_is_refused():
    """The prior term is evaluated once, so a potential over the batch cannot be honored."""
    n = 10
    y = np.zeros(n)
    with pm.Model() as model:
        theta = pm.Normal("theta", 0.0, 10.0)
        batch = pm.Data("batch", y)
        pm.Potential("batchy", -pm.math.sum(batch) * theta)
        pm.Normal("obs", theta, 1.0, observed=batch, total_size=n)
    with pytest.raises(NotImplementedError, match="depends on the minibatch"):
        fit_streaming_pathfinder(model, ArrayLoader(y, 5), num_iters=4, random_seed=0)


@pytest.mark.parametrize(
    "importance_sampling, resamples",
    [("identity", True), (None, False)],
    ids=["identity-reweights", "none-returns-proposals"],
)
def test_identity_is_a_weighting_method_not_an_off_switch(importance_sampling, resamples):
    """identity means raw log-importance weights, which the upstream sampler implements;
    treating it as 'off' silently returned unweighted proposals."""
    import pymc_extras.inference.pathfinder.streaming_pathfinder as sp

    model, packed, _ = gaussian_case()
    called = []
    original = sp.psis_fn
    sp.psis_fn = lambda *a, **k: (called.append(k.get("method")), original(*a, **k))[1]
    try:
        result = fit_streaming_pathfinder(
            model,
            ArrayLoader(packed, 20),
            num_iters=8,
            num_draws=50,
            importance_sampling=importance_sampling,
            random_seed=0,
        )
    finally:
        sp.psis_fn = original
    assert called == (["identity"] if resamples else [])
    assert result.samples.shape == (50, 2)


def test_proposal_pool_smaller_than_num_draws_still_returns_num_draws():
    """Without a floor, samples[:num_draws] quietly returned a shorter array."""
    model, packed, _ = gaussian_case()
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, 20),
        num_iters=8,
        num_draws=200,
        num_proposal_draws=20,
        importance_sampling=None,
        random_seed=0,
    )
    assert result.samples.shape == (200, 2)


def test_full_data_logp_installs_every_batch():
    """The streaming pass must install each batch, not read whatever is already there.

    A finished optimization leaves its last minibatch in the placeholder, so a loop that
    forgot ``set_data`` would sum that one subset n_batches times. Existing tests
    miss it because their fixtures happen to leave the full dataset installed, in
    which case the rescaled sum telescopes back to the right answer by accident.
    """
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs
    from pymc_extras.inference.pathfinder.streaming_pathfinder import (
        _compile_batched_logp,
        _full_data_logp,
    )

    rng = np.random.default_rng(0)
    k, n, chunk = 2, 40, 10
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    phi = rng.normal(size=(4, k))

    nlp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    model.set_data("batch", packed)
    truth = np.array([-nlp(row.astype(np.float64))[0] for row in phi])

    prior_fn = _compile_batched_logp(model, [*model.free_RVs, *model.potentials], jacobian=True)
    obs_fn = _compile_batched_logp(model, model.observed_RVs, jacobian=False)
    model.set_data("batch", packed[:chunk])  # the stale batch a finished fit leaves behind
    got = _full_data_logp(
        phi,
        [packed[i : i + chunk] for i in range(0, n, chunk)],
        model,
        "batch",
        prior_fn,
        obs_fn,
        n,
    )
    np.testing.assert_allclose(got, truth, rtol=1e-9)


def test_loader_without_total_size_is_refused():
    """A loader that cannot report N would leave the driver guessing the scale of
    the likelihood; it must refuse up front rather than mis-weight silently."""
    model, packed, _ = gaussian_case()

    class UnsizedLoader:
        def __iter__(self):
            return iter([packed[:20]])

    with pytest.raises(TypeError, match="total_size"):
        fit_streaming_pathfinder(model, UnsizedLoader(), num_iters=4, random_seed=0)
