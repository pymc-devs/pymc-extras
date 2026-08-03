"""Tests for the streaming Pathfinder driver (compiles PyTensor; small models)."""

import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.pathfinder import fit_streaming_pathfinder
from pymc_extras.inference.pathfinder.stochastic_lbfgs import StochasticLBFGSConfig

CFG = StochasticLBFGSConfig(maxcor=6)


class ArrayLoader:
    """Minimal sized re-iterable batch source over a dense (N, cols) array.

    ``len(loader)`` is the dataset row count N (what the model passes as
    ``total_size``), and every ``iter()`` is a fresh complete pass over all rows
    (no dropped tail), so the loader can serve as its own ``full_pass``.
    """

    def __init__(self, data, batch_size, shuffle=False, seed=0):
        self.data = np.asarray(data, dtype=np.float64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
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


def test_set_data_changes_compiled_objective():
    """The compiled gradient closes over the pm.Data batch: set_data changes it.

    This is the seam the whole streaming design rests on — one compiled graph,
    different minibatches fed between calls.
    """
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs

    rng = np.random.default_rng(0)
    X, y, _ = sample_logistic(k=2, n=200, rng=rng)
    model, packed = logistic_regression(X, y)
    vg = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)

    x = np.zeros(2)
    _, g_all = vg(x)
    model.set_data("batch", packed[:50])  # a sub-batch
    _, g_sub = vg(x)
    assert not np.allclose(g_all, g_sub), "gradient did not respond to set_data"


def test_smoke_streaming_logistic():
    """A short streaming fit on logistic data returns finite, correctly shaped draws."""
    rng = np.random.default_rng(1)
    X, y, _ = sample_logistic(k=2, n=400, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = ArrayLoader(packed, batch_size=64)
    res = fit_streaming_pathfinder(
        model,
        loader,
        num_iters=15,
        num_draws=500,
        eval_rows=200,
        random_seed=2,
        lbfgs_config=CFG,
    )
    assert res.samples.shape == (500, 2)
    assert np.isfinite(res.samples).all()
    assert res.elbo_trace.size == len(res.elbo_trace)
    assert 0.0 <= res.violation_rate <= 1.0


def test_crn_determinism():
    """Same seed gives an identical fit (loader, optimizer, and draws are all seeded)."""
    rng = np.random.default_rng(3)
    X, y, _ = sample_logistic(k=2, n=300, rng=rng)
    model, packed = logistic_regression(X, y)

    def run():
        loader = ArrayLoader(packed, batch_size=64)
        return fit_streaming_pathfinder(
            model,
            loader,
            num_iters=20,
            num_draws=400,
            eval_rows=150,
            random_seed=7,
            lbfgs_config=CFG,
        )

    a, b = run(), run()
    np.testing.assert_array_equal(a.elbo_trace, b.elbo_trace)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_elbo_argmax_selects_max():
    """The reported argmax is the index of the largest ELBO in the trace."""
    rng = np.random.default_rng(4)
    X, y, _ = sample_logistic(k=2, n=300, rng=rng)
    model, packed = logistic_regression(X, y)
    loader = ArrayLoader(packed, batch_size=64)
    res = fit_streaming_pathfinder(
        model,
        loader,
        num_iters=25,
        num_draws=300,
        eval_rows=150,
        random_seed=5,
        lbfgs_config=CFG,
    )
    assert res.elbo_argmax == int(np.argmax(res.elbo_trace))


def test_a_degenerate_iterate_cannot_win_the_argmax():
    """A draw with zero proposal density scores +inf, which would take the argmax
    outright; a non-finite score has to fall to -inf instead."""
    from pymc_extras.inference.pathfinder.streaming_pathfinder import _elbo

    assert _elbo(np.array([1.0, 2.0]), np.array([-np.inf, 0.0])) == -np.inf
    assert _elbo(np.array([1.0, 2.0]), np.array([0.0, 0.0])) == 1.5


def test_evaluation_batch_holds_exactly_eval_rows(monkeypatch):
    """Rows past eval_rows are ones the loader wrapped around the epoch to fetch again,
    so the selection sweep would score iterates on a batch with duplicated rows."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=50)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

    set_data = model.set_data
    installed = []
    monkeypatch.setattr(
        model,
        "set_data",
        lambda name, value, **kw: (installed.append(len(value)), set_data(name, value, **kw))[1],
    )
    fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, batch_size=16),
        num_iters=5,
        num_draws=10,
        eval_rows=40,
        importance_sampling=None,
        random_seed=0,
    )
    assert max(installed) == 40


def test_optimizer_health_counters_reach_the_result(monkeypatch):
    """Both counters are read off the trajectory, so a fit that struggled cannot come
    back reporting a clean one."""
    import pymc_extras.inference.pathfinder.streaming_pathfinder as sp

    run = sp.run_stochastic_lbfgs

    def struggling(*args, **kwargs):
        traj = run(*args, **kwargs)
        traj.n_curvature_violations = traj.n_accepted
        traj.n_ls_failures = 7
        return traj

    monkeypatch.setattr(sp, "run_stochastic_lbfgs", struggling)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, batch_size=20),
        num_iters=6,
        num_draws=10,
        eval_rows=60,
        importance_sampling=None,
        random_seed=0,
    )
    assert result.violation_rate == 0.5
    assert result.n_ls_failures == 7


def test_jitter_starts_two_seeds_at_different_points(monkeypatch):
    """Without the jitter every fit starts at the same deterministic prior point, so a
    second seed retraces the first trajectory and multi-path Pathfinder degenerates."""
    import pymc_extras.inference.pathfinder.streaming_pathfinder as sp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

    run, firsts = sp.run_stochastic_lbfgs, []

    def record(*args, **kwargs):
        traj = run(*args, **kwargs)
        firsts.append(traj.iterates[0]["x"])
        return traj

    monkeypatch.setattr(sp, "run_stochastic_lbfgs", record)
    for seed in (0, 1):
        fit_streaming_pathfinder(
            model,
            ArrayLoader(packed, 20),  # unshuffled: the seed's only entry point is the jitter
            num_iters=4,
            num_draws=10,
            eval_rows=60,
            importance_sampling=None,
            random_seed=seed,
        )
    assert not np.allclose(firsts[0], firsts[1])


def test_callbacks_reach_the_optimizer():
    """The optimizer's callback hook is only worth its lines if a user can reach it: a
    pm.fit early-stopping rule has to run against a streaming fit unchanged."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

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
        eval_rows=60,
        importance_sampling=None,
        callbacks=[stop_at_3],
        random_seed=0,
    )
    assert seen == [(None, 1, 1), (None, 2, 2), (None, 3, 3)]
    assert res.elbo_trace.size <= 3


def test_missing_batch_var_raises():
    """A model without the named placeholder gives an actionable error."""
    rng = np.random.default_rng(6)
    X, y, _ = sample_logistic(k=2, n=100, rng=rng)
    model, packed = logistic_regression(X, y)
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
            eval_rows=n,
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
    rng = np.random.default_rng(11)
    X, y, _ = sample_logistic(k=3, n=2000, rng=rng)
    model, packed = logistic_regression(X, y)

    means = {}
    for bs in (2000, 256):
        loader = ArrayLoader(packed, batch_size=bs, shuffle=(bs < 2000), seed=0)
        res = fit_streaming_pathfinder(
            model,
            loader,
            num_iters=150,
            num_draws=3000,
            eval_rows=1000,
            random_seed=0,
            lbfgs_config=CFG,
        )
        means[bs] = res.samples.mean(0)
    # minibatch mean tracks the full-batch mean to within a modest tolerance
    assert np.all(np.abs(means[256] - means[2000]) < 0.25 + 0.15 * np.abs(means[2000]))


def test_full_data_logp_exact_with_tail():
    """Streaming full-data logP equals model.logp over the whole dataset, including the
    trailing partial batch a drop-last training loader would skip."""
    from pymc_extras.inference.pathfinder.bfgs_sample import get_neg_logp_dlogp_of_ravel_inputs
    from pymc_extras.inference.pathfinder.streaming_pathfinder import (
        _compile_batched_logp,
        _full_data_logp,
    )

    rng = np.random.default_rng(0)
    k, n = 3, 350  # 350 is not divisible by 128 -> a genuine partial tail
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

    # An explicit complete pass: every row exactly once, partial tail batch included.
    full_pass = [packed[i : i + 128] for i in range(0, n, 128)]
    assert sum(b.shape[0] for b in full_pass) == n
    assert full_pass[-1].shape[0] == n % 128  # the tail is genuinely partial

    prior_fn = _compile_batched_logp(model, model.free_RVs, jacobian=True)
    obs_fn = _compile_batched_logp(model, model.observed_RVs, jacobian=False)
    nlp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=True)
    phi = rng.normal(size=(5, k))
    got = _full_data_logp(phi, full_pass, model, "batch", prior_fn, obs_fn, n)
    for i in range(phi.shape[0]):
        model.set_data("batch", packed)  # full data in the placeholder = the exact truth
        truth = -nlp(phi[i].astype(np.float64))[0]
        assert abs(got[i] - truth) < 1e-6, (i, got[i], truth)


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
    rng = np.random.default_rng(21)
    k, n = 2, 60
    X = rng.normal(size=(n, k))
    y = X @ np.array([0.8, -1.2]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
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


def test_a_drop_last_loader_is_refused_and_told_what_to_pass():
    """A DataLoader epoch yields floor(N / batch_size) * batch_size rows, so for almost
    any N the default full_pass cannot produce an exact logP. The error has to name the
    remedy: it fires only after the whole optimization has been paid for."""
    rng = np.random.default_rng(24)
    k, n = 2, 350
    X = rng.normal(size=(n, k))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

    class DropLastLoader(ArrayLoader):
        def __iter__(self):
            for start in range(0, len(self) - self.batch_size + 1, self.batch_size):
                yield self.data[start : start + self.batch_size]

    loader = DropLastLoader(packed, batch_size=128)
    assert sum(b.shape[0] for b in loader) == 256 < len(loader)

    kwargs = dict(num_iters=5, num_draws=50, eval_rows=60, random_seed=0, lbfgs_config=CFG)
    with pytest.raises(RuntimeError, match=r"pass full_pass=<the loader's dataset source>"):
        fit_streaming_pathfinder(model, loader, **kwargs)

    res = fit_streaming_pathfinder(
        model, loader, full_pass=[packed[i : i + 97] for i in range(0, n, 97)], **kwargs
    )
    assert res.samples.shape == (50, k)


def test_incomplete_full_pass_is_refused():
    """A pass that does not visit every row is rejected rather than returning a
    subset-rescaled density that silently looks like the full-data one."""
    rng = np.random.default_rng(22)
    k, n = 2, 40
    X = rng.normal(size=(n, k))
    y = X @ np.array([0.5, 0.25]) + rng.normal(0, 1.0, size=n)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    full_data_logp, prior_fn, obs_fn, _, phi = full_data_logp_parts(model, k, rng)

    with pytest.raises(RuntimeError, match=r"visited 25 rows but len\(loader\)=40"):
        full_data_logp(phi, [packed[:25]], model, "batch", prior_fn, obs_fn, n)


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
    rng = np.random.default_rng(23)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    res = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, 20),
        num_iters=8,
        num_draws=num_draws,
        num_proposal_draws=num_proposal_draws,
        eval_rows=60,
        importance_sampling=importance_sampling,
        random_seed=0,
        lbfgs_config=CFG,
    )
    assert res.samples.shape == (num_draws, 2)
    assert res.logP.shape == (n_prop,)
    assert res.logQ.shape == (n_prop,)
    assert np.isfinite(res.samples).all()
    assert (res.pareto_k is not None) == has_pareto_k


def test_psis_reweights_rather_than_permuting_the_pool():
    """The default pool has to be a multiple of num_draws. Resampling num_draws out of
    num_draws + 1 without replacement (importance_sampling.py sets replace=False for
    "psis") returns the pool minus one draw whatever the weights are, and arviz's Pareto
    tail fit needs 5 tail draws, so num_draws <= 23 raised outright."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    res = fit_streaming_pathfinder(
        model, ArrayLoader(packed, 20), num_iters=8, num_draws=20, eval_rows=60, random_seed=0
    )
    assert res.samples.shape == (20, 2)
    assert res.logP.shape == (80,)
    assert res.pareto_k is not None


def potential_model(y, n):
    """Normal-normal model carrying a pm.Potential that pulls theta away from the data."""
    with pm.Model() as model:
        theta = pm.Normal("theta", 0.0, 10.0)
        batch = pm.Data("batch", y[: len(y)])
        pm.Potential("penalty", -100.0 * (theta - 3.0) ** 2)
        pm.Normal("obs", theta, 1.0, observed=batch, total_size=n)
    return model


def test_final_target_includes_potentials():
    """model.logp(vars=[...]) drops pm.Potential terms, so the returned weights would
    target a different posterior than the one the optimizer walked."""
    n = 10
    y = np.zeros(n)
    model = potential_model(y, n)
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(y, 5),
        num_iters=10,
        num_draws=25,
        eval_rows=n,
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

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)

    called = []
    original = sp.psis_fn
    sp.psis_fn = lambda *a, **k: (called.append(k.get("method")), original(*a, **k))[1]
    try:
        result = fit_streaming_pathfinder(
            model,
            ArrayLoader(packed, 20),
            num_iters=8,
            num_draws=50,
            eval_rows=60,
            importance_sampling=importance_sampling,
            random_seed=0,
        )
    finally:
        sp.psis_fn = original
    assert called == (["identity"] if resamples else [])
    assert result.samples.shape == (50, 2)


def test_proposal_pool_smaller_than_num_draws_still_returns_num_draws():
    """Without a floor, samples[:num_draws] quietly returned a shorter array."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1.0, size=60)
    model, packed, *_ = gaussian_regression(X, y, 1.0)
    result = fit_streaming_pathfinder(
        model,
        ArrayLoader(packed, 20),
        num_iters=8,
        num_draws=200,
        num_proposal_draws=20,
        eval_rows=60,
        importance_sampling=None,
        random_seed=0,
    )
    assert result.samples.shape == (200, 2)


def test_full_data_logp_installs_every_batch():
    """The streaming pass must install each batch, not read whatever is already there.

    A finished fit leaves the evaluation batch in the placeholder, so a loop that
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
