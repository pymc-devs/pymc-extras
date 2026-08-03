"""Streaming (minibatch) Pathfinder for PyMC.

Runs Pathfinder where the log-density gradients come from minibatches yielded by
a sized re-iterable loader, rather than the full dataset. The model carries the
data in a ``pm.Data`` placeholder scaled with ``total_size=len(loader)``, so
``model.logp()`` already returns the correctly rescaled full-data log-density for
whatever batch is currently set.

The optimizer core is :func:`run_stochastic_lbfgs`; its stored iterates are scored
against one fixed evaluation batch with shared Monte-Carlo draws, so the ELBO argmax is
a paired comparison rather than a lottery over batches, and the winner is then drawn at
full width and importance-resampled against the exact full-data ``logP``. The compiled
log-density, Gaussian sampler and PSIS are reused unchanged from the deterministic
Pathfinder (``bfgs_sample``, ``importance_sampling``).

Measured accuracy, against an exact full-data Laplace reference on Bayesian logistic
regression: the posterior *scale* is right (sd ratio 0.8-1.1), but the *location* is
not, and the error grows with the dataset. Pareto-k 3.4 at N=1e5 and 11.8 at N=1.6e6,
against 0.3-0.5 for ``fit_pathfinder`` on the same data; the mean is 3.7 reference-sd
away at N=1e5 and 22 sd away at N=1.6e6. Larger batches help (Pareto-k 7.8 -> 3.9
raising the batch from 512 to 8192 rows at N=1e5); more iterations do not help at all.
Two causes are known and neither is fixed here: the stochastic trajectory never
approaches the MAP as closely as the deterministic one, and each iterate's stored
gradient is one rescaled minibatch gradient whose noise lands directly in the Gaussian
mean. Agreement with ``fit_pathfinder`` at large N is therefore unverified — treat the
draws as a proposal, and read ``pareto_k``. ``violation_rate`` reads 0.0 on every one of
those runs including the worst, so it cannot stand in for that check.

What streaming does buy is memory: 6.4x less resident memory than ``fit_pathfinder`` at
N=4e5 and 11.4x at N=1.6e6, at comparable wall clock for a matched proposal pool.
"""

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from pymc.model.core import Point
from pymc.pytensorf import compile as pm_compile
from pytensor.graph import vectorize_graph
from pytensor.graph.traversal import ancestors

from pymc_extras.inference.pathfinder.bfgs_sample import (
    get_neg_logp_dlogp_of_ravel_inputs,
    make_pathfinder_sample_fn,
)
from pymc_extras.inference.pathfinder.importance_sampling import importance_sampling as psis_fn
from pymc_extras.inference.pathfinder.stochastic_lbfgs import (
    StochasticLBFGSConfig,
    run_stochastic_lbfgs,
)

__all__ = ["StreamingPathfinderResult", "fit_streaming_pathfinder"]


@dataclass
class StreamingPathfinderResult:
    """Output of :func:`fit_streaming_pathfinder`.

    Attributes
    ----------
    samples : ndarray, shape (num_draws, N)
        Posterior draws in the model's raveled unconstrained space. When importance
        sampling is active these are resampled from a larger proposal pool.
    logP, logQ : ndarray
        Target and proposal log-densities of the *proposal* draws (length equals the
        proposal pool, ``num_proposal_draws``). They are the weights behind the
        resampling and are row-aligned with ``samples`` only when importance sampling
        is off (``None``); after resampling they are proposal-space diagnostics, not
        per-``samples``-row densities.
    elbo_trace : ndarray
        ELBO of every stored iterate, evaluated on the shared evaluation batch.
    elbo_argmax : int
        Index of the selected iterate within ``elbo_trace``.
    pareto_k : float or None
        PSIS Pareto shape diagnostic (None when importance sampling is disabled).
    violation_rate : float
        Fraction of moved optimizer steps whose ``(s, y)`` pair failed the curvature
        test. An optimizer-health counter; it says nothing about how close the draws
        are to the posterior — read ``pareto_k`` for that.
    n_ls_failures : int
        Number of line-search failures during optimization.
    """

    samples: np.ndarray
    logP: np.ndarray
    logQ: np.ndarray
    elbo_trace: np.ndarray
    elbo_argmax: int
    pareto_k: float | None
    violation_rate: float
    n_ls_failures: int


def _elbo(logP, logQ):
    """Mean ELBO over the draws, matching upstream ``LBFGSStreamingCallback``.

    A single draw with non-finite ``logP`` (proposal mass where the target has zero
    density) collapses the estimate to ``-inf`` rather than being dropped, so a
    support-violating iterate cannot outscore a valid one.
    """
    logP_safe = np.where(np.isfinite(logP), np.asarray(logP), -np.inf)
    elbo = float(np.mean(logP_safe - np.asarray(logQ)))
    return elbo if np.isfinite(elbo) else -np.inf


def _compile_batched_logp(model, terms, jacobian):
    """Compile ``phi (M, N) -> logp (M,)`` over ``terms``, vectorized across draws.

    Mirrors the vectorization in ``make_pathfinder_sample_fn``: build the single-draw
    log-density over the raveled unconstrained value vector (same coordinate order as
    the sampler's ``phi``), then batch it over a leading draw axis.
    """
    (logp_single,), single_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(), [model.logp(vars=terms, jacobian=jacobian)], model.value_vars, ()
    )
    phi = pt.matrix("phi")
    batched = vectorize_graph(logp_single, replace={single_input: phi})
    fn = pm_compile([phi], batched)
    fn.trust_input = True
    return fn


def _full_data_logp(phi, full_pass, model, batch_var, prior_fn, obs_fn, n_total):
    """Exact full-data ``logP(phi)`` for out-of-core data, in one streaming pass.

    ``prior_fn`` (prior + Jacobian) is evaluated once; the observed log-likelihood is
    summed across every batch of ``full_pass``, an iterable that must visit every row
    exactly once (it is consumed once). Each batch's observed logp carries the model's
    ``total_size=n_total`` rescaling (``(n_total / b) * batch_sum``), so it is
    multiplied back by ``b / n_total`` to recover the exact batch sum. The result equals
    ``model.logp`` on the full dataset (verified to machine precision), not the
    subset-rescaled pseudo-density.

    Assumes the single-observed-RV, ``total_size=n_total`` minibatch contract.
    """
    phi = np.asarray(phi, dtype=np.float64)
    lp = np.asarray(prior_fn(phi), dtype=np.float64)
    seen = 0
    for batch in full_pass:  # one complete epoch; order is irrelevant to the sum
        b = np.asarray(batch)
        model.set_data(batch_var, b)
        lp = lp + (b.shape[0] / n_total) * np.asarray(obs_fn(phi), dtype=np.float64)
        seen += b.shape[0]
    if seen != n_total:
        raise RuntimeError(
            f"full-data logp pass visited {seen} rows but len(loader)={n_total}; the "
            "pass must yield every row exactly once for an exact logP. A loader epoch "
            "that drops its trailing partial batch cannot do that, so pass full_pass="
            "<the loader's dataset source> instead."
        )
    return lp


def fit_streaming_pathfinder(
    model,
    loader,
    *,
    batch_var="batch",
    num_iters=200,
    num_elbo_draws=10,
    num_draws=1000,
    num_proposal_draws=None,
    eval_rows=2000,
    full_pass=None,
    jitter=2.0,
    jacobian_correction=True,
    importance_sampling="psis",
    lbfgs_config=None,
    random_seed=None,
):
    """Fit a streaming Pathfinder approximation to ``model`` using ``loader``.

    Parameters
    ----------
    model : pymc.Model
        A model whose data enter through a ``pm.Data`` placeholder named
        ``batch_var`` and whose likelihood passes ``total_size=len(loader)``.
    loader : iterable
        Yields minibatches (arrays whose leading axis is the batch rows) and
        supports ``len(loader) == N`` (the dataset row count).
    batch_var : str
        Name of the ``pm.Data`` placeholder to stream into.
    num_iters : int
        Number of stochastic L-BFGS steps.
    num_elbo_draws : int
        Monte-Carlo draws per iterate for ELBO selection.
    num_draws : int
        Draws returned from the selected Gaussian.
    num_proposal_draws : int, optional
        Size of the proposal pool importance sampling resamples down to ``num_draws``.
        Defaults to ``num_draws``, raised by one when resampling so that PSIS reweights
        rather than permuting. The final ``logP`` costs one O(``num_proposal_draws`` x N)
        pass over the data and dominates the fit, so raising this trades wall clock for
        importance-weight quality roughly one for one.
    eval_rows : int
        Rows in the fixed evaluation batch used for iterate selection.
    full_pass : iterable, optional
        An iterable yielding every row exactly once, for the final exact full-data
        ``logP``; consumed once. Defaults to iterating ``loader``, which is correct only
        when a loader epoch covers every row. ``pymc_extras.variational.DataLoader``
        drops the trailing ``len(loader) % batch_size`` rows by design, so pass its
        dataset source here instead — any iterable of row blocks will do, and the batch
        sizes need not match. A raw source bypasses the loader's ``preprocess_fn``, so
        apply that here too if one is set.
    jitter : float
        Uniform jitter added to the prior initial point.
    jacobian_correction : bool
        Include the change-of-variables Jacobian so logp is the unconstrained
        joint density (matches ``fit_pathfinder``).
    importance_sampling : {"psis", "psir", "identity", None}
        Post-hoc reweighting of the returned draws; ``None`` returns the proposal
        draws unweighted. ``"identity"`` weights by the raw log ratio, so like
        ``"psis"``/``"psir"`` it resamples from a larger pool.
    lbfgs_config : StochasticLBFGSConfig, optional
    random_seed : int, optional

    Returns
    -------
    StreamingPathfinderResult
    """
    model = pm.modelcontext(model)
    if batch_var not in model.named_vars:
        raise KeyError(
            f"batch_var {batch_var!r} is not a variable in the model; add a "
            f"pm.Data({batch_var!r}, ...) placeholder that the data stream feeds."
        )
    cfg = lbfgs_config or StochasticLBFGSConfig()
    J = cfg.maxcor

    init_ss, elbo_ss, final_ss, resample_ss = np.random.SeedSequence(random_seed).spawn(4)

    # Compile once; all functions below close over the batch_var pm.Data shared variable.
    neg_logp_dlogp = get_neg_logp_dlogp_of_ravel_inputs(model, jacobian=jacobian_correction)
    ip = Point(make_initial_point_fn(model=model)(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    N = x_base.shape[0]
    sample_logp = make_pathfinder_sample_fn(model, N=N, J=J, jacobian=jacobian_correction)
    if model.potentials and model[batch_var] in set(ancestors(model.potentials)):
        raise NotImplementedError(
            "a pm.Potential that depends on the minibatch cannot be evaluated once "
            "against the full data; move it out of the batch or fold it into the "
            "observed likelihood."
        )
    prior_logp_fn = _compile_batched_logp(
        model, [*model.free_RVs, *model.potentials], jacobian=jacobian_correction
    )
    obs_logp_fn = _compile_batched_logp(model, model.observed_RVs, jacobian=False)

    def value_grad_fn(x):
        value, grad = neg_logp_dlogp(np.asarray(x, dtype=np.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    epoch = iter(loader)

    def next_batch():
        nonlocal epoch
        try:
            return next(epoch)
        except StopIteration:
            epoch = iter(loader)
            return next(epoch)

    # One batch fixed for the whole selection sweep, so every iterate is scored on the
    # same rows. Not held out: they come off the stream the optimizer also trains on.
    n_total = len(loader)
    target_rows = min(eval_rows, n_total)
    chunks, rows = [], 0
    while rows < target_rows:
        b = next_batch()
        chunks.append(b)
        rows += b.shape[0]
    eval_batch = np.concatenate(chunks, axis=0)[:target_rows]

    init_rng = np.random.default_rng(init_ss)
    x0 = x_base + init_rng.uniform(-jitter, jitter, size=N)
    model.set_data(batch_var, next_batch())

    traj = run_stochastic_lbfgs(
        value_grad_fn, lambda: model.set_data(batch_var, next_batch()), x0, num_iters, cfg
    )
    if not traj.iterates:
        raise RuntimeError(
            "Streaming L-BFGS produced no accepted steps; try more iterations, a "
            "smaller jitter, or a larger batch size."
        )

    model.set_data(batch_var, eval_batch)
    u_elbo = np.random.default_rng(elbo_ss).standard_normal((num_elbo_draws, N))
    elbo_trace = np.empty(len(traj.iterates))
    for k, it in enumerate(traj.iterates):
        _, logQ, logP, _ = sample_logp(
            it["x"], it["g"], it["alpha"], it["s_win"], it["z_win"], u_elbo
        )
        elbo_trace[k] = _elbo(logP, logQ)
    elbo_argmax = int(np.argmax(elbo_trace))
    best = traj.iterates[elbo_argmax]

    resample = importance_sampling is not None
    # Resampling num_draws out of exactly num_draws candidates is drawing without
    # replacement from the whole pool: a permutation, not a reweighting. Hence the floor.
    n_prop = int(num_proposal_draws) if num_proposal_draws is not None else num_draws
    n_prop = max(n_prop, num_draws + 1 if resample else num_draws)
    u_final = np.random.default_rng(final_ss).standard_normal((n_prop, N))
    phi, logQ, _logP_subset, _ = sample_logp(
        best["x"], best["g"], best["alpha"], best["s_win"], best["z_win"], u_final
    )
    samples = np.asarray(phi)
    logQ = np.asarray(logQ)
    # sample_logp's logP is the eval-batch pseudo-density; the weights need the exact one.
    logP = _full_data_logp(
        samples,
        full_pass if full_pass is not None else loader,
        model,
        batch_var,
        prior_logp_fn,
        obs_logp_fn,
        n_total,
    )

    pareto_k = None
    if resample:
        result = psis_fn(
            samples[None],
            logP[None],
            logQ[None],
            num_draws,
            method=importance_sampling,
            random_seed=int(resample_ss.generate_state(1)[0]),
        )
        samples = np.asarray(result.samples)
        pareto_k = result.pareto_k
    else:
        samples = samples[:num_draws]

    return StreamingPathfinderResult(
        samples=samples,
        logP=logP,
        logQ=logQ,
        elbo_trace=elbo_trace,
        elbo_argmax=elbo_argmax,
        pareto_k=pareto_k,
        violation_rate=traj.violation_rate,
        n_ls_failures=traj.n_ls_failures,
    )
