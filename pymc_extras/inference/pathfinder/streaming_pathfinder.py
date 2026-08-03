"""Streaming (minibatch) Pathfinder for PyMC.

Runs Pathfinder where the log-density gradients come from minibatches yielded by
a sized re-iterable loader, rather than the full dataset. The model carries the
data in a ``pm.Data`` placeholder scaled with ``total_size=len(loader)``, so
``model.logp()`` already returns the correctly rescaled full-data log-density for
whatever batch is currently set.

The optimizer core is :func:`run_stochastic_lbfgs`. Its iterate positions are tail-averaged
into a single point, which is then drawn at full width and importance-resampled against the
exact full-data ``logP``. The compiled log-density, Gaussian sampler and PSIS are reused
unchanged from the deterministic Pathfinder (``bfgs_sample``, ``importance_sampling``).

The draws it returns are a proposal, not a posterior; the measured operating range, in both
dataset size and dimension, is on :func:`fit_streaming_pathfinder`.

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

# Fraction of the trajectory's iterate positions that the tail average covers. Not a
# keyword: 0.50 and 0.75 are indistinguishable and the 0.50-0.90 basin is flat, so the
# only thing a user could learn from the knob is that 1.00 is broken.
_TAIL_AVERAGE_FRAC = 0.75


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
    pareto_k: float | None
    violation_rate: float
    n_ls_failures: int


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
    num_draws=1000,
    num_proposal_draws=None,
    full_pass=None,
    jitter=2.0,
    jacobian_correction=True,
    importance_sampling="psis",
    lbfgs_config=None,
    callbacks=(),
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
    num_draws : int
        Draws returned from the selected Gaussian.
    num_proposal_draws : int, optional
        Size of the proposal pool importance sampling resamples down to ``num_draws``.
        Defaults to ``4 * num_draws`` when resampling, matching ``fit_pathfinder``'s
        ``num_paths=4`` x ``num_draws_per_path=1000`` pool, and to ``num_draws`` when
        ``importance_sampling=None``. The pool's exact full-data ``logP`` is one
        O(``num_proposal_draws`` x N) pass and was 85-94% of wall clock at N=1e5, so this
        setting is very nearly the cost of the fit.
    full_pass : iterable, optional
        An iterable of row blocks visiting every row exactly once, for the final exact
        full-data ``logP``; consumed once. Defaults to iterating ``loader``, which is
        correct only when a loader epoch covers every row: a loader that drops its
        trailing ``len(loader) % batch_size`` rows does not, and the run raises rather
        than return a subset-rescaled ``logP``. Block sizes need not match the loader's.

        .. warning::
            Blocks are installed in ``batch_var`` unchanged. If ``loader`` transforms
            its batches before yielding them, ``full_pass`` must apply the identical
            transform. Untransformed rows are indistinguishable from correct ones here,
            so they produce a wrong ``logP``, and therefore wrong importance weights and
            wrong draws, with no error and no diagnostic.
    jitter : float
        Uniform jitter added to the prior initial point.
    jacobian_correction : bool
        Include the change-of-variables Jacobian so logp is the unconstrained
        joint density (matches ``fit_pathfinder``).
    importance_sampling : {"psis", "psir", "identity", None}
        Post-hoc reweighting of the returned draws; ``None`` returns the proposal
        draws unweighted. ``"identity"`` weights by the raw log ratio, so like
        ``"psis"``/``"psir"`` it resamples from a larger pool. At the Notes configuration
        (k=8, N=1e5, three seeds) the 4000-draw pool carried a raw-weight effective sample
        size of 449-1762, so the weights inform the resampling rather than collapsing onto
        a few draws; it cost 3.4-3.7x the wall clock of ``None``, which draws a pool of
        ``num_draws`` only.
    lbfgs_config : StochasticLBFGSConfig, optional
    callbacks : iterable of callable, optional
        ``pm.fit`` callbacks, called ``(None, losses, i)`` after each optimizer step;
        one raising ``StopIteration`` stops the optimization and the fit proceeds with
        the iterates recorded so far. ``losses`` are minibatch objectives, so they are
        noisy; ``approx`` is ``None`` because there is no ``Approximation`` here.
    random_seed : int, optional

    Returns
    -------
    StreamingPathfinderResult

    Notes
    -----
    Measured operating range. Bayesian logistic regression, Normal(0, 2) prior, batch 2048,
    ``num_iters=200``, ``maxcor=6``, ``jitter=2.0``, ``num_draws=1000``,
    ``num_proposal_draws=4000``, PSIS, against an exact full-data Laplace reference. With
    k=8 coefficients, worst-coordinate error in reference-sd:

    ====== ===================== =========================
    N      ``pareto_k``          worst coordinate (ref-sd)
    ====== ===================== =========================
    1e5    0.31-0.80 (med 0.69)  0.17-0.58
    4e5    0.80-0.91 (med 0.87)  1.07-1.56
    1.6e6  2.05-2.56 (med 2.52)  2.91-4.35
    ====== ===================== =========================

    On the same trajectories before tail averaging, N=1e5 gave ``pareto_k`` 5.80-7.00 and
    9.71-11.13 ref-sd. But ``pareto_k`` still exceeds 0.7 on 6 of 12 seeds at N=1e5 and on
    5 of 6 at N=4e5, so read it on every fit: these draws remain a *proposal*, not a
    posterior, and in the large-N regime that gap is not small.

    The range is in ``k`` as much as in ``N``. At k=100 the tail-averaged ``pareto_k`` is
    2.2-4.5 and the error 4.6-17.4 ref-sd — unusable, and worse, quiet: with the
    line-search gradient-finiteness guard the k=100 case no longer raises, it returns a
    silently wrong posterior. Do not run this above a few tens of dimensions without
    checking ``pareto_k``.

    Averaging buys a level, not an exponent: the gap still grows roughly 3x per 4x in N.
    ``violation_rate`` is an optimizer-health counter and read 0.0 on all of these runs.
    """
    model = pm.modelcontext(model)
    if batch_var not in model.named_vars:
        raise KeyError(
            f"batch_var {batch_var!r} is not a variable in the model; add a "
            f"pm.Data({batch_var!r}, ...) placeholder that the data stream feeds."
        )
    cfg = lbfgs_config or StochasticLBFGSConfig()
    J = cfg.maxcor

    init_ss, final_ss, resample_ss = np.random.SeedSequence(random_seed).spawn(3)

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

    n_total = len(loader)
    init_rng = np.random.default_rng(init_ss)
    x0 = x_base + init_rng.uniform(-jitter, jitter, size=N)
    model.set_data(batch_var, next_batch())

    traj = run_stochastic_lbfgs(
        value_grad_fn,
        lambda: model.set_data(batch_var, next_batch()),
        x0,
        num_iters,
        cfg,
        callbacks,
    )
    if not traj.iterates:
        raise RuntimeError(
            "Streaming L-BFGS produced no accepted steps; try more iterations, a "
            "smaller jitter, or a larger batch size."
        )

    # Polyak-Ruppert tail averaging of the iterate positions. Each stochastic L-BFGS step is
    # a full quasi-Newton step plus line search on ONE minibatch, so its accepted point is
    # essentially that minibatch's MAP: ~sqrt(N / b) full-data posterior-sd off the true MAP
    # (measured ratio 0.971 over 27 runs), while the same batch pins the posterior sd to
    # within 1-2%. That makes it a location error, not a covariance one, which is why no rule
    # that *picks* an iterate can fix it. g is zeroed because the sampler centres the Gaussian
    # at mu = x - H_inv @ g; keeping the last iterate's gradient costs 2.0-6.6x. Curvature
    # stays from the last iterate: averaged (s, z) pairs are not a valid L-BFGS memory.
    # Ruppert 1988 (Cornell ORIE TR-781); Polyak & Juditsky 1992, SIAM J. Control Optim.
    # 30(4):838-855; Jain et al., JMLR 18(223), 2018 (tail averaging the last c*n iterates);
    # Mandt, Hoffman & Blei, JMLR 18(134), 2017; Dhaka et al., NeurIPS 2020 and Welandawe
    # et al., JMLR 25(219), 2024 (iterate averaging in VI); Byrd, Hansen, Nocedal & Singer,
    # SIAM J. Optim. 26(2):1008-1031, 2016 (averaged iterates inside stochastic L-BFGS).
    m = max(1, round(_TAIL_AVERAGE_FRAC * len(traj.iterates)))
    x_avg = np.mean([it["x"] for it in traj.iterates[-m:]], axis=0)
    last = traj.iterates[-1]
    best = {
        "x": x_avg,
        "g": np.zeros_like(x_avg),
        "alpha": last["alpha"],
        "s_win": last["s_win"],
        "z_win": last["z_win"],
    }

    resample = importance_sampling is not None
    if num_proposal_draws is None:
        # fit_pathfinder resamples num_draws out of num_paths=4 x num_draws_per_path=1000;
        # a single streaming path has to supply that 4x pool itself. Unresampled the extra
        # draws are only dropped, and the pool's logP pass is most of the wall clock.
        n_prop = 4 * num_draws if resample else num_draws
    else:
        n_prop = max(int(num_proposal_draws), num_draws)  # samples[:num_draws] must not be short
    u_final = np.random.default_rng(final_ss).standard_normal((n_prop, N))
    phi, logQ, _logP_subset, _ = sample_logp(
        best["x"], best["g"], best["alpha"], best["s_win"], best["z_win"], u_final
    )
    samples = np.asarray(phi)
    logQ = np.asarray(logQ)
    # sample_logp's logP is the installed batch's pseudo-density; the weights need the exact one.
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
        pareto_k=pareto_k,
        violation_rate=traj.violation_rate,
        n_ls_failures=traj.n_ls_failures,
    )
