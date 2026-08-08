"""Streaming (minibatch) Pathfinder for PyMC.

Runs Pathfinder where the log-density gradients come from minibatches yielded by
a sized re-iterable loader, rather than the full dataset. The model carries the
data in a ``pm.Data`` placeholder scaled with ``total_size=loader.total_size``, so
``model.logp()`` already returns the correctly rescaled full-data log-density for
whatever batch is currently set.

The optimizer core is :func:`run_stochastic_lbfgs`. Its iterate positions are tail-averaged
into a single point, which is then drawn at full width and importance-resampled against the
exact full-data ``logP``. The compiled log-density, Gaussian sampler and PSIS are reused
unchanged from the deterministic Pathfinder (``bfgs_sample``, ``importance_sampling``).

The draws it returns are a proposal, not a posterior; the measured operating range, in both
dataset size and dimension, is on :func:`fit_streaming_pathfinder`.

What streaming buys is memory: peak is O(``num_proposal_draws`` x ``full_pass`` block rows)
rather than O(N). On 200k rows with 100 proposal draws, peak traced allocation was 40 MB with
2048-row blocks and 231 MB with one block; the block size is the caller's to set.
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

# Fraction of the trajectory's iterate positions that the tail average covers. Not a keyword.
# Swept over 0.50/0.60/0.75/0.90 at the Notes k=8, N=1e5 configuration, scoring worst-coordinate
# position error |x_avg - full-data MAP| in reference-sd: medians over 8 seeds 0.86/0.80/0.68/0.80,
# against 5.06 at 1.00. The interior is flat; the only firm result is that 1.00 is broken.
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
            f"full-data logp pass visited {seen} rows but total_size={n_total}; the "
            "pass must yield every row exactly once for an exact logP. A shuffled "
            "DataLoader epoch drops its trailing partial batch, so pass full_pass="
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
        ``batch_var`` and whose likelihood passes ``total_size=loader.total_size``.
    loader : iterable
        Yields minibatches (arrays whose leading axis is the batch rows) and
        exposes ``loader.total_size == N`` (the dataset row count), as
        :class:`pymc_extras.variational.DataLoader` does.
    batch_var : str
        Name of the ``pm.Data`` placeholder to stream into. The value it held on entry is
        restored before returning, on the exception path too.
    num_iters : int
        Number of stochastic L-BFGS steps.
    num_draws : int
        Draws returned from the selected Gaussian.
    num_proposal_draws : int, optional
        Size of the proposal pool importance sampling resamples down to ``num_draws``.
        Defaults to ``4 * num_draws`` when resampling, matching ``fit_pathfinder``'s
        ``num_paths=4`` x ``num_draws_per_path=1000`` pool, and to ``num_draws`` when not.
        The pool's exact full-data ``logP`` is an O(``num_proposal_draws`` x N) pass, so on
        any large dataset this setting dominates the cost of the fit.
    full_pass : iterable, optional
        Row blocks visiting every row exactly once, for the final exact full-data ``logP``;
        consumed once. Defaults to iterating ``loader``: an unshuffled DataLoader passes its
        source through verbatim and is a valid exact pass, while a shuffled epoch drops its
        trailing partial batch and raises here rather than return a subset-rescaled ``logP``
        — pass the underlying source in that case. Block sizes need
        not match the loader's. Blocks are installed in ``batch_var`` unchanged, so if
        ``loader`` transforms its batches, ``full_pass`` must apply the identical transform:
        untransformed rows give a wrong ``logP`` and wrong draws, with no error to say so.
    jitter : float
        Uniform jitter added to the prior initial point.
    jacobian_correction : bool
        Include the change-of-variables Jacobian so logp is the unconstrained
        joint density (matches ``fit_pathfinder``).
    importance_sampling : {"psis", "psir", "identity", None}
        Post-hoc reweighting of the returned draws; ``None`` returns the proposal draws
        unweighted. ``"identity"`` weights by the raw log ratio: a weighting method that
        still resamples from the 4x pool, not an off switch.
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
    Measured operating range. Bayesian logistic regression, Normal(0, 2) prior, batch 2048 from a
    *shuffled* loader (part of the configuration, not an aside), ``num_iters=200``, ``maxcor=6``,
    ``jitter=2.0``, ``num_draws=1000``, ``num_proposal_draws=4000``, PSIS, scored against an exact
    full-data Laplace reference. At k=8, N=1e5 over 12 seeds ``pareto_k`` ran 0.60-0.80 (median
    0.70, over 0.7 on half of them) and the worst coordinate of the *resampled posterior mean* was
    0.23-0.58 reference-sd from the reference mean. At N=4e5 over 6 seeds that was 0.76-1.29 and
    0.53-2.24. At k=100, N=1e5 over 4 seeds it was 2.0-3.8 and 13-64, nothing raises, and two of
    the four came back with a nonzero ``n_ls_failures``; ``violation_rate`` read 0.0 throughout.

    These draws are a *proposal*, not a posterior. Read ``pareto_k`` on every fit, and do not run
    this above a few tens of dimensions.

    References
    ----------
    Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by
    averaging. SIAM Journal on Control and Optimization, 30(4), 838-855.

    Jain, P., Kakade, S. M., Kidambi, R., Netrapalli, P., & Sidford, A. (2018). Parallelizing
    stochastic gradient descent for least squares regression. COLT, 545-604.
    """
    model = pm.modelcontext(model)
    if batch_var not in model.named_vars:
        raise KeyError(
            f"batch_var {batch_var!r} is not a variable in the model; add a "
            f"pm.Data({batch_var!r}, ...) placeholder that the data stream feeds."
        )
    # Restored in the finally below; get_value() copies.
    saved_batch = model[batch_var].get_value()
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

    n_total = getattr(loader, "total_size", None)
    if n_total is None:
        raise TypeError(
            "loader must expose total_size (the dataset row count N), as "
            "pymc_extras DataLoader does; wrap a plain iterable in "
            "DataLoader(source, batch_size=..., total_size=N)."
        )
    n_total = int(n_total)
    init_rng = np.random.default_rng(init_ss)
    x0 = x_base + init_rng.uniform(-jitter, jitter, size=N)
    try:
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

        # Polyak-Ruppert tail averaging of the iterate positions. Each step's accepted point sits
        # near its own minibatch's MAP -- worst-coordinate relative error, median over 200 random
        # batches, ran 9.1% at b=512, 4.3% at 2048 and 2.2% at 8192, with the b=512 tail at 36% --
        # and that error is in the location, not the covariance, so no rule that *picks* an
        # iterate removes it. g is zeroed because the sampler centres at mu = x - H_inv @ g, and
        # keeping the last iterate's g moved that centre further from the full-data MAP on all of
        # 12 seeds; curvature stays the last iterate's -- averaged (s, z) is not a valid memory.
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
        # fit_pathfinder resamples out of a num_paths=4 x num_draws_per_path pool; one streaming
        # path has to supply that 4x itself. max() keeps samples[:num_draws] from coming back short.
        default_prop = (4 if resample else 1) * num_draws
        n_prop = (
            default_prop if num_proposal_draws is None else max(int(num_proposal_draws), num_draws)
        )
        u_final = np.random.default_rng(final_ss).standard_normal((n_prop, N))
        phi, logQ, _logP_subset, _ = sample_logp(
            best["x"], best["g"], best["alpha"], best["s_win"], best["z_win"], u_final
        )
        samples = np.asarray(phi)
        logQ = np.asarray(logQ)
        # sample_logp's logP is the installed batch's pseudo-density; the weights need the
        # exact one.
        logP = _full_data_logp(
            samples,
            full_pass if full_pass is not None else loader,
            model,
            batch_var,
            prior_logp_fn,
            obs_logp_fn,
            n_total,
        )
    finally:
        model.set_data(batch_var, saved_batch)

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
