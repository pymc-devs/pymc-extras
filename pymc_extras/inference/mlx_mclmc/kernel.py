import logging
import math

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from numpy.typing import ArrayLike

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MCLMC sampler requires the `mlx` package, which is only available on Apple "
        "Silicon. Install it with `pip install mlx`."
    ) from exc

_log = logging.getLogger(__name__)

# The dynamics, the integrator schedules, and the adaptation are ported from the blackjax
# isokinetic sampler, so its source is the reference for anything ambiguous here.
_MCLACHLAN_B1 = 0.1931833275037836

INTEGRATOR_COEFFICIENTS = {
    "mclachlan": [_MCLACHLAN_B1, 0.5, 1.0 - 2.0 * _MCLACHLAN_B1, 0.5, _MCLACHLAN_B1],
    "velocity_verlet": [0.5, 1.0, 0.5],
}

# Below this norm a vector is left unnormalized rather than divided by its own (near-zero)
# norm, so a vanishing gradient at the mode gives a finite result instead of 0 / 0 -> nan.
# Below this delta the textbook kinetic-energy form cancels catastrophically, and above it MLX's
# expm1 is the weaker primitive; each form covers where the other fails. The crossover is sharp
# between 0.15 and 0.18 in float32, measured against a float64 reference.
_CANCELLATION_FREE_BELOW = 0.15
_LOG2 = math.log(2.0)

# A retained direction whose gain reaches -1 annihilates that direction entirely, so a fit that
# comes anywhere near it is rejected instead.
_MIN_LOW_RANK_GAIN = 1e-3

# Floor on eigenvalues fed to a fractional matrix power, where a zero would produce inf.
_EIGENVALUE_FLOOR = 1e-30

_NORM_FLOOR = 1e-13
_SAFE_DIVISOR = 1e-30

# MLX evaluates lazily, so the trajectory loop only forces the graph this often.
_EVAL_EVERY = 64


class ChainState(NamedTuple):
    """Position of each chain and everything the next transition needs to continue from it."""

    position: mx.array
    momentum: mx.array
    logdensity: mx.array
    grad: mx.array


class AdaptationSettings(NamedTuple):
    r"""
    Everything :func:`warmup` adapts, and how.

    Attributes
    ----------
    mass_matrix : str
        ``"variance"`` for blackjax's per-coordinate posterior variance, ``"gradient"`` for
        nuts-rs's :math:`\sqrt{\mathrm{Var}[x] / \mathrm{Var}[\nabla \log p]}`, or
        ``"low_rank"`` to fit a low-rank correction on top of the latter. Only ``"low_rank"`` can
        precondition a posterior whose ridges are not axis-aligned.
    settings.diagonal_preconditioning : bool
        Whether to install the estimated metric at all.
    early_switch_freq, switch_freq : int
        Moment-estimator window lengths, before and after ``early_end``.
    early_end : int
        Phase-2 step at which the short early windows give way to the growing ones. 0 disables the
        early phase.
    window_growth : float
        Factor by which each main-phase window exceeds the last.
    low_rank_window : int
        Trailing phase-2 draws retained for a ``"low_rank"`` fit.
    low_rank_refits : int
        Refits within phase 2 before the final one. The first is always the gradient diagonal,
        which seeds the later low-rank fits with draws taken under something better than identity.
    settings.desired_energy_var : float
        Target energy variance per dimension for the step-size controller.
    settings.trust_in_estimate : float
        Width of the controller's Gaussian weighting. Larger values give more weight to single-step
        estimates far from the target.
    settings.num_effective_samples : float
        Sets the controller's exponential decay rate.
    settings.frac_tune1, settings.frac_tune2, settings.frac_tune3 : float
        Fractions of the step budget given to each adaptation phase.
    settings.l_factor : float
        Multiplier on the autocorrelation-derived ``L`` in phase 3.
    settings.optimize_steps : int
        Maximum Adam steps taken toward the mode before adaptation. Pass 0 for a log-density
        unbounded above, such as a centered hierarchical model.
    settings.optimize_learning_rate : float
        Adam learning rate for that ascent.
    """

    mass_matrix: str = "gradient"
    diagonal_preconditioning: bool = True
    early_switch_freq: int = 10
    switch_freq: int = 80
    early_end: int = 0
    window_growth: float = 1.5
    low_rank_window: int = 400
    low_rank_refits: int = 2
    desired_energy_var: float = 5e-4
    trust_in_estimate: float = 1.5
    num_effective_samples: float = 150
    frac_tune1: float = 0.1
    frac_tune2: float = 0.1
    frac_tune3: float = 0.1
    l_factor: float = 0.4
    optimize_steps: int = 200
    optimize_learning_rate: float = 0.05


class LowRankCorrection(NamedTuple):
    r"""Orthonormal directions and their :math:`\sqrt{\Lambda} - 1` gains."""

    vectors: mx.array
    scales: mx.array


class Metric(NamedTuple):
    r"""
    A diagonal inverse mass matrix, optionally corrected on a low-rank subspace.

    Without a ``correction`` this is :math:`M^{-1} = \mathrm{diag}(\sigma^2)`. With them it is
    blackjax's ``LowRankInverseMassMatrix``,

    .. math:: M^{-1} = \mathrm{diag}(\sigma)(I + U(\Lambda - I)U^\top)\mathrm{diag}(\sigma)

    which the dynamics never form: only the two O(dk) maps :func:`_whiten_gradient` and
    :func:`_unwhiten_momentum` are ever applied.

    Attributes
    ----------
    scale : mx.array
        Per-coordinate :math:`\sigma`, of shape ``(dim,)``.
    correction : LowRankCorrection or None
        The low-rank part, absent for a purely diagonal metric.
    """

    scale: mx.array
    correction: LowRankCorrection | None = None


def _whiten_gradient(metric: Metric, grad: mx.array) -> mx.array:
    """Map a gradient into the whitened frame: blackjax's ``adjoint_L``."""
    scaled = grad * metric.scale
    if metric.correction is None:
        return scaled

    vectors, scales = metric.correction

    return scaled + (scaled @ vectors * scales) @ vectors.T


def _unwhiten_momentum(metric: Metric, momentum: mx.array) -> mx.array:
    """Map a whitened momentum to a position velocity: blackjax's ``forward_L``."""
    if metric.correction is None:
        return momentum * metric.scale

    vectors, scales = metric.correction
    corrected = momentum + (momentum @ vectors * scales) @ vectors.T

    return corrected * metric.scale


class Dynamics(NamedTuple):
    """Everything a transition needs that does not change from one step to the next."""

    logp_and_grad: Callable
    step_size: float | mx.array
    coefficients: list[float]
    metric: Metric
    inverse_L: float
    dim: int


class WindowedMoments(NamedTuple):
    """Running count, mean and mean-square of the position and of the log-density gradient."""

    count: mx.array
    position_mean: mx.array
    position_mean_sq: mx.array
    grad_mean: mx.array
    grad_mean_sq: mx.array


class AdaptationState(NamedTuple):
    """Carry threaded through :func:`warmup`, held as MLX arrays so the loop body compiles."""

    position: mx.array
    momentum: mx.array
    logdensity: mx.array
    grad: mx.array
    step_size: mx.array
    step_size_max: mx.array
    time: mx.array
    x_average: mx.array
    foreground: WindowedMoments
    background: WindowedMoments


class SamplerOutput(NamedTuple):
    """Draws and per-step diagnostics returned by :func:`sample`."""

    samples: mx.array
    energy_errors: mx.array
    diverging: mx.array


class TunedParameters(NamedTuple):
    """Sampler parameters produced by :func:`warmup`."""

    position: mx.array
    L: float
    step_size: float
    metric: Metric
    num_tuning_steps: int


def _check_dim(dim: int) -> None:
    if dim < 2:
        raise ValueError(
            f"MCLMC needs at least 2 dimensions, got {dim}. The isokinetic momentum update "
            "divides by (dim - 1), which is undefined for a single parameter."
        )


def _as_metric(inverse_mass_matrix) -> Metric:
    """Accept a diagonal inverse mass matrix or an already-built :class:`Metric`."""
    if isinstance(inverse_mass_matrix, Metric):
        return inverse_mass_matrix

    return Metric(scale=mx.sqrt(mx.array(inverse_mass_matrix, dtype=mx.float32)))


def _empty_moments(dim: int) -> WindowedMoments:
    zeros = mx.zeros((1, dim))
    return WindowedMoments(mx.zeros(()), zeros, zeros, zeros, zeros)


def _accumulate(moments: WindowedMoments, position, grad, weight) -> WindowedMoments:
    """Fold one draw into a running mean and mean-square, skipping zero-weight steps."""
    count = moments.count + weight
    safe = mx.maximum(count, 1e-30)

    def blend(previous, sample):
        return mx.where(count > 0, (moments.count * previous + weight * sample) / safe, previous)

    return WindowedMoments(
        count=count,
        position_mean=blend(moments.position_mean, position),
        position_mean_sq=blend(moments.position_mean_sq, position * position),
        grad_mean=blend(moments.grad_mean, grad),
        grad_mean_sq=blend(moments.grad_mean_sq, grad * grad),
    )


def _spd_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""
    Geometric mean of two symmetric positive-definite matrices.

    :math:`A \# B = B^{-1/2}(B^{1/2} A B^{1/2})^{1/2}B^{-1/2}`, the Riemannian midpoint. This is
    how nuts-rs reconciles the draw covariance with the inverse gradient covariance when the two
    disagree, which is exactly what happens on a rotated posterior.
    """

    def symmetric_power(matrix, power):
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * np.clip(values, _EIGENVALUE_FLOOR, None) ** power) @ vectors.T

    right_sqrt = symmetric_power(right, 0.5)
    right_inv_sqrt = symmetric_power(right, -0.5)
    middle = symmetric_power(right_sqrt @ left @ right_sqrt, 0.5)

    return right_inv_sqrt @ middle @ right_inv_sqrt


def _low_rank_metric(
    draws: np.ndarray,
    grads: np.ndarray,
    *,
    gamma: float = 1e-5,
    eigval_cutoff: float = 2.0,
    max_eigenvalue: float = 1e6,
    eps: float = 1e-12,
) -> Metric | None:
    r"""
    Fit a diagonal-plus-low-rank inverse mass matrix to a window of draws and gradients.

    A port of nuts-rs's ``LowRankMassMatrixStrategy``. The diagonal is the fourth root of
    :math:`\mathrm{Var}[x] / \mathrm{Var}[\nabla \log p]`, applied as a scale to the draws and
    its reciprocal to the gradients. The correction is then fitted only inside the subspace spanned
    by the leading directions of the rescaled draws *and* gradients together, which is what lets it
    find a ridge neither set resolves alone.

    Parameters
    ----------
    draws, grads : np.ndarray
        Window of positions and log-density gradients, of shape ``(dim, n_draws)``.
    gamma : float
        Shrinkage toward the identity for the projected covariances. Default is 1e-5.
    eigval_cutoff : float
        Keep only directions whose eigenvalue is above this or below its reciprocal; the rest are
        left to the diagonal. Default is 2.0.
    max_eigenvalue : float
        Largest eigenvalue of the geometric mean the fit will accept. Those eigenvalues measure how
        far the draws and the gradients disagree about the geometry; past this the window is
        uninformative rather than the posterior extreme. nuts-rs needs no such test because it fits
        from a tuned chain. Default is 1e6.

    Returns
    -------
    Metric or None
        None when the window does not support a trustworthy fit, in which case the caller should
        keep the diagonal it already has.

    Returns
    -------
    scale : np.ndarray
        Diagonal :math:`\sigma`, of shape ``(dim,)``.
    vectors : np.ndarray
        Retained directions, of shape ``(dim, k)``.
    scales : np.ndarray
        The :math:`\sqrt{\Lambda} - 1` gains, of shape ``(k,)``.
    """
    dim, n_draws = draws.shape

    draw_mean, grad_mean = draws.mean(axis=1), grads.mean(axis=1)
    draw_var = np.clip(draws.var(axis=1), eps, None)
    grad_var = np.clip(grads.var(axis=1), eps, None)
    scale = (draw_var / grad_var) ** 0.25

    # nuts-rs centres the draws on the mean shifted along the gradient rather than on the sample
    # mean, a Newton-like correction toward the mode.
    centre = draw_mean + scale**2 * grad_mean
    rescaled_draws = (draws - centre[:, None]) / scale[:, None]
    rescaled_grads = grads * scale[:, None]
    rescaled_draws -= rescaled_draws.mean(axis=1, keepdims=True)
    rescaled_grads -= rescaled_grads.mean(axis=1, keepdims=True)

    draw_basis = np.linalg.svd(rescaled_draws, full_matrices=False)[0]
    grad_basis = np.linalg.svd(rescaled_grads, full_matrices=False)[0]
    subspace, _ = np.linalg.qr(np.concatenate([draw_basis, grad_basis], axis=1))

    projected_draws = subspace.T @ rescaled_draws
    projected_grads = subspace.T @ rescaled_grads
    identity = np.eye(subspace.shape[1])
    cov_draws = projected_draws @ projected_draws.T / gamma + identity
    cov_grads = projected_grads @ projected_grads.T / gamma + identity

    values, vectors = np.linalg.eigh(_spd_mean(cov_draws, cov_grads))
    if not np.isfinite(values).all():
        return None
    if values.max() > max_eigenvalue or values.min() < 1.0 / max_eigenvalue:
        return None

    keep = (values > eigval_cutoff) | (values < 1.0 / eigval_cutoff)
    gains = np.sqrt(values[keep]) - 1.0
    if not (np.all(scale > 0) and np.all(gains > -1.0 + _MIN_LOW_RANK_GAIN)):
        return None

    return Metric(
        scale=mx.array(scale.astype(np.float32)),
        correction=LowRankCorrection(
            vectors=mx.array((subspace @ vectors[:, keep]).astype(np.float32)),
            scales=mx.array(gains.astype(np.float32)),
        ),
    )


def _reset_step_size_controller(state: "AdaptationState") -> "AdaptationState":
    """
    Re-seed the step-size controller, which every reference does when the metric changes.

    The step tuned under the previous metric is baked into ``x_average``, whose decay is slow
    enough that the controller cannot re-tune from it under a new one.
    """
    return state._replace(
        step_size_max=mx.array([float("inf")], dtype=mx.float32),
        time=mx.array([0.0], dtype=mx.float32),
        x_average=mx.array([0.0], dtype=mx.float32),
    )


def _fit_metric(
    moments: WindowedMoments,
    retained: list,
    dim: int,
    mass_matrix: str,
    allow_low_rank: bool,
) -> tuple[Metric, np.ndarray]:
    """
    Build the metric for the next stretch of adaptation, and the diagonal it implies.

    The low-rank fit is only as good as the draws behind it, and draws taken under an identity
    metric on an ill-conditioned target do not span the posterior. So the first fit is always the
    gradient diagonal, and the low-rank correction goes on top of it once there are draws taken
    under something better than identity.
    """
    if mass_matrix == "low_rank" and allow_low_rank and len(retained) >= 3:
        mx.eval(retained)
        draws = np.stack([np.asarray(p).reshape(dim) for p, _ in retained], axis=1)
        grads = np.stack([np.asarray(g).reshape(dim) for _, g in retained], axis=1)
        fitted = _low_rank_metric(draws, grads)
        if fitted is not None:
            return fitted, (np.asarray(fitted.scale) ** 2).astype(np.float32)

        _log.warning(
            "The low-rank mass matrix fit was rejected as unreliable; keeping the diagonal."
        )

    diagonal = _diagonal_from_moments(
        moments, dim, "gradient" if mass_matrix == "low_rank" else mass_matrix
    )

    return _as_metric(diagonal), diagonal


def _window_switch_steps(
    num_steps: int, early_end: int, early_switch_freq: int, switch_freq: int, window_growth: float
) -> set[int]:
    """
    Step counts at which the background estimator should replace the foreground.

    Mirrors nuts-rs's schedule: short windows of ``early_switch_freq`` until ``early_end``, then
    windows starting at ``switch_freq`` and growing by ``window_growth``. A window that could not
    finish inside ``num_steps`` is not started, which is the reference's is-late gate.
    """
    switches, size, step = set(), switch_freq, 0
    while True:
        early = step < early_end
        target = early_switch_freq if early else size
        if step + target > num_steps:
            return switches
        step += target
        switches.add(step)
        if not early:
            size = max(size + 1, round(size * window_growth))


_MASS_MATRIX_METHODS = ("variance", "gradient", "low_rank")


def _diagonal_from_moments(
    moments: WindowedMoments, dim: int, method: str, eps: float = 1e-12
) -> np.ndarray:
    """
    Read the diagonal inverse mass matrix off a window's moments.

    ``"variance"`` is blackjax's per-coordinate posterior variance. ``"gradient"`` is nuts-rs's
    ``sqrt(var[x] / var[grad])``, which for an axis-aligned Gaussian equals the same variance but
    degrades more gracefully when the posterior is rotated, since the gradient carries the
    curvature the positions alone do not.
    """

    def variance(mean, mean_sq):
        mean = np.asarray(mean).reshape(dim)
        return np.clip(np.asarray(mean_sq).reshape(dim) - mean**2, eps, None)

    position_variance = variance(moments.position_mean, moments.position_mean_sq)
    if method == "variance":
        return position_variance.astype(np.float32)

    grad_variance = variance(moments.grad_mean, moments.grad_mean_sq)

    return np.sqrt(position_variance / grad_variance).astype(np.float32)


def _batched_value_and_grad(logdensity_fn: Callable) -> Callable:
    """
    Build the batched value-and-gradient callable the dynamics use.

    A ``logdensity_fn`` carrying its own ``value_and_grad`` supplies the gradient itself, which is
    how :class:`~pymc_extras.inference.mlx_mclmc.logp.MLXLogp` hands over PyTensor's symbolic
    gradient. Anything else is differentiated by MLX, whose reverse rules do not cover every op.
    """
    supplied = getattr(logdensity_fn, "value_and_grad", None)

    return mx.vmap(supplied if callable(supplied) else mx.value_and_grad(logdensity_fn))


def _check_initial_state(logdensity: mx.array, grad: mx.array) -> None:
    """
    Raise ValueError unless the log-density and its gradient are finite at the starting point.

    A finite log-density with a non-finite gradient is the diagnostic case: the point sits on a
    boundary or a removable singularity of the model, where the value is defined but the
    derivative is not.
    """
    if not bool(mx.all(mx.isfinite(logdensity))):
        raise ValueError(
            "The log-density is not finite at the initial point, so MCLMC cannot start. Pass an "
            "`initial_point` inside the support of the model."
        )
    if not bool(mx.all(mx.isfinite(grad))):
        raise ValueError(
            "The log-density is finite at the initial point but its gradient is not, so MCLMC "
            "cannot start. The point is most likely on a boundary or a removable singularity of "
            "the model; pass an `initial_point` away from it."
        )


def _normalize(vectors: mx.array) -> mx.array:
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)

    return mx.where(norms > _NORM_FLOOR, vectors / mx.maximum(norms, _SAFE_DIVISOR), vectors)


def _unit_vectors(shape: tuple[int, ...], key: mx.array) -> mx.array:
    return _normalize(mx.random.normal(shape=shape, key=key))


def _momentum_update(
    momentum: mx.array,
    grad: mx.array,
    effective_step: float | mx.array,
    metric: Metric,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Apply one ESH isokinetic momentum sub-step.

    Parameters
    ----------
    effective_step : float or mx.array
        Step size times this sub-step's integrator coefficient.

    Returns
    -------
    momentum : mx.array
        Updated unit momentum.
    velocity : mx.array
        Preconditioned velocity for the following position sub-step.
    kinetic_energy_change : mx.array
        Per-chain change in kinetic energy.
    """
    scaled_grad = _whiten_gradient(metric, grad)
    grad_norm = mx.linalg.norm(scaled_grad, axis=-1, keepdims=True)
    grad_direction = _normalize(scaled_grad)
    projection = mx.sum(momentum * grad_direction, axis=-1, keepdims=True)

    delta = effective_step * grad_norm / (dim - 1)
    zeta = mx.exp(-delta)

    # Written without an exp(delta) factor, so a large gradient norm cannot overflow.
    momentum = _normalize(
        grad_direction * (1 - zeta) * (1 + zeta + projection * (1 - zeta)) + 2 * zeta * momentum
    )
    # blackjax writes this as delta - log2 + log(1 + p + (1 - p) * zeta^2), which subtracts two
    # near-equal 0.693s at small delta and then scales the remainder by dim - 1 -- up to 13% of
    # the value in float32, right where the step-size controller is most sensitive. The algebraic
    # rewrite below removes that cancellation but leans on expm1, which MLX computes less
    # accurately for larger arguments, so the two are blended at the measured crossover.
    kinetic_energy_change = (
        mx.where(
            delta < _CANCELLATION_FREE_BELOW,
            delta + mx.log1p((1 - projection) * mx.expm1(-2 * delta) / 2),
            delta - _LOG2 + mx.log(1 + projection + (1 - projection) * zeta**2),
        )
        * (dim - 1)
    ).squeeze(-1)

    return momentum, _unwhiten_momentum(metric, momentum), kinetic_energy_change


def _integrate(
    position: mx.array, momentum: mx.array, grad: mx.array, dynamics: Dynamics
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """
    Run one deterministic integrator step, a palindromic momentum/position schedule.

    Returns the new position, momentum, log-density, gradient, and the accumulated change in
    kinetic energy.
    """
    kinetic_energy_change = mx.zeros((position.shape[0],))
    velocity = None
    logdensity = None

    for i, coefficient in enumerate(dynamics.coefficients[:-1]):
        if i % 2 == 0:
            momentum, velocity, energy_change = _momentum_update(
                momentum=momentum,
                grad=grad,
                effective_step=dynamics.step_size * coefficient,
                metric=dynamics.metric,
                dim=dynamics.dim,
            )
            kinetic_energy_change = kinetic_energy_change + energy_change
        else:
            position = position + dynamics.step_size * coefficient * velocity
            logdensity, grad = dynamics.logp_and_grad(position)

    momentum, velocity, energy_change = _momentum_update(
        momentum=momentum,
        grad=grad,
        effective_step=dynamics.step_size * dynamics.coefficients[-1],
        metric=dynamics.metric,
        dim=dynamics.dim,
    )
    kinetic_energy_change = kinetic_energy_change + energy_change

    return position, momentum, logdensity, grad, kinetic_energy_change


def _partial_refresh(
    momentum: mx.array, key: mx.array, step_size: float | mx.array, inverse_L: float, dim: int
) -> mx.array:
    """
    Decohere the momentum by one Maruyama half-step: add Gaussian noise, then renormalize.

    The noise scale is :math:`\\sqrt{(e^{2\\epsilon/L} - 1) / d}`, so ``inverse_L = 0`` leaves the
    momentum untouched.
    """
    noise_scale = mx.sqrt(mx.expm1(2.0 * step_size * inverse_L) / dim)

    return _normalize(momentum + noise_scale * mx.random.normal(shape=momentum.shape, key=key))


def _revert_nonfinite(
    proposed: ChainState, previous: ChainState, energy_error: mx.array, key: mx.array
) -> tuple[ChainState, mx.array, mx.array]:
    """
    Revert any chain whose step came out non-finite, and resample its momentum.

    Matches blackjax's kernel-level ``handle_nans``. Reductions are per-chain, so one chain
    diverging does not disturb the others.

    Returns
    -------
    state : ChainState
        ``proposed`` for the chains that stepped cleanly, ``previous`` with a fresh unit momentum
        for the rest.
    energy_error : mx.array
        Zeroed for the reverted chains.
    is_finite : mx.array
        Per-chain flag, False where the step was reverted.
    """
    is_finite = (
        mx.all(mx.isfinite(proposed.position), axis=-1)
        & mx.all(mx.isfinite(proposed.momentum), axis=-1)
        & mx.isfinite(proposed.logdensity)
        & mx.isfinite(energy_error)
    )
    per_chain = is_finite[:, None]
    resampled = _unit_vectors(shape=proposed.momentum.shape, key=key)

    state = ChainState(
        position=mx.where(per_chain, proposed.position, previous.position),
        momentum=mx.where(per_chain, proposed.momentum, resampled),
        logdensity=mx.where(is_finite, proposed.logdensity, previous.logdensity),
        grad=mx.where(per_chain, proposed.grad, previous.grad),
    )
    energy_error = mx.where(is_finite, energy_error, mx.zeros_like(energy_error))

    return state, energy_error, is_finite


def _transition(
    state: ChainState, keys: tuple[mx.array, mx.array], dynamics: Dynamics
) -> tuple[ChainState, mx.array]:
    """
    Take one MCLMC transition: half refresh, deterministic step, half refresh.

    Returns
    -------
    state : ChainState
        The chains after the transition.
    energy_error : mx.array
        Per-chain energy error of the step.
    """
    refresh_key, decohere_key = keys
    half_step = 0.5 * dynamics.step_size

    momentum = _partial_refresh(
        momentum=state.momentum,
        key=refresh_key,
        step_size=half_step,
        inverse_L=dynamics.inverse_L,
        dim=dynamics.dim,
    )
    position, momentum, logdensity, grad, kinetic_energy_change = _integrate(
        position=state.position, momentum=momentum, grad=state.grad, dynamics=dynamics
    )
    momentum = _partial_refresh(
        momentum=momentum,
        key=decohere_key,
        step_size=half_step,
        inverse_L=dynamics.inverse_L,
        dim=dynamics.dim,
    )
    energy_error = kinetic_energy_change - logdensity + state.logdensity

    return ChainState(position, momentum, logdensity, grad), energy_error


def sample(
    logdensity_fn: Callable[[mx.array], mx.array],
    initial_positions: ArrayLike,
    *,
    L: float,
    step_size: float,
    n_steps: int,
    integrator: str = "mclachlan",
    inverse_mass_matrix: ArrayLike | Metric = 1.0,
    discard: int = 0,
    seed: int = 0,
    compile_step: bool = True,
) -> SamplerOutput:
    """
    Run unadjusted MCLMC from fixed parameters.

    The sampler evolves an isokinetic Hamiltonian: the momentum is held on the unit sphere and
    partially refreshed each step, so trajectories decorrelate without a Metropolis accept step.
    Chains are the leading array axis and evolve independently.

    Parameters
    ----------
    logdensity_fn : callable
        Maps an MLX array of shape ``(dim,)`` to a scalar log-density. If it also has a
        ``value_and_grad`` method, that supplies the gradient instead of MLX autodiff.
    initial_positions : array
        Starting positions, of shape ``(chains, dim)``.
    L : float
        Momentum decoherence scale. Must be non-zero.
    step_size : float
        Integrator step size.
    n_steps : int
        Number of integrator steps to run, ``discard`` included.
    integrator : str
        Either ``"mclachlan"``, which takes 2 gradient evaluations per step, or
        ``"velocity_verlet"``, which takes 1. Default is ``"mclachlan"``.
    inverse_mass_matrix : float, array or Metric
        Diagonal inverse mass matrix, broadcastable to ``(dim,)``, or a :class:`Metric` carrying a
        low-rank correction as well. Default is 1.0.
    discard : int
        Number of leading steps to drop from the returned draws. Default is 0.
    seed : int
        Seed for the MLX random key. Default is 0.
    compile_step : bool
        Whether to fuse the step with ``mx.compile``. Pass False for very large graphs, whose
        fused kernel can exceed Metal's argument-buffer limit. Default is True.

    Returns
    -------
    SamplerOutput
        The ``samples`` of shape ``(n_steps - discard, chains, dim)``, the per-step
        ``energy_errors`` of shape ``(n_steps, chains)`` that the step-size adaptation steers by,
        and a ``diverging`` flag of the same shape marking the steps that were reverted.

    Raises
    ------
    ValueError
        If ``discard`` leaves no draws, or if ``L`` is zero.
    """
    if discard >= n_steps:
        raise ValueError(f"discard={discard} leaves no draws out of n_steps={n_steps}")
    if L == 0:
        raise ValueError("L must be non-zero; pass float('inf') to disable momentum decoherence")

    position = mx.array(initial_positions, dtype=mx.float32)
    n_chains, dim = position.shape
    _check_dim(dim)
    logp_and_grad = _batched_value_and_grad(logdensity_fn)
    dynamics = Dynamics(
        logp_and_grad=logp_and_grad,
        step_size=step_size,
        coefficients=INTEGRATOR_COEFFICIENTS[integrator],
        metric=_as_metric(inverse_mass_matrix),
        inverse_L=1.0 / L,
        dim=dim,
    )

    key = mx.random.key(seed)
    key, subkey = mx.random.split(key, num=2)
    logdensity, grad = logp_and_grad(position)
    _check_initial_state(logdensity, grad)
    state = ChainState(
        position=position,
        momentum=_unit_vectors(shape=(n_chains, dim), key=subkey),
        logdensity=logdensity,
        grad=grad,
    )

    def one_step(state, keys):
        refresh_key, decohere_key, resample_key = keys
        proposed, energy_error = _transition(
            state=state, keys=(refresh_key, decohere_key), dynamics=dynamics
        )

        return _revert_nonfinite(
            proposed=proposed, previous=state, energy_error=energy_error, key=resample_key
        )

    step = mx.compile(one_step) if compile_step else one_step

    # MLX has no scan primitive, so the trajectory is a Python loop over a single compiled step.
    # Lazy evaluation batches the graph between the periodic mx.eval, which hides the dispatch
    # cost of that loop.
    kept, energy_errors, diverging = [], [], []
    for t in range(n_steps):
        key, *step_keys = mx.random.split(key, num=4)
        state, energy_error, is_finite = step(state, keys=tuple(step_keys))

        energy_errors.append(energy_error)
        diverging.append(~is_finite)
        if t >= discard:
            kept.append(state.position)
        if (t + 1) % _EVAL_EVERY == 0:
            mx.eval(state)

    output = SamplerOutput(
        samples=mx.stack(kept, axis=0),
        energy_errors=mx.stack(energy_errors, axis=0),
        diverging=mx.stack(diverging, axis=0),
    )
    mx.eval(output)

    return output


def tune_step_size(
    logdensity_fn: Callable[[mx.array], mx.array],
    initial_positions: ArrayLike,
    *,
    L: float,
    integrator: str = "mclachlan",
    inverse_mass_matrix: ArrayLike | Metric = 1.0,
    desired_energy_var: float = 5e-4,
    rounds: int = 25,
    steps: int = 600,
    seed: int = 0,
) -> float:
    """
    Find a step size that hits a target energy variance per dimension.

    A cheap stand-in for :func:`warmup` when the metric and ``L`` are already known. Each round
    rescales the step by the minimal-norm integrator's :math:`\\mathrm{Var}[E] = O(\\epsilon^6)`
    law and stops once the measured variance is within a factor of two of the target.

    Parameters
    ----------
    desired_energy_var : float
        Target energy variance per dimension. Default is 5e-4.
    rounds : int
        Maximum number of rescaling rounds. Default is 25.
    steps : int
        Number of sampler steps used to measure the variance each round, half of them discarded.
        Default is 600.
    """
    dim = np.shape(initial_positions)[1]
    step_size = 0.5 * math.sqrt(dim)

    for round_index in range(rounds):
        energy_errors = sample(
            logdensity_fn,
            initial_positions,
            L=L,
            step_size=step_size,
            n_steps=steps,
            discard=steps // 2,
            integrator=integrator,
            inverse_mass_matrix=inverse_mass_matrix,
            seed=seed + round_index,
        ).energy_errors
        energy_var = (mx.mean(mx.var(energy_errors, axis=0)) / dim).item()
        _log.debug(
            "tune round %d: step_size=%.4f energy_var/dim=%.2e", round_index, step_size, energy_var
        )

        if not math.isfinite(energy_var) or energy_var <= 0.0:
            step_size *= 0.2
            continue
        if 0.5 * desired_energy_var < energy_var < 2.0 * desired_energy_var:
            break

        step_size *= min(3.0, max(0.2, (desired_energy_var / energy_var) ** (1.0 / 6.0)))

    return step_size


def _ess_per_dim(samples: np.ndarray) -> np.ndarray:
    """
    Estimate the single-chain effective sample size of each column of ``samples``.

    A NumPy port of ``blackjax.diagnostics.effective_sample_size`` specialized to one chain of
    shape ``(n_samples, dim)``: Geyer's initial positive sequence over an FFT autocovariance,
    followed by his initial monotone sequence.
    """
    n_samples, dim = samples.shape
    columns = np.arange(dim)

    centered = samples - samples.mean(axis=0, keepdims=True)
    padded_length = 1 << int(np.ceil(np.log2(2 * n_samples)))
    spectrum = np.fft.rfft(centered, n=padded_length, axis=0)
    autocov = (
        np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded_length, axis=0)[:n_samples].real
        / n_samples
    )

    # With one chain the between-chain term drops out, leaving weighted_var == autocov[0].
    var0 = autocov[0] * n_samples / (n_samples - 1.0)
    weighted_var = autocov[0]
    n_even = n_samples - n_samples % 2
    autocorr = np.concatenate([np.ones((1, dim)), 1.0 - (var0 - autocov[1:n_even]) / weighted_var])
    even, odd = autocorr[0::2], autocorr[1::2]

    # Geyer's initial positive sequence: keep the leading run of positive pair sums.
    positive = np.logical_and.accumulate((even + odd) > 0.0, axis=0)
    last = np.maximum(positive.sum(axis=0) - 1, 0)
    cutoff = np.minimum(last + 1, len(even) - 1)

    odd = np.where(positive, odd, 0.0)
    positive_even = positive.copy()
    positive_even[cutoff, columns] = even[cutoff, columns] > 0
    even = np.where(positive_even, even, 0.0)

    # Geyer's initial monotone sequence: clip the pair sums to their running minimum.
    pair_sum = even + odd
    running_min = np.minimum.accumulate(pair_sum, axis=0)
    clipped = pair_sum > np.concatenate([pair_sum[:1], running_min[:-1]])
    even = np.where(clipped, running_min / 2.0, even)
    odd = np.where(clipped, running_min / 2.0, odd)

    autocorr_time = -1.0 + 2.0 * np.sum(even + odd, axis=0) - even[cutoff, columns]
    autocorr_time = np.maximum(autocorr_time, 1.0 / np.log10(n_samples))

    return n_samples / autocorr_time


def _optimize_to_mode(
    logp_and_grad: Callable,
    position: mx.array,
    steps: int,
    learning_rate: float,
    tolerance: float = 1e-4,
    max_consecutive_skips: int = 5,
) -> mx.array:
    """
    Ascend the log-density with Adam, to concentrate the adapting chain near the mode.

    A step whose gradient or proposal is not finite is skipped whole -- the position and both Adam
    moments keep their previous values, as ``optax.apply_if_finite`` does -- because folding a nan
    gradient into the moments would poison every later step. The ascent gives up once that many
    steps in a row are skipped.

    Stops early once a block of steps improves the log-density by less than ``tolerance``. A
    log-density unbounded above, as a centered hierarchical model has, never triggers that stop,
    so ``steps`` remains a hard cap.

    Parameters
    ----------
    tolerance : float
        Smallest log-density improvement over a block of steps that counts as progress. Default
        is 1e-4.
    max_consecutive_skips : int
        How many consecutive non-finite steps to tolerate before giving up. The count is exact
        but only inspected when the lazy graph is forced, so the ascent can overshoot it slightly.
        Default is 5.
    """
    beta1, beta2 = 0.9, 0.999
    mean = mx.zeros_like(position)
    mean_sq = mx.zeros_like(position)
    previous_logdensity = -mx.inf

    # Both counters are MLX arrays so the loop body stays lazy between the periodic mx.eval.
    updates = mx.zeros(())
    consecutive_skips = mx.zeros(())

    for step in range(1, steps + 1):
        logdensity, grad = logp_and_grad(position)
        usable = mx.all(mx.isfinite(grad))

        # A skipped step must not advance the moments or their bias correction, so the optimizer
        # resumes from exactly where it was rather than from a nan-contaminated state.
        next_updates = updates + 1
        next_mean = beta1 * mean + (1 - beta1) * grad
        next_mean_sq = beta2 * mean_sq + (1 - beta2) * grad * grad
        mean_hat = next_mean / (1 - mx.power(beta1, next_updates))
        mean_sq_hat = next_mean_sq / (1 - mx.power(beta2, next_updates))
        proposed = position + learning_rate * mean_hat / (mx.sqrt(mean_sq_hat) + 1e-8)

        applied = usable & mx.all(mx.isfinite(proposed))
        position = mx.where(applied, proposed, position)
        mean = mx.where(applied, next_mean, mean)
        mean_sq = mx.where(applied, next_mean_sq, mean_sq)
        updates = mx.where(applied, next_updates, updates)
        consecutive_skips = mx.where(applied, mx.zeros(()), consecutive_skips + 1)

        if step % _EVAL_EVERY == 0:
            mx.eval(position, mean, mean_sq, updates, consecutive_skips, logdensity)
            if float(consecutive_skips) >= max_consecutive_skips:
                _log.warning(
                    "Adam ascent to the mode stopped after %d steps: the log-density gradient "
                    "kept coming back non-finite.",
                    step,
                )
                break
            if float(mx.max(logdensity - previous_logdensity)) < tolerance:
                break
            previous_logdensity = logdensity

    mx.eval(position)

    return position


def warmup(
    logdensity_fn: Callable[[mx.array], mx.array],
    initial_position: ArrayLike,
    *,
    num_steps: int,
    settings: AdaptationSettings = AdaptationSettings(),
    integrator: str = "mclachlan",
    seed: int = 0,
    compile_step: bool = True,
) -> TunedParameters:
    r"""
    Adapt the step size, the diagonal metric, and ``L``.

    A single adapting chain, carried as shape ``(1, dim)`` so the batched primitives apply, both
    tunes the parameters and moves into the typical set, so the result cold-starts sampling
    without a hand-supplied metric. Adaptation runs in three phases, sized by the ``frac_tune``
    fractions of ``num_steps``:

    1. Tune the step size alone, with a controller that drives
       :math:`\\mathrm{Var}[E]` per dimension to ``settings.desired_energy_var``.
    2. Keep tuning while accumulating a step-size-weighted running mean and mean-square of the
       position, then set the metric to the per-coordinate variance and re-adjust the step size
       under it.
    3. Hold the parameters fixed, sample, and set ``L`` from the autocorrelation length.

    Parameters
    ----------
    logdensity_fn : callable
        Maps an MLX array of shape ``(dim,)`` to a scalar log-density. If it also has a
        ``value_and_grad`` method, that supplies the gradient instead of MLX autodiff.
    initial_position : array
        Starting point of the adapting chain, of shape ``(dim,)``.
    num_steps : int
        Budget of integrator steps, of which the ``frac_tune`` fractions are taken.
    integrator : str
        Either ``"mclachlan"`` or ``"velocity_verlet"``. Default is ``"mclachlan"``.
    settings.diagonal_preconditioning : bool
        Whether to estimate a diagonal inverse mass matrix in phase 2. Default is True.
    mass_matrix : str or MassMatrixSettings
        How phase 2 estimates the metric. A bare string selects the method and leaves the estimator
        windows at their defaults; pass a :class:`MassMatrixSettings` to set those too. Default is
        ``"gradient"``.
    settings.desired_energy_var : float
        Target energy variance per dimension. Default is 5e-4.
    settings.trust_in_estimate : float
        Width of the controller's Gaussian weighting. Larger values give more weight to
        single-step estimates far from the target. Default is 1.5.
    settings.num_effective_samples : float
        Sets the controller's exponential decay rate. Default is 150.
    settings.frac_tune1, settings.frac_tune2, settings.frac_tune3 : float
        Fractions of ``num_steps`` given to each phase. Each defaults to 0.1.
    settings.l_factor : float
        Multiplier on the autocorrelation-derived ``L`` in phase 3. Default is 0.4.
    seed : int
        Seed for the MLX random key. Default is 0.
    settings.optimize_steps : int
        Maximum number of Adam steps taken toward the mode before adaptation. Pass 0 for a model
        whose log-density is unbounded above, such as a centered hierarchical model, where the
        ascent runs off into a region of high density but negligible mass. Default is 200.
    settings.optimize_learning_rate : float
        Adam learning rate for that ascent. Default is 0.05.
    compile_step : bool
        Whether to fuse the adapting step with ``mx.compile``. Default is True.

    Returns
    -------
    TunedParameters
        The adapting chain's final ``position`` of shape ``(dim,)``, which should be jittered by
        ``sqrt(inverse_mass_matrix)`` to seed the sampling chains, along with the adapted ``L``,
        ``step_size``, and ``metric``, and the ``num_tuning_steps`` spent.
    """
    if settings.mass_matrix not in _MASS_MATRIX_METHODS:
        raise ValueError(
            f"mass_matrix must be one of {_MASS_MATRIX_METHODS}, got {settings.mass_matrix!r}"
        )

    coefficients = INTEGRATOR_COEFFICIENTS[integrator]
    initial_position = np.asarray(initial_position, dtype=np.float32).ravel()
    dim = initial_position.shape[0]
    _check_dim(dim)
    logp_and_grad = _batched_value_and_grad(logdensity_fn)
    decay_rate = (settings.num_effective_samples - 1.0) / (settings.num_effective_samples + 1.0)

    # Phases 1 and 2 hold L at sqrt(dim), so their refresh rate is a compile-time constant rather
    # than threaded state.
    adaptation_inverse_L = 1.0 / math.sqrt(dim)

    num_steps1 = round(num_steps * settings.frac_tune1)
    num_steps2 = round(num_steps * settings.frac_tune2)
    num_steps3 = round(num_steps * settings.frac_tune3)

    def adapt_step(state, metric, mask, keys):
        """
        Take one adapting step: dynamics, the step-size controller, then the streaming moments.

        Non-finite steps are rejected with ``mx.where`` rather than a Python branch, so the loop
        stays lazy and this body compiles.
        """
        refresh_key, decohere_key, resample_key = keys
        dynamics = Dynamics(
            logp_and_grad=logp_and_grad,
            step_size=state.step_size,
            coefficients=coefficients,
            metric=metric,
            inverse_L=adaptation_inverse_L,
            dim=dim,
        )
        previous = ChainState(state.position, state.momentum, state.logdensity, state.grad)
        proposed, energy_error = _transition(
            state=previous, keys=(refresh_key, decohere_key), dynamics=dynamics
        )
        chain, energy_error, is_finite = _revert_nonfinite(
            proposed=proposed, previous=previous, energy_error=energy_error, key=resample_key
        )
        position, momentum, logdensity, grad = chain
        step_size_max = mx.where(is_finite, state.step_size_max, state.step_size * 0.8)

        relative_error = energy_error**2 / (dim * settings.desired_energy_var) + 1e-8
        weight = mx.exp(-0.5 * (mx.log(relative_error) / (6.0 * settings.trust_in_estimate)) ** 2)
        x_average = decay_rate * state.x_average + weight * (relative_error / state.step_size**6)
        time = decay_rate * state.time + weight
        step_size = mx.maximum(
            mx.minimum(mx.power(x_average / time, -1.0 / 6.0), step_size_max), 1e-12
        )

        # Both windows see every accepted draw; the caller swaps them on the schedule. Unlike
        # blackjax this is unweighted, matching nuts-rs, whose windowing plays the role the
        # step-size weighting played there.
        moment_weight = mask * is_finite.astype(mx.float32)
        x = position.reshape(1, dim)
        foreground = _accumulate(state.foreground, x, grad, moment_weight)
        background = _accumulate(state.background, x, grad, moment_weight)

        return AdaptationState(
            position=position,
            momentum=momentum,
            logdensity=logdensity,
            grad=grad,
            step_size=step_size,
            step_size_max=step_size_max,
            time=time,
            x_average=x_average,
            foreground=foreground,
            background=background,
        )

    step = mx.compile(adapt_step) if compile_step else adapt_step

    key = mx.random.key(seed)
    key, subkey = mx.random.split(key, num=2)

    position = mx.array(initial_position).reshape(1, dim)
    _check_initial_state(*logp_and_grad(position))

    # Without this ascent the chain wanders under the large initial step while the diagonal
    # variances are collected, which inflates the metric and forces a step size too small to mix.
    position = _optimize_to_mode(
        logp_and_grad=logp_and_grad,
        position=position,
        steps=settings.optimize_steps,
        learning_rate=settings.optimize_learning_rate,
    )
    logdensity, grad = logp_and_grad(position)

    state = AdaptationState(
        position=position,
        momentum=_unit_vectors(shape=(1, dim), key=subkey),
        logdensity=logdensity,
        grad=grad,
        step_size=mx.array([math.sqrt(dim) * 0.25], dtype=mx.float32),
        step_size_max=mx.array([float("inf")], dtype=mx.float32),
        time=mx.array([0.0], dtype=mx.float32),
        x_average=mx.array([0.0], dtype=mx.float32),
        foreground=_empty_moments(dim),
        background=_empty_moments(dim),
    )

    metric = Metric(scale=mx.ones((dim,)))
    num_tuning_steps = 0

    # Phase 1 tunes the step size alone; phase 2 also accumulates the windowed moments. The swap
    # schedule is a function of the step index alone, so it needs no synchronization -- a draw
    # rejected as non-finite simply leaves the window one sample short of its nominal length.
    # The low-rank fit needs the raw draws, not running moments, so phase 2 retains a trailing
    # window of them. nuts-rs keeps the same buffer and splits it on each switch; keeping the most
    # recent low_rank_window draws is the streaming approximation of that.
    retained: list[tuple[mx.array, mx.array]] = []

    # Phase 2 is cut into segments: at each boundary the metric is refitted from what has been seen
    # since the last one, the step-size controller is re-seeded, and the draw window is cleared so
    # the next fit only sees draws taken under the improved metric.
    refits = settings.low_rank_refits
    refit_steps = (
        {round(num_steps2 * i / (refits + 1)): i for i in range(1, refits + 1)}
        if settings.mass_matrix == "low_rank"
        else {}
    )
    switch_steps = _window_switch_steps(
        num_steps2,
        settings.early_end,
        settings.early_switch_freq,
        settings.switch_freq,
        settings.window_growth,
    )
    for n_phase_steps, mask_value in ((num_steps1, 0.0), (num_steps2, 1.0)):
        mask = mx.array([mask_value], dtype=mx.float32)
        for phase_step in range(1, n_phase_steps + 1):
            key, *keys = mx.random.split(key, num=4)
            state = step(state, metric=metric, mask=mask, keys=tuple(keys))
            if mask_value and settings.mass_matrix == "low_rank":
                retained.append((state.position, state.grad))
                del retained[: -settings.low_rank_window]
            if mask_value and phase_step in switch_steps:
                state = state._replace(foreground=state.background, background=_empty_moments(dim))
            if mask_value and phase_step in refit_steps:
                metric, _ = _fit_metric(
                    state.foreground,
                    retained,
                    dim,
                    settings.mass_matrix,
                    allow_low_rank=refit_steps[phase_step] > 1,
                )
                state = _reset_step_size_controller(state)
                retained.clear()
                mx.eval(state)
            num_tuning_steps += 1
            if num_tuning_steps % _EVAL_EVERY == 0:
                mx.eval(state)
    mx.eval(state)

    L = math.sqrt(dim)

    if num_steps2 > 1:
        fitted, variances = _fit_metric(
            state.foreground, retained, dim, settings.mass_matrix, allow_low_rank=True
        )
        L = float(np.sqrt(variances.sum()))

        if settings.diagonal_preconditioning:
            L = math.sqrt(dim)
            metric = fitted
            mask = mx.array([1.0], dtype=mx.float32)

            state = _reset_step_size_controller(state)

            for _ in range(round(num_steps2 / 3)):
                key, *keys = mx.random.split(key, num=4)
                state = step(state, metric=metric, mask=mask, keys=tuple(keys))
                num_tuning_steps += 1
                if num_tuning_steps % _EVAL_EVERY == 0:
                    mx.eval(state)
            mx.eval(state)

    step_size = float(np.asarray(state.step_size).reshape(-1)[0])
    chain = ChainState(state.position, state.momentum, state.logdensity, state.grad)

    if num_steps3 >= 2:
        # Phase 3 runs at the L phase 2 settled on, which is sqrt(dim) only under diagonal
        # preconditioning. Holding it at sqrt(dim) regardless would measure the autocorrelation
        # length under the wrong decoherence rate.
        dynamics = Dynamics(
            logp_and_grad=logp_and_grad,
            step_size=state.step_size,
            coefficients=coefficients,
            metric=metric,
            inverse_L=1.0 / L,
            dim=dim,
        )
        positions = []
        for _ in range(num_steps3):
            key, refresh_key, decohere_key = mx.random.split(key, num=3)
            chain, _ = _transition(state=chain, keys=(refresh_key, decohere_key), dynamics=dynamics)

            positions.append(chain.position)
            num_tuning_steps += 1
            if num_tuning_steps % _EVAL_EVERY == 0:
                mx.eval(chain)

        samples = mx.stack(positions, axis=0).reshape(num_steps3, dim)
        mx.eval(samples)
        ess = _ess_per_dim(np.asarray(samples))
        L = settings.l_factor * step_size * float(np.mean(num_steps3 / np.clip(ess, 1e-8, None)))

    return TunedParameters(
        position=mx.array(np.asarray(chain.position).reshape(dim)),
        L=float(L),
        step_size=step_size,
        metric=metric,
        num_tuning_steps=num_tuning_steps,
    )


def warmup_and_sample(
    logdensity_fn: Callable[[mx.array], mx.array],
    initial_position: ArrayLike,
    *,
    num_tune: int,
    draws: int,
    chains: int,
    discard: int = 0,
    settings: AdaptationSettings = AdaptationSettings(),
    integrator: str = "mclachlan",
    seed: int = 0,
    compile_step: bool = True,
) -> tuple[SamplerOutput, TunedParameters]:
    """
    Adapt on one chain, then sample ``chains`` chains jittered around the tuned position.

    Parameters
    ----------
    num_tune : int
        Budget of integrator steps for :func:`warmup`.
    draws : int
        Number of draws to keep per chain.
    chains : int
        Number of chains to run.
    discard : int
        Number of leading sampling steps to drop. Default is 0.

    Returns
    -------
    output : SamplerOutput
        Draws of shape ``(draws, chains, dim)``, with per-step diagnostics covering all
        ``draws + discard`` steps.
    tuned : TunedParameters
        The parameters :func:`warmup` settled on.
    """
    tuned = warmup(
        logdensity_fn,
        initial_position,
        num_steps=num_tune,
        settings=settings,
        integrator=integrator,
        seed=seed,
        compile_step=compile_step,
    )

    dim = tuned.position.shape[0]
    # A draw from N(0, M^-1) is forward_L(z), so the chains scatter along the fitted metric rather
    # than along its diagonal -- which for a rotated posterior points the wrong way entirely.
    jitter = _unwhiten_momentum(
        tuned.metric, mx.random.normal(shape=(chains, dim), key=mx.random.key(seed + 1))
    )

    output = sample(
        logdensity_fn,
        tuned.position[None, :] + jitter,
        L=tuned.L,
        step_size=tuned.step_size,
        n_steps=draws + discard,
        discard=discard,
        integrator=integrator,
        inverse_mass_matrix=tuned.metric,
        seed=seed + 2,
        compile_step=compile_step,
    )

    return output, tuned
