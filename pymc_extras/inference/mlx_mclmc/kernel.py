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

_LOG2 = math.log(2.0)

# Below this norm a vector is left unnormalized rather than divided by its own (near-zero)
# norm, so a vanishing gradient at the mode gives a finite result instead of 0 / 0 -> nan.
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


class Dynamics(NamedTuple):
    """Everything a transition needs that does not change from one step to the next."""

    logp_and_grad: Callable
    step_size: float | mx.array
    coefficients: list[float]
    sqrt_inverse_mass: mx.array
    inverse_L: float
    dim: int


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
    stream_weight: mx.array
    stream_mean: mx.array
    stream_mean_sq: mx.array


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
    inverse_mass_matrix: mx.array
    num_tuning_steps: int


def _check_dim(dim: int) -> None:
    if dim < 2:
        raise ValueError(
            f"MCLMC needs at least 2 dimensions, got {dim}. The isokinetic momentum update "
            "divides by (dim - 1), which is undefined for a single parameter."
        )


def _batched_value_and_grad(logdensity_fn: Callable) -> Callable:
    """
    Build the batched value-and-gradient callable the dynamics use.

    A ``logdensity_fn`` carrying its own ``value_and_grad`` supplies the gradient itself, which is
    how :class:`~pymc_extras.inference.mlx_mclmc.logp.MLXLogp` hands over PyTensor's symbolic
    gradient. Anything else is differentiated by MLX, whose reverse rules do not cover every op.
    """
    supplied = getattr(logdensity_fn, "value_and_grad", None)

    return mx.vmap(supplied if callable(supplied) else mx.value_and_grad(logdensity_fn))


def _normalize(vectors: mx.array) -> mx.array:
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)

    return mx.where(norms > _NORM_FLOOR, vectors / mx.maximum(norms, _SAFE_DIVISOR), vectors)


def _unit_vectors(shape: tuple[int, ...], key: mx.array) -> mx.array:
    return _normalize(mx.random.normal(shape=shape, key=key))


def _momentum_update(
    momentum: mx.array,
    grad: mx.array,
    effective_step: float | mx.array,
    sqrt_inverse_mass: mx.array,
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
    scaled_grad = grad * sqrt_inverse_mass
    grad_norm = mx.linalg.norm(scaled_grad, axis=-1, keepdims=True)
    grad_direction = _normalize(scaled_grad)
    projection = mx.sum(momentum * grad_direction, axis=-1, keepdims=True)

    delta = effective_step * grad_norm / (dim - 1)
    zeta = mx.exp(-delta)

    # Written without an exp(delta) factor, so a large gradient norm cannot overflow.
    momentum = _normalize(
        grad_direction * (1 - zeta) * (1 + zeta + projection * (1 - zeta)) + 2 * zeta * momentum
    )
    kinetic_energy_change = (
        (delta - _LOG2 + mx.log(1 + projection + (1 - projection) * zeta**2)) * (dim - 1)
    ).squeeze(-1)

    return momentum, momentum * sqrt_inverse_mass, kinetic_energy_change


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
                sqrt_inverse_mass=dynamics.sqrt_inverse_mass,
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
        sqrt_inverse_mass=dynamics.sqrt_inverse_mass,
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
    inverse_mass_matrix: ArrayLike = 1.0,
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
    inverse_mass_matrix : float or array
        Diagonal inverse mass matrix, broadcastable to ``(dim,)``. Default is 1.0.
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
        sqrt_inverse_mass=mx.sqrt(mx.array(inverse_mass_matrix, dtype=mx.float32)),
        inverse_L=1.0 / L,
        dim=dim,
    )

    key = mx.random.key(seed)
    key, subkey = mx.random.split(key, num=2)
    logdensity, grad = logp_and_grad(position)
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
    inverse_mass_matrix: ArrayLike = 1.0,
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
) -> mx.array:
    """
    Ascend the log-density with Adam, to concentrate the adapting chain near the mode.

    Stops early once a block of steps improves the log-density by less than ``tolerance``. A
    log-density unbounded above, as a centered hierarchical model has, never triggers that stop,
    so ``steps`` remains a hard cap.
    """
    beta1, beta2 = 0.9, 0.999
    mean = mx.zeros_like(position)
    mean_sq = mx.zeros_like(position)
    previous_logdensity = -mx.inf

    for t in range(1, steps + 1):
        logdensity, grad = logp_and_grad(position)
        mean = beta1 * mean + (1 - beta1) * grad
        mean_sq = beta2 * mean_sq + (1 - beta2) * grad * grad
        mean_hat = mean / (1 - beta1**t)
        mean_sq_hat = mean_sq / (1 - beta2**t)
        position = position + learning_rate * mean_hat / (mx.sqrt(mean_sq_hat) + 1e-8)

        if t % _EVAL_EVERY == 0:
            mx.eval(position, mean, mean_sq, logdensity)
            improvement = float(mx.max(logdensity - previous_logdensity))
            if improvement < tolerance:
                break
            previous_logdensity = logdensity

    mx.eval(position)

    return position


def warmup(
    logdensity_fn: Callable[[mx.array], mx.array],
    initial_position: ArrayLike,
    *,
    num_steps: int,
    integrator: str = "mclachlan",
    diagonal_preconditioning: bool = True,
    desired_energy_var: float = 5e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: float = 150,
    frac_tune1: float = 0.1,
    frac_tune2: float = 0.1,
    frac_tune3: float = 0.1,
    l_factor: float = 0.4,
    seed: int = 0,
    optimize_steps: int = 200,
    optimize_learning_rate: float = 0.05,
    compile_step: bool = True,
) -> TunedParameters:
    """
    Adapt the step size, the diagonal metric, and ``L``.

    A single adapting chain, carried as shape ``(1, dim)`` so the batched primitives apply, both
    tunes the parameters and moves into the typical set, so the result cold-starts sampling
    without a hand-supplied metric. Adaptation runs in three phases, sized by the ``frac_tune``
    fractions of ``num_steps``:

    1. Tune the step size alone, with a controller that drives
       :math:`\\mathrm{Var}[E]` per dimension to ``desired_energy_var``.
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
    diagonal_preconditioning : bool
        Whether to estimate a diagonal inverse mass matrix in phase 2. Default is True.
    desired_energy_var : float
        Target energy variance per dimension. Default is 5e-4.
    trust_in_estimate : float
        Width of the controller's Gaussian weighting. Larger values give more weight to
        single-step estimates far from the target. Default is 1.5.
    num_effective_samples : float
        Sets the controller's exponential decay rate. Default is 150.
    frac_tune1, frac_tune2, frac_tune3 : float
        Fractions of ``num_steps`` given to each phase. Each defaults to 0.1.
    l_factor : float
        Multiplier on the autocorrelation-derived ``L`` in phase 3. Default is 0.4.
    seed : int
        Seed for the MLX random key. Default is 0.
    optimize_steps : int
        Maximum number of Adam steps taken toward the mode before adaptation. Pass 0 for a model
        whose log-density is unbounded above, such as a centered hierarchical model, where the
        ascent runs off into a region of high density but negligible mass. Default is 200.
    optimize_learning_rate : float
        Adam learning rate for that ascent. Default is 0.05.
    compile_step : bool
        Whether to fuse the adapting step with ``mx.compile``. Default is True.

    Returns
    -------
    TunedParameters
        The adapting chain's final ``position`` of shape ``(dim,)``, which should be jittered by
        ``sqrt(inverse_mass_matrix)`` to seed the sampling chains, along with the adapted ``L``,
        ``step_size``, and ``inverse_mass_matrix``, and the ``num_tuning_steps`` spent.
    """
    coefficients = INTEGRATOR_COEFFICIENTS[integrator]
    initial_position = np.asarray(initial_position, dtype=np.float32).ravel()
    dim = initial_position.shape[0]
    _check_dim(dim)
    logp_and_grad = _batched_value_and_grad(logdensity_fn)
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    # Phases 1 and 2 hold L at sqrt(dim), so their refresh rate is a compile-time constant rather
    # than threaded state.
    adaptation_inverse_L = 1.0 / math.sqrt(dim)

    num_steps1 = round(num_steps * frac_tune1)
    num_steps2 = round(num_steps * frac_tune2)
    num_steps3 = round(num_steps * frac_tune3)

    def adapt_step(state, sqrt_inverse_mass, mask, keys):
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
            sqrt_inverse_mass=sqrt_inverse_mass,
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

        relative_error = energy_error**2 / (dim * desired_energy_var) + 1e-8
        weight = mx.exp(-0.5 * (mx.log(relative_error) / (6.0 * trust_in_estimate)) ** 2)
        x_average = decay_rate * state.x_average + weight * (relative_error / state.step_size**6)
        time = decay_rate * state.time + weight
        step_size = mx.maximum(
            mx.minimum(mx.power(x_average / time, -1.0 / 6.0), step_size_max), 1e-12
        )

        moment_weight = mask * is_finite.astype(mx.float32) * step_size
        stream_weight = state.stream_weight + moment_weight
        safe_weight = mx.maximum(stream_weight, 1e-30)
        x = position.reshape(1, dim)
        do_update = stream_weight > 0
        stream_mean = mx.where(
            do_update,
            (state.stream_weight * state.stream_mean + moment_weight * x) / safe_weight,
            state.stream_mean,
        )
        stream_mean_sq = mx.where(
            do_update,
            (state.stream_weight * state.stream_mean_sq + moment_weight * (x * x)) / safe_weight,
            state.stream_mean_sq,
        )

        return AdaptationState(
            position=position,
            momentum=momentum,
            logdensity=logdensity,
            grad=grad,
            step_size=step_size,
            step_size_max=step_size_max,
            time=time,
            x_average=x_average,
            stream_weight=stream_weight,
            stream_mean=stream_mean,
            stream_mean_sq=stream_mean_sq,
        )

    step = mx.compile(adapt_step) if compile_step else adapt_step

    key = mx.random.key(seed)
    key, subkey = mx.random.split(key, num=2)

    # Without this ascent the chain wanders under the large initial step while the diagonal
    # variances are collected, which inflates the metric and forces a step size too small to mix.
    position = _optimize_to_mode(
        logp_and_grad=logp_and_grad,
        position=mx.array(initial_position).reshape(1, dim),
        steps=optimize_steps,
        learning_rate=optimize_learning_rate,
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
        stream_weight=mx.array([0.0], dtype=mx.float32),
        stream_mean=mx.zeros((1, dim)),
        stream_mean_sq=mx.zeros((1, dim)),
    )

    sqrt_inverse_mass = mx.ones((dim,))
    num_tuning_steps = 0

    # Phase 1 tunes the step size alone; phase 2 also accumulates the streaming moments.
    for n_phase_steps, mask_value in ((num_steps1, 0.0), (num_steps2, 1.0)):
        mask = mx.array([mask_value], dtype=mx.float32)
        for _ in range(n_phase_steps):
            key, *keys = mx.random.split(key, num=4)
            state = step(state, sqrt_inverse_mass=sqrt_inverse_mass, mask=mask, keys=tuple(keys))
            num_tuning_steps += 1
            if num_tuning_steps % _EVAL_EVERY == 0:
                mx.eval(state)
    mx.eval(state)

    inverse_mass_matrix = np.ones(dim, dtype=np.float32)
    L = math.sqrt(dim)

    if num_steps2 > 1:
        stream_mean = np.asarray(state.stream_mean).reshape(dim)
        stream_mean_sq = np.asarray(state.stream_mean_sq).reshape(dim)
        variances = np.clip(stream_mean_sq - stream_mean**2, 1e-12, None).astype(np.float32)
        L = float(np.sqrt(variances.sum()))

        if diagonal_preconditioning:
            inverse_mass_matrix = variances
            L = math.sqrt(dim)
            sqrt_inverse_mass = mx.array(np.sqrt(inverse_mass_matrix))
            mask = mx.array([1.0], dtype=mx.float32)

            # Reset the controller accumulators before the re-adjustment. Otherwise the tiny
            # identity-metric step from phases 1 and 2 stays baked into x_average, whose decay is
            # slow enough that the step size cannot re-tune under the new metric.
            state = state._replace(
                step_size_max=mx.array([float("inf")], dtype=mx.float32),
                time=mx.array([0.0], dtype=mx.float32),
                x_average=mx.array([0.0], dtype=mx.float32),
            )

            for _ in range(round(num_steps2 / 3)):
                key, *keys = mx.random.split(key, num=4)
                state = step(
                    state, sqrt_inverse_mass=sqrt_inverse_mass, mask=mask, keys=tuple(keys)
                )
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
            sqrt_inverse_mass=sqrt_inverse_mass,
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
        L = l_factor * step_size * float(np.mean(num_steps3 / np.clip(ess, 1e-8, None)))

    return TunedParameters(
        position=mx.array(np.asarray(chain.position).reshape(dim)),
        L=float(L),
        step_size=step_size,
        inverse_mass_matrix=mx.array(inverse_mass_matrix),
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
    integrator: str = "mclachlan",
    seed: int = 0,
    compile_step: bool = True,
    **warmup_kwargs,
) -> tuple[SamplerOutput, TunedParameters]:
    """
    Adapt on one chain, then sample ``chains`` chains jittered around the tuned position.

    Parameters
    ----------
    num_tune : int
        Budget of integrator steps for :func:`warmup`, which takes the remaining keyword
        arguments.
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
        integrator=integrator,
        seed=seed,
        compile_step=compile_step,
        **warmup_kwargs,
    )

    dim = tuned.position.shape[0]
    jitter = mx.sqrt(tuned.inverse_mass_matrix) * mx.random.normal(
        shape=(chains, dim), key=mx.random.key(seed + 1)
    )

    output = sample(
        logdensity_fn,
        tuned.position[None, :] + jitter,
        L=tuned.L,
        step_size=tuned.step_size,
        n_steps=draws + discard,
        discard=discard,
        integrator=integrator,
        inverse_mass_matrix=tuned.inverse_mass_matrix,
        seed=seed + 2,
        compile_step=compile_step,
    )

    return output, tuned
