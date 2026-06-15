"""Optax-like gradient transformations operating on dicts of numpy parameters.

The API follows optax: an optimizer is a ``GradientTransformation`` with
``init(params) -> state`` and ``update(grads, state, params) -> (updates, state)``,
where updates are increments added to the parameters with :func:`apply_updates`.
Actual optax optimizers are plug-compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

Schedule = Callable[[int], float]
ScalarOrSchedule = float | Schedule


class GradientTransformation(NamedTuple):
    init: Callable[[dict[str, np.ndarray]], Any]
    update: Callable[..., tuple[dict[str, np.ndarray], Any]]


def apply_updates(
    params: dict[str, np.ndarray], updates: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Add the updates to the parameters."""
    return {name: np.asarray(param + updates[name]) for name, param in params.items()}


def chain(*transforms: GradientTransformation) -> GradientTransformation:
    """Compose gradient transformations, applied in the given order."""

    def init(params):
        return tuple(transform.init(params) for transform in transforms)

    def update(updates, state, params=None):
        new_state = []
        for transform, transform_state in zip(transforms, state):
            updates, transform_state = transform.update(updates, transform_state, params)
            new_state.append(transform_state)
        return updates, tuple(new_state)

    return GradientTransformation(init, update)


def clip_by_global_norm(max_norm: float) -> GradientTransformation:
    """Clip the gradients so that their global L2 norm does not exceed ``max_norm``."""

    def init(params):
        return None

    def update(updates, state, params=None):
        global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in updates.values()))
        scale = np.minimum(1.0, max_norm / (global_norm + 1e-12))
        return {name: g * scale for name, g in updates.items()}, state

    return GradientTransformation(init, update)


def scale_by_adam(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> GradientTransformation:
    """Rescale the gradients by the Adam preconditioner (Kingma & Ba, 2015)."""

    def init(params):
        return {
            "mu": {name: np.zeros_like(value) for name, value in params.items()},
            "nu": {name: np.zeros_like(value) for name, value in params.items()},
            "count": 0,
        }

    def update(updates, state, params=None):
        count = state["count"] + 1
        mu, nu = state["mu"], state["nu"]
        new_updates = {}
        for name, g in updates.items():
            mu[name] = b1 * mu[name] + (1 - b1) * g
            nu[name] = b2 * nu[name] + (1 - b2) * g**2
            mu_hat = mu[name] / (1 - b1**count)
            nu_hat = nu[name] / (1 - b2**count)
            new_updates[name] = mu_hat / (np.sqrt(nu_hat) + eps)
        return new_updates, {"mu": mu, "nu": nu, "count": count}

    return GradientTransformation(init, update)


def scale_by_learning_rate(learning_rate: ScalarOrSchedule) -> GradientTransformation:
    """Scale the gradients by ``-learning_rate``, which may be a schedule of the step count."""

    def init(params):
        return {"count": 0}

    def update(updates, state, params=None):
        count = state["count"]
        lr = learning_rate(count) if callable(learning_rate) else learning_rate
        return {name: -lr * g for name, g in updates.items()}, {"count": count + 1}

    return GradientTransformation(init, update)


def adam(
    learning_rate: ScalarOrSchedule, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> GradientTransformation:
    """Adam optimizer."""
    return chain(scale_by_adam(b1=b1, b2=b2, eps=eps), scale_by_learning_rate(learning_rate))


def clipped_adam(
    learning_rate: ScalarOrSchedule, clip_norm: float = 10.0, **adam_kwargs
) -> GradientTransformation:
    """Adam with gradient clipping by global norm, as numpyro's ClippedAdam."""
    return chain(clip_by_global_norm(clip_norm), adam(learning_rate, **adam_kwargs))


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Schedule:
    """Linear one-cycle learning rate schedule (Smith & Topin, 2018), as in optax.

    The learning rate ramps from ``peak_value / div_factor`` to ``peak_value`` over the
    first ``pct_start`` fraction of ``transition_steps``, anneals back down by
    ``pct_final``, and decays to ``peak_value / div_factor / final_div_factor`` at the end.
    """
    init_value = peak_value / div_factor
    end_value = init_value / final_div_factor
    boundaries = np.array([0.0, pct_start, pct_final, 1.0]) * transition_steps
    values = np.array([init_value, peak_value, init_value, end_value])

    def schedule(count: int) -> float:
        return float(np.interp(count, boundaries, values))

    return schedule
