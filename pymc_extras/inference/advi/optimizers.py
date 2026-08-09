from collections.abc import Callable

import numpy as np
import pytensor

from pytensor import config
from pytensor import tensor as pt

Schedule = Callable[[int], float]
ScalarOrSchedule = float | Schedule


class GradientTransformation:
    """An optax-style gradient transformation with optional PyTensor implementation.

    Parameters
    ----------
    init :
        Function ``(params: dict[str, np.ndarray]) -> state`` that initializes the
        optimizer state from the initial parameter values.
    update :
        Function ``(updates, state, params=None) -> (new_updates, new_state)`` that
        applies the transformation to a dictionary of numpy gradient updates.
    pytensor :
        Optional function ``(grads, shared_params) -> (new_grads, updates_dict)``
        that applies the transformation in a PyTensor graph.  ``grads`` is a list
        of symbolic gradient variables and ``shared_params`` the corresponding
        shared parameter variables.  Returns transformed gradients and a
        dictionary of ``{shared_var: new_value}`` updates for
        :func:`pytensor.compile`.  Transformations that depend on a schedule
        (callable learning rate) leave this ``None`` and use the Python update
        path instead.
    """

    def __init__(self, init, update, pytensor=None):
        self.init = init
        self.update = update
        self.pytensor = pytensor


def apply_updates(
    params: dict[str, np.ndarray], updates: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Add the updates to the parameters."""
    return {name: np.asarray(param + updates[name]) for name, param in params.items()}


def _chain_pt(*fns):
    """Compose PyTensor update functions."""

    def composed(grads, shared_params):
        all_updates = {}
        for fn in fns:
            grads, updates = fn(grads, shared_params)
            all_updates.update(updates)
        return grads, all_updates

    return composed


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

    pytensor_fns = [t.pytensor for t in transforms if t.pytensor is not None]
    pytensor = _chain_pt(*pytensor_fns) if len(pytensor_fns) == len(transforms) else None

    return GradientTransformation(init, update, pytensor)


def clip_by_global_norm(max_norm: float) -> GradientTransformation:
    """Clip the gradients so that their global L2 norm does not exceed ``max_norm``."""

    def init(params):
        return None

    def update(updates, state, params=None):
        global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in updates.values()))
        scale = np.minimum(1.0, max_norm / (global_norm + 1e-12))
        return {name: g * scale for name, g in updates.items()}, state

    def _pytensor_impl(grads, shared_params):
        global_norm = pt.sqrt(pt.sum([pt.sum(pt.square(g)) for g in grads]))
        scale = pt.minimum(1.0, max_norm / (global_norm + 1e-12))
        return [g * scale for g in grads], {}

    return GradientTransformation(init, update, _pytensor_impl)


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

    def _pytensor_impl(grads, shared_params):
        t = pytensor.shared(np.zeros((), dtype="int64"), name="adam_t")
        t_new = t + 1
        t_new_float = t_new.astype(config.floatX)
        updates = {t: t_new}
        new_grads = []
        for param, grad in zip(shared_params, grads):
            value = param.get_value(borrow=True)
            m = pytensor.shared(np.zeros_like(value), name=f"adam_m_{param.name}")
            v = pytensor.shared(np.zeros_like(value), name=f"adam_v_{param.name}")
            m_new = b1 * m + (1 - b1) * grad
            v_new = b2 * v + (1 - b2) * pt.square(grad)
            m_hat = m_new / (1 - b1**t_new_float)
            v_hat = v_new / (1 - b2**t_new_float)
            new_grads.append(m_hat / (pt.sqrt(v_hat) + eps))
            updates.update({m: m_new, v: v_new})
        return new_grads, updates

    return GradientTransformation(init, update, _pytensor_impl)


def scale_by_learning_rate(learning_rate: ScalarOrSchedule) -> GradientTransformation:
    """Scale the gradients by ``-learning_rate``, which may be a schedule of the step count."""

    def init(params):
        return {"count": 0}

    def update(updates, state, params=None):
        count = state["count"]
        lr = learning_rate(count) if callable(learning_rate) else learning_rate
        return {name: -lr * g for name, g in updates.items()}, {"count": count + 1}

    # PyTensor path: only works with a constant learning rate (schedules are
    # Python callables and cannot be baked into the graph).
    if callable(learning_rate):
        _pytensor_impl = None
    else:
        lr = learning_rate

        def _pytensor_impl(grads, shared_params):
            return [g * (-lr) for g in grads], {}

    return GradientTransformation(init, update, _pytensor_impl)


def adam(
    learning_rate: ScalarOrSchedule = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> GradientTransformation:
    """Adam optimizer."""
    return chain(scale_by_adam(b1=b1, b2=b2, eps=eps), scale_by_learning_rate(learning_rate))


def clipped_adam(
    learning_rate: ScalarOrSchedule = 0.01, clip_norm: float = 10.0, **adam_kwargs
) -> GradientTransformation:
    """Adam with gradient clipping by global norm, as numpyro's ClippedAdam."""
    return chain(clip_by_global_norm(clip_norm), adam(learning_rate, **adam_kwargs))


def sgd(learning_rate: ScalarOrSchedule = 0.01) -> GradientTransformation:
    """Stochastic gradient descent optimizer."""
    return scale_by_learning_rate(learning_rate)


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
