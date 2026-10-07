from collections.abc import Callable

import numpy as np
import pytensor

from pytensor import config
from pytensor import tensor as pt

Schedule = Callable[[pt.TensorVariable], pt.TensorVariable]


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
        Optional function ``(grads, shared_params, state) -> (new_grads, updates_dict)``
        that applies the transformation in a PyTensor graph.  ``grads`` is a list
        of symbolic gradient variables, ``shared_params`` the corresponding
        shared parameter variables, and ``state`` the dictionary of shared state
        variables from ``pytensor_init``.  Returns transformed gradients and a
        dictionary of ``{shared_var: new_value}`` updates for
        :func:`pytensor.compile`.  ``None`` means the transformation has no
        compiled path and can only be used through the Python update path.
    pytensor_init :
        Optional function ``(shared_params) -> dict[str, SharedVariable]`` that creates
        the transformation's state buffers, keyed by a name unique within the
        transformation. The caller owns them and passes them to every ``pytensor``
        call. Default is a function returning no state.
    """

    def __init__(self, init, update, pytensor=None, pytensor_init=None):
        self.init = init
        self.update = update
        self.pytensor = pytensor
        self.pytensor_init = pytensor_init if pytensor_init is not None else _no_state


def _no_state(shared_params) -> dict:
    return {}


def _zeros_like_shared(shared: pytensor.compile.SharedVariable) -> np.ndarray:
    return np.zeros_like(shared.get_value(borrow=True))


def apply_updates(
    params: dict[str, np.ndarray], updates: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Add the updates to the parameters."""
    return {name: np.asarray(param + updates[name]) for name, param in params.items()}


def _chain_pt(*fns):
    """Compose PyTensor update functions."""

    def composed(grads, shared_params, state):
        all_updates = {}
        for fn in fns:
            grads, updates = fn(grads, shared_params, state)
            all_updates.update(updates)
        return grads, all_updates

    return composed


def _chain_pt_init(*inits):
    """Compose state constructors into one dictionary, refusing a name two stages share."""

    def composed(shared_params):
        state = {}
        for init in inits:
            stage_state = init(shared_params)
            if duplicates := sorted(set(stage_state) & set(state)):
                raise ValueError(
                    f"The optimizer has more than one state variable named {duplicates}, so its "
                    "state cannot be snapshotted or restored unambiguously. Give each transform "
                    "in the chain state variables with distinct names."
                )
            state.update(stage_state)
        return state

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
    pytensor_init = _chain_pt_init(*(t.pytensor_init for t in transforms))

    return GradientTransformation(init, update, pytensor, pytensor_init)


def clip_by_global_norm(max_norm: float) -> GradientTransformation:
    """Clip the gradients so that their global L2 norm does not exceed ``max_norm``."""

    def init(params):
        return None

    def update(updates, state, params=None):
        global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in updates.values()))
        scale = np.minimum(1.0, max_norm / (global_norm + 1e-12))
        return {name: g * scale for name, g in updates.items()}, state

    def _pytensor_impl(grads, shared_params, state):
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

    def _pytensor_init(shared_params):
        state = {"adam_t": pytensor.shared(np.zeros((), dtype="int64"), name="adam_t")}
        for param in shared_params:
            for moment in ("m", "v"):
                name = f"adam_{moment}_{param.name}"
                state[name] = pytensor.shared(_zeros_like_shared(param), name=name)
        return state

    def _pytensor_impl(grads, shared_params, state):
        t = state["adam_t"]
        t_new = t + 1
        t_new_float = t_new.astype(config.floatX)
        updates = {t: t_new}
        new_grads = []
        for param, grad in zip(shared_params, grads):
            m = state[f"adam_m_{param.name}"]
            v = state[f"adam_v_{param.name}"]
            m_new = b1 * m + (1 - b1) * grad
            v_new = b2 * v + (1 - b2) * pt.square(grad)
            m_hat = m_new / (1 - b1**t_new_float)
            v_hat = v_new / (1 - b2**t_new_float)
            new_grads.append(m_hat / (pt.sqrt(v_hat) + eps))
            updates.update({m: m_new, v: v_new})
        return new_grads, updates

    return GradientTransformation(init, update, _pytensor_impl, _pytensor_init)


def scale(step_size: float) -> GradientTransformation:
    """Scale the gradients by ``-step_size``."""

    def init(params):
        return None

    def update(updates, state, params=None):
        return {name: -step_size * g for name, g in updates.items()}, state

    def _pytensor_impl(grads, shared_params, state):
        return [g * (-step_size) for g in grads], {}

    return GradientTransformation(init, update, _pytensor_impl)


def scale_by_schedule(step_size_fn: Schedule) -> GradientTransformation:
    """Scale the gradients by ``-step_size_fn(count)``, where ``step_size_fn`` is a schedule."""

    def init(params):
        return {"count": 0}

    def update(updates, state, params=None):
        count = state["count"]
        lr = step_size_fn(pt.constant(count, dtype="int64")).eval()
        return {name: -lr * g for name, g in updates.items()}, {"count": count + 1}

    def _pytensor_init(shared_params):
        return {"lr_t": pytensor.shared(np.zeros((), dtype="int64"), name="lr_t")}

    def _pytensor_impl(grads, shared_params, state):
        t = state["lr_t"]
        lr = step_size_fn(t)
        t_new = t + 1
        return [g * (-lr) for g in grads], {t: t_new}

    return GradientTransformation(init, update, _pytensor_impl, _pytensor_init)


def scale_by_learning_rate(learning_rate: float | Schedule) -> GradientTransformation:
    """Scale the gradients by ``-learning_rate``, a constant or a schedule of the step count."""
    if callable(learning_rate):
        return scale_by_schedule(learning_rate)
    return scale(learning_rate)


def adam(
    learning_rate: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> GradientTransformation:
    """Adam optimizer."""
    return chain(scale_by_adam(b1=b1, b2=b2, eps=eps), scale_by_learning_rate(learning_rate))


def clipped_adam(
    learning_rate: float = 0.01, clip_norm: float = 10.0, **adam_kwargs
) -> GradientTransformation:
    """Adam with gradient clipping by global norm, as numpyro's ClippedAdam."""
    return chain(clip_by_global_norm(clip_norm), adam(learning_rate, **adam_kwargs))


def sgd(learning_rate: float = 0.01) -> GradientTransformation:
    """Stochastic gradient descent optimizer."""
    return scale_by_learning_rate(learning_rate)


def scale_by_rmsprop(decay: float = 0.9, eps: float = 1e-8) -> GradientTransformation:
    """Rescale the gradients by the RMSProp preconditioner (Tieleman & Hinton, 2012)."""

    def init(params):
        return {"avg_sq": {name: np.zeros_like(value) for name, value in params.items()}}

    def update(updates, state, params=None):
        avg_sq = state["avg_sq"]
        new_avg_sq = {}
        new_updates = {}
        for k, g in updates.items():
            v = avg_sq[k]
            v_new = decay * v + (1.0 - decay) * (g * g)
            new_avg_sq[k] = v_new
            new_updates[k] = g / (np.sqrt(v_new) + eps)
        return new_updates, {"avg_sq": new_avg_sq}

    def _pytensor_init(shared_params):
        return {
            f"rmsprop_v_{param.name}": pytensor.shared(
                _zeros_like_shared(param), name=f"rmsprop_v_{param.name}"
            )
            for param in shared_params
        }

    def _pytensor_impl(grads, shared_params, state):
        updates = {}
        new_grads = []
        for param, grad in zip(shared_params, grads):
            v = state[f"rmsprop_v_{param.name}"]
            v_new = decay * v + (1.0 - decay) * pt.square(grad)
            new_grads.append(grad / (pt.sqrt(v_new) + eps))
            updates[v] = v_new
        return new_grads, updates

    return GradientTransformation(init, update, _pytensor_impl, _pytensor_init)


def rmsprop(
    learning_rate: float = 0.01,
    decay: float = 0.9,
    eps: float = 1e-8,
) -> GradientTransformation:
    """RMSProp optimizer."""
    return chain(scale_by_rmsprop(decay=decay, eps=eps), scale_by_learning_rate(learning_rate))


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

    The returned schedule maps a symbolic step count to a symbolic learning rate, so it
    can be baked into the compiled step function.
    """
    init_value = peak_value / div_factor
    end_value = init_value / final_div_factor
    boundaries = np.array([0.0, pct_start, pct_final, 1.0]) * transition_steps
    values = np.array([init_value, peak_value, init_value, end_value])

    return lambda count: pt.interp(count, boundaries, values)
