import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_extras.inference.advi.optimizers import (
    adam,
    apply_updates,
    chain,
    clip_by_global_norm,
    clipped_adam,
    linear_onecycle_schedule,
    rmsprop,
    scale_by_adam,
    scale_by_learning_rate,
    scale_by_rmsprop,
    sgd,
)


def test_adam_minimizes_quadratic():
    optimizer = adam(0.1)
    params = {"x": np.array(5.0), "y": np.array([-3.0, 2.0])}
    state = optimizer.init(params)

    for _ in range(500):
        grads = {name: 2 * value for name, value in params.items()}  # d/dx of x**2
        updates, state = optimizer.update(grads, state, params)
        params = apply_updates(params, updates)

    np.testing.assert_allclose(params["x"], 0.0, atol=1e-3)
    np.testing.assert_allclose(params["y"], 0.0, atol=1e-3)


def test_clip_by_global_norm():
    transform = clip_by_global_norm(1.0)
    grads = {"x": np.array([3.0]), "y": np.array([4.0])}  # global norm 5

    updates, _ = transform.update(grads, transform.init(grads))

    global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in updates.values()))
    np.testing.assert_allclose(global_norm, 1.0, rtol=1e-6)
    # Direction is preserved
    np.testing.assert_allclose(updates["x"] / updates["y"], 3 / 4, rtol=1e-6)

    # Gradients under the norm pass through untouched
    small = {"x": np.array([0.3]), "y": np.array([0.4])}
    updates, _ = transform.update(small, transform.init(small))
    np.testing.assert_allclose(updates["x"], 0.3)


def test_schedule_composed_with_adam_runs():
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    optimizer = chain(clip_by_global_norm(10.0), scale_by_adam(), scale_by_learning_rate(schedule))
    params = {"x": np.array(5.0)}
    state = optimizer.init(params)

    for _ in range(100):
        updates, state = optimizer.update({"x": 2 * params["x"]}, state, params)
        params = apply_updates(params, updates)

    assert abs(params["x"]) < 5.0


def test_linear_onecycle_schedule_shape():
    schedule = linear_onecycle_schedule(
        transition_steps=1000, peak_value=0.01, pct_start=0.2, div_factor=25.0
    )

    lrs = schedule(pt.constant([0, 100, 200, 500, 1000], dtype="int64")).eval()

    np.testing.assert_allclose(lrs[0], 0.01 / 25)
    np.testing.assert_allclose(lrs[2], 0.01)  # peak at pct_start
    assert lrs[1] < lrs[2]  # warmup
    assert lrs[3] < lrs[2]  # anneal
    assert lrs[4] < lrs[0]  # final decay below init


def _make_quadratic_step(optimizer, init_value=5.0):
    """Compile a single step that minimises f(x) = x² with the given optimizer."""
    x = pytensor.shared(np.array(init_value), name="x")
    loss = x**2
    grads = pt.grad(loss, wrt=[x])
    new_grads, updates = optimizer.pytensor(grads, [x])
    updates[x] = x + new_grads[0]
    return pytensor.compile.function(inputs=[], outputs=loss, updates=updates)


def test_adam_pytensor_minimizes_quadratic():
    step = _make_quadratic_step(adam(0.1))
    for _ in range(500):
        step()
    assert abs(step()) < 1e-3


def test_clipped_adam_pytensor_minimizes_quadratic():
    step = _make_quadratic_step(clipped_adam(0.1))
    for _ in range(500):
        step()
    assert abs(step()) < 1e-3


def test_sgd_pytensor_minimizes_quadratic():
    step = _make_quadratic_step(sgd(0.1))
    for _ in range(500):
        step()
    assert abs(step()) < 1e-3


def test_chain_pytensor_composes():
    """chain() composes PyTensor implementations: clip + adam."""
    opt = chain(clip_by_global_norm(1.0), adam(0.1))
    assert opt.pytensor is not None
    step = _make_quadratic_step(opt, init_value=3.0)
    for _ in range(500):
        step()
    assert abs(step()) < 1e-3


def test_schedule_has_pytensor():
    """A schedule is evaluated against a step counter, so it has a PyTensor impl."""
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    opt = scale_by_learning_rate(schedule)
    assert opt.pytensor is not None


def test_chain_with_schedule_has_pytensor():
    """A schedule composes with other transforms in a chain."""
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    opt = chain(clip_by_global_norm(1.0), scale_by_adam(), scale_by_learning_rate(schedule))
    assert opt.pytensor is not None


def test_schedule_pytensor_follows_schedule():
    """The compiled schedule evaluates the learning rate against the step counter."""
    schedule = linear_onecycle_schedule(transition_steps=10, peak_value=0.1)
    opt = scale_by_learning_rate(schedule)

    x = pytensor.shared(np.array(1.0), name="x")
    new_grads, updates = opt.pytensor([pt.constant(1.0)], [x])
    # new_grads[0] = -lr * 1.0, so -new_grads[0] is the learning rate
    lr = -new_grads[0]
    updates[x] = x + new_grads[0]
    step = pytensor.compile.function(inputs=[], outputs=lr, updates=updates)

    lrs = [float(step()) for _ in range(10)]
    expected = schedule(pt.arange(10, dtype="int64")).eval()
    np.testing.assert_allclose(lrs, expected)


def test_rmsprop_pytensor_minimizes_quadratic():
    """RMSProp with a PyTensor impl should reduce a simple quadratic.

    RMSProp can oscillate on deterministic quadratics because the effective
    step size ``lr * g / sqrt(v)`` becomes constant when ``v`` tracks ``g^2``.
    We therefore only check that the loss decreases substantially.
    """
    step = _make_quadratic_step(rmsprop(0.1))
    for _ in range(500):
        step()
    assert abs(step()) < 0.1


def test_scale_by_rmsprop_has_pytensor():
    """scale_by_rmsprop provides a PyTensor implementation."""
    opt = scale_by_rmsprop(decay=0.9, eps=1e-8)
    assert opt.pytensor is not None
