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
    scale_by_learning_rate,
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


def test_clipped_adam_with_schedule_runs():
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    optimizer = clipped_adam(schedule)
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

    np.testing.assert_allclose(schedule(0), 0.01 / 25)
    np.testing.assert_allclose(schedule(200), 0.01)  # peak at pct_start
    assert schedule(100) < schedule(200)  # warmup
    assert schedule(500) < schedule(200)  # anneal
    assert schedule(1000) < schedule(0)  # final decay below init


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


def test_schedule_has_no_pytensor():
    """A learning-rate schedule cannot be baked into the graph, so pytensor is None."""
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    opt = scale_by_learning_rate(schedule)
    assert opt.pytensor is None


def test_chain_with_schedule_has_no_pytensor():
    """If any link in the chain lacks a PyTensor impl, the whole chain's pytensor is None."""
    schedule = linear_onecycle_schedule(transition_steps=100, peak_value=0.1)
    opt = chain(clip_by_global_norm(1.0), adam(0.1), scale_by_learning_rate(schedule))
    assert opt.pytensor is None
