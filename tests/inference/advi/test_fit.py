import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_extras.inference.advi import Trainer, fit_advi
from pymc_extras.inference.advi.autoguide import AutoDiagonalNormal, AutoGuideModel


@pytest.fixture
def conjugate_model():
    # Normal-Normal conjugate model with a known posterior
    obs = np.array([1.0, 0.5, 1.5, 1.0])
    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("y", theta, 1, observed=obs)
    post_var = 1 / (1 + obs.size)
    post_mean = post_var * obs.sum()
    return model, post_mean, post_var


def test_fit_advi_recovers_conjugate_posterior(conjugate_model):
    model, post_mean, post_var = conjugate_model

    idata = fit_advi(model=model, n_steps=5_000, draws=2_000, random_seed=1)

    theta = idata["posterior"].dataset["theta"].values.ravel()
    np.testing.assert_allclose(theta.mean(), post_mean, atol=0.1)
    np.testing.assert_allclose(theta.std(), np.sqrt(post_var), rtol=0.25)


def test_fit_advi_random_seed(conjugate_model):
    model, *_ = conjugate_model

    kwargs = dict(model=model, n_steps=200, draws=100, convergence_window=None)
    draws_a = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_b = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_c = fit_advi(random_seed=13, **kwargs)["posterior"].dataset["theta"].values

    np.testing.assert_array_equal(draws_a, draws_b)
    assert not np.array_equal(draws_a, draws_c)


@pytest.mark.filterwarnings("ignore:The RandomType SharedVariables")
def test_fit_advi_random_seed_jax(conjugate_model):
    # The JAX linker replaces RNG shared variables with internal copies at compile time,
    # so seeding must reach the compiled function's own storage
    pytest.importorskip("jax")
    model, *_ = conjugate_model

    kwargs = dict(model=model, n_steps=50, draws=50, convergence_window=None, backend="jax")
    draws_a = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_b = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_c = fit_advi(random_seed=13, **kwargs)["posterior"].dataset["theta"].values

    np.testing.assert_array_equal(draws_a, draws_b)
    assert not np.array_equal(draws_a, draws_c)


def test_fit_advi_early_stopping(conjugate_model):
    model, *_ = conjugate_model

    idata = fit_advi(
        model=model,
        n_steps=1_000,
        random_seed=1,
        convergence_window=50,
        relative_tolerance=10.0,
    )

    # With a huge tolerance, training stops at the first convergence check
    assert idata["fit"].dataset.sizes["step"] == 100


def test_fit_continues_and_reset_starts_over(conjugate_model):
    model, *_ = conjugate_model
    kwargs = dict(model=model, convergence_window=None, learning_rate=0.01, random_seed=0)

    trainer = Trainer(**kwargs)
    first = trainer.fit(100, random_seed=1)
    second = trainer.fit(100, random_seed=1)

    # fit continues rather than starting over: the step count and history accumulate,
    # and the parameters keep moving
    assert (first.step, second.step) == (100, 200)
    assert (len(first.loss_history), len(second.loss_history)) == (100, 200)
    np.testing.assert_array_equal(second.loss_history[:100], first.loss_history)
    assert not np.allclose(first.params["theta_loc"], second.params["theta_loc"])

    # a fresh trainer resuming from the snapshot matches continuing in place, which
    # requires the Adam moments to travel with the state
    resumed = Trainer(**kwargs)
    resumed.load_state(first)
    resumed_state = resumed.fit(100, random_seed=1)
    np.testing.assert_allclose(resumed_state.params["theta_loc"], second.params["theta_loc"])
    np.testing.assert_allclose(resumed_state.loss_history, second.loss_history)

    # reset returns the trainer to its initial state, so the run repeats exactly
    trainer.reset()
    assert trainer.step == 0
    assert trainer.state.loss_history.size == 0
    repeated = trainer.fit(100, random_seed=1)
    np.testing.assert_allclose(repeated.params["theta_loc"], first.params["theta_loc"])
    np.testing.assert_allclose(repeated.loss_history, first.loss_history)


def test_trainer_state_is_complete_and_honest(conjugate_model):
    model, *_ = conjugate_model
    trainer = Trainer(model=model, convergence_window=None, random_seed=0)

    with pytest.raises(RuntimeError, match="not been fitted"):
        trainer.state

    state = trainer.fit(50, random_seed=1)

    # the returned state is the trainer's state, and both read the live shared variables
    for name, value in trainer.state.params.items():
        np.testing.assert_array_equal(value, state.params[name])
    assert set(state.params) == {"theta_loc", "theta_scale"}
    assert set(state.optimizer_state) == {
        "adam_t",
        "adam_m_theta_loc",
        "adam_v_theta_loc",
        "adam_m_theta_scale",
        "adam_v_theta_scale",
    }
    assert state.optimizer_state["adam_t"] == 50

    # compile-time configuration is read-only rather than silently ignored
    with pytest.raises(AttributeError):
        trainer.n_particles = 32


def test_fit_learning_rate_override(conjugate_model):
    model, *_ = conjugate_model

    trainer = Trainer(model=model, convergence_window=None, learning_rate=0.0, random_seed=0)
    frozen = trainer.fit(50, random_seed=1)
    # a zero learning rate leaves Adam's moments moving but the parameters untouched
    np.testing.assert_allclose(frozen.params["theta_loc"], trainer._init_state.params["theta_loc"])

    trainer.reset()
    moved = trainer.fit(50, learning_rate=0.05, random_seed=1)
    assert not np.allclose(moved.params["theta_loc"], frozen.params["theta_loc"])


def test_guide_initialized_at_initial_point():
    with pm.Model() as model:
        pm.LogNormal("x", mu=np.log(4.5), sigma=0.5)
        pm.Normal("y", 0, 1, shape=(2,), initval=np.array([1.5, -0.5]))

    guide = AutoDiagonalNormal(model)
    initial_point = model.initial_point()

    np.testing.assert_array_equal(
        guide.params_init_values[guide["x_loc"]], initial_point["x_log__"]
    )
    np.testing.assert_array_equal(guide.params_init_values[guide["y_loc"]], [1.5, -0.5])


def test_guide_built_inside_model_context():
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
        guide = AutoDiagonalNormal(model)

    # The guide must not register itself as a nested submodel
    assert set(model.named_vars) == {"mu"}
    assert set(guide.model.named_vars) == {"mu", "mu_z"}


def test_naive_custom_guide_does_not_leak_into_user_model():
    def naive_guide(model):
        # Written without the Model(model=None) idiom, as a user naturally would
        loc, scale = pt.scalar("mu_loc"), pt.scalar("mu_scale")
        with pm.Model() as guide_model:
            z = pm.Normal("mu_z")
            pm.Deterministic("mu", loc + pt.softplus(scale) * z)
        return AutoGuideModel(guide_model, {loc: np.array(0.0), scale: np.array(0.1)})

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=[0.5])
        trainer = Trainer(guide=naive_guide, convergence_window=None)
        trainer.fit(10)

    assert set(model.named_vars) == {"mu", "y"}


def test_discrete_free_rv_raises():
    with pm.Model() as model:
        z = pm.Bernoulli("z", 0.5)
        pm.Normal("y", mu=z, sigma=1, observed=[0.9])

    with pytest.raises(ValueError, match="continuous"):
        AutoDiagonalNormal(model)
