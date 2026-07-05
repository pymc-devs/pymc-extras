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


def test_fit_streams_batches_into_data():
    rng = np.random.default_rng(0)

    def batches():
        while True:
            yield {"batch": rng.normal(1.0, 1.0, size=64)}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        batch = pm.Data("batch", np.zeros(64))
        pm.Normal("y", theta, 1, observed=batch)

    trainer = Trainer(model=model, convergence_window=None)
    trainer.fit(1_000, batches(), random_seed=1)
    idata = trainer.sample_posterior(1_000, random_seed=2)

    theta_draws = idata["posterior"].dataset["theta"].values.ravel()
    np.testing.assert_allclose(theta_draws.mean(), 1.0, atol=0.1)
    # The last batch remains on the model
    assert not np.array_equal(model["batch"].get_value(), np.zeros(64))


def test_fit_streams_observations_into_free_rv():
    rng = np.random.default_rng(0)

    def batches():
        while True:
            yield {"y": rng.normal(1.0, 1.0, size=64)}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(64,))

    trainer = Trainer(model=model, convergence_window=None)
    trainer.fit(1_000, batches(), observeds=["y"], random_seed=1)
    idata = trainer.sample_posterior(1_000, random_seed=2)

    # y was observed, so the posterior contains only theta
    assert set(idata["posterior"].dataset.data_vars) == {"theta"}
    theta_draws = idata["posterior"].dataset["theta"].values.ravel()
    np.testing.assert_allclose(theta_draws.mean(), 1.0, atol=0.1)
    # The user's model is untouched
    assert "y" not in [rv.name for rv in model.observed_RVs]


def test_fit_rescales_likelihood_when_stream_has_len():
    rng = np.random.default_rng(0)
    full_data = rng.normal(1.0, 1.0, size=1_000)

    class Loader:
        # A torch-style dataloader: yields minibatches, len is the dataset size N
        def __len__(self):
            return full_data.shape[0]

        def __iter__(self):
            while True:
                idx = rng.integers(full_data.shape[0], size=50)
                yield {"y": full_data[idx]}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("y", theta, 1, shape=(50,))

    trainer = Trainer(model=model, convergence_window=None)
    trainer.fit(3_000, Loader(), observeds=["y"], random_seed=1)
    idata = trainer.sample_posterior(2_000, random_seed=2)
    theta_draws = idata["posterior"].dataset["theta"].values.ravel()

    # Reference: the same fit with the full dataset observed at once
    with pm.Model() as full_model:
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("y", theta, 1, observed=full_data)
    ref = fit_advi(
        model=full_model, n_steps=3_000, draws=2_000, convergence_window=None, random_seed=1
    )
    ref_draws = ref["posterior"].dataset["theta"].values.ravel()

    post_mean = full_data.sum() / (1 + full_data.size)
    np.testing.assert_allclose(theta_draws.mean(), post_mean, atol=0.1)
    # With the N / batch_rows rescaling the minibatch fit matches the full-data
    # fit, not the unscaled 50-row batch posterior (whose std would be ~0.14)
    np.testing.assert_allclose(theta_draws.std(), ref_draws.std(), rtol=0.25)
    assert theta_draws.std() < 0.1


def test_fit_stops_when_stream_runs_out():
    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(4,))

    data = ({"y": np.ones(4)} for _ in range(5))
    trainer = Trainer(model=model, convergence_window=None)
    state = trainer.fit(1_000, data, observeds=["y"], random_seed=1)

    assert state.step == 5


def test_fit_observeds_without_data_raises():
    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(4,))

    with pytest.raises(ValueError, match="observeds requires a data iterator"):
        Trainer(model=model).fit(10, observeds=["y"])


def test_discrete_free_rv_raises():
    with pm.Model() as model:
        z = pm.Bernoulli("z", 0.5)
        pm.Normal("y", mu=z, sigma=1, observed=[0.9])

    with pytest.raises(ValueError, match="continuous"):
        AutoDiagonalNormal(model)
