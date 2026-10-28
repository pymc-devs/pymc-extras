import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import adam, chain, clip_by_global_norm, rmsprop, scale_by_schedule, sgd

from pymc_extras.inference.advi import Trainer, fit_advi, linear_onecycle_shape
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

    kwargs = dict(model=model, n_steps=200, draws=100)
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

    kwargs = dict(model=model, n_steps=50, draws=50, backend="jax")
    draws_a = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_b = fit_advi(random_seed=42, **kwargs)["posterior"].dataset["theta"].values
    draws_c = fit_advi(random_seed=13, **kwargs)["posterior"].dataset["theta"].values

    np.testing.assert_array_equal(draws_a, draws_b)
    assert not np.array_equal(draws_a, draws_c)


def test_fit_continues(conjugate_model):
    model, *_ = conjugate_model

    trainer = Trainer(random_seed=0)
    with model:
        first = trainer.fit(100, random_seed=1)
        second = trainer.fit(100, random_seed=1)

    # fit continues rather than starting over: the step count and history accumulate,
    # and the parameters keep moving
    assert (first.step, second.step) == (100, 200)
    assert (len(first.loss_history), len(second.loss_history)) == (100, 200)
    np.testing.assert_array_equal(second.loss_history[:100], first.loss_history)
    assert not np.allclose(first.params["theta_loc"], second.params["theta_loc"])


@pytest.mark.parametrize(
    "make_optimizer",
    [
        sgd,
        rmsprop,
        adam,
        # a schedule keeps its clock outside the rule's own updates, so this case fails
        # if collect_clock_updates stops routing schedule clocks into the saved state
        pytest.param(
            lambda lr: chain(adam(lr), scale_by_schedule(linear_onecycle_shape(200))),
            id="scheduled_adam",
        ),
    ],
)
def test_snapshot_restore_is_optimizer_agnostic(conjugate_model, make_optimizer):
    model, *_ = conjugate_model
    optimizer = make_optimizer(0.01)

    trainer = Trainer(optimizer=optimizer)
    with model:
        first = trainer.fit(50, random_seed=1)
        second = trainer.fit(50, random_seed=1)

    resumed = Trainer(optimizer=optimizer)
    with model:
        resumed_state = resumed.fit(50, state=first, random_seed=1)

    for name in first.params:
        np.testing.assert_allclose(resumed_state.params[name], second.params[name])
    for name in second.optimizer_state:
        np.testing.assert_allclose(
            resumed_state.optimizer_state[name], second.optimizer_state[name]
        )
    np.testing.assert_allclose(resumed_state.loss_history, second.loss_history)


def test_duplicate_optimizer_state_names_are_refused(conjugate_model):
    model, *_ = conjugate_model
    schedule = linear_onecycle_shape(100)

    # both stages allocate a clock named "scale_by_schedule/step_count", so keying the
    # snapshot by name would keep one and resume the other from whatever it happened
    # to hold
    doubled = chain(adam(0.01), scale_by_schedule(schedule), scale_by_schedule(schedule))

    # the collision is detected while compiling, before any step is taken
    trainer = Trainer(optimizer=doubled)
    with model, pytest.raises(ValueError, match="more than one state variable named"):
        trainer.fit(1)


def test_posterior_draws_are_named_for_their_own_variable():
    # distinct shapes, so draws attached to the wrong name would be the wrong shape
    with pm.Model() as model:
        pm.Normal("scalar")
        pm.Normal("pair", shape=2)
        pm.HalfNormal("positive")
        pm.Normal("triple", shape=3)
        pm.Normal("y", 0, 1, observed=[1.0, 0.5])

    trainer = Trainer(random_seed=0)
    with model:
        trainer.fit(10, random_seed=1)
        idata = trainer.sample_posterior(draws=7, random_seed=2)

    posterior = idata["posterior"].dataset
    assert set(posterior.data_vars) == {"scalar", "pair", "positive", "triple"}
    assert posterior["scalar"].shape == (1, 7)
    assert posterior["pair"].shape == (1, 7, 2)
    assert posterior["positive"].shape == (1, 7)
    assert posterior["triple"].shape == (1, 7, 3)
    # the transform is applied to the variable that carries it, not to a sibling
    assert (posterior["positive"].values > 0).all()


def test_the_optimizer_is_fixed_at_construction(conjugate_model):
    model, *_ = conjugate_model
    supplied = sgd(0.01)

    default = Trainer(random_seed=0)
    chosen = Trainer(optimizer=supplied, random_seed=0)

    # both are resolved before any fit, so the properties describe the trainer from the start
    assert chosen.optimizer is supplied
    assert default.optimizer is not None
    default_before = default.optimizer

    with model:
        default.fit(10, random_seed=1)
        chosen.fit(10, random_seed=1)

    assert chosen.optimizer is supplied
    assert default.optimizer is default_before


def test_trainer_state_is_complete_and_honest(conjugate_model):
    model, *_ = conjugate_model
    trainer = Trainer(random_seed=0)

    assert trainer.state is None

    with model:
        state = trainer.fit(50, random_seed=1)

    # the returned state is the trainer's state, and both read the live shared variables
    for name, value in trainer.state.params.items():
        np.testing.assert_array_equal(value, state.params[name])
    assert set(state.params) == {"theta_loc", "theta_scale"}
    assert set(state.optimizer_state) == {
        "adam/step_count",
        "theta_loc/adam/first_moment",
        "theta_loc/adam/second_moment",
        "theta_scale/adam/first_moment",
        "theta_scale/adam/second_moment",
    }
    # the clock the bias correction reads travels with the state, so a resumed run picks
    # up where it left off rather than replaying the earliest, most heavily corrected steps
    assert state.optimizer_state["adam/step_count"] == 50

    # compile-time configuration is read-only rather than silently ignored
    with pytest.raises(AttributeError):
        trainer.n_particles = 32


def test_learning_rate_is_a_live_shared_value(conjugate_model):
    model, *_ = conjugate_model

    trainer = Trainer(random_seed=0)
    with model:
        frozen = trainer.fit(50, learning_rate=0.0, random_seed=1)

    # a zero rate leaves Adam's moments moving but the parameters untouched
    init_loc = trainer.guide.params_init_values[trainer.guide["theta_loc"]]
    np.testing.assert_allclose(frozen.params["theta_loc"], init_loc)
    assert trainer.learning_rate == 0.0

    # and fit can change it without rebuilding the step function, schedule and all
    step_fn = trainer._step_fn
    with model:
        moved = trainer.fit(50, learning_rate=0.05, random_seed=1)
    assert trainer.learning_rate == 0.05
    assert trainer._step_fn is step_fn
    assert not np.allclose(moved.params["theta_loc"], frozen.params["theta_loc"])


def test_a_supplied_optimizer_owns_its_own_rate(conjugate_model):
    model, *_ = conjugate_model

    supplied = Trainer(optimizer=rmsprop(1e-3), random_seed=0)
    with model:
        supplied.fit(10, random_seed=1)
    assert supplied.learning_rate is None

    # _step_fn is the probe for "was compiled"; the fitted trainer confirms fit sets it
    assert supplied._step_fn is not None

    # the bad learning_rate is refused before the step function is built, so the error
    # does not cost a compilation
    fresh = Trainer(optimizer=rmsprop(1e-3), random_seed=0)
    with model, pytest.raises(ValueError, match="belongs to the optimizer"):
        fresh.fit(10, learning_rate=0.01)
    assert fresh._step_fn is None


def test_fit_rejects_invalid_n_and_learning_rate(conjugate_model):
    model, *_ = conjugate_model

    with model:
        with pytest.raises(ValueError, match="positive integer"):
            Trainer().fit(0)

        with pytest.raises(ValueError, match="non-negative"):
            Trainer().fit(10, learning_rate=-1.0)


def test_a_composed_rule_shapes_the_rate_it_is_given(conjugate_model):
    model, *_ = conjugate_model
    rate = 0.05

    with model:
        flat = Trainer(optimizer=adam(rate), random_seed=0).fit(200, random_seed=1)

        # scale_by_schedule multiplies a unit-peak shape onto the step, so an annealing
        # shape has to move the parameters somewhere a bare rule does not reach
        annealed = Trainer(
            optimizer=chain(adam(rate), scale_by_schedule(linear_onecycle_shape(200))),
            random_seed=0,
        ).fit(200, random_seed=1)
    assert not np.allclose(annealed.params["theta_loc"], flat.params["theta_loc"])


def test_clipping_binds_only_on_large_gradients(conjugate_model):
    model, *_ = conjugate_model
    rate = 0.05

    with model:
        unclipped = Trainer(optimizer=adam(rate), random_seed=0).fit(100, random_seed=1)

        # a bound no gradient reaches is a no-op, so it must not perturb the run at all
        slack_rule = chain(clip_by_global_norm(1e12), adam(rate))
        slack = Trainer(optimizer=slack_rule, random_seed=0).fit(100, random_seed=1)
        np.testing.assert_allclose(slack.params["theta_loc"], unclipped.params["theta_loc"])
        np.testing.assert_allclose(slack.loss_history, unclipped.loss_history)

        # a bound that does bind changes the trajectory, since it enters Adam's moments
        tight_rule = chain(clip_by_global_norm(1e-4), adam(rate))
        tight = Trainer(optimizer=tight_rule, random_seed=0).fit(100, random_seed=1)
    assert not np.allclose(tight.params["theta_loc"], unclipped.params["theta_loc"])


def test_linear_onecycle_shape_matches_the_reference_curve():
    pct_start, pct_final, div_factor, final_div_factor = 0.2, 0.85, 25.0, 1e4
    transition_steps = 1_000

    # the three phases interpolated directly, as optax and torch define 1cycle
    breakpoints = np.array([0.0, pct_start, pct_final, 1.0]) * transition_steps
    values = np.array([1 / div_factor, 1.0, 1 / div_factor, 1 / div_factor / final_div_factor])

    step = pt.iscalar("step")
    shape = pytensor.function([step], linear_onecycle_shape(transition_steps)(step))

    probes = np.array([0, 1, 199, 200, 201, 500, 849, 850, 999, 1_000, 1_500])
    np.testing.assert_allclose(
        [shape(s) for s in probes], np.interp(probes, breakpoints, values), atol=1e-12
    )


def test_linear_onecycle_shape_accepts_a_symbolic_horizon():
    step = pt.iscalar("step")
    horizon = pytensor.shared(np.asarray(1_000, dtype="int64"), name="horizon")
    shape = pytensor.function([step], linear_onecycle_shape(horizon)(step))

    np.testing.assert_allclose(shape(200), 1.0)  # peaks at pct_start of the horizon
    horizon.set_value(np.asarray(500, dtype="int64"))
    np.testing.assert_allclose(shape(100), 1.0)  # and the peak moves with it


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(pct_start=0.0), id="empty_ramp"),
        pytest.param(dict(pct_final=1.0), id="empty_decay"),
        pytest.param(dict(pct_start=0.5, pct_final=0.5), id="empty_anneal"),
        pytest.param(dict(pct_start=0.9, pct_final=0.5), id="inverted_phases"),
    ],
)
def test_linear_onecycle_shape_rejects_degenerate_phases(kwargs):
    # a phase of zero or negative length divides by zero and yields a plausible-looking
    # but wrong curve, so it has to be refused rather than built
    with pytest.raises(ValueError, match="0 < pct_start < pct_final < 1"):
        linear_onecycle_shape(1_000, **kwargs)


def test_linear_onecycle_shape_rejects_a_horizon_with_no_room():
    with pytest.raises(ValueError, match="at least 1"):
        linear_onecycle_shape(0)


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
        trainer = Trainer(guide=naive_guide)
        trainer.fit(10)

    assert set(model.named_vars) == {"mu", "y"}


def test_discrete_free_rv_raises():
    with pm.Model() as model:
        z = pm.Bernoulli("z", 0.5)
        pm.Normal("y", mu=z, sigma=1, observed=[0.9])

    with pytest.raises(ValueError, match="continuous"):
        AutoDiagonalNormal(model)


def test_fit_streams_batches_into_data():
    rng = np.random.default_rng(0)

    def batches():
        while True:
            yield {"batch": rng.normal(1.0, 1.0, size=64)}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        batch = pm.Data("batch", np.zeros(64))
        pm.Normal("y", theta, 1, observed=batch)

    trainer = Trainer()
    with model:
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

    trainer = Trainer()
    with model:
        trainer.fit(1_000, batches(), observeds=["y"], random_seed=1)
        idata = trainer.sample_posterior(1_000, random_seed=2)

    # y was observed, so the posterior contains only theta
    assert set(idata["posterior"].dataset.data_vars) == {"theta"}
    theta_draws = idata["posterior"].dataset["theta"].values.ravel()
    np.testing.assert_allclose(theta_draws.mean(), 1.0, atol=0.1)
    # The user's model is untouched
    assert "y" not in [rv.name for rv in model.observed_RVs]


def test_fit_after_streaming_refuses_to_reuse_the_last_batch():
    rng = np.random.default_rng(0)

    def batches():
        while True:
            yield {"y": rng.normal(1.0, 1.0, size=64)}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(64,))

    trainer = Trainer()
    with model:
        streamed = trainer.fit(10, batches(), observeds=["y"], random_seed=1)

        # the stream shareds still hold the last batch, so a bare fit would train on it
        with pytest.raises(ValueError, match="keep training on the last batch"):
            trainer.fit(10)

        # observeds is baked into the observed model built by the first streamed fit
        with pytest.raises(ValueError, match="observeds is fixed by the first streamed fit"):
            trainer.fit(10, batches(), observeds=["y"])

        # streaming on without repeating observeds continues the same run
        continued = trainer.fit(10, batches())

    assert (streamed.step, continued.step) == (10, 20)
    assert not np.allclose(continued.params["theta_loc"], streamed.params["theta_loc"])


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

    trainer = Trainer()
    with model:
        trainer.fit(3_000, Loader(), observeds=["y"], random_seed=1)
        idata = trainer.sample_posterior(2_000, random_seed=2)
    theta_draws = idata["posterior"].dataset["theta"].values.ravel()

    # Reference: the same fit with the full dataset observed at once
    with pm.Model() as full_model:
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("y", theta, 1, observed=full_data)
    ref = fit_advi(model=full_model, n_steps=3_000, draws=2_000, random_seed=1)
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
    trainer = Trainer()
    with model:
        state = trainer.fit(1_000, data, observeds=["y"], random_seed=1)

    assert state.step == 5


def test_fit_observeds_without_data_raises():
    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(4,))

    with pytest.raises(ValueError, match="observeds requires a data iterator"):
        with model:
            Trainer().fit(10, observeds=["y"])


def test_sample_posterior_with_explicit_model_after_streaming():
    """sample_posterior uses the active model context when no model argument is given."""
    rng = np.random.default_rng(0)

    def batches():
        while True:
            yield {"y": rng.normal(1.0, 1.0, size=64)}

    with pm.Model() as model:
        theta = pm.Normal("theta", 0, 10)
        pm.Normal("y", theta, 1, shape=(64,))

    trainer = Trainer()
    with model:
        trainer.fit(500, batches(), observeds=["y"], random_seed=1)

    with model:
        idata = trainer.sample_posterior(500, random_seed=2)

    assert set(idata["posterior"].dataset.data_vars) == {"theta"}
    theta_draws = idata["posterior"].dataset["theta"].values.ravel()
    np.testing.assert_allclose(theta_draws.mean(), 1.0, atol=0.15)
    # The user's model is untouched
    assert "y" not in [rv.name for rv in model.observed_RVs]
