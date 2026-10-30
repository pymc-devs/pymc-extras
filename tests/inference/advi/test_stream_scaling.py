import warnings

import numpy as np
import pymc as pm
import pytest

from pymc_extras.inference.advi import Trainer, sgd
from pymc_extras.variational.dataloader import DataLoader


class SizedBatches(list):
    def __init__(self, batches, total_size):
        super().__init__(batches)
        self.total_size = total_size


def streamed_normal_model(*, total_size=None, fixed_observations=None):
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        batch = pm.Data("batch", np.zeros(3))
        pm.Normal("y", mu, 1, observed=batch, total_size=total_size)
        if fixed_observations is not None:
            pm.Normal("fixed_y", mu, 1, observed=fixed_observations)
    return model


def full_normal_model(observations, *, fixed_observations=None):
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        pm.Normal("y", mu, 1, observed=observations)
        if fixed_observations is not None:
            pm.Normal("fixed_y", mu, 1, observed=fixed_observations)
    return model


def trainer():
    # Nonzero updates make the comparisons check gradients as well as loss values.
    return Trainer(optimizer=sgd(0.001), random_seed=12)


def assert_same_training(actual, reference):
    assert actual.step == reference.step
    np.testing.assert_allclose(actual.loss_history, reference.loss_history, rtol=1e-6, atol=1e-6)
    assert actual.params.keys() == reference.params.keys()
    for name in actual.params:
        np.testing.assert_allclose(
            actual.params[name], reference.params[name], rtol=1e-6, atol=1e-6
        )


def assert_unchanged_training(fitted, previous):
    current = fitted._snapshot()
    assert current.step == previous.step
    np.testing.assert_array_equal(current.loss_history, previous.loss_history)
    assert current.params.keys() == previous.params.keys()
    for name in previous.params:
        np.testing.assert_array_equal(current.params[name], previous.params[name])


def total_size_warnings(recorded):
    return [warning for warning in recorded if "total_size" in str(warning.message)]


def test_missing_total_size_warns_for_unscaled_streamed_observations():
    model = streamed_normal_model(fixed_observations=np.array([-0.5, 0.5]))
    data = [{"batch": np.full(3, 0.25)}]

    with pytest.warns(UserWarning, match="total_size"):
        state = trainer().fit(1, data=data, model=model, random_seed=42)

    # With no N, only the current batch enters this likelihood, after the warning.
    reference_model = full_normal_model(np.full(3, 0.25), fixed_observations=np.array([-0.5, 0.5]))
    reference = trainer().fit(1, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


def test_per_rv_total_size_needs_no_warning_or_trainer_scaling_guard():
    model = streamed_normal_model(total_size=12, fixed_observations=np.array([-0.5, 0.5]))
    data = [{"batch": np.full(3, 0.25)}]
    fitted = trainer()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        fitted.fit(1, data=data, model=model, random_seed=42)
        # The model owns all scaling, so a Trainer-level N is not a new objective.
        state = fitted.fit(1, data=data, model=model, total_size=999)

    assert not total_size_warnings(recorded)
    reference_model = full_normal_model(np.full(12, 0.25), fixed_observations=np.array([-0.5, 0.5]))
    reference = trainer().fit(2, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


@pytest.mark.parametrize("total_size", [None, 20])
def test_explicitly_observed_rv_preserves_multi_axis_total_size(total_size):
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        pm.Normal("y", mu, 1, observed=np.full((3, 2), 0.25), total_size=(12, 10))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        state = trainer().fit(
            1,
            data=[{"y": np.full((3, 2), 0.5)}],
            observeds=["y"],
            model=model,
            total_size=total_size,
            random_seed=42,
        )

    assert not total_size_warnings(recorded)
    reference_model = full_normal_model(np.full((12, 10), 0.5))
    reference = trainer().fit(1, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


def test_compiled_ragged_steps_match_full_data_loss_and_updates():
    # Constant rows make N / batch_rows times each batch logp exactly equal to
    # the full-data logp, even when the second batch has a different length.
    model = streamed_normal_model()
    data = [{"batch": np.full(rows, 0.25)} for rows in (3, 5)]
    state = trainer().fit(2, data=data, model=model, total_size=20, random_seed=42)

    reference_model = full_normal_model(np.full(20, 0.25))
    reference = trainer().fit(2, model=reference_model, random_seed=42)

    assert state.step == 2
    assert model["batch"].get_value().shape == (5,)
    assert_same_training(state, reference)


def test_scalar_streamed_observation_counts_as_one_row():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        batch = pm.Data("batch", np.array(0.0))
        pm.Normal("y", mu, 1, observed=batch)

    state = trainer().fit(
        1, data=[{"batch": np.array(0.25)}], model=model, total_size=12, random_seed=42
    )
    reference_model = full_normal_model(np.full(12, 0.25))
    reference = trainer().fit(1, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


def test_real_dataloader_dict_adapter_uses_explicit_row_count():
    # Source blocks already have the requested size. Rebatching is a separate
    # DataLoader contract; this test exercises the real loader-to-Trainer path.
    loader = DataLoader([np.full(3, 0.25), np.full(3, 0.25)], batch_size=3, total_size=6)
    assert len(loader) == 2
    data = ({"batch": batch} for batch in loader)
    model = streamed_normal_model()

    # The generator adapter does not carry the loader's metadata.
    state = trainer().fit(2, data=data, model=model, total_size=loader.total_size, random_seed=42)
    reference_model = full_normal_model(np.full(6, 0.25))
    reference = trainer().fit(2, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


def test_mixed_likelihoods_preserve_per_rv_and_fixed_observations():
    fixed = np.array([-0.75, 1.25])
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        declared_data = pm.Data("declared_data", np.zeros(4))
        automatic_data = pm.Data("automatic_data", np.zeros(3))
        pm.Normal("declared_y", mu, 1, observed=declared_data, total_size=24)
        pm.Normal("automatic_y", mu, 1, observed=automatic_data)
        pm.Normal("fixed_y", mu, 1, observed=fixed)

    data = [{"declared_data": np.full(4, 0.5), "automatic_data": np.full(3, -0.25)}]
    state = trainer().fit(1, data=data, model=model, total_size=20, random_seed=42)

    # Independent full-data likelihoods encode the intended weights: the
    # per-RV declaration is 24, automatic scaling uses 20, and fixed data stay 2.
    with pm.Model() as reference_model:
        mu = pm.Normal("mu", 0, 2)
        pm.Normal("declared_y", mu, 1, observed=np.full(24, 0.5))
        pm.Normal("automatic_y", mu, 1, observed=np.full(20, -0.25))
        pm.Normal("fixed_y", mu, 1, observed=fixed)
    reference = trainer().fit(1, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


def test_resume_accepts_same_or_omitted_total_size():
    model = streamed_normal_model()
    data = [{"batch": np.full(3, 0.25)}]
    fitted = trainer()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", UserWarning)
        fitted.fit(1, data=data, model=model, total_size=20, random_seed=42)
        fitted.fit(1, data=data, model=model, total_size=20)
        state = fitted.fit(1, data=data, model=model)

    assert not total_size_warnings(recorded)
    reference_model = full_normal_model(np.full(20, 0.25))
    reference = trainer().fit(3, model=reference_model, random_seed=42)
    assert_same_training(state, reference)


@pytest.mark.parametrize("source", ["argument", "metadata"])
def test_resume_rejects_changed_total_size_before_training(source):
    model = streamed_normal_model()
    data = [{"batch": np.full(3, 0.25)}]
    fitted = trainer()
    state = fitted.fit(1, data=data, model=model, total_size=20, random_seed=42)
    previous_batch = model["batch"].get_value().copy()
    proposed_data = [{"batch": np.full(5, 9.0)}]

    kwargs = {"data": proposed_data, "total_size": 21}
    if source == "metadata":
        kwargs = {"data": SizedBatches(proposed_data, total_size=21)}
    with pytest.raises(ValueError) as error:
        fitted.fit(1, model=model, **kwargs)

    assert "total_size" in str(error.value)
    assert "new Trainer" in str(error.value)
    assert_unchanged_training(fitted, state)
    np.testing.assert_array_equal(model["batch"].get_value(), previous_batch)


def test_resume_rejects_new_total_size_after_unscaled_fit():
    model = streamed_normal_model()
    data = [{"batch": np.full(3, 0.25)}]
    fitted = trainer()
    with pytest.warns(UserWarning, match="total_size"):
        state = fitted.fit(1, data=data, model=model, random_seed=42)
    previous_batch = model["batch"].get_value().copy()

    with pytest.raises(ValueError) as error:
        fitted.fit(1, data=[{"batch": np.full(5, 9.0)}], model=model, total_size=20)

    assert "total_size" in str(error.value)
    assert "new Trainer" in str(error.value)
    assert_unchanged_training(fitted, state)
    np.testing.assert_array_equal(model["batch"].get_value(), previous_batch)


def test_streaming_cannot_add_scaling_to_an_existing_full_data_step():
    model = streamed_normal_model()
    fitted = trainer()
    state = fitted.fit(1, model=model, random_seed=42)
    previous_batch = model["batch"].get_value().copy()

    with pytest.raises(ValueError) as error:
        fitted.fit(1, data=[{"batch": np.full(3, 0.25)}], model=model, total_size=20)

    assert "total_size" in str(error.value)
    assert "new Trainer" in str(error.value)
    assert_unchanged_training(fitted, state)
    np.testing.assert_array_equal(model["batch"].get_value(), previous_batch)


def test_compiled_full_fit_rejects_new_observations_without_total_size():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 2)
        pm.Normal("y", mu, 1, shape=3)

    fitted = trainer()
    state = fitted.fit(1, model=model, random_seed=42)
    with pytest.raises(ValueError, match="new Trainer"):
        fitted.fit(1, data=[{"y": np.full(3, 9.0)}], observeds=["y"], model=model)

    assert_unchanged_training(fitted, state)
    assert not model.observed_RVs
    assert model["y"] in model.free_RVs

    # A rejected binding must not leave cached stream state behind or change the
    # original model: ordinary full-data training must still continue unchanged.
    continued = fitted.fit(1, model=model)
    reference = trainer().fit(2, model=model, random_seed=42)
    assert_same_training(continued, reference)
