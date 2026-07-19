#   Copyright 2026 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Trainer: drive variational inference over a DataLoader with no user callbacks."""

import numpy as np
import pymc as pm
import pytest

from pymc_extras.variational.dataloader import DataLoader
from pymc_extras.variational.trainer import Trainer
from tests.variational.dataloader_helpers import chunked_factory


def marked(n, rows=4):
    """``n`` blocks, block ``i`` filled with ``i``, so an installed batch names itself."""
    return [np.full((rows, 1), float(i)) for i in range(n)]


def record_installed(model, log):
    """Log the marker of every batch ``set_data`` installs, in order."""
    original = model.set_data

    def spy(name, values, *args, **kwargs):
        log.append(float(np.asarray(values)[0, 0]))
        return original(name, values, *args, **kwargs)

    model.set_data = spy


def test_trainer_end_to_end_matches_in_ram_minibatch():
    """End-to-end: Trainer-driven streaming ADVI reproduces in-RAM pm.Minibatch ADVI.

    Exercises the whole API: a pm.Data placeholder, total_size=len(loader), and a
    Trainer that streams minibatches into the placeholder with set_data while the
    user writes no callbacks. Runs long enough to cycle the loader across epochs.
    """
    seed = 0
    rng = np.random.default_rng(seed)
    N, bs = 60_000, 2048
    X = rng.normal(size=(N, 2))
    b_true = np.array([0.3, -1.1, 0.7])
    y = (rng.random(N) < 1 / (1 + np.exp(-(b_true[0] + X @ b_true[1:])))).astype("float64")
    data = np.column_stack([X, y])

    with pm.Model():
        b = pm.Normal("b", 0, 3, shape=3)
        xb, zb, yb = pm.Minibatch(X[:, 0].copy(), X[:, 1].copy(), y, batch_size=bs)
        pm.Bernoulli("o", logit_p=b[0] + b[1] * xb + b[2] * zb, observed=yb, total_size=N)
        ap = pm.fit(
            6000,
            method="advi",
            obj_optimizer=pm.adam(learning_rate=0.02),
            progressbar=False,
            random_seed=seed,
        )
        in_ram = ap.sample(400).posterior["b"].values.reshape(-1, 3).mean(0)

    loader = DataLoader(
        chunked_factory(data, 20_000),
        batch_size=bs,
        shuffle=True,
        buffer_size=40_000,
        seed=seed,
        sample_shape=(3,),
        total_size=N,
    )
    with pm.Model() as model:
        b = pm.Normal("b", 0, 3, shape=3)
        batch = pm.Data("batch", np.zeros((bs, 3)))
        pm.Bernoulli(
            "o",
            logit_p=b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1],
            observed=batch[:, 2],
            total_size=len(loader),
        )
        ap = Trainer(
            method="advi",
            dataloader=loader,
            data_name="batch",
            obj_optimizer=pm.adam(learning_rate=0.02),
        ).fit(6000, random_seed=seed)
        stream = ap.sample(400).posterior["b"].values.reshape(-1, 3).mean(0)

    np.testing.assert_allclose(in_ram, stream, atol=0.1)


def test_trainer_streams_into_placeholder():
    """The Trainer seeds the pm.Data placeholder before step 0 (pm.fit runs
    callbacks after each step) and overwrites it each step; after fitting it holds
    a real batch, not the zero seed."""
    data = np.ones((4, 1))
    loader = DataLoader(lambda: iter([data] * 100), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader, data_name="batch").fit(
            5, progressbar=False, random_seed=0
        )
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_trainer_raises_when_loader_cannot_restart():
    """A source that streams one epoch and then comes back empty cannot be cycled;
    the Trainer surfaces a clear error instead of training on stale data."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            yield np.zeros((4, 1))

    loader = DataLoader(factory, batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.raises(RuntimeError, match="yielded no batches"):
            Trainer(method="advi", dataloader=loader, data_name="batch").fit(
                5, progressbar=False, random_seed=0
            )


def test_trainer_rejects_non_dataloader():
    """The isinstance guard fires before any model lookup."""
    with pytest.raises(TypeError, match="DataLoader"):
        Trainer(method="advi", dataloader=object()).fit(10)


def test_trainer_appends_user_callbacks_and_streams_distinct_batches():
    """User callbacks (e.g. convergence trackers) compose with the internal
    advance callback instead of colliding on the keyword, and the placeholder
    holds a different batch on successive steps. Also exercises the default
    data_name ("batch")."""
    blocks = [np.full((4, 1), float(i)) for i in range(60)]
    loader = DataLoader(lambda: iter(blocks), batch_size=4, sample_shape=(1,), total_size=240)
    seen = []
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", x, 1.0, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader).fit(
            5, callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))]
        )
    assert len(seen) == 5
    assert len(set(seen)) > 1


def test_trainer_accepts_inference_instance():
    """An Inference instance is forwarded to pm.fit unchanged; it is bound to
    the model it was built under, so the Trainer only streams the batches."""
    data = np.ones((4, 1))
    loader = DataLoader(lambda: iter([data] * 50), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        approx = Trainer(method=pm.ADVI(random_seed=0), dataloader=loader).fit(5)
    assert len(approx.hist) == 5
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_constructor_fit_kwargs_take_random_seed():
    """random_seed works as a constructor default, as the docstring promises,
    and a per-call value overrides the constructor's."""
    data = np.ones((4, 1))

    def fit_with(ctor_kwargs, fit_kwargs):
        loader = DataLoader(
            lambda: iter([data] * 50), batch_size=4, sample_shape=(1,), total_size=4
        )
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            batch = pm.Data("batch", np.zeros((4, 1)))
            pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
            return Trainer(method="advi", dataloader=loader, data_name="batch", **ctor_kwargs).fit(
                5, **fit_kwargs
            )

    a = fit_with({"random_seed": 7}, {})
    b = fit_with({"random_seed": 0}, {"random_seed": 7})
    np.testing.assert_array_equal(a.hist, b.hist)


def test_fit_trains_one_batch_per_step():
    """Step i trains batch i, and the step that ends the fit loads batch n for what follows."""
    loader = DataLoader(
        lambda: iter(marked(10, rows=2)), batch_size=2, sample_shape=(1,), total_size=20
    )
    installed = []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        record_installed(model, installed)
        Trainer(method="advi", dataloader=loader).fit(3, random_seed=0)
    assert installed == [0.0, 1.0, 2.0, 3.0]


def test_refine_after_fit_continues_without_repeating_a_batch():
    """Inference.refine replays pm.fit's saved callbacks and steps before they run.

    fit therefore has to leave the *next* batch loaded, or refine's first gradient
    step would retrain the batch fit just finished with.
    """
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, sample_shape=(1,), total_size=4)
    installed = []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        record_installed(model, installed)
        inference = pm.ADVI(random_seed=0)
        Trainer(method=inference, dataloader=loader).fit(3)
        assert installed == [0.0, 1.0, 2.0, 3.0]
        installed.clear()
        inference.refine(4, progressbar=False)
    assert installed == [4.0, 5.0, 6.0, 7.0]


def test_refine_after_an_early_stop_keeps_streaming():
    """A fit cut short by a callback must not leave the advance permanently disarmed."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, sample_shape=(1,), total_size=4)
    installed = []
    armed = [True]

    def stop_once(*_):
        if armed[0]:
            armed[0] = False
            raise StopIteration("stop")

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        record_installed(model, installed)
        inference = pm.ADVI(random_seed=0)
        Trainer(method=inference, dataloader=loader).fit(10, callbacks=[stop_once], score=False)
        assert installed == [0.0]
        installed.clear()
        inference.refine(4, progressbar=False)
    assert installed == [1.0, 2.0, 3.0, 4.0]


def test_user_callbacks_see_the_batch_that_produced_the_loss():
    """A callback reading the placeholder must see batch i on step i, not batch i+1."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, sample_shape=(1,), total_size=4)
    seen = []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        Trainer(method="advi", dataloader=loader).fit(
            5,
            random_seed=0,
            callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))],
        )
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_inference_instance_bound_to_another_model_is_rejected():
    """Streaming into one model while an Inference optimizes another is silent otherwise."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model() as other:
        pm.Normal("mu", 0, 1)
        pm.Data("batch", np.zeros((4, 1)))
        elsewhere = pm.ADVI()
    assert other is not None
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.raises(ValueError, match="bound to a different model"):
            Trainer(method=elsewhere, dataloader=loader).fit(3)


@pytest.mark.parametrize(
    "declared, match",
    [(None, "no observed variable declares total_size"), (40, "does not match the data")],
    ids=["absent", "wrong-value"],
)
def test_unusable_likelihood_scaling_warns(declared, match):
    """Absent or disagreeing total_size biases the posterior without failing the fit."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, sample_shape=(1,), total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=declared)
        with pytest.warns(UserWarning, match=match):
            Trainer(method="advi", dataloader=loader).fit(3, random_seed=0)


def test_total_size_check_fires_when_fit_ends_at_pass_boundary():
    """fit(n) with n exactly the batches in one pass still runs the total_size
    sanity check: the stream is kept one batch ahead, so stopping at the
    boundary does not abandon the check right before it would fire."""
    data = np.zeros((40, 1))
    loader = DataLoader(chunked_factory(data, 10), batch_size=10, sample_shape=(1,), total_size=400)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((10, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.warns(UserWarning, match="disagrees with"):
            Trainer(method="advi", dataloader=loader).fit(4, random_seed=0)


def test_fit_rejects_nonpositive_n():
    """fit consumes the seed batch before pm.fit could reject n itself, so a
    non-positive n is refused up front, before touching the stream."""
    loader = DataLoader(
        lambda: iter([np.zeros((2, 1))]), batch_size=2, sample_shape=(1,), total_size=2
    )
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
        with pytest.raises(ValueError, match="positive integer"):
            Trainer(method="advi", dataloader=loader).fit(0)


def test_unknown_data_name_raises_before_consuming():
    """A data_name that is not in the model raises a guided KeyError before any
    batch is pulled from the loader."""
    loader = DataLoader(
        lambda: iter([np.zeros((4, 1))] * 3), batch_size=4, sample_shape=(1,), total_size=4
    )
    installed = []
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
        record_installed(model, installed)
        with pytest.raises(KeyError, match=r"pm\.Data placeholder"):
            Trainer(method="advi", dataloader=loader, data_name="nope").fit(2)
    assert installed == []
