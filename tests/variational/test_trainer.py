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
"""Trainer: drive variational inference over a DataLoader without hand-written streaming callbacks."""

import warnings

from itertools import chain, islice, repeat

import numpy as np
import pymc as pm
import pytest

from pymc_extras.variational.dataloader import DataLoader
from pymc_extras.variational.trainer import Trainer, _cycle
from tests.variational.dataloader_helpers import chunked_factory


def marked(n, rows=4):
    """``n`` blocks, block ``i`` filled with ``i``, so an installed batch names itself."""
    return [np.full((rows, 1), float(i)) for i in range(n)]


def loud(n, rows=4, spacing=1000.0):
    """``n`` blocks, block ``i`` filled with ``spacing * (i + 1)``.

    Markers this far from zero and from each other survive into the loss, which is
    quadratic in the batch value, so the loss fingerprints the batch the gradient
    was actually taken on rather than the one ``set_data`` was asked for.
    """
    return [np.full((rows, 1), spacing * (i + 1)) for i in range(n)]


def one_epoch_then_empty(rows=4):
    """A source that streams one epoch and then comes back empty.

    The third pass raises so that a ``_cycle`` which spins over the empty second one
    fails the test instead of hanging it.
    """
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            yield np.zeros((rows, 1))
        elif calls["n"] > 2:
            raise AssertionError("_cycle restarted an exhausted source instead of raising")

    return factory


def record_installed(model, log):
    """Log the marker of every batch ``set_data`` installs, in order."""
    original = model.set_data

    def spy(name, values, *args, **kwargs):
        log.append(float(np.asarray(values)[0, 0]))
        return original(name, values, *args, **kwargs)

    model.set_data = spy


def streamed(loader, k):
    """The first ``k`` markers a user iterating ``loader`` themselves gets, epoch after epoch."""
    return [float(batch[0, 0]) for batch in islice(chain.from_iterable(repeat(loader)), k)]


def test_cycle_replays_the_loaders_own_epochs():
    """_cycle is exactly "repeat the loader": no batch dropped, repeated or reordered
    at the epoch seam."""
    loader = DataLoader(lambda: iter(marked(3, rows=2)), batch_size=2, total_size=6)
    got = [float(b[0, 0]) for b in islice(_cycle(loader), 8)]
    assert got == streamed(loader, 8)


def test_cycle_raises_instead_of_spinning_on_an_empty_pass():
    """A source that cannot be replayed would make the cycle loop forever."""
    loader = DataLoader(one_epoch_then_empty(rows=2), batch_size=2, total_size=2)
    batches = _cycle(loader)
    assert next(batches).shape == (2, 1)
    with pytest.raises(RuntimeError, match="yielded no batches"):
        next(batches)


def test_trainer_end_to_end_matches_in_ram_minibatch():
    """End-to-end: Trainer-driven streaming ADVI reproduces in-RAM pm.Minibatch ADVI.

    Exercises the whole API: a pm.Data placeholder, total_size=loader.total_size, and a
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
        total_size=N,
    )
    with pm.Model() as model:
        b = pm.Normal("b", 0, 3, shape=3)
        batch = pm.Data("batch", np.zeros((bs, 3)))
        pm.Bernoulli(
            "o",
            logit_p=b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1],
            observed=batch[:, 2],
            total_size=loader.total_size,
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
    loader = DataLoader(lambda: iter([data] * 100), batch_size=4, total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        Trainer(method="advi", dataloader=loader, data_name="batch").fit(
            5, progressbar=False, random_seed=0
        )
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_trainer_raises_when_loader_cannot_restart():
    """A source that streams one epoch and then comes back empty cannot be cycled;
    the Trainer surfaces a clear error instead of training on stale data."""
    loader = DataLoader(one_epoch_then_empty(), batch_size=4, total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        with pytest.raises(RuntimeError, match="yielded no batches"):
            Trainer(method="advi", dataloader=loader, data_name="batch").fit(
                5, progressbar=False, random_seed=0
            )


def test_trainer_appends_user_callbacks_and_streams_distinct_batches():
    """User callbacks (e.g. convergence trackers) compose with the internal
    advance callback instead of colliding on the keyword, and the placeholder
    holds a different batch on successive steps. Also exercises the default
    data_name ("batch")."""
    blocks = [np.full((4, 1), float(i)) for i in range(60)]
    loader = DataLoader(lambda: iter(blocks), batch_size=4, total_size=240)
    seen = []
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", x, 1.0, observed=batch[:, 0], total_size=loader.total_size)
        Trainer(method="advi", dataloader=loader).fit(
            5, callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))]
        )
    assert len(seen) == 5
    assert len(set(seen)) > 1


def test_trainer_accepts_inference_instance():
    """An Inference instance is forwarded to pm.fit unchanged; it is bound to
    the model it was built under, so the Trainer only streams the batches."""
    data = np.ones((4, 1))
    loader = DataLoader(lambda: iter([data] * 50), batch_size=4, total_size=4)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        approx = Trainer(method=pm.ADVI(random_seed=0), dataloader=loader).fit(5)
    assert len(approx.hist) == 5
    np.testing.assert_array_equal(model["batch"].get_value(), data)


def test_constructor_fit_kwargs_take_random_seed():
    """random_seed works as a constructor default, as the docstring promises,
    and a per-call value overrides the constructor's."""
    data = np.ones((4, 1))

    def fit_with(ctor_kwargs, fit_kwargs):
        loader = DataLoader(lambda: iter([data] * 50), batch_size=4, total_size=4)
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            batch = pm.Data("batch", np.zeros((4, 1)))
            pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
            return Trainer(method="advi", dataloader=loader, data_name="batch", **ctor_kwargs).fit(
                5, **fit_kwargs
            )

    a = fit_with({"random_seed": 7}, {})
    b = fit_with({"random_seed": 0}, {"random_seed": 7})
    np.testing.assert_array_equal(a.hist, b.hist)


@pytest.mark.parametrize(
    "n, blocks, rows",
    [(1, 6, 4), (4, 4, 4), (5, 3, 2), (7, 2, 1)],
    ids=["single-step", "ends-at-seam", "wraps-once", "wraps-often"],
)
def test_steps_consume_the_loaders_own_batch_sequence(n, blocks, rows):
    """The batches trained are the loader's own pass, in order, across epoch seams.

    Three views have to agree with what iterating the loader by hand yields: the
    batches ``set_data`` installed, the placeholder a callback reads while the loss
    it was handed is still current, and the losses themselves -- with the optimizer
    frozen the loss is quadratic in the batch marker, so ``hist / hist[0]`` decodes
    the batch each gradient was taken on. A seam that repeats, drops or reorders a
    batch still passes a "``n`` distinct batches" check.
    """
    loader = DataLoader(
        lambda: iter(loud(blocks, rows=rows)),
        batch_size=rows,
        total_size=blocks * rows,
    )
    expected = streamed(loader, n + 1)
    installed, seen = [], []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((rows, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        record_installed(model, installed)
        approx = Trainer(
            method="advi",
            dataloader=loader,
            obj_optimizer=pm.sgd(learning_rate=1e-12),
        ).fit(
            n,
            random_seed=0,
            callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))],
        )
    assert installed == expected
    assert seen == expected[:n]
    trained = np.asarray(expected[:n]) / expected[0]
    np.testing.assert_allclose(approx.hist / approx.hist[0], trained**2, rtol=0.02)


def test_refine_after_fit_continues_without_repeating_a_batch():
    """Inference.refine replays pm.fit's saved callbacks and steps before they run.

    fit therefore has to leave the *next* batch loaded, or refine's first gradient
    step would retrain the batch fit just finished with.
    """
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, total_size=4)
    installed = []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        record_installed(model, installed)
        inference = pm.ADVI(random_seed=0)
        Trainer(method=inference, dataloader=loader).fit(3)
        assert installed == [0.0, 1.0, 2.0, 3.0]
        installed.clear()
        inference.refine(4, progressbar=False)
    assert installed == [4.0, 5.0, 6.0, 7.0]


def test_refine_after_an_early_stop_keeps_streaming():
    """A fit cut short by a callback must not leave the advance permanently disarmed."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, total_size=4)
    installed = []
    armed = [True]

    def stop_once(*_):
        if armed[0]:
            armed[0] = False
            raise StopIteration("stop")

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        record_installed(model, installed)
        inference = pm.ADVI(random_seed=0)
        Trainer(method=inference, dataloader=loader).fit(10, callbacks=[stop_once], score=False)
        assert installed == [0.0]
        installed.clear()
        inference.refine(4, progressbar=False)
    assert installed == [1.0, 2.0, 3.0, 4.0]


def test_a_second_fit_reseeds_and_forgets_the_first_calls_kwargs():
    """Calling fit twice on one Trainer: each call re-reads the constructor's defaults
    and installs a batch before its own step 0.

    A per-call keyword that survived into the next call would silently change a fit
    nobody passed it to, and a call that trusted the batch its predecessor left
    loaded would train it twice.
    """
    loader = DataLoader(lambda: iter(marked(6)), batch_size=4, total_size=24)
    installed, seen = [], []
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        record_installed(model, installed)
        trainer = Trainer(method="advi", dataloader=loader, random_seed=0)
        unscored = trainer.fit(3, score=False)
        left_loaded = installed[-1]
        installed.clear()
        scored = trainer.fit(
            3, callbacks=[lambda *_: seen.append(float(model["batch"].get_value()[0, 0]))]
        )
    assert len(unscored.hist) == 0
    assert len(scored.hist) == 3
    assert seen == installed[:-1]
    assert installed[0] != left_loaded


def test_inference_instance_bound_to_another_model_is_rejected():
    """Streaming into one model while an Inference optimizes another is silent otherwise."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, total_size=4)
    with pm.Model() as other:
        pm.Normal("mu", 0, 1)
        pm.Data("batch", np.zeros((4, 1)))
        elsewhere = pm.ADVI()
    assert other is not None
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        with pytest.raises(ValueError, match="bound to a different model"):
            Trainer(method=elsewhere, dataloader=loader).fit(3)


def test_inference_instance_is_matched_against_the_trained_model():
    """The identity check reads the model the Trainer was given, not the context stack,
    and accepts an instance built under it; the fit hands back that instance's own
    Approximation, so refining or sampling the return value is refining the fit."""
    loader = DataLoader(lambda: iter(marked(10)), batch_size=4, total_size=40)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        inference = pm.ADVI(random_seed=0)
    with pm.Model():
        pm.Normal("mu", 0, 1)
        pm.Data("batch", np.zeros((4, 1)))
        stranger = pm.ADVI()

    approx = Trainer(method=inference, dataloader=loader, model=model).fit(3)
    assert approx is inference.approx
    assert len(approx.hist) == 3
    with pytest.raises(ValueError, match="bound to a different model"):
        Trainer(method=stranger, dataloader=loader, model=model).fit(3)


@pytest.mark.parametrize(
    "declared, match",
    [(None, "no observed variable declares total_size"), (40, "does not match the data")],
    ids=["absent", "wrong-value"],
)
def test_unusable_likelihood_scaling_warns(declared, match):
    """Absent or disagreeing total_size biases the posterior without failing the fit."""
    loader = DataLoader(lambda: iter(marked(50)), batch_size=4, total_size=4)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=declared)
        with pytest.warns(UserWarning, match=match):
            Trainer(method="advi", dataloader=loader).fit(3, random_seed=0)


@pytest.mark.parametrize("rows, blocks", [(4, 10), (3, 7)], ids=["N=40,batch=4", "N=21,batch=3"])
@pytest.mark.parametrize(
    "declares, match",
    [
        ("absent", "no observed variable declares total_size"),
        ("mismatched", "does not match the data"),
        ("correct", None),
    ],
)
def test_scaling_warning_fires_only_on_an_unusable_total_size(rows, blocks, declares, match):
    """The scaling check warns on an absent or disagreeing total_size and stays quiet
    on a correct one, for N != batch_size as well as N == batch_size.

    A false positive would be worse than the bias the check guards against: it teaches
    users to ignore the warning. One shape cannot show the check reads the dataset
    size rather than anything else the loader could offer.
    """
    n_rows = rows * blocks
    loader = DataLoader(
        lambda: iter(marked(blocks, rows=rows)),
        batch_size=rows,
        total_size=n_rows,
    )
    total_size = {"absent": None, "mismatched": n_rows + rows, "correct": n_rows}[declares]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            batch = pm.Data("batch", np.zeros((rows, 1)))
            pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=total_size)
            Trainer(method="advi", dataloader=loader).fit(2, random_seed=0)
    scaling = [str(w.message) for w in caught if "total_size" in str(w.message)]
    if match is None:
        assert scaling == []
    else:
        assert len(scaling) == 1
        assert match in scaling[0]


@pytest.mark.parametrize("n", [0, True], ids=["zero", "bool"])
def test_fit_rejects_nonpositive_n(n):
    """fit consumes the seed batch before pm.fit could reject n itself, so a
    non-positive n is refused up front, before touching the stream. ``True`` is an
    ``int`` to Python, so it would otherwise pass as a one-step fit."""
    loader = DataLoader(lambda: iter([np.zeros((2, 1))]), batch_size=2, total_size=2)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        with pytest.raises(ValueError, match="positive integer"):
            Trainer(method="advi", dataloader=loader).fit(n)


def test_fit_accepts_a_numpy_integer_step_count():
    """A numpy integer is not an ``int``, so the two modules would otherwise disagree:
    the DataLoader takes one for ``batch_size`` and fit would refuse one for ``n``."""
    loader = DataLoader(lambda: iter(marked(10, rows=2)), batch_size=np.int64(2), total_size=20)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((2, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        approx = Trainer(method="advi", dataloader=loader).fit(np.int64(3), random_seed=0)
    assert len(approx.hist) == 3


def test_progressbar_defaults_off_and_yields_to_an_explicit_setting(monkeypatch):
    """pm.fit draws a progress bar by default, which a per-step streaming fit turns
    into a wall of output; an explicit setting from either level still wins."""
    seen = []
    monkeypatch.setattr(
        "pymc_extras.variational.trainer._fit", lambda n, **kw: seen.append(kw["progressbar"])
    )
    loader = DataLoader(lambda: iter(marked(4)), batch_size=4, total_size=16)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=loader.total_size)
        trainer = Trainer(method="advi", dataloader=loader)
        trainer.fit(2)
        trainer.fit(2, progressbar=True)
    assert seen == [False, True]


def test_trainer_accepts_a_duck_typed_loader():
    """The loader contract is duck-typed -- iterable plus total_size -- so a
    source that is not the concrete DataLoader class still trains."""

    class SizedSource:
        total_size = 4

        def __iter__(self):
            return iter([np.zeros((4, 1))] * 50)

    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", np.zeros((4, 1)))
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=4)
        approx = Trainer(method="advi", dataloader=SizedSource()).fit(3, random_seed=0)
    assert len(approx.hist) == 3
