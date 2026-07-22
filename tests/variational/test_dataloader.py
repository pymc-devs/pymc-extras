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
import numpy as np
import pymc as pm
import pytest

from pymc_extras.variational.dataloader import (
    DataLoader,
    IterableDataset,
    shuffle_buffer,
)
from tests.variational.dataloader_helpers import chunked_factory


def test_plain_loader_rebatches_arbitrary_blocks():
    """Blocks of 3 are re-batched into 4s in order, and the trailing rows are dropped."""
    data = np.arange(20, dtype="float64").reshape(10, 2)
    ds = DataLoader(chunked_factory(data, 3), batch_size=4, sample_shape=(2,), total_size=10)
    batches = list(ds)
    assert [b.shape for b in batches] == [(4, 2), (4, 2)]
    np.testing.assert_array_equal(np.concatenate(batches), data[:8])


def test_wrong_sample_shape_rejected():
    """A source whose trailing shape does not match sample_shape raises."""
    data = np.zeros((12, 3))
    ds = DataLoader(chunked_factory(data, 4), batch_size=4, sample_shape=(2,), total_size=12)
    with pytest.raises(ValueError, match="source yielded shape"):
        next(iter(ds))


def test_preprocess_fn_applied():
    """preprocess_fn transforms each batch before it is yielded."""
    data = np.ones((8, 1))
    ds = DataLoader(
        chunked_factory(data, 4),
        batch_size=4,
        sample_shape=(1,),
        total_size=8,
        preprocess_fn=lambda b: b * 3.0,
    )
    np.testing.assert_array_equal(next(iter(ds)), np.full((4, 1), 3.0))


@pytest.mark.parametrize(
    "buffer_size, n_rows",
    [(55, 140), (3, 120)],
    ids=["non-dividing-chunks", "buffer-below-batch"],
)
def test_shuffle_buffer_conserves_rows(buffer_size, n_rows):
    """Buffer sizes that do not divide batch_size, one of them below it, lose no rows."""
    data = np.arange(n_rows, dtype="float64").reshape(n_rows, 1)
    src = shuffle_buffer(chunked_factory(data, 7), buffer_size=buffer_size, batch_size=10, seed=0)
    batches = list(src())
    assert batches
    assert all(b.shape == (10, 1) for b in batches)
    seen = np.sort(np.concatenate([b.ravel() for b in batches]))
    np.testing.assert_array_equal(seen, data.ravel())


@pytest.mark.parametrize("chunk", [25, 100], ids=["several-chunks-per-fill", "one-chunk-fills-it"])
def test_shuffle_buffer_does_not_mutate_source(chunk):
    """Shuffling happens on an owned copy, even when a single chunk fills the buffer."""
    data = np.arange(100, dtype="float64").reshape(100, 1)
    original = data.copy()
    src = shuffle_buffer(chunked_factory(data, chunk), buffer_size=40, batch_size=10, seed=1)
    list(src())
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "data, make_source, sample_shape, batch_size, buffer_size, shapes",
    [
        (
            np.arange(120.0).reshape(120, 1),
            lambda d: chunked_factory(d, 8),
            (1,),
            10,
            40,
            [(10, 1)] * 12,
        ),
        (np.arange(40.0).reshape(20, 2), lambda d: d, (2,), 8, 16, [(8, 2)] * 2),
        (np.arange(12.0), lambda d: d, (), 4, 6, [(4,)] * 3),
    ],
    ids=["block-factory", "raw-2d", "raw-1d-scalars"],
)
def test_shuffle_true_yields_whole_source_rows(
    data, make_source, sample_shape, batch_size, buffer_size, shapes
):
    """shuffle=True yields only full batches, of distinct rows that all came from the source."""
    ds = DataLoader(
        make_source(data),
        batch_size=batch_size,
        shuffle=True,
        buffer_size=buffer_size,
        seed=0,
        sample_shape=sample_shape,
        total_size=len(data),
    )
    batches = list(ds)
    assert [b.shape for b in batches] == shapes
    seen = {tuple(np.atleast_1d(r)) for b in batches for r in b}
    assert len(seen) == sum(s[0] for s in shapes)
    assert seen <= {tuple(np.atleast_1d(r)) for r in data}


def test_total_size_rescales_logp_like_minibatch():
    """total_size=len(loader) scales the observed logp by exactly N / batch_size."""
    rng = np.random.default_rng(0)
    N, bs = 1000, 20
    data = rng.normal(size=(bs, 1))
    loader = DataLoader(lambda: iter([data]), batch_size=bs, sample_shape=(1,), total_size=N)

    with pm.Model() as scaled:
        mu = pm.Normal("mu", 0, 1)
        batch = pm.Data("batch", data)
        pm.Normal("y", mu, 1, observed=batch[:, 0], total_size=len(loader))
    with pm.Model() as plain:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu, 1, observed=data[:, 0])

    point = {"mu": np.array(0.3)}
    obs_scaled = scaled.compile_logp(scaled.observed_RVs)(point)
    obs_plain = plain.compile_logp(plain.observed_RVs)(point)
    np.testing.assert_allclose(obs_scaled, obs_plain * (N / bs), rtol=1e-6)


def test_len_raises_when_total_size_none():
    """total_size=None warns at construction, and len() then raises instead of guessing N."""
    data = np.ones((4, 1))
    with pytest.warns(UserWarning, match="total_size=None"):
        loader = DataLoader(
            lambda: iter([data] * 5), batch_size=4, sample_shape=(1,), total_size=None
        )
    with pytest.raises(TypeError, match="total_size=None"):
        len(loader)


def test_iter_yields_clean_batches_and_reiterates():
    """__iter__ yields (batch_size, *sample_shape) batches and can be re-iterated."""
    data = np.arange(40, dtype="float64").reshape(40, 1)
    loader = DataLoader(chunked_factory(data, 10), batch_size=10, sample_shape=(1,), total_size=40)
    e1 = list(loader)
    e2 = list(loader)
    assert len(e1) == 4 and all(b.shape == (10, 1) for b in e1)
    np.testing.assert_array_equal(np.sort(np.concatenate([b.ravel() for b in e1])), data.ravel())
    np.testing.assert_array_equal(np.sort(np.concatenate([b.ravel() for b in e2])), data.ravel())


@pytest.mark.parametrize(
    "kwargs",
    [{"total_size": 0}, {"total_size": -100}, {"batch_size": True}],
    ids=["zero", "negative", "bool"],
)
def test_nonpositive_sizes_rejected(kwargs):
    """0 skips the rescaling, a negative N flips the logp sign, and bool is not a size."""
    kw = {"batch_size": 4, "total_size": 8, **kwargs}
    data = np.zeros((8, 1))
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(chunked_factory(data, 4), sample_shape=(1,), **kw)


def test_shuffle_buffer_rejects_nonpositive_sizes():
    """Zero or negative buffer/batch sizes raise at construction."""
    data = np.zeros((10, 1))
    with pytest.raises(ValueError, match="buffer_size"):
        shuffle_buffer(chunked_factory(data, 5), buffer_size=0, batch_size=4)
    with pytest.raises(ValueError, match="batch_size"):
        shuffle_buffer(chunked_factory(data, 5), buffer_size=10, batch_size=0)


def test_accepts_numpy_integer_sizes():
    """numpy ints pass the Integral check and are stored as plain Python ints."""
    data = np.zeros((8, 1))
    ds = DataLoader(
        chunked_factory(data, 4), batch_size=np.int64(4), sample_shape=(1,), total_size=np.int64(8)
    )
    assert next(iter(ds)).shape == (4, 1)
    assert (ds.batch_size, ds.total_size) == (4, 8)
    assert type(ds.batch_size) is int and type(ds.total_size) is int


def test_shuffle_buffer_reshuffles_each_epoch_and_is_seed_reproducible():
    """Each epoch draws a fresh permutation, and the same seed replays the same epochs."""
    data = np.arange(60, dtype="float64").reshape(60, 1)

    def epochs():
        f = shuffle_buffer(chunked_factory(data, 10), buffer_size=60, batch_size=10, seed=7)
        return [np.concatenate([b.ravel() for b in f()]) for _ in range(2)]

    first, second = epochs()
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(epochs(), [first, second])
    np.testing.assert_array_equal(np.sort(first), data.ravel())
    np.testing.assert_array_equal(np.sort(second), data.ravel())


def test_factory_returning_reiterable_is_accepted():
    """A zero-arg factory may return any iterable, e.g. a list, not just an iterator."""
    data = [np.zeros((4, 1), dtype="float64")]
    ds = DataLoader(lambda: data, batch_size=4, sample_shape=(1,), total_size=4)
    assert next(iter(ds)).shape == (4, 1)


@pytest.mark.parametrize(
    "data, batch_size, shapes",
    [
        (np.arange(6, dtype="float64"), 3, [(3,), (3,)]),
        (np.arange(40, dtype="float64").reshape(20, 2), 8, [(8,)] * 5),
    ],
    ids=["1d-source", "2d-source-overrides-inference"],
)
def test_scalar_sample_shape_batches_scalars(data, batch_size, shapes):
    """sample_shape=() reads every element as one scalar sample, overriding any inference."""
    ds = DataLoader(data, batch_size=batch_size, sample_shape=(), total_size=data.size)
    batches = list(ds)
    assert [b.shape for b in batches] == shapes
    np.testing.assert_array_equal(np.concatenate(batches), data.ravel()[: len(shapes) * batch_size])


def test_iterable_dataset_base_is_abstract():
    """The base class is a contract: __iter__ must be overridden."""
    with pytest.raises(NotImplementedError):
        iter(IterableDataset())


def test_raw_2d_array_infers_sample_shape():
    """A raw 2-D array defaults sample_shape to its trailing shape, so rows are batched."""
    data = np.arange(40, dtype="float64").reshape(20, 2)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(data, batch_size=8, total_size="auto")
    assert ds.total_size == 20
    batches = list(ds)
    assert [b.shape for b in batches] == [(8, 2), (8, 2)]
    np.testing.assert_array_equal(np.concatenate(batches), data[:16])


def test_shuffle_buffer_accepts_factory_returning_reiterable():
    """A factory returning a re-iterable is normalized to one iterator, so fills don't restart."""
    data = np.arange(120, dtype="float64").reshape(120, 1)
    chunks = [data[i : i + 20] for i in range(0, 120, 20)]
    src = shuffle_buffer(lambda: chunks, buffer_size=50, batch_size=10, seed=0)
    batches = list(src())
    assert len(batches) == 12
    np.testing.assert_array_equal(
        np.sort(np.concatenate([b.ravel() for b in batches])), data.ravel()
    )
