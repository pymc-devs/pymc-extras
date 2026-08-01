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
from tests.variational.dataloader_helpers import (
    BlockDataset,
    chunked_factory,
    reused_buffer_factory,
)


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


@pytest.mark.parametrize(
    "block_rows, kwargs",
    [
        (2, {}),
        (2, {"shuffle": True, "buffer_size": 4, "seed": 0}),
        (4, {}),
    ],
    ids=["rebatched", "shuffled", "exact-batch-size"],
)
def test_source_reusing_one_buffer_streams_every_row(block_rows, kwargs):
    """A source may hand back a view into a buffer it overwrites; no row may be lost to it."""
    n_blocks = 8 // block_rows
    ds = DataLoader(
        reused_buffer_factory(n_blocks, block_rows),
        batch_size=4,
        sample_shape=(1,),
        total_size=8,
        **kwargs,
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(ds)).ravel()),
        np.repeat(np.arange(n_blocks, dtype="float64"), block_rows),
    )


@pytest.mark.parametrize("block_rows, batch_size", [(3, 2), (5, 3), (7, 4)])
def test_remainder_left_by_a_block_outlives_the_next_pull(block_rows, batch_size):
    """Rows carried over from a block that overfilled a batch are copied before the next pull.

    A block wider than one batch is sliced without ever being concatenated, so what
    is left of it is a view of the source's buffer and is read only after the source
    has been asked for -- and has refilled -- that same buffer.
    """
    n_blocks = 4
    values = np.repeat(np.arange(n_blocks, dtype="float64"), block_rows)
    ds = DataLoader(
        reused_buffer_factory(n_blocks, block_rows),
        batch_size=batch_size,
        sample_shape=(1,),
        total_size=len(values),
    )
    kept = len(values) // batch_size * batch_size
    np.testing.assert_array_equal(np.concatenate(list(ds)).ravel(), values[:kept])


def test_shuffle_buffer_copies_blocks_it_holds():
    """shuffle_buffer fills across several pulls, so it cannot alias the source's buffer."""
    src = shuffle_buffer(reused_buffer_factory(4, 2), buffer_size=8, batch_size=4, seed=0)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(src())).ravel()), np.repeat(np.arange(4, dtype="float64"), 2)
    )


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


@pytest.mark.parametrize(
    "shuffle, buffer_size",
    [(False, None), (True, 7), (True, 1), (True, 500)],
    ids=["plain", "buffer-under-chunk", "buffer-of-one", "buffer-over-dataset"],
)
@pytest.mark.parametrize(
    "n, batch_size, chunk",
    [(60, 10, 7), (37, 5, 1), (100, 100, 13), (23, 4, 23), (12, 1, 5)],
    ids=[
        "chunks-straddle-batches",
        "one-row-chunks",
        "one-batch-is-everything",
        "one-chunk-many-batches",
        "batch-of-one",
    ],
)
def test_one_epoch_conserves_source_rows(n, batch_size, chunk, shuffle, buffer_size):
    """An epoch is floor(N/batch_size) batches of distinct source rows, in source order unshuffled.

    Nothing may be duplicated or invented, and the only rows a pass is allowed to
    drop are the N mod batch_size that cannot fill a last batch -- for every mix of
    chunking, batch size and buffer size, not just where they divide each other.
    """
    data = np.arange(2 * n, dtype="float64").reshape(n, 2)
    loader = DataLoader(
        chunked_factory(data, chunk),
        batch_size=batch_size,
        shuffle=shuffle,
        buffer_size=buffer_size,
        seed=0,
        sample_shape=(2,),
        total_size=n,
    )
    batches = list(loader)
    n_batches = n // batch_size
    assert [b.shape for b in batches] == [(batch_size, 2)] * n_batches
    streamed = np.concatenate(batches)
    if shuffle:
        rows = {tuple(r) for r in streamed}
        assert len(rows) == n_batches * batch_size
        assert rows <= {tuple(r) for r in data}
    else:
        np.testing.assert_array_equal(streamed, data[: n_batches * batch_size])


@pytest.mark.parametrize("shuffle", [False, True], ids=["plain", "shuffled"])
def test_second_epoch_replays_the_same_rows(shuffle):
    """A second pass over a re-readable source streams the same rows, reordered only if shuffling."""
    data = np.arange(120, dtype="float64").reshape(60, 2)
    loader = DataLoader(
        chunked_factory(data, 7),
        batch_size=10,
        shuffle=shuffle,
        buffer_size=25,
        seed=4,
        sample_shape=(2,),
        total_size=60,
    )
    first, second = (np.concatenate(list(loader)) for _ in range(2))
    for epoch in (first, second):
        np.testing.assert_array_equal(epoch[np.argsort(epoch[:, 0])], data)
    assert np.array_equal(first, second) is not shuffle


@pytest.mark.parametrize(
    "buffer_size, effective", [(25, 25), (None, 500)], ids=["given", "default"]
)
def test_shuffled_loader_is_a_seeded_shuffle_buffer(buffer_size, effective):
    """A seeded loader reproduces itself across instances and equals the same shuffle_buffer wrap."""
    data = np.arange(120, dtype="float64").reshape(60, 2)

    def stream(seed):
        loader = DataLoader(
            chunked_factory(data, 7),
            batch_size=10,
            shuffle=True,
            buffer_size=buffer_size,
            seed=seed,
            sample_shape=(2,),
            total_size=60,
        )
        return np.concatenate(list(loader))

    manual = shuffle_buffer(chunked_factory(data, 7), buffer_size=effective, batch_size=10, seed=11)
    np.testing.assert_array_equal(stream(11), stream(11))
    np.testing.assert_array_equal(stream(11), np.concatenate(list(manual())))
    assert not np.array_equal(stream(11), stream(12))


@pytest.mark.parametrize(
    "make_source",
    [
        lambda d: d,
        lambda d: chunked_factory(d, 7),
        lambda d: [d[i : i + 7] for i in range(0, len(d), 7)],
        lambda d: BlockDataset(d, 7),
    ],
    ids=["raw-array", "factory", "reiterable-blocks", "iterable-dataset"],
)
def test_every_source_kind_streams_the_same_batches(make_source):
    """One row per yield, a block factory, a re-iterable and an IterableDataset are interchangeable."""
    data = np.arange(84, dtype="float64").reshape(42, 2)
    loader = DataLoader(make_source(data), batch_size=8, sample_shape=(2,), total_size=42)
    batches = list(loader)
    assert [b.shape for b in batches] == [(8, 2)] * 5
    np.testing.assert_array_equal(np.concatenate(batches), data[:40])


def test_preprocess_fn_runs_once_per_batch_over_the_rows_in_order():
    """preprocess_fn sees each streamed row exactly once, batched and in source order.

    It runs on the assembled batch, not on the source chunks, so a chunking that
    straddles batches cannot split or repeat a call; it may reshape the sample as
    long as the row count survives.
    """
    data = np.arange(60, dtype="float64").reshape(20, 3)
    seen = []

    def record(batch):
        seen.append(np.array(batch))
        return batch[:, :1] * 2.0

    loader = DataLoader(
        chunked_factory(data, 3),
        batch_size=6,
        sample_shape=(3,),
        total_size=20,
        preprocess_fn=record,
    )
    batches = list(loader)
    assert [b.shape for b in batches] == [(6, 1)] * 3
    np.testing.assert_array_equal(np.concatenate(seen), data[:18])
    np.testing.assert_array_equal(np.concatenate(batches), data[:18, :1] * 2.0)
    list(loader)
    assert len(seen) == 6


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32"])
def test_batches_are_cast_to_the_declared_dtype(dtype):
    """Every batch carries the requested dtype, whatever the source and preprocess_fn produced."""
    data = np.arange(24, dtype="int16").reshape(12, 2)
    loader = DataLoader(
        chunked_factory(data, 5),
        batch_size=4,
        sample_shape=(2,),
        total_size=12,
        dtype=dtype,
        preprocess_fn=lambda b: b / 2.0,
    )
    batches = list(loader)
    assert {b.dtype for b in batches} == {np.dtype(dtype)}
    np.testing.assert_array_equal(np.concatenate(batches), (data / 2.0).astype(dtype))
