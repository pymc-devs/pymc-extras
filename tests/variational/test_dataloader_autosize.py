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
"""total_size='auto' resolution, the total_size sanity warning, and parquet_source."""

from contextlib import nullcontext

import numpy as np
import pytest

from pymc_extras.variational.dataloader import (
    DataLoader,
    parquet_source,
)
from tests.variational.dataloader_helpers import BlockDataset, chunked_factory, write_parquet


def test_auto_counts_finite_source():
    """Without .n_rows, 'auto' does one counting pass and resolves the true N."""
    data = np.arange(60, dtype="float64").reshape(60, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(chunked_factory(data, 7), batch_size=10, total_size="auto")
    assert ds.total_size == 60


@pytest.mark.parametrize("shuffle", [False, True], ids=["plain", "shuffled"])
def test_auto_uses_n_rows_fast_path(shuffle):
    """A source-advertised .n_rows is trusted without a counting pass, shuffle or not."""
    src = chunked_factory(np.zeros((8, 1)), 4)
    src.n_rows = 1000
    ds = DataLoader(
        src,
        batch_size=4,
        shuffle=shuffle,
        buffer_size=8,
        seed=0,
        total_size="auto",
    )
    assert ds.total_size == 1000


def test_auto_rejects_one_shot_iterator():
    """A bare generator would be consumed by the counting pass, so 'auto' refuses it."""
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with pytest.raises(ValueError, match="re-readable"):
        DataLoader(one_shot, batch_size=4, total_size="auto")


def test_auto_rejects_factory_returning_same_one_shot_iterator():
    """A factory handing back the same consumed iterator is not re-readable."""
    data = np.zeros((20, 1))
    one_shot = (data[i : i + 4] for i in range(0, 20, 4))
    with (
        pytest.warns(UserWarning, match="counting pass"),
        pytest.raises(ValueError, match="fresh iterator"),
    ):
        DataLoader(lambda: one_shot, batch_size=4, total_size="auto")


def test_auto_accepts_any_n_rows():
    """A source .n_rows is trusted as-is (like PyTorch)."""
    f = chunked_factory(np.zeros((8, 1)), 4)
    f.n_rows = 0
    ds = DataLoader(f, batch_size=4, total_size="auto")
    assert ds.total_size == 0


@pytest.mark.parametrize(
    "n, batch_size, total_size, stray, warns",
    [
        (20, 4, 20, 0, False),
        (25, 10, 25, 0, False),
        (100, 10, 100, 3, False),
        (100, 10, 130, 3, True),
        (100, 10, 95, 0, True),
    ],
    ids=[
        "exact",
        "drop-last-truncates",
        "stray-pass-first",
        "wrong-size-after-stray",
        "fewer-rows-than-streamed",
    ],
)
def test_total_size_sanity_check(n, batch_size, total_size, stray, warns):
    """The epoch-boundary check reads the pass that just completed, not the cumulative rows."""
    data = np.arange(n, dtype="float64").reshape(n, 1)
    ds = DataLoader(chunked_factory(data, 5), batch_size=batch_size, total_size=total_size)
    partial = iter(ds)
    for _ in range(stray):
        next(partial)
    if warns:
        with pytest.warns(UserWarning, match="disagrees with"):
            list(ds)
    else:
        list(ds)


def test_pass_too_short_for_one_batch_warns():
    """A source that cannot fill a single batch streams nothing, which must not pass silently."""
    data = np.arange(5, dtype="float64").reshape(5, 1)
    ds = DataLoader(chunked_factory(data, 5), batch_size=10, total_size=5)
    with pytest.warns(UserWarning, match="no complete batch"):
        assert list(ds) == []


def test_auto_counts_unshuffled_source_when_shuffling_non_divisible():
    """With shuffle=True 'auto' counts the unshuffled source, so the dropped batch can't lower N."""
    data = np.arange(125, dtype="float64").reshape(125, 1)
    with pytest.warns(UserWarning, match="counting pass"):
        ds = DataLoader(
            chunked_factory(data, 125),
            batch_size=10,
            shuffle=True,
            buffer_size=30,
            seed=0,
            total_size="auto",
        )
    assert ds.total_size == 125


def test_total_size_warning_fires_once_across_passes():
    """A wrong total_size warns on the first full pass and stays quiet on the next."""
    data = np.arange(40, dtype="float64").reshape(20, 2)
    ds = DataLoader(chunked_factory(data, 5), batch_size=5, total_size=10_000)
    with pytest.warns(UserWarning, match="disagrees with"):
        batches = list(ds)
    assert len(batches) == 4
    assert list(ds) != []


def test_auto_rejects_factory_closing_over_consumed_iterator():
    """A generator function over a consumed iterator returns a new but empty stream."""
    data = np.zeros((20, 1))
    underlying = iter([data[i : i + 4] for i in range(0, 20, 4)])

    def gen():
        yield from underlying

    with (
        pytest.warns(UserWarning, match="counting pass"),
        pytest.raises(ValueError, match="fresh iterator"),
    ):
        DataLoader(gen, batch_size=4, total_size="auto")


def test_parquet_source_n_rows_from_metadata(tmp_path):
    """n_rows comes from file metadata, and total_size='auto' takes it without counting."""
    rng = np.random.default_rng(0)
    total = 0
    for i in range(3):
        n = 100 + 50 * i
        total += n
        block = rng.normal(size=(n, 2))
        write_parquet(tmp_path / f"part_{i:02d}.parquet", {"a": block[:, 0], "b": block[:, 1]})
    src = parquet_source(str(tmp_path))
    assert src.n_rows == total

    ds = DataLoader(src, batch_size=10, total_size="auto")
    assert ds.total_size == total


def test_parquet_source_columns_and_shard_order(tmp_path):
    """columns= selects a column subset and shards are read in sorted path order."""
    for i in range(2):
        write_parquet(
            tmp_path / f"part_{i}.parquet",
            {"a": [float(i)] * 2, "b": [9.0] * 2, "c": [float(10 + i)] * 2},
        )
    src = parquet_source(str(tmp_path), columns=["a", "c"])
    blocks = list(src)
    assert [b.shape for b in blocks] == [(2, 2), (2, 2)]
    np.testing.assert_array_equal(blocks[0][:, 0], [0.0, 0.0])
    np.testing.assert_array_equal(blocks[1][:, 1], [11.0, 11.0])


def test_parquet_source_empty_dir_raises(tmp_path):
    """A directory with no matching Parquet files raises a clear error."""
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="no Parquet files match"):
        parquet_source(str(tmp_path))


def test_parquet_source_freezes_column_order_across_permuted_shards(tmp_path):
    """A shard whose schema permutes the columns is read back in the first shard's order."""
    write_parquet(tmp_path / "p0.parquet", {"a": [1.0, 1.0], "b": [10.0, 10.0]})
    write_parquet(tmp_path / "p1.parquet", {"b": [20.0, 20.0], "a": [2.0, 2.0]})
    blocks = list(parquet_source(str(tmp_path)))
    np.testing.assert_array_equal(blocks[0], [[1.0, 10.0], [1.0, 10.0]])
    np.testing.assert_array_equal(blocks[1], [[2.0, 20.0], [2.0, 20.0]])


def test_parquet_source_streams_row_groups_not_whole_files(tmp_path):
    """A multi-row-group file is yielded one row group at a time, not one file at a time."""
    write_parquet(tmp_path / "p.parquet", {"a": np.arange(30.0)}, row_group_size=10)
    blocks = list(parquet_source(str(tmp_path)))
    assert [b.shape for b in blocks] == [(10, 1), (10, 1), (10, 1)]
    np.testing.assert_array_equal(np.concatenate(blocks).ravel(), np.arange(30.0))


def test_parquet_source_names_a_later_shard_with_a_non_numeric_column(tmp_path):
    """A later shard whose column turned non-numeric is named by path, not an opaque cast error."""
    write_parquet(tmp_path / "p0.parquet", {"a": [1.0, 2.0]})
    write_parquet(tmp_path / "p1.parquet", {"a": ["bad", "worse"]})
    src = parquet_source(str(tmp_path))
    with pytest.raises(ValueError, match=r"p1\.parquet.*not numeric"):
        list(src)


def test_parquet_source_rejects_non_numeric_columns(tmp_path):
    """A string column is rejected at construction, naming the column and the columns= remedy."""
    write_parquet(tmp_path / "p.parquet", {"x": [1.0, 2.0], "id": ["a", "b"]})
    with pytest.raises(ValueError, match="not numeric"):
        parquet_source(str(tmp_path))
    src = parquet_source(str(tmp_path), columns=["x"])
    np.testing.assert_array_equal(next(iter(src)), [[1.0], [2.0]])


def test_parquet_source_names_the_shard_missing_a_column(tmp_path):
    """read_row_group drops unknown names silently, so a shard missing a frozen column is named."""
    write_parquet(tmp_path / "p0.parquet", {"a": [1.0], "b": [2.0]})
    write_parquet(tmp_path / "p1.parquet", {"a": [3.0]})
    src = parquet_source(str(tmp_path))
    with pytest.raises(ValueError, match=r"p1\.parquet"):
        list(src)


def test_parquet_source_rejects_unknown_columns(tmp_path):
    """A typo in columns= raises at construction, not as a pyarrow error at first iteration."""
    write_parquet(tmp_path / "p.parquet", {"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="not found"):
        parquet_source(str(tmp_path), columns=["a", "nope"])


@pytest.mark.parametrize(
    "make_source, counts",
    [
        (lambda d: d, True),
        (lambda d: chunked_factory(d, 7), True),
        (lambda d: [d[i : i + 7] for i in range(0, len(d), 7)], True),
        (lambda d: BlockDataset(d, 7), True),
        (lambda d: BlockDataset(d, 7, n_rows=45), False),
    ],
    ids=["raw-array", "factory", "reiterable-blocks", "iterable-dataset", "advertised-n-rows"],
)
def test_auto_resolves_the_explicit_n_for_every_source_kind(make_source, counts):
    """'auto' lands on the same N an explicit total_size would, counting only when it must.

    A source that does not advertise n_rows is counted, a BlockDataset that
    inherits the default ``n_rows=None`` is counted too, and either way the epoch that
    follows the counting pass is the whole dataset.
    """
    data = np.arange(90, dtype="float64").reshape(45, 2)
    counting = pytest.warns(UserWarning, match="counting pass") if counts else nullcontext()
    with counting:
        ds = DataLoader(make_source(data), batch_size=6, total_size="auto")
    assert ds.total_size == len(ds) == 45
    np.testing.assert_array_equal(np.concatenate(list(ds)), data[:42])
