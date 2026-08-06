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
"""Stream out-of-core data into a PyMC model, one batch at a time.

The full dataset never has to be resident: peak memory is set by the batch, not by
the dataset size N. Iteration runs one batch ahead, so budget two batches plus the
source chunks they hold and any shuffle buffer.

The API mirrors ``torch.utils.data``: a re-iterable source of rows
(e.g. :func:`parquet_source` over a directory of shards, read a chunk at a time)
is turned into fixed-size, optionally shuffled batches by a :class:`DataLoader`. A source yields blocks of rows: the
leading axis is the rows, so ``block.shape[1:]`` is one sample and nothing has to
be declared. One difference from torch: ``len(loader)`` is the row count ``N``,
not the batch count.

Every batch has exactly ``batch_size`` rows, so each pass drops the final
``N mod batch_size`` rows (torch's ``drop_last``). Shuffling is only as good as
the order the source yields rows in; a bounded buffer just block-shuffles
time-ordered data, so pre-shuffle on disk (or interleave shards) for well-mixed
batches, and/or pass ``shuffle=True``.
"""

from __future__ import annotations

import glob
import os
import warnings

from collections.abc import Callable, Iterable, Iterator

import numpy as np

__all__ = ["DataLoader", "parquet_source", "shuffle_buffer"]


def _as_source(
    dataset: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
) -> Callable[[], Iterator[np.ndarray]]:
    """Normalize any accepted source into a zero-arg factory returning a fresh iterator."""
    if isinstance(dataset, Iterator):
        used = {"done": False}

        def new_iter() -> Iterator[np.ndarray]:
            if used["done"]:
                raise RuntimeError(
                    "source is a bare iterator and was already consumed; the loader "
                    "restarts the stream each epoch, so pass a zero-arg factory or a "
                    "re-iterable instead"
                )
            used["done"] = True
            return dataset

        return new_iter

    if isinstance(dataset, np.ndarray):
        # Iterating an array yields its rows one at a time, which loses the
        # distinction between a row and a block; hand it over whole instead.
        return lambda: iter((dataset,))

    make = dataset if callable(dataset) else (lambda: dataset)
    return lambda: iter(make())


def _auto_total_size(
    dataset: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
    new_iter: Callable[[], Iterator[np.ndarray]],
) -> int:
    """Resolve ``total_size="auto"``: trust a source ``.n_rows``, else count once."""
    n_rows = getattr(dataset, "n_rows", None)
    if n_rows is not None:
        return int(n_rows)
    if isinstance(dataset, Iterator):
        raise ValueError(
            "total_size='auto' needs a re-readable source (a zero-arg factory or an "
            "iterable), not a one-shot iterator; pass total_size=N explicitly instead."
        )
    warnings.warn(
        "total_size='auto' is doing a full counting pass over the source; for a cheap "
        "path use a source exposing .n_rows (e.g. parquet_source, from Parquet metadata).",
        UserWarning,
        stacklevel=3,
    )
    first = new_iter()
    count = 0
    for chunk in first:
        count += int(np.asarray(chunk).shape[0])
    if count <= 0:
        raise ValueError("total_size='auto' counted 0 rows (empty or non-re-readable source).")
    second = new_iter()
    if second is first or next(second, None) is None:
        raise ValueError(
            "total_size='auto' counted rows but the source's next stream was empty "
            "(it returns the same one-shot iterator, or closes over an already-"
            "consumed one); pass a source that makes a fresh iterator each epoch, "
            "or total_size=N explicitly."
        )
    return count


def _rebatch(blocks: Iterable[np.ndarray], batch_size: int) -> Iterator[np.ndarray]:
    """Slice a stream of row blocks into exact ``batch_size``-row batches, in order.

    Blocks may be any number of rows; remainders carry across them so no row is
    lost mid-stream. Trailing rows that do not fill a final batch are dropped when
    the stream ends (``drop_last``; the model observes a fixed-shape placeholder).
    A block that has to survive another pull is copied, since a source may hand
    back a view into a buffer it overwrites; sources that already yield exact
    ``batch_size`` blocks pass through uncopied.
    """
    buf: list[np.ndarray] = []
    have = 0
    for arr in blocks:
        a = np.asarray(arr)
        have += a.shape[0]
        if have < batch_size:
            buf.append(np.array(a))
            continue
        buf.append(a)
        merged = np.concatenate(buf, axis=0) if len(buf) > 1 else buf[0]
        n_full = merged.shape[0] // batch_size
        for i in range(n_full):
            yield merged[i * batch_size : (i + 1) * batch_size]
        rem = merged.shape[0] - n_full * batch_size
        buf = [merged[n_full * batch_size :].copy()] if rem else []
        have = rem


def shuffle_buffer(
    chunk_source: Callable[[], Iterator[np.ndarray]],
    *,
    buffer_size: int,
    batch_size: int,
    seed: int | None = None,
) -> Callable[[], Iterator[np.ndarray]]:
    """Wrap a block source into a shuffled, fixed-size batch source.

    Fills a buffer of at least ``buffer_size`` rows from ``chunk_source`` (a
    zero-arg factory yielding blocks), shuffles it, and yields ``batch_size``
    slices; a remainder carries into the next fill and a final partial batch is
    dropped. Each epoch (each call of the returned factory) draws a fresh
    permutation from ``seed``, reproducible per seed.

    ``DataLoader(shuffle=True)`` calls this for you; use it directly only to control
    ``buffer_size`` explicitly. A bounded buffer cannot fix strongly ordered data --
    pre-shuffle on disk for that. ``buffer_size`` is a lower bound, and the chunk that
    crosses it is kept whole, so peak allocation is about twice
    ``max(buffer_size, batch_size)`` plus one chunk. Blocks are copied as they fill
    the buffer, since a source may hand back a view into a buffer it overwrites.
    """
    seed_seq = np.random.SeedSequence(seed)

    def factory() -> Iterator[np.ndarray]:
        rng = np.random.default_rng(seed_seq.spawn(1)[0])
        # chunk_source() may be any re-iterable; normalize to one iterator so each
        # fill continues the stream instead of restarting it.
        it = iter(chunk_source())
        carry: np.ndarray | None = None
        exhausted = False
        # Accumulate at least one batch even if buffer_size < batch_size, else the
        # guard below would discard the whole stream.
        target = max(buffer_size, batch_size)
        while not exhausted:
            bufs: list[np.ndarray] = []
            have = 0
            if carry is not None:
                bufs.append(carry)
                have += carry.shape[0]
                carry = None
            for arr in it:
                a = np.array(arr)
                bufs.append(a)
                have += a.shape[0]
                if have >= target:
                    break
            else:
                exhausted = True
            if have < batch_size:
                return
            buf = np.concatenate(bufs, axis=0)
            rng.shuffle(buf)
            n_full = buf.shape[0] // batch_size
            for i in range(n_full):
                yield buf[i * batch_size : (i + 1) * batch_size]
            rem = buf.shape[0] - n_full * batch_size
            carry = buf[n_full * batch_size :].copy() if rem else None

    return factory


class DataLoader:
    """Turn an out-of-core dataset into fixed-size minibatches for variational inference.

    Parameters
    ----------
    dataset : iterable of ndarray, or zero-arg factory
        The source of rows. A factory is preferred: it restarts the stream each
        epoch. It may yield single samples or blocks of any size.
    batch_size : int
        Leading dimension of every yielded minibatch.
    shuffle : bool, default False
        Wrap the source in a bounded :func:`shuffle_buffer`. See the module
        docstring for what that does and does not fix, and for ``drop_last``.
    buffer_size : int, optional
        Shuffle-buffer size in rows when ``shuffle=True``; defaults to
        ``50 * batch_size``. A buffer as large as the dataset is a full shuffle.
    seed : int, optional
        Seed for the shuffle buffer (ignored when ``shuffle=False``).
    dtype : str, default "float64"
        Dtype each batch is cast to; match the ``pm.Data`` placeholder's dtype.
    total_size : int or "auto", default "auto"
        The dataset size ``N``, or ``"auto"`` to infer it (from the source's
        ``n_rows`` if available, else one counting pass). ``None`` warns and
        disables the rescaling; a non-positive value raises.
    preprocess_fn : callable, optional
        Applied to each batch before it is yielded; must preserve the row count.

    Examples
    --------
    .. code-block:: python

        loader = DataLoader(
            parquet_source("shuffled/"),  # a re-iterable source over the shards
            batch_size=4096,  # each row group is (rows, 4): 3 features + 1 observed
            total_size="auto",  # infer N from the source; N == len(loader)
        )

        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 3.0, shape=4)
            batch = pm.Data("batch", np.zeros((4096, 4)))
            logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
            pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))

        with model:
            for next_batch in loader:  # each epoch yields (batch_size, 4) arrays
                model.set_data("batch", next_batch)
                ...  # one optimization step over this batch
    """

    def __init__(
        self,
        dataset: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        shuffle: bool = False,
        buffer_size: int | None = None,
        seed: int | None = None,
        dtype: str = "float64",
        total_size: int | str | None = "auto",
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._new_iter = new_iter = _as_source(dataset)
        self._batch_size = int(batch_size)
        self._dtype = dtype
        self._preprocess_fn = preprocess_fn

        if total_size == "auto":
            total_size = _auto_total_size(dataset, new_iter)
        elif total_size is None:
            warnings.warn(
                "DataLoader created with total_size=None: the minibatch "
                "log-likelihood will not be rescaled and the posterior will be "
                "biased. Pass total_size=N (the true dataset size) or total_size='auto'.",
                UserWarning,
                stacklevel=2,
            )
        self._total_size = None if total_size is None else int(total_size)

        if shuffle:
            if buffer_size is None:
                buffer_size = 50 * self._batch_size
            self._batch_source = shuffle_buffer(
                self._blocks, buffer_size=buffer_size, batch_size=self._batch_size, seed=seed
            )
        else:
            self._batch_source = self._blocks

        self._warned_size = False

    def _blocks(self) -> Iterator[np.ndarray]:
        """One epoch of source blocks, checked for a consistent trailing shape.

        Rows are the leading axis, so ``block.shape[1:]`` is one sample. A source
        that changes that shape mid-stream cannot be concatenated into batches, and
        numpy would only say so once the shapes happened to collide.
        """
        trailing = None
        for arr in self._new_iter():
            a = np.asarray(arr)
            if a.ndim == 0:
                raise ValueError(
                    "source yielded a scalar; blocks must be arrays whose leading "
                    "axis is the rows, so a single sample of shape S is shape (1, *S)"
                )
            if trailing is None:
                trailing = a.shape[1:]
            elif a.shape[1:] != trailing:
                raise ValueError(
                    f"source yielded a block of shape {a.shape}, but earlier blocks had "
                    f"trailing shape {trailing}; every block must be (rows, *sample_shape) "
                    f"with the same sample shape"
                )
            yield a

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_size(self) -> int | None:
        """The dataset size ``N`` (pass to the distribution's ``total_size``)."""
        return self._total_size

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of ``batch_size``-row minibatches.

        Stream each into the model's ``pm.Data`` placeholder with ``model.set_data``
        before a step; re-iterate for another epoch. The pass runs one batch ahead so
        the ``total_size`` check still fires when a fit stops exactly at the epoch
        boundary, which also means the source is pulled one batch further than the
        last batch yielded.
        """
        seen = 0
        it = _rebatch(self._batch_source(), self._batch_size)
        batch = next(it, None)
        if batch is None:
            self._maybe_warn_total_size(0)
        while batch is not None:
            if self._preprocess_fn is not None:
                batch = self._preprocess_fn(batch)
            prepared = np.array(batch, dtype=self._dtype)
            seen += int(prepared.shape[0])
            following = next(it, None)
            if following is None:
                self._maybe_warn_total_size(seen)
            yield prepared
            batch = following

    def __len__(self) -> int:
        """The dataset size ``N`` -- rows, not batches. Same value as :attr:`total_size`."""
        if self._total_size is None:
            raise TypeError(
                "len(DataLoader) is the dataset size N, but this loader was built with "
                "total_size=None; construct it with total_size=N or total_size='auto'."
            )
        return self._total_size

    def _maybe_warn_total_size(self, seen: int) -> None:
        """Warn once if ``total_size`` disagrees with the rows of one full pass.

        A correct ``N`` satisfies ``seen <= N < seen + batch_size``, since the
        trailing partial batch is dropped. Streaming more rows than ``N`` always
        warns; over-declaring gets 10% slack to absorb approximate sizes.
        """
        if self._warned_size or self._total_size is None:
            return
        self._warned_size = True
        if not seen:
            warnings.warn(
                f"the source yielded no complete batch of {self._batch_size} rows, so "
                f"this pass streamed nothing. Lower batch_size, or check that the "
                f"source is not empty.",
                UserWarning,
                stacklevel=3,
            )
            return
        if seen <= self._total_size < seen + self._batch_size:
            return
        if self._total_size < seen or self._total_size - seen > 0.1 * seen:
            warnings.warn(
                f"total_size={self._total_size} disagrees with the {seen} rows streamed "
                f"in one full pass; the N/batch_size rescaling is likely wrong. Pass the "
                f"true dataset size, or fix the source's n_rows if 'auto' read it there.",
                UserWarning,
                stacklevel=3,
            )


def _check_columns(schema, columns: list[str], path: str) -> None:
    """Reject a shard whose schema cannot supply ``columns`` as a float batch.

    ``read_row_group(columns=...)`` silently drops unknown names, and a non-numeric
    column only blows up later at the float cast, so both are named against ``path``.
    """
    import pyarrow as pa

    missing = [c for c in columns if c not in schema.names]
    if missing:
        raise ValueError(
            f"columns {missing} not found in {path!r}; available: {sorted(schema.names)}"
        )
    numeric = (pa.types.is_integer, pa.types.is_floating, pa.types.is_boolean)
    bad = [c for c in columns if not any(t(schema.field(c).type) for t in numeric)]
    if bad:
        raise ValueError(
            f"columns {bad} in {path!r} are not numeric and cannot be streamed into a "
            f"float batch; select numeric columns with columns=."
        )


class _ParquetDataset:
    def __init__(self, paths: list[str], columns: list[str], n_rows: int):
        self._paths = paths
        self._columns = columns
        self.n_rows = n_rows

    def __iter__(self) -> Iterator[np.ndarray]:
        import pyarrow.parquet as pq

        for path in self._paths:
            file = pq.ParquetFile(path)
            # parquet_source only ever sees shard 0, so re-check every shard here.
            _check_columns(file.schema_arrow, self._columns, path)
            for i in range(file.metadata.num_row_groups):
                table = file.read_row_group(i, columns=self._columns)
                # Stack by the frozen names: a permuted shard must not swap features.
                yield np.column_stack([table.column(c).to_numpy() for c in self._columns])


def parquet_source(
    directory: str,
    *,
    columns: list[str] | None = None,
    pattern: str = "*.parquet",
) -> _ParquetDataset:
    """A re-iterable source over a directory of Parquet files.

    Yields one ``(rows, n_columns)`` array per row group (one or more per file),
    so peak read memory is one row group, not one file. The column order is
    frozen at construction (``columns`` if given, else the first file's schema
    order) and every shard is read in that order, so a shard with a permuted
    schema cannot silently reorder features. Carries an ``n_rows`` from Parquet
    metadata (no data scan) so ``total_size="auto"`` resolves the dataset size for
    free. Pass ``shuffle=True`` to the :class:`DataLoader` for shuffled batches.
    """
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise ValueError(f"no Parquet files match {os.path.join(directory, pattern)!r}")
    schema = pq.read_schema(paths[0])
    if columns is None:
        columns = list(schema.names)
    _check_columns(schema, columns, paths[0])
    n_rows = sum(pq.read_metadata(p).num_rows for p in paths)
    return _ParquetDataset(paths, columns, n_rows)
