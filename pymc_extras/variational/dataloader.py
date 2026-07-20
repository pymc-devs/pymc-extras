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
"""Out-of-core minibatching for variational inference.

``pm.Minibatch`` indexes an array that is fully in memory, so its peak memory is
O(N). This streams minibatches from an out-of-core source into a ``pm.Data``
placeholder instead, so peak memory is set by the batch (plus the source chunk
and any shuffle buffer), not by N.

The API mirrors ``torch.utils.data``: an :class:`IterableDataset` is a
re-iterable source of rows (e.g. :func:`parquet_source` over a directory of
shards, read a chunk at a time), and a :class:`DataLoader` turns it into
fixed-size, optionally shuffled minibatches. One difference: ``len(loader)`` is
the row count ``N`` (what the observed distribution needs for ``total_size``),
not the batch count torch returns.

The rescaling is the same as ``pm.Minibatch``: the observed log-likelihood is
scaled by ``N / batch_size`` via ``create_minibatch_rv``, and ``N`` is
``len(loader)``, so the model passes ``total_size=len(loader)``.

Every batch has exactly ``batch_size`` rows, so each pass drops the final
``N mod batch_size`` rows (torch's ``drop_last``). Shuffling is only as good as
the order the source yields rows in; a bounded buffer just block-shuffles
time-ordered data, so pre-shuffle on disk (or interleave shards) for well-mixed
minibatches, and/or pass ``shuffle=True``.

Examples
--------
.. code-block:: python

    import numpy as np
    import pymc as pm
    from pymc_extras.variational.dataloader import DataLoader, parquet_source

    loader = DataLoader(
        parquet_source("shuffled/"),  # an IterableDataset over the shards
        batch_size=4096,
        sample_shape=(4,),  # 3 features + 1 observed column
        total_size="auto",  # infer N from Parquet metadata; N == len(loader)
    )

    with pm.Model() as model:
        b = pm.Normal("b", 0.0, 3.0, shape=4)
        batch = pm.Data("batch", np.zeros((4096, 4)))
        logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
        pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))

    with model:
        for minibatch in loader:  # each epoch yields validated minibatches
            model.set_data("batch", minibatch)
            ...  # one variational step over this minibatch
"""

from __future__ import annotations

import glob
import numbers
import os
import warnings

from collections.abc import Callable, Iterable, Iterator

import numpy as np

__all__ = ["DataLoader", "IterableDataset", "parquet_source", "shuffle_buffer"]


def _is_positive_int(value: object) -> bool:
    """True for a strictly positive integer (incl. numpy ints), excluding bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) > 0


def _promote_to_block(a: np.ndarray, sample_shape: tuple[int, ...]) -> np.ndarray:
    """Return ``a`` as a ``(rows, *sample_shape)`` block; a single sample becomes one row."""
    if a.shape == sample_shape:
        return a[None, ...]
    if a.ndim != len(sample_shape) + 1 or a.shape[1:] != sample_shape:
        raise ValueError(
            f"source yielded shape {a.shape}; expected one sample of shape "
            f"{sample_shape} or a (rows, *sample_shape) block; if the source is "
            f"right, declare its trailing shape with DataLoader(sample_shape=...)"
        )
    return a


class IterableDataset:
    """A re-iterable, out-of-core source of rows, like ``torch.utils.data.IterableDataset``.

    Subclass and implement :meth:`__iter__` to yield ``np.ndarray`` blocks of rows
    (shape ``(rows, *sample_shape)``); a :class:`DataLoader` re-batches those into
    fixed-size minibatches. ``__iter__`` must return a fresh iterator each call so
    the dataset can be replayed across epochs. Set :attr:`n_rows` if the row count
    is known cheaply (e.g. from file metadata) so ``total_size="auto"`` can skip a
    counting pass.

    A plain array, a zero-arg factory, or any re-iterable also works directly as a
    ``DataLoader`` source; this base class is only for attaching behavior or
    ``n_rows`` to a custom source.
    """

    n_rows: int | None = None

    def __iter__(self) -> Iterator[np.ndarray]:
        raise NotImplementedError("IterableDataset subclasses must implement __iter__")


def _as_source(
    dataset: IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
) -> tuple[Callable[[], Iterator[np.ndarray]], int | None, bool]:
    """Normalize any accepted source into ``(new_iter, n_rows, reiterable)``.

    ``new_iter()`` returns a fresh iterator for one epoch. ``n_rows`` is the
    source's advertised row count (its ``.n_rows``) if it has one, else ``None``.
    ``reiterable`` is ``False`` only for a bare iterator, which can be walked once.
    """
    n_rows = getattr(dataset, "n_rows", None)

    if callable(dataset) and not isinstance(dataset, Iterator):
        # A factory may return any iterable (a list, a generator, ...); normalize
        # so the loader always pulls from a true iterator.
        def new_iter() -> Iterator[np.ndarray]:
            return iter(dataset())  # type: ignore[operator]

        return new_iter, n_rows, True

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

        return new_iter, n_rows, False

    def new_iter() -> Iterator[np.ndarray]:
        return iter(dataset)

    return new_iter, n_rows, True


def _auto_total_size(
    new_iter: Callable[[], Iterator[np.ndarray]],
    n_rows: int | None,
    reiterable: bool,
    sample_shape: tuple[int, ...],
) -> int:
    """Resolve ``total_size="auto"``: trust a source ``.n_rows``, else count once.

    A ``.n_rows`` (e.g. from Parquet metadata) is used directly. Otherwise a single
    counting pass runs over a re-readable source; a one-shot iterator cannot be
    counted (counting consumes it) and must pass ``total_size`` explicitly.
    """
    if n_rows is not None:
        if not _is_positive_int(n_rows):
            raise ValueError(f"source.n_rows must be a positive integer, got {n_rows!r}")
        return int(n_rows)
    if not reiterable:
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
        a = np.asarray(chunk)
        # A yield of shape exactly sample_shape is one sample, not a block.
        count += 1 if a.shape == sample_shape else int(a.shape[0])
    if count <= 0:
        raise ValueError("total_size='auto' counted 0 rows (empty or non-re-readable source).")
    # A genuine source yields a fresh, non-empty stream each call; one that returns
    # the same exhausted iterator would leave nothing to stream. The probe costs one
    # chunk, which the counting pass has already dwarfed.
    second = new_iter()
    if second is first or next(second, None) is None:
        raise ValueError(
            "total_size='auto' counted rows but the source's next stream was empty "
            "(it returns the same one-shot iterator, or closes over an already-"
            "consumed one); pass a source that makes a fresh iterator each epoch, "
            "or total_size=N explicitly."
        )
    return count


def _rebatch(
    blocks: Iterable[np.ndarray],
    batch_size: int,
    sample_shape: tuple[int, ...],
) -> Iterator[np.ndarray]:
    """Slice a stream of samples/blocks into exact ``batch_size``-row batches, in order.

    Accepts single samples (shape ``sample_shape``) and blocks of any size (shape
    ``(rows, *sample_shape)``), carrying remainders across blocks so no row is lost
    mid-stream. Trailing rows that do not fill a final batch are dropped when the
    stream ends (``drop_last``; the model observes a fixed-shape placeholder).
    Sources that already yield exact ``batch_size`` blocks pass through uncopied.
    """
    buf: list[np.ndarray] = []
    have = 0
    for arr in blocks:
        a = _promote_to_block(np.asarray(arr), sample_shape)
        buf.append(a)
        have += a.shape[0]
        if have < batch_size:
            continue
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

    ``DataLoader(shuffle=True)`` calls this for you; use it directly only to
    control ``buffer_size`` explicitly. It only approximates i.i.d. batches for an
    already-unordered stream: a bounded buffer cannot fix strongly ordered data
    (pre-shuffle on disk for that). ``buffer_size`` is a lower bound, and the chunk
    that crosses it is kept whole, so peak allocation is about twice
    ``max(buffer_size, batch_size)`` plus one chunk.
    """
    if not _is_positive_int(batch_size):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not _is_positive_int(buffer_size):
        raise ValueError(f"buffer_size must be a positive integer, got {buffer_size!r}")
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
                a = np.asarray(arr)
                bufs.append(a)
                have += a.shape[0]
                if have >= target:
                    break
            else:
                exhausted = True
            if have < batch_size:
                return  # source exhausted: drop the final partial batch
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

    Like ``torch.utils.data.DataLoader``, it batches (and optionally shuffles) an
    :class:`IterableDataset` into a minibatch stream. It is iterable and sized
    (``len(loader)`` is the dataset size ``N``). With bounded source chunks the
    full dataset is never resident at once.

    Parameters
    ----------
    dataset : IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]]
        The source of rows: an :class:`IterableDataset`, a re-iterable (including a
        plain ``np.ndarray``), or a zero-arg factory returning a fresh iterator
        (preferred, so the stream restarts each epoch). It may yield single samples
        or blocks of any size; the loader re-batches them in order to exactly
        ``batch_size`` rows and drops a trailing partial batch (``drop_last``).
    batch_size : int
        Leading dimension of every yielded minibatch.
    shuffle : bool, default False
        Wrap the source in a bounded :func:`shuffle_buffer`. This only approximates
        i.i.d. batches for an already-unordered stream (pre-shuffle on disk for
        strongly ordered data; see the module docstring).
    buffer_size : int, optional
        Shuffle-buffer size in rows when ``shuffle=True``; defaults to
        ``50 * batch_size``. A buffer as large as the dataset is a full shuffle.
    seed : int, optional
        Seed for the shuffle buffer (ignored when ``shuffle=False``).
    sample_shape : tuple of int, optional
        Trailing shape of one observation. ``()`` for scalars, ``(k,)`` to stream
        ``k`` columns. Defaults to ``dataset.shape[1:]`` for a raw ``np.ndarray``
        (its rows are the samples, like torch's ``TensorDataset``), else ``()``.
    dtype : str, default "float64"
        Dtype each batch is cast to; match the ``pm.Data`` placeholder's dtype.
    total_size : int or "auto", default "auto"
        The dataset size ``N``, or ``"auto"`` to infer it (from the source's
        ``n_rows`` if available, else a single counting pass). Pass it on as
        ``total_size=len(loader)`` so the minibatch log-likelihood is rescaled by
        ``N / batch_size``. ``None`` warns and disables the rescaling (biased
        posterior); a non-positive value raises.
    preprocess_fn : callable, optional
        Pure transform applied to each batch before it is yielded (e.g.
        normalization); it must preserve the row count and ``sample_shape``.
    """

    def __init__(
        self,
        dataset: IterableDataset | Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        shuffle: bool = False,
        buffer_size: int | None = None,
        seed: int | None = None,
        sample_shape: tuple[int, ...] | None = None,
        dtype: str = "float64",
        total_size: int | str | None = "auto",
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if not _is_positive_int(batch_size):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if sample_shape is None:
            # A raw array is rows-of-samples; without this a 2-D array would be read
            # as blocks of scalars and silently flattened.
            sample_shape = dataset.shape[1:] if isinstance(dataset, np.ndarray) else ()
        sample_shape = tuple(sample_shape)

        new_iter, source_n_rows, reiterable = _as_source(dataset)
        self._new_iter = new_iter
        self._batch_size = int(batch_size)
        self._sample_shape = sample_shape
        self._dtype = dtype
        self._preprocess_fn = preprocess_fn

        if total_size == "auto":
            total_size = _auto_total_size(new_iter, source_n_rows, reiterable, sample_shape)
        elif total_size is None:
            warnings.warn(
                "DataLoader created with total_size=None: the minibatch "
                "log-likelihood will not be rescaled and the posterior will be "
                "biased. Pass total_size=N (the true dataset size) or total_size='auto'.",
                UserWarning,
                stacklevel=2,
            )
        elif not _is_positive_int(total_size):
            # 0 is falsy (rescaling silently skipped) and a negative value flips the
            # sign of the data log-likelihood; raise on both.
            raise ValueError(
                "total_size must be a positive integer (the true dataset size N) so "
                "the minibatch log-likelihood is rescaled by N / batch_size; got "
                f"{total_size!r}."
            )
        self._total_size = None if total_size is None else int(total_size)

        # shuffle_buffer needs blocks, so promote single samples to one-row blocks
        # before it; a plain source is re-batched as-is.
        if shuffle:
            if buffer_size is None:
                buffer_size = 50 * self._batch_size

            def blocks() -> Iterator[np.ndarray]:
                for arr in self._new_iter():
                    yield _promote_to_block(np.asarray(arr), self._sample_shape)

            self._batch_source = shuffle_buffer(
                blocks, buffer_size=buffer_size, batch_size=self._batch_size, seed=seed
            )
        else:
            self._batch_source = self._new_iter

        self._batches_seen = 0
        self._rows_streamed = 0
        self._warned_size = False

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_size(self) -> int | None:
        """The dataset size ``N`` (pass to the distribution's ``total_size``)."""
        return self._total_size

    @property
    def batches_seen(self) -> int:
        return self._batches_seen

    @property
    def rows_streamed(self) -> int:
        """Total rows streamed into the model (grows past ``N`` across epochs)."""
        return self._rows_streamed

    def _rebatched(self) -> Iterator[np.ndarray]:
        """A fresh pass of exactly ``batch_size``-row batches from the source."""
        return _rebatch(self._batch_source(), self._batch_size, self._sample_shape)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of ``(batch_size, *sample_shape)`` minibatches.

        Stream each into the model's ``pm.Data`` placeholder with ``model.set_data``
        before a step. Plain iteration is side-effect-free (it does not touch the
        :attr:`batches_seen` / :attr:`rows_streamed` counters); re-iterate for
        another epoch.
        """
        for batch in self._rebatched():
            yield self._prepare(batch)

    def __len__(self) -> int:
        """The dataset size ``N``; pass it to the distribution's ``total_size``.

        This is the row count ``N`` (what ``total_size`` needs), not the batch count
        ``torch.utils.data.DataLoader.__len__`` returns. Same value as
        :attr:`total_size`.
        """
        if self._total_size is None:
            raise TypeError(
                "len(DataLoader) is the dataset size N, but this loader was built with "
                "total_size=None; construct it with total_size=N or total_size='auto'."
            )
        return self._total_size

    def _stream_batches(self) -> Iterator[np.ndarray]:
        """One epoch, but updating the counters and running the total_size check.

        Like :meth:`__iter__`, but it updates :attr:`batches_seen` /
        :attr:`rows_streamed` and fires the one-shot ``total_size`` sanity check on
        the pass's final batch. The Trainer consumes this; plain :meth:`__iter__`
        stays side-effect-free. Kept one batch ahead so the check still fires when a
        fit stops exactly at the pass boundary.
        """
        seen_this_pass = 0
        it = self._rebatched()
        batch = next(it, None)
        while batch is not None:
            following = next(it, None)
            prepared = self._prepare(batch)
            self._batches_seen += 1
            self._rows_streamed += int(prepared.shape[0])
            seen_this_pass += int(prepared.shape[0])
            if following is None:
                self._maybe_warn_total_size(seen_this_pass)
            yield prepared
            batch = following

    def _prepare(self, batch: np.ndarray) -> np.ndarray:
        """Apply ``preprocess_fn`` and return an owned, correctly-typed copy.

        A source may hand back a view into a reused buffer, so the copy prevents the
        consumer from aliasing it (torch does the collate/transform here too).
        """
        if self._preprocess_fn is not None:
            batch = self._preprocess_fn(batch)
        return np.array(batch, dtype=self._dtype)

    def _maybe_warn_total_size(self, seen: int) -> None:
        """Warn once if ``total_size`` is inconsistent with the rows of one full pass.

        ``seen`` is the row count of the pass that just completed. A correct ``N``
        satisfies ``seen <= N < seen + batch_size`` (the trailing partial batch is
        dropped), so that window never warns; outside it a 10% slack absorbs sources
        that are only approximately sized.
        """
        if self._warned_size or self._total_size is None:
            return
        self._warned_size = True
        if not seen or seen <= self._total_size < seen + self._batch_size:
            return
        if abs(self._total_size - seen) > 0.1 * seen:
            warnings.warn(
                f"total_size={self._total_size} disagrees with the {seen} rows streamed "
                f"in one full pass; the N/batch_size rescaling, and therefore the "
                f"posterior width, is likely wrong. Pass the true dataset size (or, if "
                f"'auto' resolved it from the source's n_rows, fix that attribute).",
                UserWarning,
                stacklevel=3,
            )


class _ParquetDataset(IterableDataset):
    """An :class:`IterableDataset` over a directory of Parquet shards.

    Yields one ``(rows, n_columns)`` array per row group (so peak read memory is
    one row group, not one file), in the fixed column order chosen at
    construction, and exposes :attr:`n_rows` read from Parquet metadata (no data
    scan).
    """

    def __init__(self, paths: list[str], columns: list[str], n_rows: int):
        self._paths = paths
        self._columns = columns
        self.n_rows = n_rows

    def __iter__(self) -> Iterator[np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for path in self._paths:
            file = pq.ParquetFile(path)
            schema = file.schema_arrow
            missing = [c for c in self._columns if c not in schema.names]
            if missing:
                # read_row_group(columns=...) silently drops unknown names, so a
                # malformed shard must be named here, not surface as a bare
                # KeyError with no path.
                raise ValueError(f"columns {missing} not found in {path!r}")
            non_numeric = [
                c
                for c in self._columns
                if not (
                    pa.types.is_integer(schema.field(c).type)
                    or pa.types.is_floating(schema.field(c).type)
                    or pa.types.is_boolean(schema.field(c).type)
                )
            ]
            if non_numeric:
                # parquet_source validates types against the first shard only; a
                # later shard whose column turned non-numeric would otherwise
                # become an object array and fail at the batch cast with no path.
                raise ValueError(
                    f"columns {non_numeric} in {path!r} are not numeric and cannot be "
                    f"streamed into a float batch; select numeric columns with columns=."
                )
            for i in range(file.metadata.num_row_groups):
                table = file.read_row_group(i, columns=self._columns)
                # Stack by the frozen column names, not the file's own order, so
                # a shard with a permuted schema cannot silently swap features.
                yield np.column_stack([table.column(c).to_numpy() for c in self._columns])


def parquet_source(
    directory: str,
    *,
    columns: list[str] | None = None,
    pattern: str = "*.parquet",
) -> _ParquetDataset:
    """An :class:`IterableDataset` over a directory of Parquet files.

    Yields one ``(rows, n_columns)`` array per row group (one or more per file),
    so peak read memory is one row group, not one file. The column order is
    frozen at construction (``columns`` if given, else the first file's schema
    order) and every shard is read in that order, so a shard with a permuted
    schema cannot silently reorder features. Carries an ``n_rows`` from Parquet
    metadata (no data scan) so ``total_size="auto"`` resolves the dataset size for
    free. Pass ``shuffle=True`` to the :class:`DataLoader` for shuffled batches.
    """
    # pyarrow is an optional dependency, so it is imported on use.
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise ValueError(f"no Parquet files match {os.path.join(directory, pattern)!r}")
    schema = pq.read_schema(paths[0])
    if columns is None:
        columns = list(schema.names)
    else:
        missing = sorted(set(columns) - set(schema.names))
        if missing:
            raise ValueError(
                f"columns {missing} not found in {paths[0]!r}; available: {sorted(schema.names)}"
            )
    non_numeric = [
        c
        for c in columns
        if not (
            pa.types.is_integer(schema.field(c).type)
            or pa.types.is_floating(schema.field(c).type)
            or pa.types.is_boolean(schema.field(c).type)
        )
    ]
    if non_numeric:
        # A string/dictionary column would turn whole chunks object-dtype and only
        # fail later at the batch cast, without naming the column.
        raise ValueError(
            f"columns {non_numeric} in {paths[0]!r} are not numeric and cannot be "
            f"streamed into a float batch; select numeric columns with columns=."
        )
    n_rows = sum(pq.read_metadata(p).num_rows for p in paths)
    return _ParquetDataset(paths, columns, n_rows)
