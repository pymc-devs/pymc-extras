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

The API mirrors ``torch.utils.data``: a re-iterable source of rows is turned into
fixed-size, optionally shuffled batches by a :class:`DataLoader`. A source yields
blocks of rows: the leading axis is the rows, so ``block.shape[1:]`` is one sample.
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
        raise ValueError("total_size='auto' counted 0 rows (empty source).")
    if next(new_iter(), None) is None:
        raise ValueError(
            "total_size='auto' counted rows but the source is not re-readable "
            "(it returns a one-shot iterator, or closes over an already-consumed one); "
            "pass a source that makes a fresh iterator each epoch, "
            "or total_size=N explicitly."
        )
    return count


def shuffle_buffer(
    chunk_source: Callable[[], Iterator[np.ndarray]],
    *,
    buffer_size: int,
    batch_size: int,
    seed: int | None = None,
) -> Callable[[], Iterator[np.ndarray]]:
    """Wrap a block source into a shuffled, fixed-size batch source.

    Fills a buffer of at least ``buffer_size`` rows, shuffles, and yields
    ``batch_size`` slices. Each epoch draws a fresh permutation from ``seed``.
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
        Wrap the source in a bounded :func:`shuffle_buffer`.
    buffer_size : int, optional
        Shuffle-buffer size in rows when ``shuffle=True``; defaults to
        ``50 * batch_size``.
    seed : int, optional
        Seed for the shuffle buffer (ignored when ``shuffle=False``).
    total_size : int or "auto", default "auto"
        The dataset size ``N``, or ``"auto"`` to infer it (from the source's
        ``n_rows`` if available, else one counting pass). ``None`` disables
        the rescaling.

    Examples
    --------
    .. code-block:: python

        loader = DataLoader(
            parquet_source("shuffled/"),
            batch_size=4096,
            total_size="auto",
        )

        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 3.0, shape=4)
            batch = pm.Data("batch", np.zeros((4096, 4)))
            logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
            pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=loader.total_size)

        with model:
            for next_batch in loader:
                model.set_data("batch", next_batch)
                ...
    """

    def __init__(
        self,
        dataset: Iterable[np.ndarray] | Callable[[], Iterator[np.ndarray]],
        *,
        batch_size: int,
        shuffle: bool = False,
        buffer_size: int | None = None,
        seed: int | None = None,
        total_size: int | str | None = "auto",
    ):
        self._new_iter = new_iter = _as_source(dataset)
        self._batch_size = int(batch_size)

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
                self._new_iter, buffer_size=buffer_size, batch_size=self._batch_size, seed=seed
            )
        else:
            self._batch_source = self._new_iter

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_size(self) -> int | None:
        """The dataset size ``N`` (pass to the distribution's ``total_size``)."""
        return self._total_size

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one epoch of ``batch_size``-row minibatches."""
        yield from self._batch_source()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self._total_size is None:
            raise TypeError(
                "len(DataLoader) requires total_size; "
                "construct with total_size=N or total_size='auto'."
            )
        return self._total_size // self._batch_size


def _check_columns(schema, columns: list[str], path: str) -> None:
    """Reject a shard whose schema cannot supply ``columns`` as a float batch."""
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

    Yields one ``(rows, n_columns)`` array per row group. Carries ``n_rows`` from
    Parquet metadata so ``total_size="auto"`` is free.
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
