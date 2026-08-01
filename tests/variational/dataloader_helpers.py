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
"""Shared helpers for the DataLoader tests."""

import numpy as np
import pytest


def chunked_factory(data, size):
    """A zero-arg factory replaying ``data`` in ``size``-row chunks, fresh each epoch."""

    def factory():
        for i in range(0, len(data), size):
            yield data[i : i + size]

    return factory


def reused_buffer_factory(n_blocks, rows):
    """A factory that refills and re-yields one array, as an out-of-core reader may.

    Block ``i`` is ``rows`` copies of ``i``, so a batch that aliases an overwritten
    buffer shows up as a repeated or missing value.
    """

    def factory():
        buf = np.empty((rows, 1))
        for value in range(n_blocks):
            buf.fill(value)
            yield buf

    return factory


def write_parquet(path, columns, **kwargs):
    """Write ``columns`` to ``path``, skipping the calling test if pyarrow is not installed."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table(columns), str(path), **kwargs)
