import numpy as np
import pymc as pm
import pytensor
import pytest

from numpy.testing import assert_allclose
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import MvNormalRV

import pymc_extras.statespace.filters  # registers local_mvnormal_zero_cov


def _has_mvnormal(draw) -> bool:
    canonical = rewrite_graph(draw, include=("canonicalize",))
    return any(
        var.owner is not None and isinstance(var.owner.op, MvNormalRV)
        for var in ancestors([canonical])
    )


@pytest.mark.parametrize("size", [None, (3,)], ids=["no_size", "size"])
def test_zero_cov_mvnormal_is_replaced_by_its_mean(size):
    mean = np.array([1.0, 2.0])
    draw = pm.MvNormal.dist(mu=mean, cov=np.zeros((2, 2)), method="cholesky", size=size)

    assert not _has_mvnormal(draw)
    assert_allclose(pytensor.function([], draw)(), np.broadcast_to(mean, (*(size or ()), 2)))


def test_nonzero_cov_mvnormal_is_kept():
    draw = pm.MvNormal.dist(mu=np.zeros(2), cov=np.diag([0.0, 1.0]), method="svd")

    assert _has_mvnormal(draw)
