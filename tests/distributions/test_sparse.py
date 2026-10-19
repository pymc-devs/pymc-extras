#   Copyright 2026 The PyMC Developers
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
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose

from pymc_extras import SparseBetaBinomial, SparseDirichletMultinomial, SparseNegativeBinomial


@pytest.mark.parametrize(
    "value,parameter",
    [
        ([2, 0, 3], [1.0, 2.0, 3.0]),
        ([[2, 0, 3], [0, 0, 5]], [1.0, 2.0, 3.0]),
        ([2, 0, 3], [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        ([[[2, 0, 3]], [[0, 0, 5]]], [[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]),
        ([0, 0, 0], [1.0, 2.0, 3.0]),
        ([5], [2.0]),
        (np.empty((0, 3), dtype="int64"), np.ones((1, 3))),
    ],
)
def test_dm_logp_and_gradient_match_stock(value, parameter):
    value = np.asarray(value, dtype="int64")
    parameter = np.asarray(parameter, dtype=pytensor.config.floatX)
    theta = pt.tensor("theta", shape=parameter.shape)
    kwargs = {"a": theta, "n": value.sum(axis=-1)}
    dense_logp = pm.logp(pm.DirichletMultinomial.dist(**kwargs), value)
    sparse_logp = pm.logp(SparseDirichletMultinomial.dist(**kwargs), value)
    fn = pytensor.function(
        [theta],
        [
            dense_logp,
            sparse_logp,
            pt.grad(dense_logp.sum(), theta),
            pt.grad(sparse_logp.sum(), theta),
        ],
    )
    dense_logp, sparse_logp, dense_grad, sparse_grad = fn(parameter)
    tol = 1e-5 if pytensor.config.floatX == "float32" else 1e-10
    assert_allclose(sparse_logp, dense_logp, atol=tol, rtol=tol)
    assert_allclose(sparse_grad, dense_grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("name", ["NegativeBinomial", "BetaBinomial"])
@pytest.mark.parametrize("scalar", [True, False])
def test_scalar_count_logp_and_gradient_match_stock(name, scalar):
    theta = pt.scalar("theta") if scalar else pt.matrix("theta")
    value = pt.lscalar("value") if scalar else pt.lmatrix("value")
    log_scale = pt.scalar("log_scale")
    scale = pt.exp(log_scale)
    if name == "NegativeBinomial":
        parameters = {"mu": pt.exp(theta), "alpha": scale}
        stock, sparse = pm.NegativeBinomial, SparseNegativeBinomial
    else:
        p = pt.sigmoid(theta)
        parameters = {"n": 5, "alpha": scale * p, "beta": scale * (1 - p)}
        stock, sparse = pm.BetaBinomial, SparseBetaBinomial
    inputs = [theta, log_scale]
    outputs = []
    for dist in [stock, sparse]:
        logp = pm.logp(dist.dist(**parameters), value)
        outputs.extend([logp, *pt.grad(logp.sum(), inputs)])
    fn = pytensor.function([value, *inputs], outputs)
    counts = np.array(2 if scalar else [[0, 2, 5], [3, 0, 1]], dtype="int64")
    point = np.asarray(
        0.2 if scalar else [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]], dtype=pytensor.config.floatX
    )
    scales = [2.0, 1e12] if name == "NegativeBinomial" else [2.0, 20.0]
    for counts in [counts, np.zeros_like(counts)]:
        for scale in scales:
            result = fn(counts, point, np.log(scale))
            for dense, sparse in zip(result[:3], result[3:], strict=True):
                assert_allclose(sparse, dense, rtol=1e-5, atol=1e-8)
