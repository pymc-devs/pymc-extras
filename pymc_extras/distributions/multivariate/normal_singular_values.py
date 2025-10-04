#   Copyright 2025 The PyMC Developers
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

import pytensor.tensor as pt

from pymc.distributions.continuous import Continuous
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.shape_utils import (
    rv_size_is_none,
)
from pymc.distributions.transforms import _default_transform
from pymc.pytensorf import normalize_rng_param
from pytensor.tensor import get_underlying_scalar_constant_value
from pytensor.tensor.random.utils import (
    normalize_size_param,
)

__all__ = ["NormalSingularValues"]

from pymc.logprob.transforms import Transform


# TODO: this is a lot of work to just get a list normally distributed variables
class NormalSingularValuesRV(SymbolicRandomVariable):
    name = "normalsingularvalues"
    extended_signature = "[rng],[size],(),(m)->[rng],(m)"  # TODO: check if this is correct
    _print_name = ("NormalSingularValuesRV", "\\operatorname{NormalSingularValuesRV}")

    def make_node(self, rng, size, n, m):
        n = pt.as_tensor_variable(n)
        m = pt.as_tensor_variable(m)
        if not all(n.type.broadcastable) or not all(m.type.broadcastable):
            raise ValueError("n and m must be scalars.")

        return super().make_node(rng, size, n, m)

    @classmethod
    def rv_op(cls, n: int, m: int, *, rng=None, size=None):
        # We flatten the size to make operations easier, and then rebuild it
        n = pt.as_tensor(n, ndim=0, dtype=int)
        m = pt.as_tensor(m, ndim=0, dtype=int)

        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        # TODO: currently assume size = 1. Fix this once everything is working
        D = get_underlying_scalar_constant_value(n)
        Q = get_underlying_scalar_constant_value(m)

        # Perform a direct computation via SVD of a normal matrix
        sz = [] if rv_size_is_none(size) else size
        next_rng, z = pt.random.normal(0, 1, size=(*sz, D, Q), rng=rng).owner.outputs
        _, samples, _ = pt.linalg.svd(z)

        return cls(
            inputs=[rng, size, n, m],
            outputs=[next_rng, samples],
        )(rng, size, n, m)

        return samples


# This is adapted from ordered transform.
# Might make sense to just make that transform more generic by
# allowing it to take parameters "positive" and "ascending"
# and then just use that here.
class PosRevOrdered(Transform):
    name = "posrevordered"

    def __init__(self, ndim_supp=None):
        pass

    def backward(self, value, *inputs):
        return pt.cumsum(pt.exp(value[..., ::-1]), axis=-1)[..., ::-1]

    def forward(self, value, *inputs):
        y = pt.zeros(value.shape)
        y = pt.set_subtensor(y[..., -1], pt.log(value[..., -1]))
        y = pt.set_subtensor(y[..., :-1], pt.log(value[..., :-1] - value[..., 1:]))
        return y

    def log_jac_det(self, value, *inputs):
        return pt.sum(value, axis=-1)


class NormalSingularValues(Continuous):
    rv_type = NormalSingularValuesRV
    rv_op = NormalSingularValuesRV.rv_op

    @classmethod
    def dist(cls, n, m, **kwargs):
        n = pt.as_tensor_variable(n).astype(int)
        m = pt.as_tensor_variable(m).astype(int)
        return super().dist([n, m], **kwargs)

    def support_point(rv, *args):
        return pt.linspace(1, 0.5, rv.shape[-1])

    def logp(sigma, n, m):
        # First term: prod[exp(-0.5*sigma**2)]
        log_p = -0.5 * pt.sum(sigma**2)

        # Second + Fourth term (ignoring constant factor)
        # prod(sigma**(D-Q-1)) + prod(2*sigma)) = prod(2*sigma**(D-Q))
        log_p += (n - m) * pt.sum(pt.log(sigma))

        # Third term: prod[prod[ |s1**2-s2**2| ]]
        # li = pt.triu_indices(m,k=1)
        # log_p += pt.log((sigma[:,None]**2 - sigma[None,:]**2)[li]).sum()
        log_p += (
            pt.log(pt.eye(m) + pt.abs(sigma[:, None] ** 2 - sigma[None, :] ** 2) + 1e-6).sum() / 2.0
        )

        return log_p


@_default_transform.register(NormalSingularValues)
def lkjcorr_default_transform(op, rv):
    return PosRevOrdered()
