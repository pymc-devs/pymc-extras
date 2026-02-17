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

import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.continuous import Continuous
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.shape_utils import (
    rv_size_is_none,
)
from pymc.pytensorf import normalize_rng_param
from pytensor.tensor import get_underlying_scalar_constant_value
from pytensor.tensor.random.utils import (
    normalize_size_param,
)

__all__ = ["Spherical"]


class SphericalRV(SymbolicRandomVariable):
    name = "spherical"
    extended_signature = "[rng],[size],(n)->[rng],(n)"  # TODO: check if this is correct
    _print_name = ("SphericalRV", "\\operatorname{SphericalRV}")

    def make_node(self, rng, size, n):
        n = pt.as_tensor_variable(n)
        return super().make_node(rng, size, n)

    @classmethod
    def rv_op(cls, n, *, rng=None, size=None):
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        n = pt.as_tensor(n, ndim=0, dtype=int)
        nv = get_underlying_scalar_constant_value(n)

        # Perform a direct computation via SVD of a normal matrix
        sz = [] if rv_size_is_none(size) else size

        next_rng, z = pt.random.normal(0, 1, size=(*sz, nv), rng=rng).owner.outputs
        samples = z / pt.sqrt(z * z.sum(axis=-1, keepdims=True) + 1e-6)
        # TODO: scale by the .dist given

        return cls(
            inputs=[rng, size, n],
            outputs=[next_rng, samples],
        )(rng, size, n)

        return samples


class Spherical(Continuous):
    rv_type = SphericalRV
    rv_op = SphericalRV.rv_op

    @classmethod
    def dist(cls, n, **kwargs):
        n = pt.as_tensor_variable(n).astype(int)
        return super().dist([n], **kwargs)

    def support_point(rv, size, n, *args):
        return pt.ones(rv.shape) / pt.sqrt(n)

    def logp(value, n):
        # TODO: take dist as a parameter instead of hardcoding
        dist = pm.Gamma.dist(50, 50)

        # Get the radius
        r = pt.sqrt(pt.sum(value**2))

        # Get the log prior of the radius
        log_p = pm.logp(dist, r)
        # log_p = pm.logp(pm.TruncatedNormal.dist(1,lower=0),r)

        # Add the log det jacobian for radius
        log_p += (value.shape[-1] - 1) * pt.log(r)

        return log_p
