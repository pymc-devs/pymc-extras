#   Copyright 2024 The PyMC Developers
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

from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _support_point,
)
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.utils import normalize_size_param

__all__ = ["JointCategorical"]


def _unravel_states(
    flat: TensorVariable, n_states: TensorVariable, n_lags: int
) -> list[TensorVariable]:
    """Base-``n_states`` digits of ``flat`` (most-significant state first)."""
    return [(flat // n_states ** (n_lags - 1 - i)) % n_states for i in range(n_lags)]


class JointCategoricalRV(SymbolicRandomVariable):
    """Joint distribution over ``n_lags`` categorical states."""

    n_lags: int
    default_output = 1
    _print_name = ("JointCategorical", "\\operatorname{JointCategorical}")

    def __init__(self, *args, n_lags, **kwargs):
        self.n_lags = n_lags
        # One core axis per lag, all of the same length (the repeated name enforces that).
        self.extended_signature = f"[size],({','.join('s' * n_lags)}),[rng]->[rng],(l)"
        super().__init__(*args, **kwargs)

    def rebuild_rv(self, *args, **kwargs):
        # rv_op is a classmethod, so it can't see n_lags on the instance; inject it so the default
        # change_dist_size (which calls rebuild_rv) can resize the RV automatically.
        return self.rv_op(*args, n_lags=self.n_lags, **kwargs)

    @classmethod
    def rv_op(cls, logit_p, *, n_lags, size=None, rng=None):
        logit_p = pt.as_tensor_variable(logit_p)
        size = normalize_size_param(size)
        if rng is None:
            rng = pytensor.shared(np.random.default_rng())

        # A leading size broadcasts logit_p to (*size, *core_shape); raveling the state axes turns
        # it into a flat Categorical, which draws one joint state per batch element.
        if rv_size_is_none(size):
            bcast_logit_p = logit_p
        else:
            bcast_logit_p = pt.broadcast_to(
                logit_p, pt.concatenate([size, logit_p.shape[-n_lags:]])
            )

        next_rng, flat = pm.Categorical.dist(
            logit_p=pt.join_dims(bcast_logit_p, start_axis=-n_lags), rng=rng, return_next_rng=True
        )
        states = pt.stack(_unravel_states(flat, logit_p.shape[-1], n_lags), axis=-1)

        return cls(
            inputs=[size, logit_p, rng],
            outputs=[next_rng, states],
            n_lags=n_lags,
        )(size, logit_p, rng)


class JointCategorical(Distribution):
    r"""Joint distribution over ``n_lags`` categorical states.

    A draw is a vector :math:`(x_0, \dots, x_{n\_lags - 1})`, each entry taking one of ``n_states``
    values. Unlike a product of independent categoricals, the entries can be arbitrarily correlated:
    the distribution is parameterized by the probability of every one of the ``n_states ** n_lags``
    configurations, one array axis per entry.

    .. math::

        \Pr(x_0, \dots, x_{n\_lags - 1}) = p[x_0, \dots, x_{n\_lags - 1}]

    Parameters
    ----------
    p : tensor_like of float
        Probability of each state configuration, with shape ``(n_states,) * n_lags``: one axis per
        entry of a draw, so ``p[i, j, ...]`` is the probability of drawing ``[i, j, ...]``. Any
        leading axes are batch dimensions. Note the parameter grows exponentially in ``n_lags``,
        as the joint is stored explicitly.
    logit_p : tensor_like of float
        Alternative parametrization, as unnormalized log-probabilities of the same shape. Only one
        of ``p`` or ``logit_p`` may be given.
    n_lags : int
        Number of states in the joint. This is the support dimensionality: a single draw is a
        vector of length ``n_lags``. The number of values each entry can take is taken from the
        trailing axes of ``p`` / ``logit_p``.

    Examples
    --------
    A joint over two binary states. Of the four possible paths, only the two that stay put,
    ``[0, 0]`` and ``[1, 1]``, carry appreciable mass, so the pair is strongly correlated even
    though each state is marginally 50/50:

    .. code-block:: python

        import numpy as np
        import pymc as pm
        import pymc_extras as pmx

        # One axis per lag, so p[i, j] is the probability of drawing [i, j].
        p = np.array(
            [
                [0.45, 0.05],  # [0, 0], [0, 1]
                [0.05, 0.45],  # [1, 0], [1, 1]
            ]
        )
        dist = pmx.JointCategorical.dist(p=p, n_lags=2)

        pm.draw(dist, draws=5)
        # array([[1, 1],
        #        [0, 0],
        #        [0, 0],
        #        [1, 1],
        #        [0, 1]])

    Each row is one draw: a vector of ``n_lags=2`` states, the first entry being :math:`x_0` and
    the second :math:`x_1`. Switching paths like ``[0, 1]`` show up in roughly one draw in ten.
    Setting those entries to exactly ``0.0`` instead would make them impossible, leaving a joint
    supported on two configurations only.

    Notes
    -----
    This is the distribution of the smoothed initial states of a marginalized
    :class:`~pymc_extras.distributions.DiscreteMarkovChain` with ``n_lags > 1``, where the first
    ``n_lags`` states are correlated a posteriori.
    """

    rv_type = JointCategoricalRV
    rv_op = JointCategoricalRV.rv_op

    @classmethod
    def dist(cls, p=None, *, logit_p=None, n_lags, **kwargs):
        if (p is None) == (logit_p is None):
            raise ValueError("Must specify exactly one of p or logit_p.")
        if p is not None:
            logit_p = pt.log(p)
        logit_p = pt.as_tensor_variable(logit_p)
        return super().dist([logit_p], n_lags=n_lags, **kwargs)


@_support_point.register(JointCategoricalRV)
def joint_categorical_moment(op, rv, size, logit_p, state_rng):
    n_lags = op.n_lags
    k = logit_p.shape[-1]
    if not rv_size_is_none(size):
        logit_p = pt.broadcast_to(logit_p, pt.concatenate([size, logit_p.shape[-n_lags:]]))
    flat = pt.argmax(pt.join_dims(logit_p, start_axis=-n_lags), axis=-1)
    return pt.stack(_unravel_states(flat, k, n_lags), axis=-1)


@_logprob.register(JointCategoricalRV)
def joint_categorical_logp(op, values, size, logit_p, state_rng, **kwargs):
    value = values[0]
    n_lags = op.n_lags
    k = logit_p.shape[-1]
    flat = sum(value[..., i] * k ** (n_lags - 1 - i) for i in range(n_lags))
    return logp(pm.Categorical.dist(logit_p=pt.join_dims(logit_p, start_axis=-n_lags)), flat)
