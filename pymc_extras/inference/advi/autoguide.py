#   Copyright 2025 - present The PyMC Developers
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
from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from pymc.distributions import Normal
from pymc.logprob.basic import conditional_logp
from pymc.model.core import Deterministic, Model
from pytensor.graph.basic import Variable

from pymc_extras.inference.advi.pytensorf import get_symbolic_rv_shapes


@dataclass(frozen=True)
class AutoGuideModel:
    model: Model
    params_init_values: dict[Variable, np.ndarray]
    name_to_param: dict[str, Variable] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "name_to_param",
            {x.name: x for x in self.params_init_values.keys()},
        )

    @property
    def params(self) -> tuple[Variable, ...]:
        return tuple(self.params_init_values.keys())

    def __getitem__(self, name: str) -> Variable:
        return self.name_to_param[name]

    def stochastic_logq(self):
        """Returns a graph representing the logp of the guide model, evaluated under draws from its random variables."""
        # This allows arbitrary
        logp_terms = conditional_logp(
            {rv: rv for rv in self.model.deterministics},
            warn_rvs=False,
        )
        return pt.sum([logp_term.sum() for logp_term in logp_terms.values()])


def AutoDiagonalNormal(model) -> AutoGuideModel:
    coords = model.coords
    free_rvs = model.free_RVs

    free_rv_shapes = dict(zip(free_rvs, get_symbolic_rv_shapes(free_rvs)))
    params_init_values = {}

    with Model(coords=coords) as guide_model:
        for rv in free_rvs:
            loc = pt.tensor(f"{rv.name}_loc", shape=rv.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=rv.type.shape)
            # TODO: Make these customizable
            params_init_values[loc] = pt.random.uniform(-1, 1, size=free_rv_shapes[rv]).eval()
            params_init_values[scale] = pt.full(free_rv_shapes[rv], 0.1).eval()

            z = Normal(
                f"{rv.name}_z",
                mu=0,
                sigma=1,
                shape=free_rv_shapes[rv],
            )
            Deterministic(
                rv.name,
                loc + pt.softplus(scale) * z,
                dims=model.named_vars_to_dims.get(rv.name, None),
            )

    return AutoGuideModel(guide_model, params_init_values)
