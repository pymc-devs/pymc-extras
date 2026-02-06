from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from pymc.distributions import Normal
from pymc.logprob.basic import conditional_logp
from pymc.model.core import Deterministic, Model
from pytensor import graph_replace
from pytensor.gradient import disconnected_grad
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

    def stochastic_logq(self, stick_the_landing: bool = True) -> pt.TensorVariable:
        """Returns a graph representing the logp of the guide model, evaluated under draws from its random variables."""
        logp_terms = conditional_logp(
            {rv: rv for rv in self.model.deterministics},
            warn_rvs=False,
        )
        logq = pt.sum([logp_term.sum() for logp_term in logp_terms.values()])

        if stick_the_landing:
            # Detach variational parameters from the gradient computation of logq
            repl = {p: disconnected_grad(p) for p in self.params}
            logq = graph_replace(logq, repl)

        return logq


def AutoDiagonalNormal(model: Model) -> AutoGuideModel:
    """
    Create a guide model for ADVI with a mean-field normal distribution.

    A guide model is a variational distribution that approximates the posterior distribution of the model's free
    random variables. In this case, we use a mean-field normal distribution, which assumes that the free random
    variables are independent and normally distributed. For details, see _[1].

    For each free random variable in the model, we create a corresponding random variable in the guide model with a
    normal distribution. The mean and standard deviation of each normal distribution are parameterized by learnable
    parameters (loc and scale), which are initialized to small random values.

    Parameters
    ----------
    model : Model
        The probabilistic model for which to create the guide.

    Returns
    -------
    guide_model : AutoGuideModel
        An AutoGuideModel containing the guide model and the initial values for its parameters.

    References
    ----------
    .. [1] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic Differentiation
           Variational Inference. Journal of Machine Learning Research, 18(14):1–45, 2017.
    """
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
