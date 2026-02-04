from pymc import Model
from pytensor import graph_replace
from pytensor.tensor import TensorVariable

from pymc_extras.inference.advi.autoguide import AutoGuideModel


def get_logp_logq(model: Model, guide: AutoGuideModel):
    inputs_to_guide_rvs = {
        model_value_var: guide.model[rv.name]
        for rv, model_value_var in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }

    logp = graph_replace(model.logp(), inputs_to_guide_rvs)
    logq = guide.stochastic_logq()

    return logp, logq


def advi_objective(logp: TensorVariable, logq: TensorVariable):
    """Compute the negative ELBO objective for ADVI.

    Parameters
    ----------
    logp : TensorVariable
        Log probability of the model.
    logq : TensorVariable
        Log probability of the guide.

    Returns
    -------
    TensorVariable
        The negative ELBO.
    """
    negative_elbo = logq - logp
    return negative_elbo
