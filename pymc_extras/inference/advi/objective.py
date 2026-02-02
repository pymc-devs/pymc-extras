from __future__ import annotations

from pymc import Model
from pytensor import graph_replace
from pytensor.tensor import TensorVariable

from pymc_extras.inference.advi.autoguide import AutoGuideModel


def get_logp_logq(model: Model, guide: AutoGuideModel, stick_the_landing: bool = True):
    """
    Compute the log probability of the model and the guide.

    Parameters
    ----------
    model : Model
        The probabilistic model.
    guide : AutoGuideModel
        The variational guide.
    stick_the_landing : bool, optional
        Whether to use the stick-the-landing (STL) gradient estimator, by default True.
        The STL estimator has lower gradient variance by removing the score function term
        from the gradient. When True, gradients are stopped from flowing through logq.

    Returns
    -------
    logp : TensorVariable
        Log probability of the model.
    logq : TensorVariable
        Log probability of the guide.
    """

    inputs_to_guide_rvs = {
        model_value_var: guide.model[rv.name]
        for rv, model_value_var in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }

    logp = graph_replace(model.logp(), inputs_to_guide_rvs)
    logq = guide.stochastic_logq(stick_the_landing=stick_the_landing)

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
