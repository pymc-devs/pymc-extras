from __future__ import annotations

from pymc import Model
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable

from pymc_extras.inference.advi.autoguide import AutoGuideModel


def get_logp_logq(model: Model, guide: AutoGuideModel, path_derivative_gradient: bool = True):
    """
    Compute the log probability of the model and the guide, evaluated under draws from the guide.

    Parameters
    ----------
    model : Model
        The probabilistic model.
    guide : AutoGuideModel
        The variational guide.
    path_derivative_gradient : bool, optional
        Whether the variational parameters are detached from the density evaluation of logq,
        so that gradients flow only through the random draws, by default True. This does not
        change the value of logq, only its gradients: the score-function term, which has zero
        expectation, is dropped, yielding the lower-variance path-derivative gradient
        estimator of _[1] (also known as "sticking the landing").

    Returns
    -------
    logp : TensorVariable
        Log probability of the model.
    logq : TensorVariable
        Log probability of the guide.

    References
    ----------
    .. [1] Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the Landing: Simple,
           Lower-Variance Gradient Estimators for Variational Inference. NeurIPS, 2017.
    """

    inputs_to_guide_rvs = {
        model_value_var: guide.model[rv.name]
        for rv, model_value_var in model.rvs_to_values.items()
        if rv not in model.observed_RVs
    }

    logp = graph_replace(model.logp(), inputs_to_guide_rvs)
    logq = guide.stochastic_logq(path_derivative_gradient=path_derivative_gradient)

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
