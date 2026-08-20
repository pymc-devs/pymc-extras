from __future__ import annotations

import pytensor.tensor as pt

from pymc import Model
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable

from pymc_extras.inference.advi.autoguide import AutoGuideModel


def get_logp_logq(
    model: Model,
    guide: AutoGuideModel,
    path_derivative_gradient: bool = True,
    logp_scalings: dict[TensorVariable, float] | None = None,
):
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
    logp_scalings : dict, optional
        Maps model variables to a factor multiplying their logp term, e.g. the
        ``total_size / batch_size`` rescaling that makes a minibatch log-likelihood an
        unbiased estimate of the full-data one.

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

    if logp_scalings:
        all_vars = model.free_RVs + model.observed_RVs + model.potentials
        logps = model.logp(vars=all_vars, sum=False)
        scales = pt.constant([logp_scalings.get(var, 1.0) for var in all_vars])
        model_logp = pt.sum([pt.sum(logp) * scale for logp, scale in zip(logps, scales)])
    else:
        model_logp = model.logp()

    logp = graph_replace(model_logp, inputs_to_guide_rvs)
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
