import arviz as az
import pymc as pm

from pytensor.tensor import TensorLike, TensorVariable

from pymc_extras.model.marginal.marginal_model import marginalize


def fit_INLA(
    x: TensorVariable,
    Q: TensorLike,
    minimizer_seed: int = 42,
    model: pm.Model | None = None,
    minimizer_kwargs: dict = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}},
    return_latent_posteriors: bool = False,
    **sampler_kwargs,
) -> az.InferenceData:
    r"""
    Performs inference over a linear mixed model using Integrated Nested Laplace Approximations (INLA). Assumes a model of the form:

    .. math::

        \theta \rightarrow x \rightarrow y

    Where the prior on the hyperparameters :math:`\pi(\theta)` is arbitrary, the prior on the latent field is Gaussian (and in precision form): :math:`\pi(x) = N(\mu, Q^{-1})` and the latent field is linked to the observables $y$ through some linear map.

    As it stands, INLA in PyMC Extras has three main limitations:

    - Does not support inference over the latent field, only the hyperparameters.
    - Optimisation for :math:`\mu^*` is bottlenecked by calling `minimize`, and to a lesser extent, computing the hessian :math:`f^"(x)`.
    - Does not offer sparse support which can provide significant speedups.

    Parameters
    ----------
    x: TensorVariable
        The latent gaussian to marginalize out.
    Q: TensorLike
        Precision matrix of the latent field.
    minimizer_seed: int
        Seed for random initialisation of the minimum point x*.
    model: pm.Model
        PyMC model.
    minimizer_kwargs:
        Kwargs to pass to pytensor.optimize.minimize during the optimization step maximizing logp(x | y, params).
    returned_latent_posteriors:
        If True, also return posteriors for the latent Gaussian field (currently unsupported).
    sampler_kwargs:
        Kwargs to pass to pm.sample.
    """
    model = pm.modelcontext(model)

    # TODO is there a better way to check if it's a RV?
    # print(vars(Q.owner))
    # if isinstance(Q, TensorVariable) and "module" in vars(Q.owner):
    Q = model.rvs_to_values[Q] if isinstance(Q, TensorVariable) else Q

    # Marginalize out the latent field
    marginalize_kwargs = {
        "Q": Q,
        "minimizer_seed": minimizer_seed,
        "minimizer_kwargs": minimizer_kwargs,
    }
    marginal_model = marginalize(model, x, use_laplace=True, **marginalize_kwargs)

    # Sample over the hyperparameters
    if not return_latent_posteriors:
        idata = pm.sample(model=marginal_model, **sampler_kwargs)
        return idata

    # Unmarginalize stuff
    raise NotImplementedError(
        "Inference over the latent field with INLA is currently unsupported. Set return_latent_posteriors to False"
    )
