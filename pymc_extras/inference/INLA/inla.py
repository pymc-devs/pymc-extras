import arviz as az
import pymc as pm

from pytensor.tensor import TensorVariable

from pymc_extras.model.marginal.marginal_model import marginalize


def fit_INLA(
    x: TensorVariable,
    Q: TensorVariable,
    minimizer_seed: int = 42,
    model: pm.Model | None = None,
    minimizer_kwargs: dict = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}},
    return_latent_posteriors: bool = False,
    **sampler_kwargs,
) -> az.InferenceData:
    """
    TODO ADD DOCSTRING
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
    idata = pm.sample(model=marginal_model, **sampler_kwargs)

    if not return_latent_posteriors:
        return idata

    # Unmarginalize stuff
    raise NotImplementedError(
        "Inference over the latent field with INLA is currently unsupported. Set return_latent_posteriors to False"
    )
