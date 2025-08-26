import arviz as az
import pymc as pm

from pytensor.tensor import TensorVariable

from pymc_extras.model.marginal.marginal_model import marginalize


def fit_INLA(
    x: TensorVariable,
    Q: TensorVariable,
    minimizer_seed: int = 42,
    model: pm.Model | None = None,
    minimizer_kwargs: dict | None = None,
    return_latent_posteriors: bool = True,
    **sampler_kwargs,
) -> az.InferenceData:
    model = pm.modelcontext(model)

    # Check if latent field is Gaussian
    # if not isinstance(x.owner.op, MvNormal):
    #     raise ValueError(
    #         f"Latent field {x} is not instance of MvNormal. Has distribution {x.owner.op}."
    #     )

    # _, _, _, tau = x.owner.inputs

    # # Latent field should use precison rather than covariance
    # if not (tau.owner and tau.owner.op == matrix_inverse):
    #     raise ValueError(
    #         f"Latent field {x} is not in precision matrix form. Use MvNormal(tau=Q) instead."
    #     )

    # Q = tau.owner.inputs[0]

    # TODO is there a better way to check if it's a RV?
    # print(vars(Q.owner))
    # if isinstance(Q, TensorVariable) and "module" in vars(Q.owner):
    Q = model.rvs_to_values[Q]

    # Marginalize out the latent field
    minimizer_kwargs = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}}
    marginalize_kwargs = {
        "Q": Q,
        "minimizer_seed": minimizer_seed,
        "minimizer_kwargs": minimizer_kwargs,
    }
    marginal_model = marginalize(model, x, use_laplace=True, **marginalize_kwargs)

    # Sample over the hyperparameters
    # marginal_model.logp().dprint()
    idata = pm.sample(model=marginal_model, **sampler_kwargs)

    if not return_latent_posteriors:
        return idata

    # TODO Unmarginalize stuff
    raise NotImplementedError("Latent posteriors not supported yet, WIP.")
