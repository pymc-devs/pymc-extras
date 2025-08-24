import warnings

import arviz as az
import pymc as pm

from pymc.distributions.multivariate import MvNormal
from pytensor.tensor import TensorVariable
from pytensor.tensor.linalg import inv as matrix_inverse

from pymc_extras.model.marginal.marginal_model import marginalize


def fit_INLA(
    x: TensorVariable,
    temp_kwargs=None,  # TODO REMOVE. DEBUGGING TOOL
    model: pm.Model | None = None,
    minimizer_kwargs: dict | None = None,
    return_latent_posteriors: bool = True,
    **sampler_kwargs,
) -> az.InferenceData:
    warnings.warn("Currently only valid for a nested normal model. WIP.", UserWarning)

    model = pm.modelcontext(model)

    # Check if latent field is Gaussian
    if not isinstance(x.owner.op, MvNormal):
        raise ValueError(
            f"Latent field {x} is not instance of MvNormal. Has distribution {x.owner.op}."
        )

    _, _, _, tau = x.owner.inputs

    # Latent field should use precison rather than covariance
    if not (tau.owner and tau.owner.op == matrix_inverse):
        raise ValueError(
            f"Latent field {x} is not in precision matrix form. Use MvNormal(tau=Q) instead."
        )

    Q = tau.owner.inputs[0]

    # Marginalize out the latent field
    minimizer_kwargs = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}}
    marginalize_kwargs = {"Q": Q, "temp_kwargs": temp_kwargs, "minimizer_kwargs": minimizer_kwargs}
    marginal_model = marginalize(model, x, use_laplace=True, **marginalize_kwargs)

    # Sample over the hyperparameters
    idata = pm.sample(model=marginal_model, **sampler_kwargs)

    if not return_latent_posteriors:
        return idata

    # TODO Unmarginalize stuff
    raise NotImplementedError("Latent posteriors not supported yet, WIP.")
