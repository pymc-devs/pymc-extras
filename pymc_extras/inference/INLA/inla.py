import warnings

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

    As it stands, INLA in PyMC Extras is currently experimental.

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

    Returns
    -------
    idata: az.InferenceData
        Standard PyMC InferenceData instance.

    Examples
    --------
    .. code:: ipython

        In [1]: rng = np.random.default_rng(123)
           ...: n = 10000
           ...: d = 3
           ...: mu_mu = 10 * rng.random(d)
           ...: mu_true = rng.random(d)
           ...: tau = np.identity(d)
           ...: cov = np.linalg.inv(tau)
           ...: y_obs = rng.multivariate_normal(mean=mu_true, cov=cov, size=n)

        In [2]: with pm.Model() as model:
           ...:     mu = pm.MvNormal("mu", mu=mu_mu, tau=tau)
           ...:     x = pm.MvNormal("x", mu=mu, tau=tau)
           ...:     y = pm.MvNormal("y", mu=x, tau=tau, observed=y_obs)

           ...:     idata = pmx.fit(
           ...:     method="INLA",
           ...:     x=x,
           ...:     Q=tau,
           ...:     return_latent_posteriors=False,
           ...:     )

        In[3]: posterior_mean_true = (mu_mu + mu_true) / 2
           ...: posterior_mean_inla = idata.posterior.mu.mean(axis=(0, 1)).values
           ...: print(posterior_mean_true)
           ...: print(posterior_mean_inla)

        Out[3]:
            [3.50394522 0.35705804 1.50784662]
            [3.48732847 0.35738072 1.46851421]

    """
    warnings.warn(
        "INLA is currently experimental. Please see the INLA Roadmap for more info: https://github.com/pymc-devs/pymc-extras/issues/340.",
        UserWarning,
    )
    model = pm.modelcontext(model)

    # Get the TensorVariable if Q is provided as an RV
    if isinstance(Q, TensorVariable) and Q in model.rvs_to_values.keys():
        Q = model.rvs_to_values[Q]

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
