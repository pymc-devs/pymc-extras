import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from better_optimize.constants import minimize_method
from numpy.typing import ArrayLike
from pytensor.tensor import TensorVariable
from pytensor.tensor.optimize import minimize


def get_conditional_gaussian_approximation(
    x: TensorVariable,
    Q: TensorVariable | ArrayLike,
    mu: TensorVariable | ArrayLike,
    model: pm.Model | None = None,
    method: minimize_method = "BFGS",
    use_jac: bool = True,
    use_hess: bool = False,
    optimizer_kwargs: dict | None = None,
) -> list[TensorVariable]:
    """
    Returns an estimate the a posteriori probability of a latent Gaussian field x and its mode x0 using the Laplace approximation.

    That is:
    y | x, sigma ~ N(Ax, sigma^2 W)
    x | params ~ N(mu, Q(params)^-1)

    We seek to estimate p(x | y, params) with a Gaussian:

    log(p(x | y, params)) = log(p(y | x, params)) + log(p(x | params)) + const

    Let f(x) = log(p(y | x, params)). From the definition of our model above, we have log(p(x | params)) = -0.5*(x - mu).T Q (x - mu) + 0.5*logdet(Q).

    This gives log(p(x | y, params)) = f(x) - 0.5*(x - mu).T Q (x - mu) + 0.5*logdet(Q). We will estimate this using the Laplace approximation by Taylor expanding f(x) about the mode.

    Thus:

    1. Maximize log(p(x | y, params)) = f(x) - 0.5*(x - mu).T Q (x - mu) wrt x (note that logdet(Q) does not depend on x) to find the mode x0.

    2. Use the Laplace approximation expanded about the mode: p(x | y, params) ~= N(mu=x0, tau=Q - f''(x0)).

    Parameters
    ----------
    x: TensorVariable
        The parameter with which to maximize wrt (that is, find the mode in x). In INLA, this is the latent Gaussian field x~N(mu,Q^-1).
    Q: TensorVariable | ArrayLike
        The precision matrix of the latent field x.
    mu: TensorVariable | ArrayLike
        The mean of the latent field x.
    model: Model
        PyMC model to use.
    method: minimize_method
        Which minimization algorithm to use.
    use_jac: bool
        If true, the minimizer will compute the gradient of log(p(x | y, params)).
    use_hess: bool
        If true, the minimizer will compute the Hessian log(p(x | y, params)).
    optimizer_kwargs: dict
        Kwargs to pass to scipy.optimize.minimize.

    Returns
    -------
    x0, p(x | y, params): list[TensorVariable]
        Mode and Laplace approximation for posterior.
    """
    model = pm.modelcontext(model)

    # f = log(p(y | x, params))
    f_x = model.logp()

    # log(p(x | y, params)) only including terms that depend on x for the minimization step (logdet(Q) ignored as it is a constant wrt x)
    log_x_posterior = f_x - 0.5 * (x - mu).T @ Q @ (x - mu)

    # Maximize log(p(x | y, params)) wrt x to find mode x0
    x0, _ = minimize(
        objective=-log_x_posterior,
        x=x,
        method=method,
        jac=use_jac,
        hess=use_hess,
        optimizer_kwargs=optimizer_kwargs,
    )

    # require f''(x0) for Laplace approx
    hess = pytensor.gradient.hessian(f_x, x)
    hess = pytensor.graph.replace.graph_replace(hess, {x: x0})

    # Could be made more efficient with adding diagonals only
    tau = Q - hess

    # Currently x is passed both as the query point for f(x, args) = logp(x | y, params) AND as an initial guess for x0. This may cause issues if the query point is
    # far from the mode x0 or in a neighbourhood which results in poor convergence.
    return x0, pm.MvNormal(f"{x.name}_laplace_approx", mu=x0, tau=tau)


def get_log_marginal_likelihood(
    x: TensorVariable,
    Q: TensorVariable | ArrayLike,
    mu: TensorVariable | ArrayLike,
    model: pm.Model | None = None,
    method: minimize_method = "BFGS",
    use_jac: bool = True,
    use_hess: bool = False,
    optimizer_kwargs: dict | None = None,
) -> TensorVariable:
    model = pm.modelcontext(model)

    x0, laplace_approx = get_conditional_gaussian_approximation(
        x, Q, mu, model, method, use_jac, use_hess, optimizer_kwargs
    )
    log_laplace_approx = pm.logp(laplace_approx, model.rvs_to_values[x])

    _, logdetQ = pt.nlinalg.slogdet(Q)
    log_x_likelihood = (
        -0.5 * (x - mu).T @ Q @ (x - mu) + 0.5 * logdetQ - 0.5 * x.shape[0] * np.log(2 * np.pi)
    )

    log_likelihood = (  # logp(y | params) =
        model.logp()  # logp(y | x, params)
        + log_x_likelihood  # * logp(x | params)
        - log_laplace_approx  # / logp(x | y, params)
    )

    return log_likelihood


def fit_INLA(
    x: TensorVariable,
    Q: TensorVariable | ArrayLike,
    mu: TensorVariable | ArrayLike,
    model: pm.Model | None = None,
    method: minimize_method = "BFGS",
    use_jac: bool = True,
    use_hess: bool = False,
    optimizer_kwargs: dict | None = None,
) -> az.InferenceData:
    model = pm.modelcontext(model)

    # logp(y | params)
    log_likelihood = get_log_marginal_likelihood(
        x, Q, mu, model, method, use_jac, use_hess, optimizer_kwargs
    )

    # TODO How to obtain prior? It can parametrise Q, mu, y, etc. Not sure if we could extract from model.logp somehow. Otherwise simply specify as a user input
    prior = None
    params = None
    log_prior = pm.logp(prior, model.rvs_to_values[params])

    # logp(params | y) = logp(y | params) + logp(params) + const
    log_posterior = log_likelihood + log_prior

    # TODO log_marginal_x_likelihood is almost the same as log_likelihood, but need to do some sampling?
    log_marginal_x_likelihood = None
    log_marginal_x_posterior = log_marginal_x_likelihood + log_prior

    # TODO can we sample over log likelihoods?
    # Marginalize params
    idata_params = log_posterior.sample()  # TODO something like NUTS, QMC, etc.?
    idata_x = log_marginal_x_posterior.sample()

    # Bundle up idatas somehow
    return idata_params, idata_x
