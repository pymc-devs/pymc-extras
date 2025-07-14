#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import logging

from collections.abc import Callable
from functools import partial
from typing import Literal
from typing import cast as type_cast

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from better_optimize.constants import minimize_method
from numpy.typing import ArrayLike
from pymc.blocking import DictToArrayBijection
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import join_nonshared_inputs
from pymc.util import get_default_varnames
from pytensor.graph import vectorize_graph
from pytensor.tensor import TensorVariable
from pytensor.tensor.optimize import minimize
from pytensor.tensor.type import Variable

from pymc_extras.inference.laplace_approx.find_map import (
    _compute_inverse_hessian,
    _make_initial_point,
    find_MAP,
)
from pymc_extras.inference.laplace_approx.scipy_interface import (
    GradientBackend,
    scipy_optimize_funcs_from_loss,
)

_log = logging.getLogger(__name__)


def get_conditional_gaussian_approximation(
    x: TensorVariable,
    Q: TensorVariable | ArrayLike,
    mu: TensorVariable | ArrayLike,
    args: list[TensorVariable] | None = None,
    model: pm.Model | None = None,
    method: minimize_method = "BFGS",
    use_jac: bool = True,
    use_hess: bool = False,
    optimizer_kwargs: dict | None = None,
) -> Callable:
    """
    Returns a function to estimate the a posteriori log probability of a latent Gaussian field x and its mode x0 using the Laplace approximation.

    That is:
    y | x, sigma ~ N(Ax, sigma^2 W)
    x | params ~ N(mu, Q(params)^-1)

    We seek to estimate log(p(x | y, params)):

    log(p(x | y, params)) = log(p(y | x, params)) + log(p(x | params)) + const

    Let f(x) = log(p(y | x, params)). From the definition of our model above, we have log(p(x | params)) = -0.5*(x - mu).T Q (x - mu) + 0.5*logdet(Q).

    This gives log(p(x | y, params)) = f(x) - 0.5*(x - mu).T Q (x - mu) + 0.5*logdet(Q). We will estimate this using the Laplace approximation by Taylor expanding f(x) about the mode.

    Thus:

    1. Maximize log(p(x | y, params)) = f(x) - 0.5*(x - mu).T Q (x - mu) wrt x (note that logdet(Q) does not depend on x) to find the mode x0.

    2. Substitute x0 into the Laplace approximation expanded about the mode: log(p(x | y, params)) ~= -0.5*x.T (-f''(x0) + Q) x + x.T (Q.mu + f'(x0) - f''(x0).x0) + 0.5*logdet(Q).

    Parameters
    ----------
    x: TensorVariable
        The parameter with which to maximize wrt (that is, find the mode in x). In INLA, this is the latent field x~N(mu,Q^-1).
    Q: TensorVariable | ArrayLike
        The precision matrix of the latent field x.
    mu: TensorVariable | ArrayLike
        The mean of the latent field x.
    args: list[TensorVariable]
        Args to supply to the compiled function. That is, (x0, logp) = f(x, *args). If set to None, assumes the model RVs are args.
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
    f: Callable
        A function which accepts a value of x and args and returns [x0, log(p(x | y, params))], where x0 is the mode. x is currently both the point at which to evaluate logp and the initial guess for the minimizer.
    """
    model = pm.modelcontext(model)

    if args is None:
        args = model.continuous_value_vars + model.discrete_value_vars

    # f = log(p(y | x, params))
    f_x = model.logp()
    jac = pytensor.gradient.grad(f_x, x)
    hess = pytensor.gradient.jacobian(jac.flatten(), x)

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

    # require f'(x0) and f''(x0) for Laplace approx
    jac = pytensor.graph.replace.graph_replace(jac, {x: x0})
    hess = pytensor.graph.replace.graph_replace(hess, {x: x0})

    # Full log(p(x | y, params)) using the Laplace approximation (up to a constant)
    _, logdetQ = pt.nlinalg.slogdet(Q)
    conditional_gaussian_approx = (
        -0.5 * x.T @ (-hess + Q) @ x + x.T @ (Q @ mu + jac - hess @ x0) + 0.5 * logdetQ
    )

    # Currently x is passed both as the query point for f(x, args) = logp(x | y, params) AND as an initial guess for x0. This may cause issues if the query point is
    # far from the mode x0 or in a neighbourhood which results in poor convergence.
    return pytensor.function(args, [x0, conditional_gaussian_approx])


def _unconstrained_vector_to_constrained_rvs(model):
    outputs = get_default_varnames(model.unobserved_value_vars, include_transformed=True)
    constrained_names = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    ]
    names = [x.name for x in outputs]

    unconstrained_names = [name for name in names if name not in constrained_names]

    new_outputs, unconstrained_vector = join_nonshared_inputs(
        model.initial_point(),
        inputs=model.value_vars,
        outputs=outputs,
    )

    constrained_rvs = [x for x, name in zip(new_outputs, names) if name in constrained_names]
    value_rvs = [x for x in new_outputs if x not in constrained_rvs]

    unconstrained_vector.name = "unconstrained_vector"

    # Redo the names list to ensure it is sorted to match the return order
    names = [*constrained_names, *unconstrained_names]

    return names, constrained_rvs, value_rvs, unconstrained_vector


def model_to_laplace_approx(
    model: pm.Model, unpacked_variable_names: list[str], chains: int = 1, draws: int = 500
):
    initial_point = model.initial_point()
    raveled_vars = DictToArrayBijection.map(initial_point)
    raveled_shape = raveled_vars.data.shape[0]

    # temp_chain and temp_draw are a hack to allow sampling from the Laplace approximation. We only have one mu and cov,
    # so we add batch dims (which correspond to chains and draws). But the names "chain" and "draw" are reserved.
    names, constrained_rvs, value_rvs, unconstrained_vector = (
        _unconstrained_vector_to_constrained_rvs(model)
    )

    coords = model.coords | {
        "temp_chain": np.arange(chains),
        "temp_draw": np.arange(draws),
        "unpacked_variable_names": unpacked_variable_names,
    }

    with pm.Model(coords=coords, model=None) as laplace_model:
        mu = pm.Flat("mean_vector", shape=(raveled_shape,))
        cov = pm.Flat("covariance_matrix", shape=(raveled_shape, raveled_shape))
        laplace_approximation = pm.MvNormal(
            "laplace_approximation",
            mu=mu,
            cov=cov,
            dims=["temp_chain", "temp_draw", "unpacked_variable_names"],
            method="svd",
        )

        cast_to_var = partial(type_cast, Variable)
        batched_rvs = vectorize_graph(
            type_cast(list[Variable], constrained_rvs),
            replace={cast_to_var(unconstrained_vector): cast_to_var(laplace_approximation)},
        )

        for name, batched_rv in zip(names, batched_rvs):
            batch_dims = ("temp_chain", "temp_draw")
            if batched_rv.ndim == 2:
                dims = batch_dims
            elif name in model.named_vars_to_dims:
                dims = (*batch_dims, *model.named_vars_to_dims[name])
            else:
                dims = (*batch_dims, *[f"{name}_dim_{i}" for i in range(batched_rv.ndim - 2)])
                initval = initial_point.get(name, None)
                dim_shapes = initval.shape if initval is not None else batched_rv.type.shape[2:]
                laplace_model.add_coords(
                    {name: np.arange(shape) for name, shape in zip(dims[2:], dim_shapes)}
                )

            pm.Deterministic(name, batched_rv, dims=dims)

    return laplace_model


def unstack_laplace_draws(laplace_data, model, chains=2, draws=500):
    """
    The `model_to_laplace_approx` function returns a model with a single MvNormal distribution, draws from which are
    in the unconstrained variable space. These might be interesting to the user, but since they come back stacked in a
    single vector, it's not easy to work with.

    This function unpacks each component of the vector into its own DataArray, with the appropriate dimensions and
    coordinates, where possible.
    """
    initial_point = DictToArrayBijection.map(model.initial_point())

    cursor = 0
    unstacked_laplace_draws = {}
    coords = model.coords | {"chain": range(chains), "draw": range(draws)}

    # There are corner cases where the value_vars will not have the same dimensions as the random variable (e.g.
    # simplex transform of a Dirichlet). In these cases, we don't try to guess what the labels should be, and just
    # add an arviz-style default dim and label.
    for rv, (name, shape, size, dtype) in zip(model.free_RVs, initial_point.point_map_info):
        rv_dims = []
        for i, dim in enumerate(
            model.named_vars_to_dims.get(rv.name, [f"{name}_dim_{i}" for i in range(len(shape))])
        ):
            if coords.get(dim) and shape[i] == len(coords[dim]):
                rv_dims.append(dim)
            else:
                rv_dims.append(f"{name}_dim_{i}")
                coords[f"{name}_dim_{i}"] = np.arange(shape[i])

        dims = ("chain", "draw", *rv_dims)

        values = (
            laplace_data[..., cursor : cursor + size].reshape((chains, draws, *shape)).astype(dtype)
        )
        unstacked_laplace_draws[name] = xr.DataArray(
            values, dims=dims, coords={dim: list(coords[dim]) for dim in dims}
        )

        cursor += size

    unstacked_laplace_draws = xr.Dataset(unstacked_laplace_draws)

    return unstacked_laplace_draws


def fit_laplace(
    optimize_method: minimize_method | Literal["basinhopping"] = "BFGS",
    *,
    model: pm.Model | None = None,
    use_grad: bool | None = None,
    use_hessp: bool | None = None,
    use_hess: bool | None = None,
    initvals: dict | None = None,
    random_seed: int | np.random.Generator | None = None,
    jitter_rvs: list[pt.TensorVariable] | None = None,
    progressbar: bool = True,
    include_transformed: bool = True,
    gradient_backend: GradientBackend = "pytensor",
    chains: int = 2,
    draws: int = 500,
    optimizer_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> az.InferenceData:
    """
    Create a Laplace (quadratic) approximation for a posterior distribution.

    This function generates a Laplace approximation for a given posterior distribution using a specified
    number of draws. This is useful for obtaining a parametric approximation to the posterior distribution
    that can be used for further analysis.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to be fit. If None, the current model context is used.
    optimize_method : str
        The optimization method to use. Valid choices are: Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, TNC, SLSQP,
        trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov, and basinhopping.

        See scipy.optimize.minimize documentation for details.
    use_grad : bool | None, optional
        Whether to use gradients in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    use_hessp : bool | None, optional
        Whether to use Hessian-vector products in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    use_hess : bool | None, optional
        Whether to use the Hessian matrix in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    initvals : None | dict, optional
        Initial values for the model parameters, as str:ndarray key-value pairs. Paritial initialization is permitted.
         If None, the model's default initial values are used.
    random_seed : None | int | np.random.Generator, optional
        Seed for the random number generator or a numpy Generator for reproducibility
    jitter_rvs : list of TensorVariables, optional
        Variables whose initial values should be jittered. If None, all variables are jittered.
    progressbar : bool, optional
        Whether to display a progress bar during optimization. Defaults to True.
    include_transformed: bool, default True
        Whether to include transformed variables in the output. If True, transformed variables will be included in the
        output InferenceData object. If False, only the original variables will be included.
    gradient_backend: str, default "pytensor"
        The backend to use for gradient computations. Must be one of "pytensor" or "jax".
    chains: int, default: 2
        The number of chain dimensions to sample. Note that this is *not* the number of chains to run in parallel,
        because the Laplace approximation is not an MCMC method. This argument exists to ensure that outputs are
        compatible with the ArviZ library.
    draws: int, default: 500
        The number of samples to draw from the approximated posterior. Totals samples will be chains * draws.
    optimizer_kwargs
        Additional keyword arguments to pass to the ``scipy.optimize`` function being used. Unless
        ``method = "basinhopping"``, ``scipy.optimize.minimize`` will be used. For ``basinhopping``,
        ``scipy.optimize.basinhopping`` will be used. See the documentation of these functions for details.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to pytensor.function.

    Returns
    -------
    :class:`~arviz.InferenceData`
        An InferenceData object containing the approximated posterior samples.

    Examples
    --------
    >>> from pymc_extras.inference import fit_laplace
    >>> import numpy as np
    >>> import pymc as pm
    >>> import arviz as az
    >>> y = np.array([2642, 3503, 4358]*10)
    >>> with pm.Model() as m:
    >>>     logsigma = pm.Uniform("logsigma", 1, 100)
    >>>     mu = pm.Uniform("mu", -10000, 10000)
    >>>     yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
    >>>     idata = fit_laplace()

    Notes
    -----
    This method of approximation may not be suitable for all types of posterior distributions,
    especially those with significant skewness or multimodality.

    See Also
    --------
    fit : Calling the inference function 'fit' like pmx.fit(method="laplace", model=m)
          will forward the call to 'fit_laplace'.

    """
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    model = pm.modelcontext(model) if model is None else model

    idata = find_MAP(
        method=optimize_method,
        model=model,
        use_grad=use_grad,
        use_hessp=use_hessp,
        use_hess=use_hess,
        initvals=initvals,
        random_seed=random_seed,
        jitter_rvs=jitter_rvs,
        progressbar=progressbar,
        include_transformed=include_transformed,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
        **optimizer_kwargs,
    )

    unpacked_variable_names = idata.fit["mean_vector"].coords["rows"].values.tolist()

    if "covariance_matrix" not in idata.fit:
        # The user didn't use `use_hess` or `use_hessp` (or an optimization method that returns an inverse Hessian), so
        # we have to go back and compute the Hessian at the MAP point now.
        frozen_model = freeze_dims_and_data(model)
        initial_params = _make_initial_point(frozen_model, initvals, random_seed, jitter_rvs)

        _, f_hessp = scipy_optimize_funcs_from_loss(
            loss=-frozen_model.logp(jacobian=False),
            inputs=frozen_model.continuous_value_vars + frozen_model.discrete_value_vars,
            initial_point_dict=DictToArrayBijection.rmap(initial_params),
            use_grad=False,
            use_hess=False,
            use_hessp=True,
            gradient_backend=gradient_backend,
            compile_kwargs=compile_kwargs,
        )
        H_inv = _compute_inverse_hessian(
            optimizer_result=None,
            optimal_point=idata.fit.mean_vector.values,
            f_fused=None,
            f_hessp=f_hessp,
            use_hess=False,
            method=optimize_method,
        )

        idata.fit["covariance_matrix"] = xr.DataArray(
            H_inv,
            dims=("rows", "columns"),
            coords={"rows": unpacked_variable_names, "columns": unpacked_variable_names},
        )

    with model_to_laplace_approx(model, unpacked_variable_names, chains, draws) as laplace_model:
        new_posterior = (
            pm.sample_posterior_predictive(
                idata.fit.expand_dims(chain=[0], draw=[0]),
                extend_inferencedata=False,
                random_seed=random_seed,
                var_names=[
                    "laplace_approximation",
                    *[x.name for x in laplace_model.deterministics],
                ],
            )
            .posterior_predictive.squeeze(["chain", "draw"])
            .drop_vars(["chain", "draw"])
            .rename({"temp_chain": "chain", "temp_draw": "draw"})
        )

        if include_transformed:
            idata.unconstrained_posterior = unstack_laplace_draws(
                new_posterior.laplace_approximation.values, model, chains=chains, draws=draws
            )

        idata.posterior = new_posterior.drop_vars(
            ["laplace_approximation", "unpacked_variable_names"]
        )

    return idata
