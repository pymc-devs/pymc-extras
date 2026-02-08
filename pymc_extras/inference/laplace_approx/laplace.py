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
from typing import Literal

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from arviz import dict_to_dataset
from better_optimize.constants import minimize_method
from numpy.typing import ArrayLike
from pymc import Model
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.util import get_untransformed_name, is_transformed_name
from pytensor.graph import vectorize_graph
from pytensor.tensor import TensorVariable
from pytensor.tensor.optimize import minimize
from xarray import Dataset

from pymc_extras.inference.laplace_approx.find_map import (
    _compute_inverse_hessian, _make_initial_point, find_MAP)
from pymc_extras.inference.laplace_approx.scipy_interface import (
    GradientBackend, scipy_optimize_funcs_from_loss)

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
    _, logdetQ = pt.linalg.slogdet(Q)
    conditional_gaussian_approx = (
        -0.5 * x.T @ (-hess + Q) @ x + x.T @ (Q @ mu + jac - hess @ x0) + 0.5 * logdetQ
    )

    # Currently x is passed both as the query point for f(x, args) = logp(x | y, params) AND as an initial guess for x0. This may cause issues if the query point is
    # far from the mode x0 or in a neighbourhood which results in poor convergence.
    return pytensor.function(args, [x0, conditional_gaussian_approx])


def unpack_last_axis(packed_input, packed_shapes):
    if len(packed_shapes) == 1:
        # Single case currently fails in unpack
        return [pt.split_dims(packed_input, packed_shapes[0], axis=-1)]

    axes = tuple(range(packed_input.ndim))[:-1]
    return pt.unpack(packed_input, axes=axes, packed_shapes=packed_shapes)


def draws_from_laplace_approx(
    *,
    mean,
    covariance=None,
    standard_deviation=None,
    draws: int,
    model: Model,
    vectorize_draws: bool = True,
    return_unconstrained: bool = True,
    random_seed=None,
    compile_kwargs: dict | None = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Generate draws from the Laplace approximation of the posterior.

    Parameters
    ----------
    mean : np.ndarray
        The mean of the Laplace approximation (MAP estimate).
    covariance : np.ndarray, optional
        The covariance matrix of the Laplace approximation.
        Mutually exclusive with `standard_deviation`.
    standard_deviation : np.ndarray, optional
        The standard deviation of the Laplace approximation (diagonal approximation).
        Mutually exclusive with `covariance`.
    draws : int
        The number of draws.
    model : pm.Model
        The PyMC model.
    vectorize_draws : bool, default True
        Whether to vectorize the draws.
    return_unconstrained : bool, default True
        Whether to return the unconstrained draws in addition to the constrained ones.
    random_seed : int, optional
        Random seed for reproducibility.
    compile_kwargs: dict, optional
        Optional compile kwargs

    Returns
    -------
    tuple[Dataset, Dataset | None]
        A tuple containing the constrained draws (trace) and optionally the unconstrained draws.

    Raises
    ------
    ValueError
        If neither `covariance` nor `standard_deviation` is provided,
        or if both are provided.
    """
    # This function assumes that mean/covariance/standard_deviation are aligned with model.initial_point()
    if covariance is None and standard_deviation is None:
        raise ValueError("Must specify either covariance or standard_deviation")
    if covariance is not None and standard_deviation is not None:
        raise ValueError("Cannot specify both covariance and standard_deviation")
    if compile_kwargs is None:
        compile_kwargs = {}

    initial_point = model.initial_point()
    n = int(np.sum([np.prod(v.shape) for v in initial_point.values()]))
    assert mean.shape == (n,)
    if covariance is not None:
        assert covariance.shape == (n, n)
    elif standard_deviation is not None:
        assert standard_deviation.shape == (n,)

    vars_to_sample = [v for v in model.free_RVs + model.deterministics]
    var_names = [v.name for v in vars_to_sample]

    orig_constrained_vars = model.value_vars
    orig_outputs = model.replace_rvs_by_values(vars_to_sample)
    if return_unconstrained:
        orig_outputs.extend(model.value_vars)

    mu_pt = pt.vector("mu", shape=(n,), dtype=mean.dtype)
    size = (draws,) if vectorize_draws else ()
    if covariance is not None:
        sigma_pt = pt.matrix("cov", shape=(n, n), dtype=covariance.dtype)
        laplace_approximation = pm.MvNormal.dist(
            mu=mu_pt, cov=sigma_pt, size=size, method="svd"
        )
    else:
        sigma_pt = pt.vector("sigma", shape=(n,), dtype=standard_deviation.dtype)
        laplace_approximation = pm.Normal.dist(
            mu=mu_pt, sigma=sigma_pt, size=(*size, n)
        )

    constrained_vars = unpack_last_axis(
        laplace_approximation,
        [initial_point[v.name].shape for v in orig_constrained_vars],
    )
    outputs = vectorize_graph(
        orig_outputs, replace=dict(zip(orig_constrained_vars, constrained_vars))
    )

    fn = pm.pytensorf.compile(
        [mu_pt, sigma_pt],
        outputs,
        random_seed=random_seed,
        trust_input=True,
        **compile_kwargs,
    )
    sigma = covariance if covariance is not None else standard_deviation
    if vectorize_draws:
        output_buffers = fn(mean, sigma)
    else:
        # Take one draw to find the shape of the outputs
        output_buffers = []
        for out_draw in fn(mean, sigma):
            output_buffer = np.empty((draws, *out_draw.shape), dtype=out_draw.dtype)
            output_buffer[0] = out_draw
            output_buffers.append(output_buffer)
        # Fill one draws at a time
        for i in range(1, draws):
            for out_buffer, out_draw in zip(output_buffers, fn(mean, sigma)):
                out_buffer[i] = out_draw

    model_coords, model_dims = coords_and_dims_for_inferencedata(model)
    posterior = {
        var_name: out_buffer[None]
        for var_name, out_buffer in (
            zip(var_names, output_buffers, strict=not return_unconstrained)
        )
    }
    posterior_dataset = dict_to_dataset(
        posterior, coords=model_coords, dims=model_dims, inference_library=pm
    )
    unconstrained_posterior_dataset = None

    if return_unconstrained:
        unconstrained_posterior = {
            var.name: out_buffer[None]
            for var, out_buffer in zip(
                model.value_vars, output_buffers[len(posterior) :], strict=True
            )
        }
        # Attempt to map constrained dims to unconstrained dims
        for var_name, var_draws in unconstrained_posterior.items():
            if not is_transformed_name(var_name):
                # constrained == unconstrained, dims already shared
                continue
            constrained_dims = model_dims.get(get_untransformed_name(var_name))
            if constrained_dims is None or (
                len(constrained_dims) != (var_draws.ndim - 2)
            ):
                continue
            # Reuse dims from constrained variable if they match in length with unconstrained draws
            inferred_dims = []
            for i, (constrained_dim, unconstrained_dim_length) in enumerate(
                zip(constrained_dims, var_draws.shape[2:], strict=True)
            ):
                if model_coords.get(constrained_dim) is not None and (
                    len(model_coords[constrained_dim]) == unconstrained_dim_length
                ):
                    # Assume coordinates map. This could be fooled, by e.g., having a transform that reverses values
                    inferred_dims.append(constrained_dim)
                else:
                    # Size mismatch (e.g., Simplex), make no assumption about mapping
                    inferred_dims.append(f"{var_name}_dim_{i}")
            model_dims[var_name] = inferred_dims

        unconstrained_posterior_dataset = dict_to_dataset(
            unconstrained_posterior,
            coords=model_coords,
            dims=model_dims,
            inference_library=pm,
        )

    return posterior_dataset, unconstrained_posterior_dataset


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
    freeze_model: bool = True,
    gradient_backend: GradientBackend = "pytensor",
    chains: None | int = None,
    draws: int = 500,
    vectorize_draws: bool = True,
    optimizer_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
) -> xr.DataTree:
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
    freeze_model: bool, optional
        If True, freeze_dims_and_data will be called on the model before compiling the loss functions. This is
        sometimes necessary for JAX, and can sometimes improve performance by allowing constant folding. Defaults to
        True.
    gradient_backend: str, default "pytensor"
        The backend to use for gradient computations. Must be one of "pytensor" or "jax".
    draws: int, default: 500
        The number of samples to draw from the approximated posterior.
    optimizer_kwargs
        Additional keyword arguments to pass to the ``scipy.optimize`` function being used. Unless
        ``method = "basinhopping"``, ``scipy.optimize.minimize`` will be used. For ``basinhopping``,
        ``scipy.optimize.basinhopping`` will be used. See the documentation of these functions for details.
    vectorize_draws: bool, default True
        Whether to natively vectorize the random function or take one at a time in a python loop.
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
    >>> y = np.array([2642, 3503, 4358] * 10)
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
    if chains is not None:
        raise ValueError(
            "chains argument has been deprecated. "
            "The behavior can be recreated by unstacking draws into multiple chains after fitting"
        )

    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    model = pm.modelcontext(model) if model is None else model

    if freeze_model:
        model = freeze_dims_and_data(model)

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
        freeze_model=False,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
        compute_hessian=True,
        **optimizer_kwargs,
    )

    if "covariance_matrix" not in idata.fit:
        # The user didn't use `use_hess` or `use_hessp` (or an optimization method that returns an inverse Hessian), so
        # we have to go back and compute the Hessian at the MAP point now.
        unpacked_variable_names = (
            idata.fit["mean_vector"].coords["rows"].values.tolist()
        )
        frozen_model = freeze_dims_and_data(model)
        initial_params = _make_initial_point(
            frozen_model, initvals, random_seed, jitter_rvs
        )

        _, f_hessp = scipy_optimize_funcs_from_loss(
            loss=-frozen_model.logp(jacobian=False),
            inputs=frozen_model.continuous_value_vars
            + frozen_model.discrete_value_vars,
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
            coords={
                "rows": unpacked_variable_names,
                "columns": unpacked_variable_names,
            },
        )

    # We override the posterior/unconstrained_posterior from find_MAP
    idata["posterior"], unconstrained_posterior = draws_from_laplace_approx(
        mean=idata.fit["mean_vector"].values,
        covariance=idata.fit["covariance_matrix"].values,
        draws=draws,
        return_unconstrained=include_transformed,
        model=model,
        vectorize_draws=vectorize_draws,
        random_seed=random_seed,
    )
    if include_transformed:
        idata.unconstrained_posterior = unconstrained_posterior
    return idata
