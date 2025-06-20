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

from functools import reduce
from importlib.util import find_spec
from itertools import product
from typing import Literal

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from arviz import dict_to_dataset
from better_optimize.constants import minimize_method
from pymc import DictToArrayBijection
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.blocking import RaveledVars
from pymc.model.transform.conditioning import remove_value_transforms
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.util import get_default_varnames
from scipy import stats

from pymc_extras.inference.find_map import (
    GradientBackend,
    _unconstrained_vector_to_constrained_rvs,
    find_MAP,
    get_nearest_psd,
    scipy_optimize_funcs_from_loss,
)

_log = logging.getLogger(__name__)


def laplace_draws_to_inferencedata(
    posterior_draws: list[np.ndarray[float | int]], model: pm.Model | None = None
) -> az.InferenceData:
    """
    Convert draws from a posterior estimated with the Laplace approximation to an InferenceData object.


    Parameters
    ----------
    posterior_draws: list of np.ndarray
        A list of arrays containing the posterior draws. Each array should have shape (chains, draws, *shape), where
        shape is the shape of the variable in the posterior.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples
    """
    model = pm.modelcontext(model)
    chains, draws, *_ = posterior_draws[0].shape

    def make_rv_coords(name):
        coords = {"chain": range(chains), "draw": range(draws)}
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return coords
        return coords | {dim: list(model.coords[dim]) for dim in extra_dims}

    def make_rv_dims(name):
        dims = ["chain", "draw"]
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return dims
        return dims + list(extra_dims)

    names = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    ]
    idata = {
        name: xr.DataArray(
            data=draws,
            coords=make_rv_coords(name),
            dims=make_rv_dims(name),
            name=name,
        )
        for name, draws in zip(names, posterior_draws)
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.convert_to_inference_data(idata, coords=coords, dims=dims)

    return idata


def add_fit_to_inferencedata(
    idata: az.InferenceData, mu: RaveledVars, H_inv: np.ndarray, model: pm.Model | None = None
) -> az.InferenceData:
    """
    Add the mean vector and covariance matrix of the Laplace approximation to an InferenceData object.


    Parameters
    ----------
    idata: az.InfereceData
        An InferenceData object containing the approximated posterior samples.
    mu: RaveledVars
        The MAP estimate of the model parameters.
    H_inv: np.ndarray
        The inverse Hessian matrix of the log-posterior evaluated at the MAP estimate.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with the mean vector and covariance matrix added to the "fit" group.
    """
    model = pm.modelcontext(model)
    coords = model.coords

    variable_names, *_ = zip(*mu.point_map_info)

    def make_unpacked_variable_names(name):
        value_to_dim = {
            x.name: model.named_vars_to_dims.get(model.values_to_rvs[x].name, None)
            for x in model.value_vars
        }
        value_to_dim = {k: v for k, v in value_to_dim.items() if v is not None}

        rv_to_dim = model.named_vars_to_dims
        dims_dict = rv_to_dim | value_to_dim

        dims = dims_dict.get(name)
        if dims is None:
            return [name]
        labels = product(*(coords[dim] for dim in dims))
        return [f"{name}[{','.join(map(str, label))}]" for label in labels]

    unpacked_variable_names = reduce(
        lambda lst, name: lst + make_unpacked_variable_names(name), variable_names, []
    )

    mean_dataarray = xr.DataArray(mu.data, dims=["rows"], coords={"rows": unpacked_variable_names})
    cov_dataarray = xr.DataArray(
        H_inv,
        dims=["rows", "columns"],
        coords={"rows": unpacked_variable_names, "columns": unpacked_variable_names},
    )

    dataset = xr.Dataset({"mean_vector": mean_dataarray, "covariance_matrix": cov_dataarray})
    idata.add_groups(fit=dataset)

    return idata


def add_data_to_inferencedata(
    idata: az.InferenceData,
    progressbar: bool = True,
    model: pm.Model | None = None,
    compile_kwargs: dict | None = None,
) -> az.InferenceData:
    """
    Add observed and constant data to an InferenceData object.

    Parameters
    ----------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples.
    progressbar: bool
        Whether to display a progress bar during computations. Default is True.
    model: Model, optional
        A PyMC model. If None, the model is taken from the current model context.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to pytensor.function.

    Returns
    -------
    idata: az.InferenceData
        The provided InferenceData, with observed and constant data added.
    """
    model = pm.modelcontext(model)

    if model.deterministics:
        idata.posterior = pm.compute_deterministics(
            idata.posterior,
            model=model,
            merge_dataset=True,
            progressbar=progressbar,
            compile_kwargs=compile_kwargs,
        )

    coords, dims = coords_and_dims_for_inferencedata(model)

    observed_data = dict_to_dataset(
        find_observations(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    constant_data = dict_to_dataset(
        find_constants(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    idata.add_groups(
        {"observed_data": observed_data, "constant_data": constant_data},
        coords=coords,
        dims=dims,
    )

    return idata


def fit_mvn_at_MAP(
    optimized_point: dict[str, np.ndarray],
    model: pm.Model | None = None,
    on_bad_cov: Literal["warn", "error", "ignore"] = "ignore",
    transform_samples: bool = False,
    gradient_backend: GradientBackend = "pytensor",
    zero_tol: float = 1e-8,
    diag_jitter: float | None = 1e-8,
    compile_kwargs: dict | None = None,
) -> tuple[RaveledVars, np.ndarray]:
    """
    Create a multivariate normal distribution using the inverse of the negative Hessian matrix of the log-posterior
    evaluated at the MAP estimate. This is the basis of the Laplace approximation.

    Parameters
    ----------
    optimized_point : dict[str, np.ndarray]
        Local maximum a posteriori (MAP) point returned from pymc.find_MAP or jax_tools.fit_map
    model : Model, optional
        A PyMC model. If None, the model is taken from the current model context.
    on_bad_cov : str, one of 'ignore', 'warn', or 'error', default: 'ignore'
        What to do when ``H_inv`` (inverse Hessian) is not positive semi-definite.
        If 'ignore' or 'warn', the closest positive-semi-definite matrix to ``H_inv`` (in L1 norm) will be returned.
        If 'error', an error will be raised.
    transform_samples : bool
        Whether to transform the samples back to the original parameter space. Default is True.
    gradient_backend: str, default "pytensor"
        The backend to use for gradient computations. Must be one of "pytensor" or "jax".
    zero_tol: float
        Value below which an element of the Hessian matrix is counted as 0.
        This is used to stabilize the computation of the inverse Hessian matrix. Default is 1e-8.
    diag_jitter: float | None
        A small value added to the diagonal of the inverse Hessian matrix to ensure it is positive semi-definite.
        If None, no jitter is added. Default is 1e-8.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to pytensor.function when compiling loss functions

    Returns
    -------
    map_estimate: RaveledVars
        The MAP estimate of the model parameters, raveled into a 1D array.

    inverse_hessian: np.ndarray
        The inverse Hessian matrix of the log-posterior evaluated at the MAP estimate.
    """
    if gradient_backend == "jax" and not find_spec("jax"):
        raise ImportError("JAX must be installed to use JAX gradients")

    model = pm.modelcontext(model)
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    frozen_model = freeze_dims_and_data(model)

    if not transform_samples:
        untransformed_model = remove_value_transforms(frozen_model)
        logp = untransformed_model.logp(jacobian=False)
        variables = untransformed_model.continuous_value_vars
    else:
        logp = frozen_model.logp(jacobian=True)
        variables = frozen_model.continuous_value_vars

    variable_names = {var.name for var in variables}
    optimized_free_params = {k: v for k, v in optimized_point.items() if k in variable_names}
    mu = DictToArrayBijection.map(optimized_free_params)

    _, f_hess, _ = scipy_optimize_funcs_from_loss(
        loss=-logp,
        inputs=variables,
        initial_point_dict=optimized_free_params,
        use_grad=True,
        use_hess=True,
        use_hessp=False,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
    )

    H = -f_hess(mu.data)
    H_inv = np.linalg.pinv(np.where(np.abs(H) < zero_tol, 0, -H))

    def stabilize(x, jitter):
        return x + np.eye(x.shape[0]) * jitter

    H_inv = H_inv if diag_jitter is None else stabilize(H_inv, diag_jitter)

    try:
        np.linalg.cholesky(H_inv)
    except np.linalg.LinAlgError:
        if on_bad_cov == "error":
            raise np.linalg.LinAlgError(
                "Inverse Hessian not positive-semi definite at the provided point"
            )
        H_inv = get_nearest_psd(H_inv)
        if on_bad_cov == "warn":
            _log.warning(
                "Inverse Hessian is not positive semi-definite at the provided point, using the closest PSD "
                "matrix in L1-norm instead"
            )

    return mu, H_inv


def sample_laplace_posterior(
    mu: RaveledVars,
    H_inv: np.ndarray,
    model: pm.Model | None = None,
    chains: int = 2,
    draws: int = 500,
    transform_samples: bool = False,
    progressbar: bool = True,
    random_seed: int | np.random.Generator | None = None,
    compile_kwargs: dict | None = None,
) -> az.InferenceData:
    """
    Generate samples from a multivariate normal distribution with mean `mu` and inverse covariance matrix `H_inv`.

    Parameters
    ----------
    mu: RaveledVars
        The MAP estimate of the model parameters.
    H_inv: np.ndarray
        The inverse Hessian matrix of the log-posterior evaluated at the MAP estimate.
    model : Model
        A PyMC model
    chains : int
        The number of sampling chains running in parallel. Default is 2.
    draws : int
        The number of samples to draw from the approximated posterior. Default is 500.
    transform_samples : bool
        Whether to transform the samples back to the original parameter space. Default is True.
    progressbar : bool
        Whether to display a progress bar during computations. Default is True.
    random_seed: int | np.random.Generator | None
        Seed for the random number generator or a numpy Generator for reproducibility

    Returns
    -------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples.
    """
    model = pm.modelcontext(model)
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs
    rng = np.random.default_rng(random_seed)

    posterior_dist = stats.multivariate_normal(
        mean=mu.data, cov=H_inv, allow_singular=True, seed=rng
    )

    posterior_draws = posterior_dist.rvs(size=(chains, draws))
    if mu.data.shape == (1,):
        posterior_draws = np.expand_dims(posterior_draws, -1)

    if transform_samples:
        constrained_rvs, unconstrained_vector = _unconstrained_vector_to_constrained_rvs(model)
        batched_values = pt.tensor(
            "batched_values",
            shape=(chains, draws, *unconstrained_vector.type.shape),
            dtype=unconstrained_vector.type.dtype,
        )
        batched_rvs = pytensor.graph.vectorize_graph(
            constrained_rvs, replace={unconstrained_vector: batched_values}
        )

        f_constrain = pm.compile(inputs=[batched_values], outputs=batched_rvs, **compile_kwargs)
        posterior_draws = f_constrain(posterior_draws)

    else:
        info = mu.point_map_info
        flat_shapes = [size for _, _, size, _ in info]
        slices = [
            slice(sum(flat_shapes[:i]), sum(flat_shapes[: i + 1])) for i in range(len(flat_shapes))
        ]

        posterior_draws = [
            posterior_draws[..., idx].reshape((chains, draws, *shape)).astype(dtype)
            for idx, (name, shape, _, dtype) in zip(slices, info)
        ]

    idata = laplace_draws_to_inferencedata(posterior_draws, model)
    idata = add_fit_to_inferencedata(idata, mu, H_inv)
    idata = add_data_to_inferencedata(idata, progressbar, model, compile_kwargs)

    return idata


def fit_laplace(
    optimize_method: minimize_method | Literal["basinhopping"] = "BFGS",
    *,
    model: pm.Model | None = None,
    use_grad: bool | None = None,
    use_hessp: bool | None = None,
    use_hess: bool | None = None,
    initvals: dict | None = None,
    random_seed: int | np.random.Generator | None = None,
    return_raw: bool = False,
    jitter_rvs: list[pt.TensorVariable] | None = None,
    progressbar: bool = True,
    include_transformed: bool = True,
    gradient_backend: GradientBackend = "pytensor",
    chains: int = 2,
    draws: int = 500,
    on_bad_cov: Literal["warn", "error", "ignore"] = "ignore",
    fit_in_unconstrained_space: bool = False,
    zero_tol: float = 1e-8,
    diag_jitter: float | None = 1e-8,
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
    method : str
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
    return_raw: bool | False, optinal
        Whether to also return the full output of `scipy.optimize.minimize`
    jitter_rvs : list of TensorVariables, optional
        Variables whose initial values should be jittered. If None, all variables are jittered.
    progressbar : bool, optional
        Whether to display a progress bar during optimization. Defaults to True.
    fit_in_unconstrained_space: bool, default False
        Whether to fit the Laplace approximation in the unconstrained parameter space. If True, samples will be drawn
        from a mean and covariance matrix computed at a point in the **unconstrained** parameter space. Samples will
        then be transformed back to the original parameter space. This will guarantee that the samples will respect
        the domain of prior distributions (for exmaple, samples from a Beta distribution will be strictly between 0
        and 1).

        .. warning::
            This argument should be considered highly experimental. It has not been verified if this method produces
            valid draws from the posterior. **Use at your own risk**.

    gradient_backend: str, default "pytensor"
        The backend to use for gradient computations. Must be one of "pytensor" or "jax".
    chains: int, default: 2
        The number of chain dimensions to sample. Note that this is *not* the number of chains to run in parallel,
        because the Laplace approximation is not an MCMC method. This argument exists to ensure that outputs are
        compatible with the ArviZ library.
    draws: int, default: 500
        The number of samples to draw from the approximated posterior. Totals samples will be chains * draws.
    on_bad_cov : str, one of 'ignore', 'warn', or 'error', default: 'ignore'
        What to do when ``H_inv`` (inverse Hessian) is not positive semi-definite.
        If 'ignore' or 'warn', the closest positive-semi-definite matrix to ``H_inv`` (in L1 norm) will be returned.
        If 'error', an error will be raised.
    zero_tol: float
        Value below which an element of the Hessian matrix is counted as 0.
        This is used to stabilize the computation of the inverse Hessian matrix. Default is 1e-8.
    diag_jitter: float | None
        A small value added to the diagonal of the inverse Hessian matrix to ensure it is positive semi-definite.
        If None, no jitter is added. Default is 1e-8.
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
    >>> from pymc_extras.inference.laplace import fit_laplace
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

    optimized_point = find_MAP(
        method=optimize_method,
        model=model,
        use_grad=use_grad,
        use_hessp=use_hessp,
        use_hess=use_hess,
        initvals=initvals,
        random_seed=random_seed,
        return_raw=return_raw,
        jitter_rvs=jitter_rvs,
        progressbar=progressbar,
        include_transformed=include_transformed,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
        **optimizer_kwargs,
    )

    mu, H_inv = fit_mvn_at_MAP(
        optimized_point=optimized_point,
        model=model,
        on_bad_cov=on_bad_cov,
        transform_samples=fit_in_unconstrained_space,
        gradient_backend=gradient_backend,
        zero_tol=zero_tol,
        diag_jitter=diag_jitter,
        compile_kwargs=compile_kwargs,
    )

    return sample_laplace_posterior(
        mu=mu,
        H_inv=H_inv,
        model=model,
        chains=chains,
        draws=draws,
        transform_samples=fit_in_unconstrained_space,
        progressbar=progressbar,
        random_seed=random_seed,
        compile_kwargs=compile_kwargs,
    )
