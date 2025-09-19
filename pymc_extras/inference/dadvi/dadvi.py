import arviz as az
import numpy as np
import pymc
import pytensor
import pytensor.tensor as pt
import xarray

from better_optimize import minimize
from better_optimize.constants import minimize_method
from pymc import DictToArrayBijection, Model, join_nonshared_inputs
from pymc.backends.arviz import (
    PointFunc,
    apply_function_over_dataset,
    coords_and_dims_for_inferencedata,
)
from pymc.blocking import RaveledVars
from pymc.util import RandomSeed, get_default_varnames
from pytensor.tensor.variable import TensorVariable

from pymc_extras.inference.laplace_approx.idata import (
    add_data_to_inference_data,
    add_optimizer_result_to_inference_data,
)
from pymc_extras.inference.laplace_approx.laplace import unstack_laplace_draws
from pymc_extras.inference.laplace_approx.scipy_interface import (
    scipy_optimize_funcs_from_loss,
    set_optimizer_function_defaults,
)


def fit_dadvi(
    model: Model | None = None,
    n_fixed_draws: int = 30,
    random_seed: RandomSeed = None,
    n_draws: int = 1000,
    include_transformed: bool = False,
    optimizer_method: minimize_method = "trust-ncg",
    use_grad: bool | None = None,
    use_hessp: bool | None = None,
    use_hess: bool | None = None,
    gradient_backend: str = "pytensor",
    compile_kwargs: dict | None = None,
    **minimize_kwargs,
) -> az.InferenceData:
    """
    Does inference using Deterministic ADVI (Automatic Differentiation Variational Inference), DADVI for short.

    For full details see the paper cited in the references: https://www.jmlr.org/papers/v25/23-1015.html

    Parameters
    ----------
    model : pm.Model
        The PyMC model to be fit. If None, the current model context is used.

    n_fixed_draws : int
        The number of fixed draws to use for the optimisation. More draws will result in more accurate estimates, but
        also increase inference time. Usually, the default of 30 is a good tradeoff between speed and accuracy.

    random_seed: int
        The random seed to use for the fixed draws. Running the optimisation twice with the same seed should arrive at
        the same result.

    n_draws: int
        The number of draws to return from the variational approximation.

    include_transformed: bool
        Whether or not to keep the unconstrained variables (such as logs of positive-constrained parameters) in the
        output.

    optimizer_method: str
        Which optimization method to use. The function calls ``scipy.optimize.minimize``, so any of the methods there
        can be used. The default is trust-ncg, which uses second-order information and is generally very reliable.
        Other methods such as L-BFGS-B might be faster but potentially more brittle and may not converge exactly to
        the optimum.

    gradient_backend: str
        Which backend to use to compute gradients. Must be one of "jax" or "pytensor". Default is "pytensor".

    compile_kwargs: dict, optional
        Additional keyword arguments to pass to `pytensor.function`

    minimize_kwargs:
        Additional keyword arguments to pass to the ``scipy.optimize.minimize`` function. See the documentation of
        that function for details.

    use_grad: bool, optional
        If True, pass the gradient function to `scipy.optimize.minimize` (where it is referred to as `jac`).

    use_hessp: bool, optional
        If True, pass the hessian vector product to `scipy.optimize.minimize`.

    use_hess: bool, optional
        If True, pass the hessian to `scipy.optimize.minimize`. Note that this is generally not recommended since its
        computation can be slow and memory-intensive if there are many parameters.

    Returns
    -------
    :class:`~arviz.InferenceData`
        The inference data containing the results of the DADVI algorithm.

    References
    ----------
    Giordano, R., Ingram, M., & Broderick, T. (2024). Black Box Variational Inference with a Deterministic Objective:
    Faster, More Accurate, and Even More Black Box. Journal of Machine Learning Research, 25(18), 1–39.
    """

    model = pymc.modelcontext(model) if model is None else model

    initial_point_dict = model.initial_point()
    initial_point = DictToArrayBijection.map(initial_point_dict)
    n_params = initial_point.data.shape[0]

    var_params, objective = create_dadvi_graph(
        model,
        n_fixed_draws=n_fixed_draws,
        random_seed=random_seed,
        n_params=n_params,
    )

    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(
        optimizer_method, use_grad, use_hess, use_hessp
    )

    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss=objective,
        inputs=[var_params],
        initial_point_dict=None,
        use_grad=use_grad,
        use_hessp=use_hessp,
        use_hess=use_hess,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
        inputs_are_flat=True,
    )

    dadvi_initial_point = {
        f"{var_name}_mu": np.zeros_like(value).ravel()
        for var_name, value in initial_point_dict.items()
    }
    dadvi_initial_point.update(
        {
            f"{var_name}_sigma__log": np.zeros_like(value).ravel()
            for var_name, value in initial_point_dict.items()
        }
    )

    dadvi_initial_point = DictToArrayBijection.map(dadvi_initial_point)

    result = minimize(
        f=f_fused,
        x0=dadvi_initial_point.data,
        method=optimizer_method,
        hessp=f_hessp,
        **minimize_kwargs,
    )

    raveled_optimized = RaveledVars(result.x, dadvi_initial_point.point_map_info)

    opt_var_params = result.x
    opt_means, opt_log_sds = np.split(opt_var_params, 2)

    # Make the draws:
    generator = np.random.default_rng(seed=random_seed)
    draws_raw = generator.standard_normal(size=(n_draws, n_params))

    draws = opt_means + draws_raw * np.exp(opt_log_sds)
    draws_arviz = unstack_laplace_draws(draws, model, chains=1, draws=n_draws)

    idata = dadvi_result_to_idata(draws_arviz, model, include_transformed=include_transformed)

    var_name_to_model_var = {f"{var_name}_mu": var_name for var_name in initial_point_dict.keys()}
    var_name_to_model_var.update(
        {f"{var_name}_sigma__log": var_name for var_name in initial_point_dict.keys()}
    )

    idata = add_optimizer_result_to_inference_data(
        idata=idata,
        result=result,
        method=optimizer_method,
        mu=raveled_optimized,
        model=model,
        var_name_to_model_var=var_name_to_model_var,
    )

    idata = add_data_to_inference_data(
        idata=idata, progressbar=False, model=model, compile_kwargs=compile_kwargs
    )

    return idata


def create_dadvi_graph(
    model: Model,
    n_params: int,
    n_fixed_draws: int = 30,
    random_seed: RandomSeed = None,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Sets up the DADVI graph in pytensor and returns it.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to be fit.

    n_params: int
        The total number of parameters in the model.

    n_fixed_draws : int
        The number of fixed draws to use.

    random_seed: int
        The random seed to use for the fixed draws.

    Returns
    -------
    Tuple[TensorVariable, TensorVariable]
        A tuple whose first element contains the variational parameters,
        and whose second contains the DADVI objective.
    """

    # Make the fixed draws
    generator = np.random.default_rng(seed=random_seed)
    draws = generator.standard_normal(size=(n_fixed_draws, n_params))

    inputs = model.continuous_value_vars + model.discrete_value_vars
    initial_point_dict = model.initial_point()
    logp = model.logp()

    # Graph in terms of a flat input
    [logp], flat_input = join_nonshared_inputs(
        point=initial_point_dict, outputs=[logp], inputs=inputs
    )

    var_params = pt.vector(name="eta", shape=(2 * n_params,))

    means, log_sds = pt.split(var_params, axis=0, splits_size=[n_params, n_params], n_splits=2)

    draw_matrix = pt.constant(draws)
    samples = means + pt.exp(log_sds) * draw_matrix

    logp_vectorized_draws = pytensor.graph.vectorize_graph(logp, replace={flat_input: samples})

    mean_log_density = pt.mean(logp_vectorized_draws)
    entropy = pt.sum(log_sds)

    objective = -mean_log_density - entropy

    return var_params, objective


def dadvi_result_to_idata(
    unstacked_draws: xarray.Dataset,
    model: Model,
    include_transformed: bool = False,
):
    """
    Transforms the unconstrained draws back into the constrained space.

    Parameters
    ----------
    unstacked_draws : xarray.Dataset
        The draws to constrain back into the original space.

    model : Model
        The PyMC model the variables were derived from.

    n_draws: int
        The number of draws to return from the variational approximation.

    include_transformed: bool
        Whether or not to keep the unconstrained variables in the output.

    Returns
    -------
    :class:`~arviz.InferenceData`
        Draws from the original constrained parameters.
    """

    filtered_var_names = model.unobserved_value_vars
    vars_to_sample = list(
        get_default_varnames(filtered_var_names, include_transformed=include_transformed)
    )
    fn = pytensor.function(model.value_vars, vars_to_sample)
    point_func = PointFunc(fn)

    coords, dims = coords_and_dims_for_inferencedata(model)

    transformed_result = apply_function_over_dataset(
        point_func,
        unstacked_draws,
        output_var_names=[x.name for x in vars_to_sample],
        coords=coords,
        dims=dims,
    )

    constrained_names = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=False)
    ]
    all_varnames = [
        x.name for x in get_default_varnames(model.unobserved_value_vars, include_transformed=True)
    ]
    unconstrained_names = set(all_varnames) - set(constrained_names)

    idata = az.InferenceData(posterior=transformed_result[constrained_names])

    if unconstrained_names and include_transformed:
        idata["unconstrained_posterior"] = transformed_result[unconstrained_names]

    return idata
