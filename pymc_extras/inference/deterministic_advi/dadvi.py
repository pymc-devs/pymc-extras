from collections import defaultdict
from typing import Tuple, Optional

import pymc
from pymc import Model
import arviz as az
import numpy as np
from scipy.optimize import minimize
import pytensor
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
import xarray

from pymc import join_nonshared_inputs, DictToArrayBijection
from pymc.util import get_default_varnames
from pymc.backends.arviz import (
    apply_function_over_dataset,
    PointFunc,
    coords_and_dims_for_inferencedata,
)
from pymc_extras.inference.laplace_approx.scipy_interface import (
    _compile_functions_for_scipy_optimize,
)
from pymc_extras.inference.laplace_approx.laplace import unstack_laplace_draws


def fit_deterministic_advi(
    model: Optional[Model] = None,
    n_fixed_draws: int = 30,
    random_seed: int = 2,
    n_draws: int = 1000,
    keep_untransformed: bool = False,
):
    """
    Does inference using deterministic ADVI (automatic differentiation
    variational inference).

    For full details see the paper cited in the references:
    https://www.jmlr.org/papers/v25/23-1015.html

    Parameters
    ----------
    model : pm.Model
        The PyMC model to be fit. If None, the current model context is used.

    n_fixed_draws : int
        The number of fixed draws to use for the optimisation. More
        draws will result in more accurate estimates, but also
        increase inference time. Usually, the default of 30 is a good
        tradeoff.between speed and accuracy.

    random_seed: int
        The random seed to use for the fixed draws. Running the optimisation
        twice with the same seed should arrive at the same result.

    n_draws: int
        The number of draws to return from the variational approximation.

    keep_untransformed: bool
        Whether or not to keep the unconstrained variables (such as
        logs of positive-constrained parameters) in the output.

    Returns
    -------
    :class:`~arviz.InferenceData`
        The inference data containing the results of the DADVI algorithm.

    References
    ----------
    Giordano, R., Ingram, M., & Broderick, T. (2024). Black Box
    Variational Inference with a Deterministic Objective: Faster, More
    Accurate, and Even More Black Box. Journal of Machine Learning
    Research, 25(18), 1–39.
    """

    model = pymc.modelcontext(model) if model is None else model

    initial_point_dict = model.initial_point()
    n_params = DictToArrayBijection.map(initial_point_dict).data.shape[0]

    var_params, objective = create_dadvi_graph(
        model,
        n_fixed_draws=n_fixed_draws,
        random_seed=random_seed,
        n_params=n_params,
    )

    f_fused, f_hessp = _compile_functions_for_scipy_optimize(
        objective,
        [var_params],
        compute_grad=True,
        compute_hessp=True,
        compute_hess=False,
    )

    result = minimize(
        f_fused, np.zeros(2 * n_params), method="trust-ncg", jac=True, hessp=f_hessp
    )

    opt_var_params = result.x
    opt_means, opt_log_sds = np.split(opt_var_params, 2)

    # Make the draws:
    draws_raw = np.random.randn(n_draws, n_params)
    draws = opt_means + draws_raw * np.exp(opt_log_sds)
    draws_arviz = unstack_laplace_draws(draws, model, chains=1, draws=n_draws)

    transformed_draws = transform_draws(
        draws_arviz, model, keep_untransformed=keep_untransformed
    )

    return transformed_draws


def create_dadvi_graph(
    model: Model,
    n_params: int,
    n_fixed_draws: int = 30,
    random_seed: int = 2,
) -> Tuple[TensorVariable, TensorVariable]:
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
    state = np.random.RandomState(random_seed)
    draws = state.randn(n_fixed_draws, n_params)

    inputs = model.continuous_value_vars + model.discrete_value_vars
    initial_point_dict = model.initial_point()
    logp = model.logp()

    # Graph in terms of a flat input
    [logp], flat_input = join_nonshared_inputs(
        point=initial_point_dict, outputs=[logp], inputs=inputs
    )

    var_params = pt.vector(name="eta", shape=(2 * n_params,))

    means = var_params[:n_params]
    log_sds = var_params[n_params:]

    draw = pt.vector(name="draw", shape=(n_params,))
    sample = means + pt.exp(log_sds) * draw

    # Graph in terms of a single sample
    logp_draw = pytensor.clone_replace(logp, replace={flat_input: sample})
    draw_matrix = pt.constant(draws)

    # Vectorise
    logp_vectorized_draws = pytensor.graph.vectorize_graph(
        logp_draw, replace={draw: draw_matrix}
    )

    mean_log_density = pt.mean(logp_vectorized_draws)
    entropy = pt.sum(log_sds)

    objective = -mean_log_density - entropy

    return var_params, objective


def transform_draws(
    unstacked_draws: xarray.Dataset,
    model: Model,
    keep_untransformed: bool = False,
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

    keep_untransformed: bool
        Whether or not to keep the unconstrained variables in the output.

    Returns
    -------
    :class:`~arviz.InferenceData`
        Draws from the original constrained parameters.
    """

    filtered_var_names = model.unobserved_value_vars
    vars_to_sample = list(
        get_default_varnames(filtered_var_names, include_transformed=keep_untransformed)
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

    return transformed_result
