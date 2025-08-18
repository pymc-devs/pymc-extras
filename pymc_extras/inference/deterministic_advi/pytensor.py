from collections import defaultdict

import pymc
import arviz as az
import numpy as np
from scipy.optimize import minimize
import pytensor
import pytensor.tensor as pt
from pymc import join_nonshared_inputs, DictToArrayBijection
from pymc.util import get_default_varnames

from pymc_extras.inference.laplace_approx.scipy_interface import (
    _compile_functions_for_scipy_optimize,
)
from pymc_extras.inference.laplace_approx.laplace import unstack_laplace_draws


def create_dadvi_graph(
    pymc_model, n_params: int, n_fixed_draws: int = 30, random_seed: int = 2
):

    state = np.random.RandomState(random_seed)

    inputs = pymc_model.continuous_value_vars + pymc_model.discrete_value_vars
    initial_point_dict = pymc_model.initial_point()
    logp = pymc_model.logp()

    # Graph in terms of a flat input
    [logp], flat_input = join_nonshared_inputs(
        point=initial_point_dict, outputs=[logp], inputs=inputs
    )

    draws = state.randn(n_fixed_draws, n_params)
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

    return var_params, objective, n_params


def transform_draws(unstacked_draws, model, n_draws, keep_untransformed=False):

    filtered_var_names = model.unobserved_value_vars

    vars_to_sample = list(
        get_default_varnames(filtered_var_names, include_transformed=keep_untransformed)
    )

    fn = pytensor.function(model.value_vars, vars_to_sample)

    d = {name: data.values for name, data in unstacked_draws.data_vars.items()}

    transformed_draws = defaultdict(list)
    vars_to_sample_names = [x.name for x in vars_to_sample]
    raw_var_names = [x.name for x in model.value_vars]

    for i in range(n_draws):

        cur_draw = {x: y[0, i] for x, y in d.items()}
        to_pass_in = [
            cur_draw[cur_variable_name] for cur_variable_name in raw_var_names
        ]
        transformed = fn(*to_pass_in)

        for cur_name, cur_value in zip(vars_to_sample_names, transformed):
            transformed_draws[cur_name].append(cur_value)

    final_dict = {
        # Add a draw dimension
        x: np.expand_dims(np.stack(y), axis=0)
        for x, y in transformed_draws.items()
    }

    transformed_result = az.from_dict(posterior=final_dict)

    return transformed_result


def fit_deterministic_advi(
    model=None,
    n_fixed_draws: int = 30,
    random_seed: int = 2,
    n_draws: int = 1000,
    keep_untransformed=False,
):

    model = pymc.modelcontext(model) if model is None else model

    initial_point_dict = model.initial_point()
    n_params = DictToArrayBijection.map(initial_point_dict).data.shape[0]

    var_params, objective, n_params = create_dadvi_graph(
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
        draws_arviz, model, n_draws=n_draws, keep_untransformed=keep_untransformed
    )

    return transformed_draws
