from typing import Callable, Dict

import numpy as np
import pymc
import arviz as az
from jax import vmap

from pymc_extras.inference.deterministic_advi.jax import build_dadvi_funs
from pymc_extras.inference.deterministic_advi.pymc_to_jax import (
    get_jax_functions_from_pymc,
    transform_dadvi_draws,
)
from pymc_extras.inference.deterministic_advi.core import (
    find_dadvi_optimum,
    get_dadvi_draws,
    DADVIFuns,
)
from pymc_extras.inference.deterministic_advi.utils import opt_callback_fun


class DADVIResult:
    def __init__(
        self,
        fixed_draws: np.ndarray,
        var_params: np.ndarray,
        unflattening_fun: Callable[[np.ndarray], Dict[str, np.ndarray]],
        dadvi_funs: DADVIFuns,
        pymc_model: pymc.Model,  # TODO Check the type here
    ):

        self.fixed_draws = fixed_draws
        self.var_params = var_params
        self.unflattening_fun = unflattening_fun
        self.dadvi_funs = dadvi_funs
        self.n_params = self.fixed_draws.shape[1]
        self.pymc_model = pymc_model

    def get_posterior_means(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with posterior means for all parameters.
        """

        means = np.split(self.var_params, 2)[0]
        return self.unflattening_fun(means)

    def get_posterior_standard_deviations_mean_field(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with posterior standard deviations (not LRVB-corrected, but mean field).
        """

        log_sds = np.split(self.var_params, 2)[1]
        sds = np.exp(log_sds)
        return self.unflattening_fun(sds)

    def get_posterior_draws_mean_field(
        self,
        n_draws: int = 1000,
        seed: int = 2,
        transform_draws: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with draws from the posterior.
        """

        np.random.seed(seed)
        z = np.random.randn(n_draws, self.n_params)
        dadvi_draws_flat = get_dadvi_draws(self.var_params, z)

        if transform_draws:

            dadvi_draws = transform_dadvi_draws(
                self.pymc_model, dadvi_draws_flat, self.unflattening_fun
            )

        else:

            dadvi_draws = vmap(self.unflattening_fun)(dadvi_draws_flat)

        return dadvi_draws

    def compute_function_on_mean_field_draws(
        self,
        function_to_run: Callable[[Dict], np.ndarray],
        n_draws: int = 1000,
        seed: int = 2,
    ):
        dadvi_dict = self.get_posterior_draws_mean_field(n_draws, seed)

        return vmap(function_to_run)(dadvi_dict)


def fit_pymc_dadvi_with_jax(pymc_model, num_fixed_draws=30, seed=2):
    np.random.seed(seed)

    jax_funs = get_jax_functions_from_pymc(pymc_model)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

    opt_callback_fun.opt_sequence = []

    init_means = np.zeros(jax_funs["n_params"])
    init_log_vars = np.zeros(jax_funs["n_params"]) - 3
    init_var_params = np.concatenate([init_means, init_log_vars])
    zs = np.random.randn(num_fixed_draws, jax_funs["n_params"])
    opt = find_dadvi_optimum(
        init_params=init_var_params,
        zs=zs,
        dadvi_funs=dadvi_funs,
        verbose=True,
        callback_fun=opt_callback_fun,
    )

    dadvi_result = DADVIResult(
        fixed_draws=zs,
        var_params=opt["opt_result"].x,
        unflattening_fun=jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
        pymc_model=pymc_model,
    )

    # Get draws and turn into arviz format expected
    draws = dadvi_result.get_posterior_draws_mean_field(transform_draws=True)

    az_draws = az.convert_to_inference_data(draws)

    return az_draws
