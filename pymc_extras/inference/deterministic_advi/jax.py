from typing import Callable, Dict, Tuple
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jvp, grad
from functools import partial

from pymc_extras.inference.deterministic_advi.core import DADVIFuns
from pymc_extras.inference.deterministic_advi.optimization import count_decorator


@partial(jit, static_argnums=0)
def hvp(f, primals, tangents):
    # Taken (and slightly modified) from:
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    return jvp(grad(f), (primals,), (tangents,))[1]


@jit
def _make_draws(z, mean, log_sd):

    draw = z * jnp.exp(log_sd) + mean

    return draw


@jit
def _calculate_entropy(log_sds):

    return jnp.sum(log_sds)


def build_dadvi_funs(log_posterior_fn: Callable[[jnp.ndarray], float]) -> DADVIFuns:
    """
    Builds the DADVIFuns from a log posterior density function written in JAX.
    """

    def single_log_posterior_fun(cur_z, var_params):
        means, log_sds = jnp.split(var_params, 2)
        cur_theta = _make_draws(cur_z, means, log_sds)
        return log_posterior_fn(cur_theta)

    def log_posterior_expectation(zs, var_params):
        single_curried = partial(single_log_posterior_fun, var_params=var_params)
        log_posts = vmap(single_curried)(zs)
        return jnp.mean(log_posts)

    def full_kl_est(var_params, zs):
        _, log_sds = jnp.split(var_params, 2)
        log_posterior = log_posterior_expectation(zs, var_params)
        entropy = _calculate_entropy(log_sds)
        return -log_posterior - entropy

    @jit
    def kl_est_hvp_fun(var_params, zs, b):
        rel_kl_est = partial(full_kl_est, zs=zs)
        rel_hvp = lambda x, y: hvp(rel_kl_est, x, y)
        return rel_hvp(var_params, b)

    kl_est_and_grad_fun = jit(value_and_grad(full_kl_est))

    return DADVIFuns(
        kl_est_and_grad_fun=kl_est_and_grad_fun, kl_est_hvp_fun=kl_est_hvp_fun
    )
