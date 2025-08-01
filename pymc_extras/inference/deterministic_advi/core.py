"""
Core computations for DADVI.
"""

from typing import NamedTuple, Callable, Optional, Dict

from scipy.sparse.linalg import LinearOperator

import numpy as np
from pymc_extras.inference.deterministic_advi.optimization import optimize_with_hvp


class DADVIFuns(NamedTuple):
    """
    This NamedTuple holds the functions required to run DADVI.

    Args:
    kl_est_and_grad_fun: Function of eta [variational parameters] and zs [draws].
        zs should have shape [M, D], where M is number of fixed draws and D is
        problem dimension. Returns a tuple whose first argument is the estimate
        of the KL divergence, and the second is its gradient w.r.t. eta.
    kl_est_hvp_fun: Function of eta, zs, and b, a vector to compute the hvp
        with. This should return a vector -- the result of the hvp with b.
    """

    kl_est_and_grad_fun: Callable[[np.ndarray, np.ndarray], np.ndarray]
    kl_est_hvp_fun: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]


def find_dadvi_optimum(
    init_params: np.ndarray,
    zs: np.ndarray,
    dadvi_funs: DADVIFuns,
    opt_method: str = "trust-ncg",
    callback_fun: Optional[Callable] = None,
    verbose: bool = False,
) -> Dict:
    """
    Optimises the DADVI objective.

    Args:
    init_params: The initial variational parameters to use. This should be a
        vector of length 2D, where D is the problem dimension. The first D
        entries specify the variational means, while the last D specify the log
        standard deviations.
    zs: The fixed draws to use in the optimisation. They must be of shape
        [M, D], where D is the problem dimension and M is the number of fixed
        draws.
    dadvi_funs: The objective to optimise. See the definition of DADVIFuns for
        more information. The kl_est_and_grad_fun is required for optimisation;
        the kl_est_hvp_fun is needed only for some optimisers.
    opt_method: The optimisation method to use. This must be one of the methods
        listed for scipy.optimize.minimize
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html].
        Defaults to trust-ncg, which requires the hvp to be available. For
        gradient-only optimisation, L-BFGS-B generally works well.
    callback_fun: If provided, this callback function is passed to
        scipy.optimize.minimize. See that function's documentation for more.
    verbose: If True, prints the progress of the optimisation by showing the
        value and gradient norm at each iteration of the optimizer.

    Returns:
    A dictionary with entries "opt_result", containing the results of running
    scipy.optimize.minimize, and "evaluation_count", containing the number of
    times the hvp and gradient functions were called.
    """

    val_and_grad_fun = lambda var_params: dadvi_funs.kl_est_and_grad_fun(var_params, zs)
    hvp_fun = (
        None
        if dadvi_funs.kl_est_hvp_fun is None
        else lambda var_params, b: dadvi_funs.kl_est_hvp_fun(var_params, zs, b)
    )

    opt_result, eval_count = optimize_with_hvp(
        val_and_grad_fun,
        hvp_fun,
        init_params,
        opt_method=opt_method,
        callback_fun=callback_fun,
        verbose=verbose,
    )

    to_return = {
        "opt_result": opt_result,
        "evaluation_count": eval_count,
    }

    # TODO: Here I originally had a Newton step check to assess
    # convergence. Could add this back in.

    return to_return


def get_dadvi_draws(var_params: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """
    Computes draws from the mean-field variational approximation given
    variational parameters and a matrix of fixed draws.

    Args:
        var_params: A vector of shape 2D, the first D entries specifying the
            means for the D model parameters, and the last D the log standard
            deviations.
        zs: A matrix of shape [N, D], containing the draws to use to sample the
            variational approximation.

    Returns:
    A matrix of shape [N, D] containing N draws from the variational
    approximation.
    """

    # TODO: Could use JAX here
    means, log_sds = np.split(var_params, 2)
    sds = np.exp(log_sds)

    draws = means.reshape(1, -1) + zs * sds.reshape(1, -1)

    return draws


# TODO -- I think the functions above cover the basic functionality of
# fixed-draw ADVI. But I have not yet included the LRVB portion of the
# code, in the interest of keeping it simple. Can add later.
