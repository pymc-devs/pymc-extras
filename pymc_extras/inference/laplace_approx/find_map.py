import logging

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pymc as pm

from better_optimize import basinhopping, minimize
from better_optimize.constants import MINIMIZE_MODE_KWARGS, minimize_method
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.util import get_default_varnames
from pytensor.tensor import TensorVariable
from scipy.optimize import OptimizeResult

from pymc_extras.inference.laplace_approx.idata import (
    add_data_to_inference_data,
    add_fit_to_inference_data,
    add_optimizer_result_to_inference_data,
    map_results_to_inference_data,
)
from pymc_extras.inference.laplace_approx.scipy_interface import (
    GradientBackend,
    scipy_optimize_funcs_from_loss,
)

_log = logging.getLogger(__name__)


def set_optimizer_function_defaults(method, use_grad, use_hess, use_hessp):
    method_info = MINIMIZE_MODE_KWARGS[method].copy()

    if use_hess and use_hessp:
        _log.warning(
            'Both "use_hess" and "use_hessp" are set to True, but scipy.optimize.minimize never uses both at the '
            'same time. When possible "use_hessp" is preferred because its is computationally more efficient. '
            'Setting "use_hess" to False.'
        )
        use_hess = False

    use_grad = use_grad if use_grad is not None else method_info["uses_grad"]

    if use_hessp is not None and use_hess is None:
        use_hess = not use_hessp

    elif use_hess is not None and use_hessp is None:
        use_hessp = not use_hess

    elif use_hessp is None and use_hess is None:
        use_hessp = method_info["uses_hessp"]
        use_hess = method_info["uses_hess"]
        if use_hessp and use_hess:
            # If a method could use either hess or hessp, we default to using hessp
            use_hess = False

    return use_grad, use_hess, use_hessp


def get_nearest_psd(A: np.ndarray) -> np.ndarray:
    """
    Compute the nearest positive semi-definite matrix to a given matrix.

    This function takes a square matrix and returns the nearest positive semi-definite matrix using
    eigenvalue decomposition. It ensures all eigenvalues are non-negative. The "nearest" matrix is defined in terms
    of the Frobenius norm.

    Parameters
    ----------
    A : np.ndarray
        Input square matrix.

    Returns
    -------
    np.ndarray
        The nearest positive semi-definite matrix to the input matrix.
    """
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(C)
    eigval[eigval < 0] = 0

    return eigvec @ np.diag(eigval) @ eigvec.T


def _make_initial_point(model, initvals=None, random_seed=None, jitter_rvs=None):
    jitter_rvs = [] if jitter_rvs is None else jitter_rvs

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(jitter_rvs),
        return_transformed=True,
        overrides=initvals,
    )

    start_dict = ipfn(random_seed)
    vars_dict = {var.name: var for var in model.continuous_value_vars}
    initial_params = DictToArrayBijection.map(
        {var_name: value for var_name, value in start_dict.items() if var_name in vars_dict}
    )

    return initial_params


def _compute_inverse_hessian(
    optimizer_result: OptimizeResult | None,
    optimal_point: np.ndarray | None,
    f_fused: Callable | None,
    f_hessp: Callable | None,
    use_hess: bool,
    method: minimize_method | Literal["BFGS", "L-BFGS-B"],
):
    """
    Compute the Hessian matrix or its inverse based on the optimization result and the method used.

    Downstream functions (e.g. laplace approximation) will need the inverse Hessian matrix. This function computes it
    in the cheapest way possible, depending on the optimization method used and the available compiled functions.

    Parameters
    ----------
    optimizer_result: OptimizeResult, optional
        The result of the optimization, containing the optimized parameters and possibly an approximate inverse Hessian.
    optimal_point: np.ndarray, optional
        The optimal point found by the optimizer, used to compute the Hessian if necessary. If not provided, it will be
        extracted from the optimizer result.
    f_fused: callable, optional
        The compiled function representing the loss and possibly its gradient and Hessian.
    f_hessp: callable, optional
        The compiled function for Hessian-vector products, if available.
    use_hess: bool
        Whether the Hessian matrix was used in the optimization.
    method: minimize_method
        The optimization method used, which determines how the Hessian is computed.

    Returns
    -------
    H_inv: np.ndarray
        The inverse Hessian matrix, computed based on the optimization method and available functions.
    """
    if optimal_point is None and optimizer_result is None:
        raise ValueError("At least one of `optimal_point` or `optimizer_result` must be provided.")

    x_star = optimizer_result.x if optimizer_result is not None else optimal_point
    n_vars = len(x_star)

    if method == "BFGS" and optimizer_result is not None:
        # If we used BFGS, the optimizer result will contain the inverse Hessian -- we can just use that rather than
        # re-computing something
        if hasattr(optimizer_result, "lowest_optimization_result"):
            # We did basinhopping, need to get the inner optimizer results
            H_inv = getattr(optimizer_result.lowest_optimization_result, "hess_inv", None)
        else:
            H_inv = getattr(optimizer_result, "hess_inv", None)

    elif method == "L-BFGS-B" and optimizer_result is not None:
        # Here we will have a LinearOperator representing the inverse Hessian-Vector product.
        if hasattr(optimizer_result, "lowest_optimization_result"):
            # We did basinhopping, need to get the inner optimizer results
            f_hessp_inv = getattr(optimizer_result.lowest_optimization_result, "hess_inv", None)
        else:
            f_hessp_inv = getattr(optimizer_result, "hess_inv", None)

        if f_hessp_inv is not None:
            basis = np.eye(n_vars)
            H_inv = np.stack([f_hessp_inv(basis[:, i]) for i in range(n_vars)], axis=-1)
        else:
            H_inv = None

    elif f_hessp is not None:
        # In the case that hessp was used, the results object will not save the inverse Hessian, so we can compute it from
        # the hessp function, using euclidian basis vector.
        basis = np.eye(n_vars)
        H = np.stack([f_hessp(x_star, basis[:, i]) for i in range(n_vars)], axis=-1)
        H_inv = np.linalg.inv(get_nearest_psd(H))

    elif use_hess and f_fused is not None:
        # If we compiled a hessian function, just use it
        _, _, H = f_fused(x_star)
        H_inv = np.linalg.inv(get_nearest_psd(H))

    else:
        H_inv = None

    return H_inv


def find_MAP(
    method: minimize_method | Literal["basinhopping"] = "L-BFGS-B",
    *,
    model: pm.Model | None = None,
    use_grad: bool | None = None,
    use_hessp: bool | None = None,
    use_hess: bool | None = None,
    initvals: dict | None = None,
    random_seed: int | np.random.Generator | None = None,
    jitter_rvs: list[TensorVariable] | None = None,
    progressbar: bool = True,
    include_transformed: bool = True,
    gradient_backend: GradientBackend = "pytensor",
    compile_kwargs: dict | None = None,
    **optimizer_kwargs,
) -> (
    dict[str, np.ndarray]
    | tuple[dict[str, np.ndarray], np.ndarray]
    | tuple[dict[str, np.ndarray], OptimizeResult]
    | tuple[dict[str, np.ndarray], OptimizeResult, np.ndarray]
):
    """
    Fit a PyMC model via maximum a posteriori (MAP) estimation using JAX and scipy.optimize.

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
        Initial values for the model parameters, as str:ndarray key-value pairs. Partial initialization is permitted.
         If None, the model's default initial values are used.
    random_seed : None | int | np.random.Generator, optional
        Seed for the random number generator or a numpy Generator for reproducibility
    jitter_rvs : list of TensorVariables, optional
        Variables whose initial values should be jittered. If None, all variables are jittered.
    progressbar : bool, optional
        Whether to display a progress bar during optimization. Defaults to True.
    include_transformed: bool, optional
        Whether to include transformed variable values in the returned dictionary. Defaults to True.
    gradient_backend: str, default "pytensor"
        Which backend to use to compute gradients. Must be one of "pytensor" or "jax".
    compile_kwargs: dict, optional
        Additional options to pass to the ``pytensor.function`` function when compiling loss functions.
    **optimizer_kwargs
        Additional keyword arguments to pass to the ``scipy.optimize`` function being used. Unless
        ``method = "basinhopping"``, ``scipy.optimize.minimize`` will be used. For ``basinhopping``,
        ``scipy.optimize.basinhopping`` will be used. See the documentation of these functions for details.

    Returns
    -------
    map_result: az.InferenceData
        Results of Maximum A Posteriori (MAP) estimation, including the optimized point, inverse Hessian, transformed
        latent variables, and optimizer results.
    """
    model = pm.modelcontext(model) if model is None else model
    frozen_model = freeze_dims_and_data(model)
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    initial_params = _make_initial_point(frozen_model, initvals, random_seed, jitter_rvs)

    do_basinhopping = method == "basinhopping"
    minimizer_kwargs = optimizer_kwargs.pop("minimizer_kwargs", {})

    if do_basinhopping:
        # For a nice API, we let the user set method="basinhopping", but if we're doing basinhopping we still need
        # another method for the inner optimizer. This will be set in the minimizer_kwargs, but also needs a default
        # if one isn't provided.

        method = minimizer_kwargs.pop("method", "L-BFGS-B")
        minimizer_kwargs["method"] = method

    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(
        method, use_grad, use_hess, use_hessp
    )

    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss=-frozen_model.logp(),
        inputs=frozen_model.continuous_value_vars + frozen_model.discrete_value_vars,
        initial_point_dict=DictToArrayBijection.rmap(initial_params),
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        gradient_backend=gradient_backend,
        compile_kwargs=compile_kwargs,
    )

    args = optimizer_kwargs.pop("args", ())

    # better_optimize.minimize will check if f_logp is a fused loss+grad Op, and automatically assign the jac argument
    # if so. That is why the jac argument is not passed here in either branch.

    if do_basinhopping:
        if "args" not in minimizer_kwargs:
            minimizer_kwargs["args"] = args
        if "hessp" not in minimizer_kwargs:
            minimizer_kwargs["hessp"] = f_hessp
        if "method" not in minimizer_kwargs:
            minimizer_kwargs["method"] = method

        optimizer_result = basinhopping(
            func=f_fused,
            x0=cast(np.ndarray[float], initial_params.data),
            progressbar=progressbar,
            minimizer_kwargs=minimizer_kwargs,
            **optimizer_kwargs,
        )

    else:
        optimizer_result = minimize(
            f=f_fused,
            x0=cast(np.ndarray[float], initial_params.data),
            args=args,
            hessp=f_hessp,
            progressbar=progressbar,
            method=method,
            **optimizer_kwargs,
        )

    H_inv = _compute_inverse_hessian(
        optimizer_result=optimizer_result,
        optimal_point=None,
        f_fused=f_fused,
        f_hessp=f_hessp,
        use_hess=use_hess,
        method=method,
    )

    raveled_optimized = RaveledVars(optimizer_result.x, initial_params.point_map_info)
    unobserved_vars = get_default_varnames(model.unobserved_value_vars, include_transformed=True)
    unobserved_vars_values = model.compile_fn(unobserved_vars, mode="FAST_COMPILE")(
        DictToArrayBijection.rmap(raveled_optimized)
    )

    optimized_point = {
        var.name: value for var, value in zip(unobserved_vars, unobserved_vars_values)
    }

    idata = map_results_to_inference_data(
        map_point=optimized_point, model=frozen_model, include_transformed=include_transformed
    )

    idata = add_fit_to_inference_data(
        idata=idata, mu=raveled_optimized, H_inv=H_inv, model=frozen_model
    )

    idata = add_optimizer_result_to_inference_data(
        idata=idata, result=optimizer_result, method=method, mu=raveled_optimized, model=model
    )

    idata = add_data_to_inference_data(
        idata=idata, progressbar=False, model=model, compile_kwargs=compile_kwargs
    )

    return idata
