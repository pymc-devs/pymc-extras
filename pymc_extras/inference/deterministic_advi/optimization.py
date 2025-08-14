import time
import numpy as np
from functools import wraps, partial
from scipy.optimize import minimize


def print_decorator(fun, verbose=True):
    def result(x):

        value, grad = fun(x)

        if verbose:
            print(f"'f': {value}, ||grad(f)||: {np.linalg.norm(grad)}", flush=True)

        return value, grad

    return result


def count_decorator(function):
    # If wrapped around a function, the number of calls of the function can be
    # accessed by calling function.calls on the decorated result.
    @wraps(function)
    def new_fun(*args, **kwargs):
        new_fun.calls += 1
        return function(*args, **kwargs)

    new_fun.calls = 0
    return new_fun


def time_decorator(function):
    @wraps(function)
    def new_fun(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        difference = end_time - start_time
        new_fun.wall_time = difference
        return result

    new_fun.wall_time = None
    return new_fun


def optimize_with_hvp(
    val_grad_fun,
    hvp_fun,
    start_params,
    opt_method="trust-ncg",
    verbose=False,
    additional_decorator=None,
    minimize_kwargs={},
    callback_fun=None,
):
    # Note: "additional_decorator" will be called on the val_and_grad_fun and
    # its purpose is to allow for additional side effects, particularly saving
    # call results to files.

    val_grad_fun = (
        val_grad_fun
        if additional_decorator is None
        else additional_decorator(val_grad_fun)
    )

    decorated = count_decorator(partial(print_decorator, verbose=verbose)(val_grad_fun))
    hvp_fun = count_decorator(hvp_fun)

    if callback_fun is not None:
        callback = lambda cur_theta: callback_fun(cur_theta, decorated, hvp_fun)
    else:
        callback = None

    result = minimize(
        decorated,
        start_params,
        method=opt_method,
        hessp=hvp_fun,
        jac=True,
        callback=callback,
        **minimize_kwargs,
    )

    n_hvp_calls = hvp_fun.calls
    n_val_and_grad_calls = decorated.calls

    return (
        result,
        {"n_hvp_calls": n_hvp_calls, "n_val_and_grad_calls": n_val_and_grad_calls},
    )
