from collections.abc import Callable
from importlib.util import find_spec
from typing import Literal, get_args

import numpy as np
import pymc as pm
import pytensor

from pymc import join_nonshared_inputs
from pytensor import tensor as pt
from pytensor.compile import Function
from pytensor.tensor import TensorVariable

GradientBackend = Literal["pytensor", "jax"]
VALID_BACKENDS = get_args(GradientBackend)


def _compile_grad_and_hess_to_jax(
    f_fused: Function, use_hess: bool, use_hessp: bool
) -> tuple[Callable | None, Callable | None]:
    """
    Compile loss function gradients using JAX.

    Parameters
    ----------
    f_fused: Function
        The loss function to compile gradients for. Expected to be a pytensor function that returns a scalar loss,
        compiled with mode="JAX".
    use_hess: bool
        Whether to compile a function to compute the hessian of the loss function.
    use_hessp: bool
        Whether to compile a function to compute the hessian-vector product of the loss function.

    Returns
    -------
    f_fused: Callable
        The compiled loss function and gradient function, which may also compute the hessian if requested.
    f_hessp: Callable | None
        The compiled hessian-vector product function, or None if use_hessp is False.
    """
    import jax

    f_hessp = None

    orig_loss_fn = f_fused.vm.jit_fn

    if use_hess:

        @jax.jit
        def loss_fn_fused(x):
            loss_and_grad = jax.value_and_grad(lambda x: orig_loss_fn(x)[0])(x)
            hess = jax.hessian(lambda x: orig_loss_fn(x)[0])(x)
            return *loss_and_grad, hess

    else:

        @jax.jit
        def loss_fn_fused(x):
            return jax.value_and_grad(lambda x: orig_loss_fn(x)[0])(x)

    if use_hessp:

        def f_hessp_jax(x, p):
            y, u = jax.jvp(lambda x: loss_fn_fused(x)[1], (x,), (p,))
            return jax.numpy.stack(u)

        f_hessp = jax.jit(f_hessp_jax)

    return loss_fn_fused, f_hessp


def _compile_functions_for_scipy_optimize(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    compute_grad: bool,
    compute_hess: bool,
    compute_hessp: bool,
    compile_kwargs: dict | None = None,
) -> list[Function] | list[Function, Function | None, Function | None]:
    """
    Compile loss functions for use with scipy.optimize.minimize.

    Parameters
    ----------
    loss: TensorVariable
        The loss function to compile.
    inputs: list[TensorVariable]
        A single flat vector input variable, collecting all inputs to the loss function. Scipy optimize routines
        expect the function signature to be f(x, *args), where x is a 1D array of parameters.
    compute_grad: bool
        Whether to compile a function that computes the gradients of the loss function.
    compute_hess: bool
        Whether to compile a function that computes the Hessian of the loss function.
    compute_hessp: bool
        Whether to compile a function that computes the Hessian-vector product of the loss function.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to the ``pm.compile`` function.

    Returns
    -------
    f_fused: Function
        The compiled loss function, which may also include gradients and hessian if requested.
    f_hessp: Function | None
        The compiled hessian-vector product function, or None if compute_hessp is False.
    """
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    loss = pm.pytensorf.rewrite_pregrad(loss)
    f_hessp = None

    # In the simplest case, we only compile the loss function. Return it as a list to keep the return type consistent
    # with the case where we also compute gradients, hessians, or hessian-vector products.
    if not (compute_grad or compute_hess or compute_hessp):
        f_loss = pm.compile(inputs, loss, **compile_kwargs)
        return [f_loss]

    # Otherwise there are three cases. If the user only wants the loss function and gradients, we compile a single
    # fused function and return it. If the user also wants the hessian, the fused function will return the loss,
    # gradients and hessian. If the user wants gradients and hess_p, we return a fused function that returns the loss
    # and gradients, and a separate function for the hessian-vector product.

    if compute_hessp:
        # Handle this first, since it can be compiled alone.
        p = pt.tensor("p", shape=inputs[0].type.shape)
        hessp = pytensor.gradient.hessian_vector_product(loss, inputs, p)
        f_hessp = pm.compile([*inputs, p], hessp[0], **compile_kwargs)

    outputs = [loss]

    if compute_grad:
        grads = pytensor.gradient.grad(loss, inputs)
        grad = pt.concatenate([grad.ravel() for grad in grads])
        outputs.append(grad)

    if compute_hess:
        hess = pytensor.gradient.jacobian(grad, inputs)[0]
        outputs.append(hess)

    f_fused = pm.compile(inputs, outputs, **compile_kwargs)

    return [f_fused, f_hessp]


def scipy_optimize_funcs_from_loss(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    initial_point_dict: dict[str, np.ndarray | float | int],
    use_grad: bool,
    use_hess: bool,
    use_hessp: bool,
    gradient_backend: GradientBackend = "pytensor",
    compile_kwargs: dict | None = None,
) -> tuple[Callable, ...]:
    """
    Compile loss functions for use with scipy.optimize.minimize.

    Parameters
    ----------
    loss: TensorVariable
        The loss function to compile.
    inputs: list[TensorVariable]
        The input variables to the loss function.
    initial_point_dict: dict[str, np.ndarray | float | int]
        Dictionary mapping variable names to initial values. Used to determine the shapes of the input variables.
    use_grad: bool
        Whether to compile a function that computes the gradients of the loss function.
    use_hess: bool
        Whether to compile a function that computes the Hessian of the loss function.
    use_hessp: bool
        Whether to compile a function that computes the Hessian-vector product of the loss function.
    gradient_backend: str, default "pytensor"
        Which backend to use to compute gradients. Must be one of "jax" or "pytensor"
    compile_kwargs:
        Additional keyword arguments to pass to the ``pm.compile`` function.

    Returns
    -------
    f_fused: Callable
        The compiled loss function, which may also include gradients and hessian if requested.
    f_hessp: Callable | None
        The compiled hessian-vector product function, or None if use_hessp is False.
    """

    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    if use_hess and not use_grad:
        raise ValueError("Cannot compute hessian without also computing the gradient")

    if gradient_backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid gradient backend: {gradient_backend}. Must be one of {VALID_BACKENDS}"
        )

    use_jax_gradients = (gradient_backend == "jax") and use_grad
    if use_jax_gradients and not find_spec("jax"):
        raise ImportError("JAX must be installed to use JAX gradients")

    mode = compile_kwargs.get("mode", None)
    if mode is None and use_jax_gradients:
        compile_kwargs["mode"] = "JAX"
    elif mode != "JAX" and use_jax_gradients:
        raise ValueError(
            'jax gradients can only be used when ``compile_kwargs["mode"]`` is set to "JAX"'
        )

    if not isinstance(inputs, list):
        inputs = [inputs]

    [loss], flat_input = join_nonshared_inputs(
        point=initial_point_dict, outputs=[loss], inputs=inputs
    )

    # If we use pytensor gradients, we will use the pytensor function wrapper that handles shared variables. When
    # computing jax gradients, we discard the function wrapper, so we can't handle shared variables --> rewrite them
    # away.
    if use_jax_gradients:
        from pymc.sampling.jax import _replace_shared_variables

        [loss] = _replace_shared_variables([loss])

    compute_grad = use_grad and not use_jax_gradients
    compute_hess = use_hess and not use_jax_gradients
    compute_hessp = use_hessp and not use_jax_gradients

    funcs = _compile_functions_for_scipy_optimize(
        loss=loss,
        inputs=[flat_input],
        compute_grad=compute_grad,
        compute_hess=compute_hess,
        compute_hessp=compute_hessp,
        compile_kwargs=compile_kwargs,
    )

    # Depending on the requested functions, f_fused will either be the loss function, the loss function with gradients,
    # or the loss function with gradients and hessian.
    f_fused = funcs.pop(0)
    f_hessp = funcs.pop(0) if compute_hessp else None

    if use_jax_gradients:
        f_fused, f_hessp = _compile_grad_and_hess_to_jax(f_fused, use_hess, use_hessp)

    return f_fused, f_hessp
