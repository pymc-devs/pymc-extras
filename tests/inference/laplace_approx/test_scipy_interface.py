import numpy as np
import pytest

from pytensor import tensor as pt

from pymc_extras.inference.laplace_approx import scipy_interface


@pytest.fixture
def simple_loss_and_inputs():
    x = pt.vector("x")
    loss = pt.sum(x**2)
    return loss, [x]


def test_compile_functions_for_scipy_optimize_loss_only(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    funcs = scipy_interface._compile_functions_for_scipy_optimize(
        loss, inputs, compute_grad=False, compute_hess=False, compute_hessp=False
    )
    assert len(funcs) == 1
    f_loss = funcs[0]
    x_val = np.array([1.0, 2.0, 3.0])
    result = f_loss(x_val)
    assert np.isclose(result, np.sum(x_val**2))


def test_compile_functions_for_scipy_optimize_with_grad(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    funcs = scipy_interface._compile_functions_for_scipy_optimize(
        loss, inputs, compute_grad=True, compute_hess=False, compute_hessp=False
    )
    f_fused = funcs[0]
    x_val = np.array([1.0, 2.0, 3.0])
    loss_val, grad_val = f_fused(x_val)
    assert np.isclose(loss_val, np.sum(x_val**2))
    assert np.allclose(grad_val, 2 * x_val)


def test_compile_functions_for_scipy_optimize_with_hess(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    funcs = scipy_interface._compile_functions_for_scipy_optimize(
        loss, inputs, compute_grad=True, compute_hess=True, compute_hessp=False
    )
    f_fused = funcs[0]
    x_val = np.array([1.0, 2.0])
    loss_val, grad_val, hess_val = f_fused(x_val)
    assert np.isclose(loss_val, np.sum(x_val**2))
    assert np.allclose(grad_val, 2 * x_val)
    assert np.allclose(hess_val, 2 * np.eye(len(x_val)))


def test_compile_functions_for_scipy_optimize_with_hessp(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    funcs = scipy_interface._compile_functions_for_scipy_optimize(
        loss, inputs, compute_grad=True, compute_hess=False, compute_hessp=True
    )
    f_fused, f_hessp = funcs
    x_val = np.array([1.0, 2.0])
    p_val = np.array([1.0, 0.0])

    loss_val, grad_val = f_fused(x_val)
    assert np.isclose(loss_val, np.sum(x_val**2))
    assert np.allclose(grad_val, 2 * x_val)

    hessp_val = f_hessp(x_val, p_val)
    assert np.allclose(hessp_val, 2 * p_val)


def test_scipy_optimize_funcs_from_loss_invalid_backend(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    with pytest.raises(ValueError, match="Invalid gradient backend"):
        scipy_interface.scipy_optimize_funcs_from_loss(
            loss,
            inputs,
            {"x": np.array([1.0, 2.0])},
            use_grad=True,
            use_hess=False,
            use_hessp=False,
            gradient_backend="not_a_backend",
        )


def test_scipy_optimize_funcs_from_loss_hess_without_grad(simple_loss_and_inputs):
    loss, inputs = simple_loss_and_inputs
    with pytest.raises(
        ValueError, match="Cannot compute hessian without also computing the gradient"
    ):
        scipy_interface.scipy_optimize_funcs_from_loss(
            loss,
            inputs,
            {"x": np.array([1.0, 2.0])},
            use_grad=False,
            use_hess=True,
            use_hessp=False,
        )


@pytest.mark.parametrize("backend", ["pytensor", "jax"], ids=str)
def test_scipy_optimize_funcs_from_loss_backend(backend, simple_loss_and_inputs):
    if backend == "jax":
        pytest.importorskip("jax", reason="JAX is not installed")

    loss, inputs = simple_loss_and_inputs
    f_fused, f_hessp = scipy_interface.scipy_optimize_funcs_from_loss(
        loss,
        inputs,
        {"x": np.array([1.0, 2.0])},
        use_grad=True,
        use_hess=False,
        use_hessp=False,
        gradient_backend=backend,
    )
    x_val = np.array([1.0, 2.0])
    loss_val, grad_val = f_fused(x_val)
    assert np.isclose(loss_val, np.sum(x_val**2))
    assert np.allclose(grad_val, 2 * x_val)
    assert f_hessp is None
