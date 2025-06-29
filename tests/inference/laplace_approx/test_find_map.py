import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_extras.inference.laplace_approx.find_map import (
    find_MAP,
)
from pymc_extras.inference.laplace_approx.scipy_interface import scipy_optimize_funcs_from_loss
from pymc_extras.inference.laplace_approx.utilities import GradientBackend

pytest.importorskip("jax")


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_fit_map"))
    return np.random.default_rng(seed)


@pytest.mark.parametrize("gradient_backend", ["jax", "pytensor"], ids=str)
def test_jax_functions_from_graph(gradient_backend: GradientBackend):
    x = pt.tensor("x", shape=(2,))

    def compute_z(x):
        z1 = x[0] ** 2 + 2
        z2 = x[0] * x[1] + 3
        return z1, z2

    z = pt.stack(compute_z(x))
    f_fused, f_hessp = scipy_optimize_funcs_from_loss(
        loss=z.sum(),
        inputs=[x],
        initial_point_dict={"x": np.array([1.0, 2.0])},
        use_grad=True,
        use_hess=True,
        use_hessp=True,
        gradient_backend=gradient_backend,
        compile_kwargs=dict(mode="JAX"),
    )

    x_val = np.array([1.0, 2.0])
    expected_z = sum(compute_z(x_val))

    z_jax, grad_val, hess_val = f_fused(x_val)
    np.testing.assert_allclose(z_jax, expected_z)
    np.testing.assert_allclose(grad_val.squeeze(), np.array([2 * x_val[0] + x_val[1], x_val[0]]))

    hess_val = np.array(hess_val)
    np.testing.assert_allclose(hess_val.squeeze(), np.array([[2, 1], [1, 0]]))

    hessp_val = np.array(f_hessp(x_val, np.array([1.0, 0.0])))
    np.testing.assert_allclose(hessp_val.squeeze(), np.array([2, 1]))


@pytest.mark.parametrize(
    "method, use_grad, use_hess, use_hessp",
    [
        (
            "Newton-CG",
            True,
            True,
            False,
        ),
        ("Newton-CG", True, False, True),
        ("BFGS", True, False, False),
        ("L-BFGS-B", True, False, False),
    ],
)
@pytest.mark.parametrize(
    "backend, gradient_backend",
    [("jax", "jax"), ("jax", "pytensor")],
    ids=str,
)
def test_find_MAP(
    method, use_grad, use_hess, use_hessp, backend, gradient_backend: GradientBackend, rng
):
    with pm.Model() as m:
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y_hat", mu=mu, sigma=sigma, observed=rng.normal(loc=3, scale=1.5, size=10))

        idata = find_MAP(
            method=method,
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            progressbar=False,
            gradient_backend=gradient_backend,
            compile_kwargs={"mode": backend.upper()},
            maxiter=5,
        )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")
    assert hasattr(idata, "observed_data")

    posterior = idata.posterior
    assert "mu" in posterior and "sigma_log__" in posterior and "sigma" in posterior
    assert posterior["mu"].shape == ()
    assert posterior["sigma_log__"].shape == ()
    assert posterior["sigma"].shape == ()


@pytest.mark.parametrize(
    "backend, gradient_backend",
    [("jax", "jax")],
    ids=str,
)
def test_map_shared_variables(backend, gradient_backend: GradientBackend):
    with pm.Model() as m:
        data = pm.Data("data", np.random.normal(loc=3, scale=1.5, size=10))
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=data)

        idata = find_MAP(
            method="L-BFGS-B",
            use_grad=True,
            use_hess=False,
            use_hessp=False,
            progressbar=False,
            gradient_backend=gradient_backend,
            compile_kwargs={"mode": backend.upper()},
        )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")
    assert hasattr(idata, "observed_data")
    assert hasattr(idata, "constant_data")

    posterior = idata.posterior
    assert "mu" in posterior and "sigma_log__" in posterior and "sigma" in posterior
    assert posterior["mu"].shape == ()
    assert posterior["sigma_log__"].shape == ()
    assert posterior["sigma"].shape == ()


@pytest.mark.parametrize(
    "method, use_grad, use_hess, use_hessp",
    [
        ("Newton-CG", True, True, False),
        ("Newton-CG", True, False, True),
    ],
)
@pytest.mark.parametrize(
    "backend, gradient_backend",
    [("jax", "pytensor")],
    ids=str,
)
def test_find_MAP_basinhopping(
    method, use_grad, use_hess, use_hessp, backend, gradient_backend, rng
):
    with pm.Model() as m:
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y_hat", mu=mu, sigma=sigma, observed=rng.normal(loc=3, scale=1.5, size=10))

        idata = find_MAP(
            method="basinhopping",
            use_grad=use_grad,
            use_hess=use_hess,
            use_hessp=use_hessp,
            progressbar=False,
            gradient_backend=gradient_backend,
            compile_kwargs={"mode": backend.upper()},
            minimizer_kwargs=dict(method=method),
            niter=1,
        )

    assert hasattr(idata, "posterior")
    posterior = idata.posterior
    assert "mu" in posterior and "sigma_log__" in posterior
    assert posterior["mu"].shape == ()
    assert posterior["sigma_log__"].shape == ()


def test_find_MAP_with_coords():
    with pm.Model(coords={"group": [1, 2, 3, 4, 5]}) as m:
        mu_loc = pm.Normal("mu_loc", 0, 1)
        mu_scale = pm.HalfNormal("mu_scale", 1)

        mu = pm.Normal("mu", mu_loc, mu_scale, dims=["group"])
        sigma = pm.HalfNormal("sigma", 1, dims=["group"])

        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.random.normal(size=(10, 5)))

        idata = find_MAP(progressbar=False, method="L-BFGS-B")

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")

    posterior = idata.posterior
    assert (
        "mu_loc" in posterior
        and "mu_scale" in posterior
        and "mu" in posterior
        and "sigma_log__" in posterior
    )
    assert posterior["mu_loc"].shape == ()
    assert posterior["mu_scale"].shape == ()
    assert posterior["mu"].shape == (5,)
    assert posterior["sigma_log__"].shape == (5,)
    assert posterior["sigma"].shape == (5,)
