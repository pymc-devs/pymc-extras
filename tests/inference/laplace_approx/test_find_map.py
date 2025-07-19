import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_extras.inference.laplace_approx.find_map import (
    find_MAP,
    get_nearest_psd,
    set_optimizer_function_defaults,
)
from pymc_extras.inference.laplace_approx.scipy_interface import (
    GradientBackend,
    scipy_optimize_funcs_from_loss,
)


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_fit_map"))
    return np.random.default_rng(seed)


def test_get_nearest_psd_returns_psd(rng):
    # Matrix with negative eigenvalues
    A = np.array([[2, -3], [-3, 2]])
    psd = get_nearest_psd(A)

    # Should be symmetric
    np.testing.assert_allclose(psd, psd.T)

    # All eigenvalues should be >= 0
    eigvals = np.linalg.eigvalsh(psd)
    assert np.all(eigvals >= -1e-12), "All eigenvalues should be non-negative"


def test_get_nearest_psd_given_psd_input(rng):
    L = rng.normal(size=(2, 2))
    A = L @ L.T
    psd = get_nearest_psd(A)

    # Given PSD input, should return the same matrix
    assert np.allclose(psd, A)


def test_set_optimizer_function_defaults_warns_and_prefers_hessp(caplog):
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"
    with caplog.at_level("WARNING"):
        use_grad, use_hess, use_hessp = set_optimizer_function_defaults(method, True, True, True)

    message = caplog.messages[0]
    assert message.startswith('Both "use_hess" and "use_hessp" are set to True')

    assert use_grad
    assert not use_hess
    assert use_hessp


def test_set_optimizer_function_defaults_infers_hess_and_hessp():
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"

    # If only use_hessp is set, use_hess should be False but use_grad should be inferred as True
    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(method, None, None, True)
    assert use_grad
    assert not use_hess
    assert use_hessp

    # Only use_hess is set
    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(method, None, True, None)
    assert use_hess
    assert not use_hessp


def test_set_optimizer_function_defaults_defaults():
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"
    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(method, None, None, None)
    assert use_grad
    assert not use_hess
    assert use_hessp


@pytest.mark.parametrize("gradient_backend", ["jax", "pytensor"], ids=str)
def test_jax_functions_from_graph(gradient_backend: GradientBackend):
    pytest.importorskip("jax")

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
    "backend, gradient_backend, include_transformed",
    [("jax", "jax", True), ("jax", "pytensor", False)],
    ids=str,
)
def test_find_MAP(
    method,
    use_grad,
    use_hess,
    use_hessp,
    backend,
    gradient_backend: GradientBackend,
    include_transformed,
    rng,
):
    pytest.importorskip("jax")

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
            include_transformed=include_transformed,
            compile_kwargs={"mode": backend.upper()},
            maxiter=5,
        )

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")
    assert hasattr(idata, "observed_data")

    posterior = idata.posterior.squeeze(["chain", "draw"])
    assert "mu" in posterior and "sigma" in posterior
    assert posterior["mu"].shape == ()
    assert posterior["sigma"].shape == ()

    if include_transformed:
        assert hasattr(idata, "unconstrained_posterior")
        unconstrained_posterior = idata.unconstrained_posterior.squeeze(["chain", "draw"])
        assert "sigma_log__" in unconstrained_posterior
        assert unconstrained_posterior["sigma_log__"].shape == ()
    else:
        assert not hasattr(idata, "unconstrained_posterior")


def test_find_map_outside_model_context():
    """
    Test that find_MAP can be called outside of a model context.
    """
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.Exponential("sigma", 1)
        y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=np.random.normal(size=10))

    idata = find_MAP(model=m, method="L-BFGS-B", use_grad=True, progressbar=False)

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")


@pytest.mark.parametrize(
    "backend, gradient_backend",
    [("jax", "jax")],
    ids=str,
)
def test_map_shared_variables(backend, gradient_backend: GradientBackend):
    pytest.importorskip("jax")

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
    assert hasattr(idata, "unconstrained_posterior")
    assert hasattr(idata, "fit")
    assert hasattr(idata, "optimizer_result")
    assert hasattr(idata, "observed_data")
    assert hasattr(idata, "constant_data")

    posterior = idata.posterior.squeeze(["chain", "draw"])
    unconstrained_posterior = idata.unconstrained_posterior.squeeze(["chain", "draw"])

    assert "mu" in posterior and "sigma" in posterior
    assert posterior["mu"].shape == ()
    assert posterior["sigma"].shape == ()

    assert "sigma_log__" in unconstrained_posterior
    assert unconstrained_posterior["sigma_log__"].shape == ()


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
    pytest.importorskip("jax")

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
    assert hasattr(idata, "unconstrained_posterior")

    posterior = idata.posterior.squeeze(["chain", "draw"])
    unconstrained_posterior = idata.unconstrained_posterior.squeeze(["chain", "draw"])
    assert "mu" in posterior
    assert posterior["mu"].shape == ()

    assert "sigma_log__" in unconstrained_posterior
    assert unconstrained_posterior["sigma_log__"].shape == ()


def test_find_MAP_with_coords():
    with pm.Model(coords={"group": [1, 2, 3, 4, 5]}) as m:
        mu_loc = pm.Normal("mu_loc", 0, 1)
        mu_scale = pm.HalfNormal("mu_scale", 1)

        mu = pm.Normal("mu", mu_loc, mu_scale, dims=["group"])
        sigma = pm.HalfNormal("sigma", 1, dims=["group"])

        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.random.normal(size=(10, 5)))

        idata = find_MAP(progressbar=False, method="L-BFGS-B")

    assert hasattr(idata, "posterior")
    assert hasattr(idata, "unconstrained_posterior")
    assert hasattr(idata, "fit")

    posterior = idata.posterior.squeeze(["chain", "draw"])
    unconstrained_posterior = idata.unconstrained_posterior.squeeze(["chain", "draw"])

    assert (
        "mu_loc" in posterior
        and "mu_scale" in posterior
        and "mu" in posterior
        and "sigma" in posterior
    )
    assert "mu_scale_log__" in unconstrained_posterior and "sigma_log__" in unconstrained_posterior

    assert posterior["mu_loc"].shape == ()
    assert posterior["mu_scale"].shape == ()
    assert posterior["mu"].shape == (5,)
    assert posterior["sigma"].shape == (5,)

    assert unconstrained_posterior["mu_scale_log__"].shape == ()
    assert unconstrained_posterior["sigma_log__"].shape == (5,)


def test_map_nonscalar_rv_without_dims():
    with pm.Model(coords={"test": ["A", "B", "C"]}) as model:
        x_loc = pm.Normal("x_loc", mu=0, sigma=1, dims=["test"])
        x = pm.Normal("x", mu=x_loc, sigma=1, shape=(2, 3))
        y = pm.Normal("y", mu=x, sigma=1, observed=np.random.randn(10, 2, 3))

        idata = find_MAP(method="L-BFGS-B", progressbar=False)

    assert idata.posterior["x"].shape == (1, 1, 2, 3)
    assert all(f"x_dim_{i}" in idata.posterior.coords for i in range(2))

    assert idata.fit.rows.values.tolist() == [
        "x_loc[A]",
        "x_loc[B]",
        "x_loc[C]",
        "x[0,0]",
        "x[0,1]",
        "x[0,2]",
        "x[1,0]",
        "x[1,1]",
        "x[1,2]",
    ]
