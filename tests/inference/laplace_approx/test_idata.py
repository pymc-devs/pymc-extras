import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc.blocking import RaveledVars
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import LinearOperator

from pymc_extras.inference.laplace_approx.idata import (
    add_data_to_inference_data,
    add_fit_to_inference_data,
    laplace_draws_to_inferencedata,
    optimizer_result_to_dataset,
)


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def simple_model(rng):
    with pm.Model() as model:
        x = pm.Data("data", rng.normal(size=(10,)))
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        obs = pm.Normal("obs", mu + x, sigma, observed=rng.normal(size=(10,)))

    mu_val = np.array([0.5, 1.0])
    H_inv = np.eye(2)

    point_map_info = (("mu", (), 1, "float64"), ("sigma", (), 1, "float64"))
    test_point = RaveledVars(mu_val, point_map_info)

    return model, mu_val, H_inv, test_point


@pytest.fixture
def hierarchical_model(rng):
    with pm.Model(coords={"group": [1, 2, 3, 4, 5]}) as model:
        mu_loc = pm.Normal("mu_loc", 0, 1)
        mu_scale = pm.HalfNormal("mu_scale", 1)
        mu = pm.Normal("mu", mu_loc, mu_scale, dims="group")
        sigma = pm.HalfNormal("sigma", 1)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=rng.normal(size=(5, 10)))

        mu_val = rng.normal(size=(8,))
        H_inv = np.eye(8)

        point_map_info = (
            ("mu_loc", (), 1, "float64"),
            ("mu_scale", (), 1, "float64"),
            ("mu", (5,), 5, "float64"),
            ("sigma", (), 1, "float64"),
        )

        test_point = RaveledVars(mu_val, point_map_info)

    return model, mu_val, H_inv, test_point


def test_laplace_draws_to_inferencedata(simple_model, rng):
    # Simulate posterior draws: 2 variables, each (chains, draws)
    chains, draws = 2, 5
    mu_draws = rng.normal(size=(chains, draws))
    sigma_draws = np.abs(rng.normal(size=(chains, draws)))
    model, *_ = simple_model

    idata = laplace_draws_to_inferencedata([mu_draws, sigma_draws], model=model)

    assert isinstance(idata, az.InferenceData)
    assert "mu" in idata.posterior
    assert "sigma" in idata.posterior

    assert idata.posterior["mu"].shape == (chains, draws)
    assert idata.posterior["sigma"].shape == (chains, draws)


class TestFittoInferenceData:
    def check_idata(self, idata, var_names, n_vars):
        assert "fit" in idata.groups()

        fit = idata.fit
        assert "mean_vector" in fit
        assert "covariance_matrix" in fit
        assert fit["mean_vector"].shape[0] == n_vars
        assert fit["covariance_matrix"].shape == (n_vars, n_vars)

        assert list(fit.coords.keys()) == ["rows", "columns"]
        assert fit.coords["rows"].values.tolist() == var_names
        assert fit.coords["columns"].values.tolist() == var_names

    def test_add_fit_to_inferencedata(self, simple_model, rng):
        model, mu_val, H_inv, test_point = simple_model
        idata = az.from_dict(posterior={"mu": rng.normal(size=()), "sigma": rng.normal(size=())})
        idata2 = add_fit_to_inference_data(idata, test_point, H_inv, model=model)

        self.check_idata(idata2, ["mu", "sigma"], 2)

    def test_add_fit_with_coords_to_inferencedata(self, hierarchical_model, rng):
        model, mu_val, H_inv, test_point = hierarchical_model
        idata = az.from_dict(
            posterior={
                "mu_loc": rng.normal(size=()),
                "mu_scale": rng.normal(size=()),
                "mu": rng.normal(size=(5,)),
                "sigma": rng.normal(size=()),
            }
        )

        idata2 = add_fit_to_inference_data(idata, test_point, H_inv, model=model)

        self.check_idata(
            idata2, ["mu_loc", "mu_scale", "mu[1]", "mu[2]", "mu[3]", "mu[4]", "mu[5]", "sigma"], 8
        )


def test_add_data_to_inferencedata(simple_model, rng):
    model, *_ = simple_model

    idata = az.from_dict(
        posterior={"mu": rng.standard_normal((1, 1)), "sigma": rng.standard_normal((1, 1))}
    )
    idata2 = add_data_to_inference_data(idata, model=model)
    assert "observed_data" in idata2.groups()
    assert "constant_data" in idata2.groups()
    assert "obs" in idata2.observed_data


def test_optimizer_result_to_dataset_basic(simple_model, rng):
    model, mu_val, H_inv, test_point = simple_model
    result = OptimizeResult(
        x=np.array([1.0, 2.0]),
        fun=0.5,
        success=True,
        message="Optimization succeeded",
        jac=np.array([0.1, 0.2]),
        nit=5,
        nfev=10,
        njev=3,
        status=0,
    )

    ds = optimizer_result_to_dataset(result, method="BFGS", model=model, mu=test_point)
    assert isinstance(ds, xr.Dataset)
    assert all(
        key in ds
        for key in [
            "x",
            "fun",
            "success",
            "message",
            "jac",
            "nit",
            "nfev",
            "njev",
            "status",
            "method",
        ]
    )

    assert list(ds["x"].coords.keys()) == ["variables"]
    assert ds["x"].coords["variables"].values.tolist() == ["mu", "sigma"]

    assert list(ds["jac"].coords.keys()) == ["variables"]
    assert ds["jac"].coords["variables"].values.tolist() == ["mu", "sigma"]


def test_optimizer_result_to_dataset_hess_inv_matrix(hierarchical_model, rng):
    model, mu_val, H_inv, test_point = hierarchical_model
    result = OptimizeResult(
        x=np.zeros((8,)),
        hess_inv=np.eye(8),
    )
    ds = optimizer_result_to_dataset(result, method="BFGS", model=model, mu=test_point)

    assert "hess_inv" in ds
    assert ds["hess_inv"].shape == (8, 8)
    assert list(ds["hess_inv"].coords.keys()) == ["variables", "variables_aux"]

    expected_names = ["mu_loc", "mu_scale", "mu[1]", "mu[2]", "mu[3]", "mu[4]", "mu[5]", "sigma"]
    assert ds["hess_inv"].coords["variables"].values.tolist() == expected_names
    assert ds["hess_inv"].coords["variables_aux"].values.tolist() == expected_names


def test_optimizer_result_to_dataset_hess_inv_linear_operator(simple_model, rng):
    model, mu_val, H_inv, test_point = simple_model
    n = mu_val.shape[0]

    def matvec(x):
        return np.array([2 * xi for xi in x])

    linop = LinearOperator((n, n), matvec=matvec)
    result = OptimizeResult(
        x=np.ones(n),
        hess_inv=linop,
    )

    with model:
        ds = optimizer_result_to_dataset(result, method="BFGS", mu=test_point)

    assert "hess_inv" in ds
    assert ds["hess_inv"].shape == (n, n)
    assert list(ds["hess_inv"].coords.keys()) == ["variables", "variables_aux"]

    expected_names = ["mu", "sigma"]
    assert ds["hess_inv"].coords["variables"].values.tolist() == expected_names
    assert ds["hess_inv"].coords["variables_aux"].values.tolist() == expected_names

    np.testing.assert_allclose(ds["hess_inv"].values, 2 * np.eye(n))


def test_optimizer_result_to_dataset_extra_fields(simple_model, rng):
    model, mu_val, H_inv, test_point = simple_model

    result = OptimizeResult(
        x=np.array([1.0, 2.0]),
        custom_stat=np.array([42, 43]),
    )

    with model:
        ds = optimizer_result_to_dataset(result, method="BFGS", mu=test_point)

    assert "custom_stat" in ds
    assert ds["custom_stat"].shape == (2,)
    assert list(ds["custom_stat"].coords.keys()) == ["custom_stat_dim_0"]
    assert ds["custom_stat"].coords["custom_stat_dim_0"].values.tolist() == [0, 1]
