import numpy as np
import pymc as pm
import pytest

from xarray import DataTree

from pymc_extras.inference.advi.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)
from pymc_extras.inference.advi.idata import add_fit_to_inference_data


@pytest.fixture
def model():
    with pm.Model() as model:
        pm.Normal("a")
        pm.HalfNormal("s")
        pm.Normal("b", shape=2)
        pm.Normal("y", 0, 1, observed=[1.0, 2.0])
    return model


def initial_params(guide):
    return {param.name: value for param, value in guide.params_init_values.items()}


def test_mean_field_fit_group_holds_a_marginal_standard_deviation(model):
    guide = AutoDiagonalNormal(model, random_seed=0)
    params = initial_params(guide)

    fit = add_fit_to_inference_data(DataTree(), guide, params, model=model)["fit"].dataset

    assert set(fit.data_vars) == {"mean_vector", "standard_deviation"}
    assert fit["mean_vector"].dims == ("rows",)
    assert fit["standard_deviation"].dims == ("rows",)

    # the guide stores the scale unconstrained; the group reports the standard deviation
    expected = np.exp(np.concatenate([params["a_scale"].ravel(), params["s_scale"].ravel()]))
    np.testing.assert_allclose(fit["standard_deviation"].values[:2], expected)


def test_full_rank_fit_group_holds_a_cholesky_factor(model):
    guide = AutoMultivariateNormal(model, random_seed=0)

    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
        "fit"
    ].dataset

    assert set(fit.data_vars) == {"mean_vector", "cholesky_lower"}
    assert fit["cholesky_lower"].dims == ("rows", "columns")

    cholesky = fit["cholesky_lower"].values
    np.testing.assert_allclose(cholesky, np.tril(cholesky))
    assert (np.diagonal(cholesky) > 0).all()


def test_low_rank_fit_group_holds_the_factor_and_diagonal(model):
    guide = AutoLowRankMultivariateNormal(model, rank=2, random_seed=0)

    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
        "fit"
    ].dataset

    assert set(fit.data_vars) == {"mean_vector", "cov_factor", "cov_diag"}
    assert fit["cov_factor"].dims == ("rows", "factors")
    assert fit["cov_factor"].shape[1] == 2
    assert (fit["cov_diag"].values > 0).all()


def test_fit_group_rows_are_labelled_in_unconstrained_space(model):
    guide = AutoDiagonalNormal(model, random_seed=0)

    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
        "fit"
    ].dataset

    # the transformed variable is labelled by its value variable, and the vector one
    # element per scalar, matching how the Laplace fit group labels its rows
    assert list(fit.coords["rows"].values) == ["a", "s_log__", "b[0]", "b[1]"]


def test_fit_group_holds_only_arrays_every_backend_can_store(model):
    for guide in (
        AutoDiagonalNormal(model, random_seed=0),
        AutoMultivariateNormal(model, random_seed=0),
        AutoLowRankMultivariateNormal(model, rank=2, random_seed=0),
    ):
        fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
            "fit"
        ].dataset
        for name, array in fit.data_vars.items():
            assert array.dtype.kind in "fiu", f"{name} has non-numeric dtype {array.dtype}"
        assert fit.attrs == {}


def test_mean_field_fit_group_is_linear_in_the_parameter_count():
    # a dense covariance would be quadratic here, which is the representation the mean-field
    # guide exists to avoid
    n_dim = 5_000
    with pm.Model() as wide_model:
        pm.Normal("wide", shape=n_dim)
        pm.Normal("y", 0, 1, observed=[1.0])

    guide = AutoDiagonalNormal(wide_model, random_seed=0)
    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=wide_model)[
        "fit"
    ].dataset

    assert fit["mean_vector"].shape == (n_dim,)
    assert fit["standard_deviation"].shape == (n_dim,)
    assert sum(array.size for array in fit.data_vars.values()) == 2 * n_dim
