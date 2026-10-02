import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from xarray import DataTree

from pymc_extras.inference.advi.autoguide import (
    AutoDiagonalNormal,
    AutoGuideModel,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)
from pymc_extras.inference.advi.idata import (
    add_fit_stats_to_inference_data,
    add_fit_to_inference_data,
)


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
    params = initial_params(guide)

    fit = add_fit_to_inference_data(DataTree(), guide, params, model=model)["fit"].dataset

    assert set(fit.data_vars) == {"mean_vector", "cov_factor", "diagonal_standard_deviation"}
    assert fit["cov_factor"].dims == ("rows", "factors")
    assert fit["cov_factor"].shape[1] == 2
    # d itself, not d ** 2: the covariance is W @ W.T + diag(d ** 2), so storing the
    # squared term under a name that says standard deviation would silently mislead
    np.testing.assert_allclose(
        fit["diagonal_standard_deviation"].values,
        np.exp(params["cov_diag_unconstrained"]),
    )


def test_fit_group_rows_are_labelled_in_unconstrained_space(model):
    guide = AutoDiagonalNormal(model, random_seed=0)

    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
        "fit"
    ].dataset

    # the transformed variable is labelled by its value variable, and the vector one
    # element per scalar, matching how the Laplace fit group labels its rows
    assert list(fit.coords["rows"].values) == ["a", "s_log__", "b[0]", "b[1]"]


@pytest.mark.parametrize(
    "make_guide",
    [
        AutoDiagonalNormal,
        AutoMultivariateNormal,
        lambda m, random_seed: AutoLowRankMultivariateNormal(m, rank=2, random_seed=random_seed),
    ],
    ids=["mean_field", "full_rank", "low_rank"],
)
def test_fit_group_mean_vector_element_matches_its_row_label(make_guide):
    # the mean-field guide concatenates per-RV params in model.free_RVs order while the
    # multivariate guides ravel in point_map_info order; the rows coord is built from
    # free_RVs, so a divergence between the two would mislabel every element silently
    with pm.Model() as model:
        pm.Normal("a", initval=1.0)
        pm.HalfNormal("s", initval=np.exp(2.0))
        pm.Normal("b", shape=2, initval=[3.0, 4.0])
        pm.Normal("y", 0, 1, observed=[1.0, 2.0])

    guide = make_guide(model, random_seed=0)
    fit = add_fit_to_inference_data(DataTree(), guide, initial_params(guide), model=model)[
        "fit"
    ].dataset

    assert list(fit.coords["rows"].values) == ["a", "s_log__", "b[0]", "b[1]"]
    np.testing.assert_allclose(fit["mean_vector"].values, [1.0, 2.0, 3.0, 4.0])


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


def test_fit_group_defaults_to_the_raw_parameters_of_a_custom_guide(model):
    # a custom guide implements nothing, so its fit group is its parameters as fitted,
    # each over dims of its own so that differently shaped parameters do not collide
    loc, scale = pt.vector("a_loc", shape=(2,)), pt.vector("a_scale", shape=(3,))
    with pm.Model() as guide_model:
        z = pm.Normal("a_z")
        pm.Deterministic("a", loc.sum() + pt.softplus(scale).sum() * z)
    params = {"a_loc": np.array([1.0, 2.0]), "a_scale": np.array([0.1, 0.2, 0.3])}
    guide = AutoGuideModel(guide_model, {loc: params["a_loc"], scale: params["a_scale"]})

    fit = add_fit_to_inference_data(DataTree(), guide, params, model=model)["fit"].dataset

    assert fit["a_loc"].dims == ("a_loc_dim_0",)
    assert fit["a_scale"].dims == ("a_scale_dim_0",)
    np.testing.assert_allclose(fit["a_loc"].values, params["a_loc"])
    np.testing.assert_allclose(fit["a_scale"].values, params["a_scale"])
    assert "rows" not in fit.coords


def test_fit_stats_group_holds_the_elbo_over_steps():
    loss_history = np.array([5.0, 4.0, 3.5])

    fit_stats = add_fit_stats_to_inference_data(DataTree(), loss_history)["fit_stats"].dataset

    # the trace is reported as the ELBO, which is the negated loss the optimizer minimizes
    np.testing.assert_allclose(fit_stats["elbo"].values, -loss_history)
    assert all(array.dims == ("step",) for array in fit_stats.data_vars.values())
