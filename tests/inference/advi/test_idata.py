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
    add_fit_to_inference_data,
    add_optimizer_result_to_inference_data,
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


def test_fit_group_refuses_a_guide_that_does_not_report_its_parameterization(model):
    # this guide names its params like the mean-field one but links the scale through
    # softplus, so exponentiating its scale would report a plausible, wrong number
    loc, scale = pt.scalar("a_loc"), pt.scalar("a_scale")
    with pm.Model() as guide_model:
        z = pm.Normal("a_z")
        pm.Deterministic("a", loc + pt.softplus(scale) * z)
    guide = AutoGuideModel(guide_model, {loc: np.array(0.0), scale: np.array(0.1)})

    with pytest.raises(NotImplementedError, match="does not report a fitted mean"):
        add_fit_to_inference_data(
            DataTree(), guide, {"a_loc": np.array(0.0), "a_scale": np.array(0.1)}, model=model
        )


def test_optimizer_result_group_holds_the_trace_and_the_buffers():
    loss_history = np.array([5.0, 4.0, 3.5])
    optimizer_state = {
        "adam_t": np.asarray(3),
        "adam_m_theta_loc": np.zeros(2),
        "adam_v_theta_loc": np.ones(2),
        "adam_m_theta_scale": np.asarray(0.5),
        "adam_v_theta_scale": np.asarray(1.5),
    }

    idata = add_optimizer_result_to_inference_data(
        DataTree(),
        loss_history=loss_history,
        step=3,
        optimizer_state=optimizer_state,
        parameter_names=["theta_loc", "theta_scale"],
    )
    group = idata["optimizer_result"].dataset

    # the trace is reported as the ELBO, which is the negated loss the optimizer minimizes
    np.testing.assert_allclose(group["elbo"].values, -loss_history)
    assert group["elbo"].dims == ("step",)
    assert group["step_count"].item() == 3

    # one variable per buffer kind over a labelled parameter dim, not one per buffer, and
    # the clock stays a scalar because it belongs to no parameter
    assert set(group.data_vars) == {"elbo", "step_count", "adam_t", "adam_m", "adam_v"}
    assert group["adam_t"].shape == ()
    assert group["adam_m"].dims == ("parameter",)
    assert list(group.coords["parameter"].values) == [
        "theta_loc[0]",
        "theta_loc[1]",
        "theta_scale",
    ]
    np.testing.assert_allclose(group["adam_m"].values, [0.0, 0.0, 0.5])
    np.testing.assert_allclose(group["adam_v"].values, [1.0, 1.0, 1.5])


def test_optimizer_result_refuses_a_kind_that_covers_some_parameters():
    with pytest.raises(ValueError, match="cover only some of the guide's parameters"):
        add_optimizer_result_to_inference_data(
            DataTree(),
            loss_history=np.zeros(1),
            step=1,
            optimizer_state={"adam_m_theta_loc": np.zeros(2)},
            parameter_names=["theta_loc", "theta_scale"],
        )


def test_optimizer_result_splits_a_parameter_whose_name_suffixes_another():
    # a hierarchical model gives the guide both "beta_loc" and "mu_beta_loc", so
    # "adam_m_mu_beta_loc" ends with two parameter names and only the longer one is right
    idata = add_optimizer_result_to_inference_data(
        DataTree(),
        loss_history=np.zeros(1),
        step=1,
        optimizer_state={
            "adam_m_beta_loc": np.full(2, 1.0),
            "adam_m_mu_beta_loc": np.full(1, 2.0),
        },
        parameter_names=["beta_loc", "mu_beta_loc"],
    )
    group = idata["optimizer_result"].dataset

    assert set(group.data_vars) == {"elbo", "step_count", "adam_m"}
    assert list(group.coords["parameter"].values) == [
        "beta_loc[0]",
        "beta_loc[1]",
        "mu_beta_loc[0]",
    ]
    np.testing.assert_allclose(group["adam_m"].values, [1.0, 1.0, 2.0])


def test_optimizer_result_parameter_dim_follows_the_guide_order_and_labels_every_element():
    # the buffers arrive in the optimizer's order, not the guide's, and a full-rank guide
    # carries a matrix parameter whose elements must line up with their labels
    idata = add_optimizer_result_to_inference_data(
        DataTree(),
        loss_history=np.zeros(1),
        step=1,
        optimizer_state={
            "adam_m_theta_cholesky": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "adam_m_theta_loc": np.array([5.0, 6.0]),
        },
        parameter_names=["theta_loc", "theta_cholesky"],
    )
    group = idata["optimizer_result"].dataset

    assert list(group.coords["parameter"].values) == [
        "theta_loc[0]",
        "theta_loc[1]",
        "theta_cholesky[0,0]",
        "theta_cholesky[0,1]",
        "theta_cholesky[1,0]",
        "theta_cholesky[1,1]",
    ]
    np.testing.assert_allclose(group["adam_m"].values, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0])


def test_optimizer_result_group_holds_only_arrays_every_backend_can_store():
    idata = add_optimizer_result_to_inference_data(
        DataTree(),
        loss_history=np.array([1.0, 2.0]),
        step=2,
        optimizer_state={"adam_t": np.asarray(2), "adam_m_x": np.zeros(3)},
        parameter_names=["x"],
    )
    group = idata["optimizer_result"].dataset

    for name, array in group.data_vars.items():
        assert array.dtype.kind in "fiu", f"{name} has non-numeric dtype {array.dtype}"
    assert group.attrs == {}
