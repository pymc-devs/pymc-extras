import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from numpy.testing import assert_allclose
from pymc.exceptions import ImputationWarning
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import graph_inputs
from pytensor_ml.optim import zoom_line_search
from scipy import stats

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    gauss_newton_curvature,
    observed_state_names,
    run_filter,
)


def level_statespace(observed_state_names):
    return st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=observed_state_names
    ).build(verbose=False)


def test_observed_state_names_use_coords_where_they_exist():
    with pm.Model(coords={"group": ["a", "b"], "side": ["x", "y"]}) as model:
        model.add_coord("unnamed", length=2)
        pm.Normal("alpha")
        pm.Normal("beta", dims="group")
        pm.Normal("gamma", shape=2)
        pm.Normal("matrix", dims=("group", "side"))
        pm.Normal("delta", dims="unnamed")

    names = observed_state_names(
        model=model, rv_names=["alpha", "beta", "gamma", "matrix", "delta"]
    )

    assert names == {
        "alpha": ["alpha"],
        "beta": ["beta[a]", "beta[b]"],
        "gamma": ["gamma[0]", "gamma[1]"],
        "matrix": ["matrix[a, x]", "matrix[a, y]", "matrix[b, x]", "matrix[b, y]"],
        "delta": ["delta[0]", "delta[1]"],
    }


def test_unpack_follows_the_statespace_order():
    with pm.Model(coords={"group": ["a", "b"], "side": ["x", "y"]}) as model:
        alpha = pm.Normal("alpha")
        matrix = pm.Normal("matrix", dims=("group", "side"))
        pm.Normal("y", alpha + matrix.sum(), 1, observed=np.zeros(3))

    names = observed_state_names(model=model, rv_names=["alpha", "matrix"])
    shuffled = [names["matrix"][3], *names["alpha"], *names["matrix"][:3]]
    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(shuffled),
        rv_names=["alpha", "matrix"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(5,))
    values = fused.unpack(theta)
    unpacked = {name: value.eval({theta: np.arange(5.0)}) for name, value in values.items()}
    assert unpacked["alpha"] == 1.0
    np.testing.assert_array_equal(unpacked["matrix"], [[2.0, 3.0], [4.0, 0.0]])


def test_unlisted_free_rv_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha, sigma, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"\['sigma'\] are neither in rv_names nor frozen"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_statespace_with_other_observed_states_raises():
    with pm.Model(coords={"group": ["a", "b"]}) as model:
        beta = pm.Normal("beta", dims="group")
        pm.Normal("y", beta.sum(), 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"missing: \['beta\[b\]'\], unexpected: \['delta'\]"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["beta[a]", "delta"]),
            rv_names=["beta"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_rv_names_that_are_not_free_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        double = pm.Deterministic("double", 2 * alpha)
        pm.Normal("y", double, 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"\['double'\] are not free random variables"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha", "double"]),
            rv_names=["alpha", "double"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_frozen_rvs_are_constants():
    with pm.Model() as model:
        model.add_coord("row", length=2)
        model.add_coord("column", length=2)
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        weights = pm.Normal("weights", dims=("row", "column"))
        pm.Normal("y", alpha + weights.sum(), sigma, observed=np.zeros(3))

    # The two-dimensional variable has no coordinate values, which only a mapped variable needs.
    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={"sigma": 2.0, "weights": np.ones((2, 2))},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    y = pt.vector("y")
    logp, _ = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    np.testing.assert_allclose(
        logp.eval({theta: [0.5], y: np.zeros(3)}),
        stats.norm.logpdf(np.zeros(3), loc=4.5, scale=2.0),
    )
    inputs = [variable for variable in graph_inputs([logp]) if not isinstance(variable, Constant)]
    assert set(inputs) == {theta, y}


def test_rv_both_mapped_and_frozen_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"\['alpha'\] are both in rv_names and frozen"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={"alpha": 0.0},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_frozen_value_with_wrong_shape_raises():
    with pm.Model(coords={"group": ["a", "b"]}) as model:
        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta", dims="group")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha + beta.sum(), sigma, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"'beta': \(\(3,\), \(2,\)\), 'sigma': \(\(2,\), \(\)\)"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={"beta": np.ones(3), "sigma": np.ones(2)},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_frozen_value_may_be_symbolic():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha, sigma, observed=np.zeros(3))

    sigma_value = pt.scalar("sigma_value")
    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={"sigma": sigma_value},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    y = pt.vector("y")
    logp, _ = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    y_values = np.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(
        logp.eval({theta: [0.5], sigma_value: 3.0, y: y_values}),
        stats.norm.logpdf(y_values, loc=0.5, scale=3.0),
    )


def test_positive_rv_state_is_log_scale():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha, sigma, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["sigma", "alpha"]),
        rv_names=["alpha", "sigma"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    logp, _ = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    y_values = np.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(
        logp.eval({theta: [np.log(3.0), 0.5], y: y_values}),
        stats.norm.logpdf(y_values, loc=0.5, scale=3.0),
    )


def test_prior_logp_is_on_the_unconstrained_scale_with_frozen_values():
    with pm.Model() as model:
        mu = pm.Normal("mu")
        alpha = pm.Normal("alpha", mu=mu, sigma=2.0)
        sigma = pm.HalfNormal("sigma", sigma=1.5)
        offset = pm.HalfNormal("offset")
        pm.Normal("y", alpha + offset, sigma, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["sigma", "alpha"]),
        rv_names=["alpha", "sigma"],
        frozen={"mu": 0.5, "offset": 0.3},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=1.0,
    )

    theta = pt.vector("theta", shape=(2,))
    log_sigma, alpha_value = np.log(0.7), -1.2
    np.testing.assert_allclose(
        fused.prior_logp(theta).eval({theta: [log_sigma, alpha_value]}),
        stats.norm.logpdf(alpha_value, loc=0.5, scale=2.0)
        + stats.halfnorm.logpdf(0.7, scale=1.5)
        + log_sigma,
    )


def test_prior_logp_with_a_frozen_transform_depending_on_another_frozen_rv():
    with pm.Model() as model:
        upper = pm.HalfNormal("upper")
        location = pm.Uniform("location", lower=0.0, upper=upper)
        alpha = pm.Normal("alpha", mu=location, sigma=2.0)
        pm.Normal("y", alpha, 1.0, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={"upper": 3.0, "location": 1.5},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=1.0,
    )

    theta = pt.vector("theta", shape=(1,))
    np.testing.assert_allclose(
        fused.prior_logp(theta).eval({theta: [0.4]}), stats.norm.logpdf(0.4, loc=1.5, scale=2.0)
    )


def test_filter_output_sampling_is_not_implemented():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1.0, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(NotImplementedError, match="run_filter or sample_conditional_posterior"):
        fused.sample_filter_outputs(None)


def test_transform_depending_on_another_rv():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.Uniform("beta", lower=alpha, upper=alpha + 2)
        pm.Normal("y", beta, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha", "beta"]),
        rv_names=["alpha", "beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    logp, _ = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    y_values = np.array([0.3, -1.0, 2.0])
    # beta = alpha + 2 * sigmoid(theta_beta), so a zero logit is the midpoint.
    np.testing.assert_allclose(
        logp.eval({theta: [1.0, 0.0], y: y_values}), stats.norm.logpdf(y_values, loc=2.0)
    )


def test_transform_that_changes_shape_raises():
    with pm.Model(coords={"component": ["a", "b", "c"]}) as model:
        weights = pm.Dirichlet("weights", a=np.ones(3), dims="component")
        pm.Normal("y", weights[0], 1, observed=np.zeros(3))

    names = observed_state_names(model=model, rv_names=["weights"])
    with pytest.raises(ValueError, match=r"\['weights'\] have transforms that change their shape"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(names["weights"]),
            rv_names=["weights"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_observed_length_follows_data():
    with pm.Model(coords={"obs": range(3)}) as model:
        x = pm.Data("x", np.ones(3), dims="obs")
        y_data = pm.Data("y_data", np.zeros(3), dims="obs")
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha * x, 1, observed=y_data, dims="obs")

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    x_input = pt.vector("x")
    y_input = pt.vector("y_data")
    logp, _ = fused.observation_logp(parameters=theta, data={"x": x_input, "y_data": y_input})["y"]
    x_values = np.arange(5.0)
    y_values = np.linspace(-1.0, 1.0, 5)
    np.testing.assert_allclose(
        logp.eval({theta: [0.7], x_input: x_values, y_input: y_values}),
        stats.norm.logpdf(y_values, loc=0.7 * x_values),
    )


def test_raw_numpy_observed_is_replaced():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.MvNormal("w", mu=alpha + np.zeros(2), cov=np.eye(2), observed=np.zeros((4, 2)))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    w = pt.matrix("w")
    logp, _ = fused.observation_logp(parameters=theta, data={"w": w})["w"]
    w_values = np.linspace(-1.0, 1.0, 12).reshape(6, 2)
    np.testing.assert_allclose(
        logp.eval({theta: [0.5], w: w_values}),
        stats.multivariate_normal.logpdf(w_values, mean=[0.5, 0.5]),
    )


def test_data_with_other_names_raises():
    with pm.Model() as model:
        x = pm.Data("x", np.ones(3))
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha * x, 1, observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"missing: \['y'\], unexpected: \['z'\]"):
        fused.observation_logp(
            parameters=pt.vector("theta", shape=(1,)),
            data={"x": pt.vector("x"), "z": pt.vector("z")},
        )


def test_deterministics_are_inlined():
    with pm.Model(coords={"obs": range(3)}) as model:
        x = pm.Data("x", np.ones(3), dims="obs")
        y_data = pm.Data("y_data", np.zeros(3), dims="obs")
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        mean = pm.Deterministic("mean", alpha * x, dims="obs")
        pm.Normal("y", mean, sigma, observed=y_data, dims="obs")

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["sigma", "alpha"]),
        rv_names=["alpha", "sigma"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(2,))
    x_input = pt.vector("x")
    y_input = pt.vector("y_data")
    logp, _ = fused.observation_logp(parameters=theta, data={"x": x_input, "y_data": y_input})["y"]
    x_values = np.arange(4.0)
    y_values = np.array([0.3, -1.0, 2.0, 0.5])
    np.testing.assert_allclose(
        logp.eval({theta: [np.log(1.5), 0.7], x_input: x_values, y_input: y_values}),
        stats.norm.logpdf(y_values, loc=0.7 * x_values, scale=1.5),
    )


def test_distribution_parameters_are_those_that_depend_on_theta():
    with pm.Model(coords={"obs": range(3)}) as model:
        x = pm.Data("x", np.ones(3), dims="obs")
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha * x, sigma, observed=np.zeros(3), dims="obs")

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={"sigma": 2.0},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    x_input = pt.vector("x")
    y = pt.vector("y")
    logp, distribution_parameters = fused.observation_logp(
        parameters=theta, data={"x": x_input, "y": y}
    )["y"]
    [mean] = distribution_parameters
    curvature = gauss_newton_curvature(
        pointwise_logp=logp, distribution_parameters=[mean], parameters=theta
    )
    x_values = np.arange(4.0)
    np.testing.assert_allclose(
        curvature.eval({theta: [0.7], x_input: x_values, y: np.zeros(4)}),
        [[np.sum(x_values**2) / 4.0]],
    )


def test_opaque_likelihood_has_no_distribution_parameters():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Censored("y", pm.Normal.dist(alpha, 1), lower=-1, upper=1, observed=np.zeros(2))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    y = pt.vector("y")
    logp, distribution_parameters = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    assert distribution_parameters is None
    y_values = np.array([-1.0, 0.2, 1.0])
    expected = pm.logp(pm.Censored.dist(pm.Normal.dist(0.5, 1), lower=-1, upper=1), y_values)
    np.testing.assert_allclose(logp.eval({theta: [0.5], y: y_values}), expected.eval())


def test_each_observed_rv_gets_its_own_logp():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", alpha, sigma, observed=np.zeros(3))
        pm.Poisson("counts", pm.math.exp(alpha), observed=np.zeros(2))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha", "sigma"]),
        rv_names=["alpha", "sigma"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    counts = pt.vector("counts", dtype="int64")
    observation_logp = fused.observation_logp(parameters=theta, data={"y": y, "counts": counts})
    y_logp, _ = observation_logp["y"]
    counts_logp, [counts_rate] = observation_logp["counts"]
    theta_values = [0.5, np.log(2.0)]
    np.testing.assert_allclose(
        y_logp.eval({theta: theta_values, y: [0.3, -1.0, 2.0]}),
        stats.norm.logpdf([0.3, -1.0, 2.0], loc=0.5, scale=2.0),
    )
    np.testing.assert_allclose(
        counts_logp.eval({theta: theta_values, counts: [1, 4]}),
        stats.poisson.logpmf([1, 4], mu=np.exp(0.5)),
    )
    np.testing.assert_allclose(
        counts_rate.eval({theta: theta_values, counts: [1, 4]}), np.exp([0.5, 0.5])
    )


def test_statespace_matrices_at_given_parameters():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Autoregressive(order=2, name="ar", observed_state_names=["alpha"]).build(
        verbose=False
    )
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    initial_covariance = [[2.0, 0.3], [0.3, 1.0]]
    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters={"params_ar": [0.5, -0.2], "sigma_ar": 0.5, "P0": initial_covariance},
        statespace_data={},
    )
    np.testing.assert_allclose(x0.eval(), [0.0, 0.0])
    np.testing.assert_allclose(P0.eval(), initial_covariance)
    np.testing.assert_allclose(c.eval(), [0.0, 0.0])
    np.testing.assert_allclose(d.eval(), [0.0])
    np.testing.assert_allclose(T.eval(), [[0.5, -0.2], [1.0, 0.0]])
    np.testing.assert_allclose(Z.eval(), [[1.0, 0.0]])
    np.testing.assert_allclose(R.eval(), [[1.0], [0.0]])
    np.testing.assert_allclose(H.eval(), [[0.0]])
    np.testing.assert_allclose(Q.eval(), [[0.25]])


def test_statespace_matrices_accept_symbolic_parameters():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Autoregressive(order=1, name="ar", observed_state_names=["alpha"]).build(
        verbose=False
    )
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    rho = pt.vector("rho", shape=(1,))
    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters={"params_ar": rho, "sigma_ar": 0.5, "P0": [[2.0]]},
        statespace_data={},
    )
    np.testing.assert_allclose(T.eval({rho: [-0.3]}), [[-0.3]])


def test_statespace_matrices_with_other_names_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Autoregressive(order=1, name="ar", observed_state_names=["alpha"]).build(
        verbose=False
    )
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    with pytest.raises(ValueError, match=r"missing: \['P0'\], unexpected: \['rho'\]"):
        fused.statespace_matrices(
            statespace_parameters={"params_ar": [0.8], "sigma_ar": 0.5, "rho": 1.0},
            statespace_data={},
        )


def test_statespace_matrices_take_statespace_data():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    statespace = st.Regression(
        name="exog", state_names=["x1"], observed_state_names=["alpha"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    exog = pt.matrix("exog", shape=(None, 1))
    x0, P0, c, d, T, Z, R, H, Q = fused.statespace_matrices(
        statespace_parameters={"beta_exog": [0.5], "P0": [[1.0]]},
        statespace_data={"data_exog": exog},
    )
    np.testing.assert_allclose(Z.eval({exog: [[2.0], [3.0]]}), [[[2.0]], [[3.0]]])
    with pytest.raises(ValueError, match=r"statespace_data .* missing: \['data_exog'\]"):
        fused.statespace_matrices(
            statespace_parameters={"beta_exog": [0.5], "P0": [[1.0]]}, statespace_data={}
        )


def test_observations_without_scalar_parameters_or_vector_batch_have_none():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.MvNormal("w", mu=alpha + np.zeros(2), cov=np.eye(2), observed=np.zeros((4, 2)))
        pm.Normal("grid", alpha, 1, observed=np.zeros((3, 2)))
        pm.Categorical("choice", p=pm.math.softmax(pt.stack([alpha, 0.0])), observed=np.zeros(3))

    fused = FusedStateSpace(
        model=model,
        statespace=level_statespace(["alpha"]),
        rv_names=["alpha"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )

    theta = pt.vector("theta", shape=(1,))
    grid = pt.matrix("grid")
    observation_logp = fused.observation_logp(
        parameters=theta,
        data={"w": pt.matrix("w"), "grid": grid, "choice": pt.lvector("choice")},
    )
    assert observation_logp["w"][1] is None
    assert observation_logp["grid"][1] is None
    assert observation_logp["choice"][1] is None

    grid_logp, _ = observation_logp["grid"]
    np.testing.assert_allclose(
        grid_logp.eval({theta: [0.5], grid: np.ones((3, 2))}),
        stats.norm.logpdf(np.ones((3, 2)), loc=0.5),
    )


def test_potentials_raise():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        pm.Potential("penalty", -(alpha**2))
        pm.Normal("y", alpha, 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"Potentials \['penalty'\]"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_observed_rv_depending_on_another_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        total = pm.Normal("total", alpha, 1, observed=np.zeros(3))
        pm.Normal("share", total, 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"\{'share': \['total'\]\}"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_imputed_observations_raise():
    with pm.Model() as model, pytest.warns(ImputationWarning):
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.array([1.0, np.nan, 2.0]))

    with pytest.raises(ValueError, match=r"\['y'\] impute missing observations"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={"y_unobserved": np.zeros(1)},
            curvature="gauss_newton",
            line_search=zoom_line_search(),
            tolerance=1e-10,
            max_iterations=10,
            prior_weight=0.0,
        )


def test_model_logp_matches_build_statespace_graph():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        beta = pm.Normal("beta", dims="coef")
        pm.Normal("y", mu=beta, sigma=0.5, observed=np.zeros(2))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=10,
        prior_weight=0.0,
    )
    linear = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=["x1", "x2"])
        + st.MeasurementError(name="noise", observed_state_names=["x1", "x2"])
    ).build(verbose=False)

    y = np.random.default_rng(0).normal(size=(6, 2))
    with pm.Model() as fused_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        fused.build_statespace_graph(data={"y": list(y)}, dates=np.arange(6))
    with pm.Model() as linear_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        pm.Deterministic("sigma_noise", pt.as_tensor([0.5, 0.5]))
        linear.build_statespace_graph(data=y)

    fused_logp, linear_logp = fused_model.compile_logp(), linear_model.compile_logp()
    point = {
        "initial_level_trend": np.array([[0.3], [-0.2]]),
        "sigma_level_trend_log__": np.log([[0.4], [0.2]]),
    }
    assert_allclose(fused_logp(point), linear_logp(point), atol=1e-10)


def test_dlogp_vanishes_at_stored_mode():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    beta_path = np.cumsum(np.array([0.2, 0.4]) * rng.normal(size=(20, 2)), axis=0)
    X_dates = rng.normal(scale=0.5, size=(20, 20, 2))
    y_dates = rng.poisson(lam=np.exp(np.einsum("tnk,tk->tn", X_dates, beta_path)))
    with pm.Model() as fused_model:
        pm.Normal("initial_level_trend", shape=(2, 1))
        pm.HalfNormal("sigma_level_trend", shape=(2, 1))
        pm.Deterministic("P0", pt.eye(2))
        fused.build_statespace_graph(
            data={"X": list(X_dates), "y": list(y_dates)}, dates=np.arange(20)
        )

    # The mode find_MAP reached with BFGS at tol=1e-12. Shifting both log scales by 0.01 moves
    # their gradients to 0.04 or more.
    mode = {
        "initial_level_trend": np.array([[0.063079986435], [0.019299647647]]),
        "sigma_level_trend_log__": np.log([[0.185623519401], [0.140406251664]]),
    }
    dlogp = fused_model.compile_dlogp(jacobian=False)
    assert_allclose(dlogp(mode), 0.0, atol=1e-6)


def test_conditional_posterior_draws_match_run_filter():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [3, 1, 0, 4]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.poisson(lam=2.0, size=size) for size in sizes],
    }
    parameters = {
        "initial_level_trend": np.array([[0.3], [-0.1]]),
        "sigma_level_trend": np.array([[0.3], [0.2]]),
        "P0": np.eye(2),
    }
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(4))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    n_draws = 4000
    idata["posterior"] = xr.Dataset(
        {
            name: (
                ("chain", "draw", *fused.param_dims[name]),
                np.broadcast_to(value, (1, n_draws, *value.shape)),
            )
            for name, value in parameters.items()
        }
    )
    conditional = fused.sample_conditional_posterior(idata, random_seed=1, progressbar=False)
    reference = run_filter(
        fused=fused,
        statespace_parameters=parameters,
        statespace_data={},
        data=data,
        dates=np.arange(4),
    )["posterior_predictive"].isel(chain=0, draw=0)

    for kind in ["filtered", "smoothed"]:
        draws = conditional[f"{kind}_posterior"].values[0]
        mean = reference[f"{kind}_states"].values
        sd = np.sqrt(np.diagonal(reference[f"{kind}_covariances"].values, axis1=1, axis2=2))
        assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * sd / np.sqrt(n_draws))
        assert_allclose(draws.std(axis=0), sd, rtol=0.05)


def test_conditional_observations_follow_the_observation_model_at_each_draw():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=X @ beta, sigma=sigma, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={"sigma": 1e-8},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [3, 1, 2]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.normal(size=size) for size in sizes],
    }
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(3))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    theta = rng.normal(size=(1, 5, 3, 2))
    conditional = xr.Dataset(
        {"smoothed_posterior_observed": (("chain", "draw", "time", "observed_state"), theta)},
        coords={"chain": [0], "draw": np.arange(5), "time": np.arange(3)},
    )
    draws = fused.sample_conditional_observations(
        idata=idata, conditional=conditional, kind="smoothed", random_seed=1
    )["y"].values

    for date, (X_date, size) in enumerate(zip(data["X"], sizes, strict=True)):
        assert_allclose(draws[:, :, date, :size], theta[:, :, date] @ X_date.T, atol=1e-6)
        assert np.isnan(draws[:, :, date, size:]).all()


def test_conditional_observations_carry_independent_observation_noise_on_each_date():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=X @ beta, sigma=sigma, observed=np.zeros(4))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={"sigma": 0.5},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    X_date = rng.normal(size=(2, 2))
    data = {"X": [X_date, X_date], "y": [rng.normal(size=2), rng.normal(size=2)]}
    with pm.Model(coords=fused.coords):
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(2))
        idata = pm.sample_prior_predictive(draws=1, random_seed=0)

    n_draws = 20000
    theta = np.array([0.4, -0.7])
    conditional = xr.Dataset(
        {
            "smoothed_posterior_observed": (
                ("chain", "draw", "time", "observed_state"),
                np.broadcast_to(theta, (1, n_draws, 2, 2)),
            )
        },
        coords={"chain": [0], "draw": np.arange(n_draws), "time": np.arange(2)},
    )
    draws = fused.sample_conditional_observations(
        idata=idata, conditional=conditional, kind="smoothed", random_seed=1
    )["y"].values[0]

    for date in range(2):
        assert np.all(
            np.abs(draws[:, date].mean(axis=0) - X_date @ theta) < 4 * 0.5 / np.sqrt(n_draws)
        )
        assert_allclose(draws[:, date].std(axis=0), 0.5, rtol=0.03)
    across_dates = np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1]
    assert np.abs(across_dates) < 4 / np.sqrt(n_draws)


@pytest.mark.xfail(
    reason="statespace_matrices cannot convert float32 parameter values to the statespace's types",
    raises=TypeError,
    strict=True,
)
@pytensor.config.change_flags(floatX="float32")
def test_large_integer_observations_survive_packing_in_float32():
    with pm.Model(coords={"coef": ["x1"]}) as model:
        X = pm.Data("X", np.zeros((2, 1)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(2, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    # 2**24 + 1 is the smallest positive integer float32 cannot represent.
    counts = np.array([2**24 + 1, 2**25 + 3])
    with pm.Model(coords=fused.coords) as fused_model:
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(1), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(
            data={"X": [np.ones((2, 1))], "y": [counts]}, dates=np.arange(1)
        )

    observations = fused_model.rvs_to_values[fused_model["obs"]].data
    np.testing.assert_array_equal(observations[0], counts)


def test_prior_predictive_observations_have_the_statespace_moments():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )

    design = np.array([[1.0, 0.5], [1.0, -1.0], [0.0, 1.0]])
    mean = np.array([0.5, -0.3])
    initial_covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    sigma = np.array([0.4, 0.3])
    with pm.Model(coords=fused.coords):
        pm.Deterministic("initial_level_trend", pt.as_tensor(mean[:, None]))
        pm.Deterministic("sigma_level_trend", pt.as_tensor(sigma[:, None]))
        pm.Deterministic("P0", pt.as_tensor(initial_covariance))
        fused.build_statespace_graph(
            data={"X": [design, design], "y": [np.zeros(3, dtype=int)] * 2}, dates=np.arange(2)
        )
        draws = pm.sample_prior_predictive(draws=40_000, random_seed=1).prior_predictive["obs"]

    # The coefficients on the first date have the initial covariance, and on the second the
    # random walk has added its variance, so each station's count has a lognormal Poisson mean.
    for date, covariance in enumerate([initial_covariance, initial_covariance + np.diag(sigma**2)]):
        expected = np.exp(
            design @ mean + 0.5 * np.einsum("ik,kl,il->i", design, covariance, design)
        )
        assert_allclose(draws.values[0, :, date].mean(axis=0), expected, rtol=0.02)


def test_per_date_logp_with_several_observed_variables_matches_run_filter():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        x = pm.Data("x", np.zeros(3))
        pm.Poisson("counts", mu=pm.math.exp(alpha + x), observed=np.zeros(3, dtype=int))
        pm.Normal("sizes", mu=alpha, sigma=0.5, observed=np.zeros(2))

    fused = FusedStateSpace(
        model=model,
        statespace=st.LevelTrend(
            order=1, innovations_order=1, observed_state_names=["alpha"]
        ).build(verbose=False),
        rv_names=["alpha"],
        frozen={},
        curvature="exact",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=20,
        prior_weight=0.0,
    )
    rng = np.random.default_rng(0)
    counts_sizes, sizes_sizes = [3, 1, 0, 2], [2, 0, 1, 2]
    data = {
        "x": [rng.normal(size=size) for size in counts_sizes],
        "counts": [rng.poisson(lam=2.0, size=size) for size in counts_sizes],
        "sizes": [rng.normal(size=size) for size in sizes_sizes],
    }
    parameters = {"initial_level_trend": [0.2], "sigma_level_trend": [0.3], "P0": [[1.0]]}

    with pm.Model(coords=fused.coords) as fused_model:
        for name, value in parameters.items():
            pm.Deterministic(name, pt.as_tensor(np.array(value)))
        fused.build_statespace_graph(data=data, dates=np.arange(4))
    [per_date_logp] = fused_model.compile_logp(vars=[fused_model["obs"]], sum=False)({})

    reference = run_filter(
        fused=fused,
        statespace_parameters=parameters,
        statespace_data={},
        data=data,
        dates=np.arange(4),
    )
    assert_allclose(per_date_logp, reference["diagnostics/loglike_obs"].values, atol=1e-10)


def test_model_gradient_matches_finite_differences():
    with pm.Model(coords={"coef": ["x1", "x2"]}) as model:
        X = pm.Data("X", np.zeros((4, 2)))
        beta = pm.Normal("beta", dims="coef")
        pm.Poisson("y", mu=pm.math.exp(X @ beta), observed=np.zeros(4, dtype=int))

    names = observed_state_names(model=model, rv_names=["beta"])
    statespace = st.LevelTrend(
        order=1, innovations_order=1, observed_state_names=names["beta"]
    ).build(verbose=False)
    fused = FusedStateSpace(
        model=model,
        statespace=statespace,
        rv_names=["beta"],
        frozen={},
        curvature="gauss_newton",
        line_search=zoom_line_search(),
        tolerance=1e-10,
        max_iterations=50,
        prior_weight=0.0,
    )

    rng = np.random.default_rng(0)
    sizes = [6, 4, 0, 5, 6]
    data = {
        "X": [rng.normal(size=(size, 2)) for size in sizes],
        "y": [rng.poisson(lam=3.0, size=size) for size in sizes],
    }
    with pm.Model(coords=fused.coords) as fused_model:
        pm.Normal("initial_level_trend", dims=fused.param_dims["initial_level_trend"])
        pm.HalfNormal("sigma_level_trend", sigma=0.5, dims=fused.param_dims["sigma_level_trend"])
        pm.Deterministic("P0", pt.eye(2), dims=fused.param_dims["P0"])
        fused.build_statespace_graph(data=data, dates=np.arange(len(sizes)))

    # PyMC rewrites the log-probability before differentiating it, which is what this checks.
    point = {
        "initial_level_trend": np.array([[0.4], [-0.3]]),
        "sigma_level_trend_log__": np.log([[0.3], [0.2]]),
    }
    logp, dlogp = fused_model.compile_logp(), fused_model.compile_dlogp()
    flat = np.concatenate([value.ravel() for value in point.values()])

    def logp_at(vector):
        return logp(
            {
                "initial_level_trend": vector[:2].reshape(2, 1),
                "sigma_level_trend_log__": vector[2:].reshape(2, 1),
            }
        )

    step = 1e-6
    finite_differences = [
        (logp_at(flat + step * unit) - logp_at(flat - step * unit)) / (2 * step)
        for unit in np.eye(flat.size)
    ]
    assert_allclose(dlogp(point), finite_differences, atol=1e-5)
