import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc.exceptions import ImputationWarning
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import graph_inputs
from scipy import stats

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.nonlinear import (
    FusedStateSpace,
    gauss_newton_curvature,
    observed_state_names,
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
        model=model, statespace=level_statespace(shuffled), rv_names=["alpha", "matrix"], frozen={}
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
            model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
    )

    theta = pt.vector("theta", shape=(2,))
    y = pt.vector("y")
    logp, _ = fused.observation_logp(parameters=theta, data={"y": y})["y"]
    y_values = np.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(
        logp.eval({theta: [np.log(3.0), 0.5], y: y_values}),
        stats.norm.logpdf(y_values, loc=0.5, scale=3.0),
    )


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
        )


def test_observed_length_follows_data():
    with pm.Model(coords={"obs": range(3)}) as model:
        x = pm.Data("x", np.ones(3), dims="obs")
        y_data = pm.Data("y_data", np.zeros(3), dims="obs")
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha * x, 1, observed=y_data, dims="obs")

    fused = FusedStateSpace(
        model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
        model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
        model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
        model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["alpha"], frozen={})

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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["alpha"], frozen={})

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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["alpha"], frozen={})

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
    fused = FusedStateSpace(model=model, statespace=statespace, rv_names=["alpha"], frozen={})

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
        model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
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
            model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
        )


def test_observed_rv_depending_on_another_raises():
    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        total = pm.Normal("total", alpha, 1, observed=np.zeros(3))
        pm.Normal("share", total, 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"\{'share': \['total'\]\}"):
        FusedStateSpace(
            model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"], frozen={}
        )


def test_imputed_observations_raise():
    with pm.Model() as model, pytest.warns(ImputationWarning):
        alpha = pm.Normal("alpha")
        pm.Normal("y", alpha, 1, observed=np.array([1.0, np.nan, 2.0]))

    with pytest.raises(ValueError, match=r"\['y_unobserved'\] impute missing observations"):
        FusedStateSpace(
            model=model,
            statespace=level_statespace(["alpha"]),
            rv_names=["alpha"],
            frozen={"y_unobserved": np.zeros(1)},
        )
