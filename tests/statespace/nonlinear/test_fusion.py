import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc.model.fgraph import ModelFreeRV, ModelObservedRV
from pytensor.compile.sharedvalue import SharedVariable

from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.nonlinear import FusedStateSpace, observed_state_names


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
    fgraph = fused._rewrite(parameters=theta)
    nodes = fgraph.apply_nodes
    assert not any(isinstance(node.op, ModelFreeRV) for node in nodes)
    [observed] = [node for node in nodes if isinstance(node.op, ModelObservedRV)]
    *_, mean, scale = observed.inputs[0].owner.inputs
    assert scale.eval() == 2.0
    assert mean.eval({theta: [0.5]}) == 4.5
    shared_names = {
        variable.name for variable in fgraph.inputs if isinstance(variable, SharedVariable)
    }
    assert not shared_names & {"sigma", "weights"}


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
    [observed] = [
        node
        for node in fused._rewrite(parameters=theta).apply_nodes
        if isinstance(node.op, ModelObservedRV)
    ]
    *_, scale = observed.inputs[0].owner.inputs
    np.testing.assert_allclose(scale.eval({sigma_value: 3.0}), 3.0)


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
    [observed] = [
        node
        for node in fused._rewrite(parameters=theta).apply_nodes
        if isinstance(node.op, ModelObservedRV)
    ]
    *_, mean, scale = observed.inputs[0].owner.inputs
    np.testing.assert_allclose(mean.eval({theta: [np.log(3.0), 0.5]}), 0.5)
    np.testing.assert_allclose(scale.eval({theta: [np.log(3.0), 0.5]}), 3.0)


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
    nodes = fused._rewrite(parameters=theta).apply_nodes
    assert not any(isinstance(node.op, ModelFreeRV) for node in nodes)
    [observed] = [node for node in nodes if isinstance(node.op, ModelObservedRV)]
    *_, mean, _ = observed.inputs[0].owner.inputs
    # beta = alpha + 2 * sigmoid(theta_beta), so a zero logit is the midpoint.
    np.testing.assert_allclose(mean.eval({theta: [1.0, 0.0]}), 2.0)


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
