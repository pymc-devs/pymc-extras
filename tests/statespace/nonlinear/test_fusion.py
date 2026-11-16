import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

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
        model=model, statespace=level_statespace(shuffled), rv_names=["alpha", "matrix"]
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

    with pytest.raises(ValueError, match=r"\['sigma'\] are not in rv_names"):
        FusedStateSpace(model=model, statespace=level_statespace(["alpha"]), rv_names=["alpha"])


def test_statespace_with_other_observed_states_raises():
    with pm.Model(coords={"group": ["a", "b"]}) as model:
        beta = pm.Normal("beta", dims="group")
        pm.Normal("y", beta.sum(), 1, observed=np.zeros(3))

    with pytest.raises(ValueError, match=r"missing: \['beta\[b\]'\], unexpected: \['delta'\]"):
        FusedStateSpace(
            model=model, statespace=level_statespace(["beta[a]", "delta"]), rv_names=["beta"]
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
        )
