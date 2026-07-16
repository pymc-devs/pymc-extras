"""Marginalization of ``pymc.dims`` models.

The reference for a dims model is always the equivalent plain tensor model: marginalizing it
should produce the same computation, and leave the dims on the model's variables.
"""

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytest
import scipy.special

from pymc.testing import equal_computations_up_to_root
from pytensor.graph import rewrite_graph
from pytensor.graph.replace import graph_replace
from pytensor.printing import debugprint
from pytensor.xtensor import as_xtensor

from pymc_extras.marginal import conditional, marginalize, unmarginalize


def assert_equivalent_logp_graph(model, reference_model):
    """Assert a dims model's logp graph matches that of an equivalent tensor model.

    Mirrors ``tests/dims/utils.py`` in pymc, which is not importable from here.
    """
    replacements = {
        var: as_xtensor(var.values.clone(name=var.name), dims=var.dims) for var in model.value_vars
    }
    lowered = rewrite_graph(
        [graph_replace(model.logp(), replacements)],
        include=("lower_xtensor", "canonicalize", "local_remove_all_assert"),
    )
    reference_lowered = rewrite_graph(
        [reference_model.logp()],
        include=("canonicalize", "local_remove_all_assert"),
    )
    assert equal_computations_up_to_root(lowered, reference_lowered, ignore_rng_values=False), (
        debugprint(lowered + reference_lowered, print_type=True)
    )


COORDS = {"trial": range(3), "obs": range(5), "cat": range(2)}
P = np.array([0.3, 0.7])


def test_marginalize_finite_discrete():
    with pm.Model(coords=COORDS) as m:
        idx = pmd.Categorical(
            "idx", p=pmd.as_xtensor(P, dims=("cat",)), core_dims="cat", dims=("trial",)
        )
        pmd.Normal("y", mu=idx * 2.0, sigma=1.0, dims=("obs", "trial"))

    with pm.Model(coords=COORDS) as ref:
        ref_idx = pm.Categorical("idx", p=P, dims=("trial",))
        pm.Normal("y", mu=ref_idx * 2.0, sigma=1.0, dims=("obs", "trial"))

    marginal_m = marginalize(m, ["idx"])
    assert_equivalent_logp_graph(marginal_m, marginalize(ref, ["idx"]))

    # The model keeps its dims, on the RVs and on their values
    [y] = marginal_m.free_RVs
    assert y.type.dims == ("obs", "trial")
    assert marginal_m.value_vars[0].type.dims == ("obs", "trial")


def test_marginalize_multiple_dependents():
    """The joint logp of several dependents cannot be split across them.

    Deriving it per dependent silently computes the density of a subset, which is why this
    asserts the joint against a reference rather than only that a logp comes out.
    """
    with pm.Model(coords=COORDS) as m:
        idx = pmd.Categorical(
            "idx", p=pmd.as_xtensor(P, dims=("cat",)), core_dims="cat", dims=("trial",)
        )
        pmd.Normal("y1", mu=idx * 2.0, sigma=1.0, dims=("obs", "trial"))
        pmd.Normal("y2", mu=idx * 3.0, sigma=1.0, dims=("trial",))

    with pm.Model(coords=COORDS) as ref:
        ref_idx = pm.Categorical("idx", p=P, dims=("trial",))
        pm.Normal("y1", mu=ref_idx * 2.0, sigma=1.0, dims=("obs", "trial"))
        pm.Normal("y2", mu=ref_idx * 3.0, sigma=1.0, dims=("trial",))

    point = {"y1": np.zeros((5, 3)), "y2": np.zeros(3)}
    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        got = marginalize(m, ["idx"]).compile_logp()(point)
        expected = marginalize(ref, ["idx"]).compile_logp()(point)
    np.testing.assert_allclose(got, expected)


def test_marginalize_normal_normal():
    with pm.Model(coords=COORDS) as m:
        x = pmd.Normal("x", mu=0.0, sigma=1.0, dims=("trial",))
        pmd.Normal("y", mu=x + 1.0, sigma=2.0, dims=("trial",))

    with pm.Model(coords=COORDS) as ref:
        ref_x = pm.Normal("x", mu=0.0, sigma=1.0, dims=("trial",))
        pm.Normal("y", mu=ref_x + 1.0, sigma=2.0, dims=("trial",))

    assert_equivalent_logp_graph(marginalize(m, ["x"]), marginalize(ref, ["x"]))


def test_unmarginalize_roundtrip():
    def build():
        with pm.Model(coords=COORDS) as m:
            idx = pmd.Categorical(
                "idx", p=pmd.as_xtensor(P, dims=("cat",)), core_dims="cat", dims=("trial",)
            )
            pmd.Normal("y", mu=idx * 2.0, sigma=1.0, dims=("obs", "trial"))
        return m

    recovered = unmarginalize(marginalize(build(), ["idx"]))

    # The recovered variable comes back with its dims, not as a bare tensor
    assert recovered["idx"].type.dims == ("trial",)

    # ...and the model is the one we started with: the dependents must be rewired onto the
    # recovered variable, not left drawing their own copy of it
    point = {"idx": np.array([0, 1, 0]), "y": np.zeros((5, 3))}
    np.testing.assert_allclose(recovered.compile_logp()(point), build().compile_logp()(point))


def test_conditional():
    with pm.Model(coords=COORDS) as m:
        idx = pmd.Categorical(
            "idx", p=pmd.as_xtensor(P, dims=("cat",)), core_dims="cat", dims=("trial",)
        )
        pmd.Normal("y", mu=idx * 2.0, sigma=1.0, dims=("obs", "trial"))

    cond_m = conditional(marginalize(m, ["idx"]))
    assert cond_m["idx"].type.dims == ("trial",)

    # p(idx | y) per trial, against the same enumeration done by hand
    y_val = np.tile(np.array([0.0, 2.0, 1.0]), (5, 1))
    logp_idx = cond_m.compile_logp(vars=[cond_m["idx"]], sum=False)
    for k in (0, 1):
        [got] = logp_idx({"idx": np.full(3, k), "y": y_val})
        unnormalized = np.array(
            [
                np.log(w) - 0.5 * ((y_val - 2.0 * j) ** 2).sum(0) - 5 * 0.5 * np.log(2 * np.pi)
                for j, w in enumerate(P)
            ]
        )
        expected = unnormalized[k] - scipy.special.logsumexp(unnormalized, axis=0)
        np.testing.assert_allclose(np.asarray(got).ravel(), expected, atol=1e-8)
