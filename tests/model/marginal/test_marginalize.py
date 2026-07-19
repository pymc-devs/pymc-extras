import itertools

from contextlib import suppress as does_not_warn

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc import Model, draw
from pymc.distributions import transforms
from pymc.initial_point import make_initial_point_expression
from pymc.pytensorf import constant_fold, inputvars
from pymc.util import UNSET
from scipy.special import logsumexp

from pymc_extras.marginal import marginalize, unmarginalize
from pymc_extras.model.marginal.distributions.core import MarginalRV
from pymc_extras.utils.model_equivalence import equal_computations_up_to_root, equivalent_models

# FIXME: A Blockwise of Reshape should be rewritten into a Reshape, as it's rather inneficient
# This shows up in `test_one_to_many_unaligned_marginalized_rvs`
pytestmark = pytest.mark.filterwarnings(
    "ignore:Numba will use object mode to run Blockwise{Reshapep*:UserWarning"
)


def test_basic_marginalized_rv():
    data = [2] * 5

    with Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Categorical("idx", p=[0.1, 0.3, 0.6])
        mu = pt.switch(
            pt.eq(idx, 0),
            -1.0,
            pt.switch(
                pt.eq(idx, 1),
                0.0,
                1.0,
            ),
        )
        y = pm.Normal("y", mu=mu, sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    marginal_m = marginalize(m, [idx])
    assert isinstance(marginal_m["y"].owner.op, MarginalRV)
    assert ["idx"] not in [rv.name for rv in marginal_m.free_RVs]

    # Test forward draws
    y_draws, z_draws = draw(
        [marginal_m["y"], marginal_m["z"]],
        # Make sigma very small to make draws deterministic
        givens={marginal_m["sigma"]: 0.001},
        draws=1000,
        random_seed=54,
    )
    assert sorted(np.unique(y_draws.round()) == [-1.0, 0.0, 1.0])
    assert z_draws[y_draws < 0].mean() < z_draws[y_draws > 0].mean()

    # Test initial_point
    ips = make_initial_point_expression(
        # Use basic_RVs to include the observed RV
        free_rvs=marginal_m.basic_RVs,
        rvs_to_transforms=marginal_m.rvs_to_transforms,
        initval_strategies={},
    )
    # After simplification, we should have only constants in the graph (expect alloc which isn't constant folded):
    ip_sigma, ip_y, ip_z = constant_fold(ips)
    np.testing.assert_allclose(ip_sigma, 1.0)
    np.testing.assert_allclose(ip_y, 1.0)
    np.testing.assert_allclose(ip_z, np.full((5,), 1.0))

    marginal_ip = marginal_m.initial_point()
    expected_ip = m.initial_point()
    expected_ip.pop("idx")
    assert marginal_ip == expected_ip

    # Test logp
    with pm.Model() as ref_m:
        sigma = pm.HalfNormal("sigma")
        y = pm.NormalMixture("y", w=[0.1, 0.3, 0.6], mu=[-1, 0, 1], sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    np.testing.assert_almost_equal(
        marginal_m.compile_logp()(marginal_ip),
        ref_m.compile_logp()(marginal_ip),
    )
    np.testing.assert_almost_equal(
        marginal_m.compile_dlogp([marginal_m["y"]])(marginal_ip),
        ref_m.compile_dlogp([ref_m["y"]])(marginal_ip),
    )


def test_one_to_one_marginalized_rvs():
    """Test case with multiple, independent marginalized RVs."""
    with Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx1 = pm.Bernoulli("idx1", p=0.75)
        x = pm.Normal("x", mu=idx1, sigma=sigma)
        idx2 = pm.Bernoulli("idx2", p=0.75, shape=(5,))
        y = pm.Normal("y", mu=(idx2 * 2 - 1), sigma=sigma, shape=(5,))

    marginal_m = marginalize(m, [idx1, idx2])
    assert isinstance(marginal_m["x"].owner.op, MarginalRV)
    assert isinstance(marginal_m["y"].owner.op, MarginalRV)
    assert marginal_m["x"].owner is not marginal_m["y"].owner

    with pm.Model() as ref_m:
        sigma = pm.HalfNormal("sigma")
        x = pm.NormalMixture("x", w=[0.25, 0.75], mu=[0, 1], sigma=sigma)
        y = pm.NormalMixture("y", w=[0.25, 0.75], mu=[-1, 1], sigma=sigma, shape=(5,))

    # Test logp
    test_point = ref_m.initial_point()
    x_logp, y_logp = marginal_m.compile_logp(vars=[marginal_m["x"], marginal_m["y"]], sum=False)(
        test_point
    )
    x_ref_log, y_ref_logp = ref_m.compile_logp(vars=[ref_m["x"], ref_m["y"]], sum=False)(test_point)
    np.testing.assert_array_almost_equal(x_logp, x_ref_log.sum())
    np.testing.assert_array_almost_equal(y_logp, y_ref_logp)


def test_one_to_many_marginalized_rvs():
    """Test that marginalization works when there is more than one dependent RV"""
    with Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.75)
        x = pm.Normal("x", mu=idx, sigma=sigma)
        y = pm.Normal("y", mu=(idx * 2 - 1), sigma=sigma, shape=(5,))

    marginal_m = marginalize(m, [idx])

    marginal_x = marginal_m["x"]
    marginal_y = marginal_m["y"]
    assert isinstance(marginal_x.owner.op, MarginalRV)
    assert isinstance(marginal_y.owner.op, MarginalRV)
    assert marginal_x.owner is marginal_y.owner

    ref_logp_x_y_fn = m.compile_logp([idx, x, y])
    tp = marginal_m.initial_point()
    ref_logp_x_y = logsumexp([ref_logp_x_y_fn({**tp, **{"idx": idx}}) for idx in (0, 1)])
    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        logp_x_y = marginal_m.compile_logp([marginal_x, marginal_y])(tp)
    np.testing.assert_array_almost_equal(logp_x_y, ref_logp_x_y)


def test_one_to_many_unaligned_marginalized_rvs():
    """Test that marginalization works when there is more than one dependent RV with batch dimensions that are not aligned"""

    def build_model(build_batched: bool):
        with Model() as m:
            if build_batched:
                idx = pm.Bernoulli("idx", p=[0.75, 0.4], shape=(3, 2))
            else:
                idxs = [pm.Bernoulli(f"idx_{i}", p=(0.75 if i % 2 == 0 else 0.4)) for i in range(6)]
                idx = pt.stack(idxs, axis=0).reshape((3, 2))

            x = pm.Normal("x", mu=idx.T[:, :, None], shape=(2, 3, 1))
            y = pm.Normal("y", mu=(idx * 2 - 1), shape=(1, 3, 2))

        return m

    marginal_m = marginalize(build_model(build_batched=True), ["idx"])
    ref_m = marginalize(build_model(build_batched=False), [f"idx_{i}" for i in range(6)])

    test_point = marginal_m.initial_point()

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        np.testing.assert_allclose(
            marginal_m.compile_logp()(test_point),
            ref_m.compile_logp()(test_point),
        )


def test_many_to_one_marginalized_rvs():
    """Test when random variables depend on multiple marginalized variables"""
    with Model() as m:
        x = pm.Bernoulli("x", 0.1)
        y = pm.Bernoulli("y", 0.3)
        z = pm.DiracDelta("z", c=x + y)

    logp_fn = marginalize(m, [x, y]).compile_logp()

    np.testing.assert_allclose(np.exp(logp_fn({"z": 0})), 0.9 * 0.7)
    np.testing.assert_allclose(np.exp(logp_fn({"z": 1})), 0.9 * 0.3 + 0.1 * 0.7)
    np.testing.assert_allclose(np.exp(logp_fn({"z": 2})), 0.1 * 0.3)


@pytest.mark.parametrize("batched", (False, "left", "right"))
def test_nested_marginalized_rvs(batched):
    """Test that marginalization works when there are nested marginalized RVs"""

    def build_model(build_batched: bool) -> Model:
        idx_shape = (3,) if build_batched else ()
        sub_idx_shape = (5,) if not build_batched else (5, 3) if batched == "left" else (3, 5)

        with Model() as m:
            sigma = pm.HalfNormal("sigma")

            idx = pm.Bernoulli("idx", p=0.75, shape=idx_shape)
            dep = pm.Normal("dep", mu=pt.switch(pt.eq(idx, 0), -1000.0, 1000.0), sigma=sigma)

            sub_idx_p = pt.switch(pt.eq(idx, 0), 0.15, 0.95)
            if build_batched and batched == "right":
                sub_idx_p = sub_idx_p[..., None]
                dep = dep[..., None]
            sub_idx = pm.Bernoulli("sub_idx", p=sub_idx_p, shape=sub_idx_shape)
            sub_dep = pm.Normal("sub_dep", mu=dep + sub_idx * 100, sigma=sigma)

        return m

    marginal_m = marginalize(build_model(build_batched=batched), ["idx", "sub_idx"])
    assert all(rv.name not in ("idx", "sub_idx") for rv in marginal_m.free_RVs)

    # Test forward draws and initial_point, shouldn't depend on batching, so we only test one case
    if not batched:
        # Test forward draws
        dep_draws, sub_dep_draws = draw(
            [marginal_m["dep"], marginal_m["sub_dep"]],
            # Make sigma very small to make draws deterministic
            givens={marginal_m["sigma"]: 0.001},
            draws=1000,
            random_seed=214,
        )
        assert sorted(np.unique(dep_draws.round()) == [-1000.0, 1000.0])
        assert sorted(np.unique(sub_dep_draws.round()) == [-1000.0, -900.0, 1000.0, 1100.0])

        # Test initial_point
        ips = make_initial_point_expression(
            free_rvs=[marginal_m["sigma"], marginal_m["dep"], marginal_m["sub_dep"]],
            rvs_to_transforms=marginal_m.rvs_to_transforms,
            initval_strategies={},
        )
        # After simplification, we should have only constants in the graph
        ip_sigma, ip_dep, ip_sub_dep = constant_fold(ips)
        np.testing.assert_allclose(ip_sigma, 1.0)
        np.testing.assert_allclose(ip_dep, 1000.0)
        np.testing.assert_allclose(ip_sub_dep, np.full((5,), 1100.0))

    # Test logp
    ref_m = build_model(build_batched=False)
    ref_logp_fn = ref_m.compile_logp(
        vars=[ref_m["idx"], ref_m["dep"], ref_m["sub_idx"], ref_m["sub_dep"]]
    )

    test_point = ref_m.initial_point()
    test_point["dep"] = np.full_like(test_point["dep"], 1000)
    test_point["sub_dep"] = np.full_like(test_point["sub_dep"], 1000 + 100)
    ref_logp = logsumexp(
        [
            ref_logp_fn({**test_point, **{"idx": idx, "sub_idx": np.array(sub_idxs)}})
            for idx in (0, 1)
            for sub_idxs in itertools.product((0, 1), repeat=5)
        ]
    )
    if batched:
        ref_logp *= 3

    test_point = marginal_m.initial_point()
    test_point["dep"] = np.full_like(test_point["dep"], 1000)
    test_point["sub_dep"] = np.full_like(test_point["sub_dep"], 1000 + 100)

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        logp = marginal_m.compile_logp(vars=[marginal_m["dep"], marginal_m["sub_dep"]])(test_point)

    np.testing.assert_almost_equal(logp, ref_logp)


def test_sequential_marginalization():
    """Test that sequential marginalization is equivalent to joint marginalization."""

    def build_model():
        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
            x = pm.Normal("x", mu=(idx + sub_idx) - 1)
        return m

    joint_m = marginalize(build_model(), ["idx", "sub_idx"])

    # idx first: sub_idx becomes a dependent of idx's marginalization
    seq_idx_first = marginalize(marginalize(build_model(), "idx"), "sub_idx")
    assert equivalent_models(seq_idx_first, joint_m)

    # sub_idx first: idx remains a plain free RV (sub_idx depends on idx, not vice versa)
    seq_sub_first = marginalize(marginalize(build_model(), "sub_idx"), "idx")
    assert equivalent_models(seq_sub_first, joint_m)


def test_three_level_nested_marginalization():
    """Chained marginalized variables three levels deep, joint and sequential."""
    from pymc.logprob.basic import _find_unallowed_rvs_in_graph

    def build_model():
        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.6)
            sub = pm.Bernoulli("sub", p=pt.as_tensor([0.3, 0.7])[idx])
            subsub = pm.Bernoulli("subsub", p=pt.as_tensor([0.2, 0.9])[sub])
            pm.Normal("y", mu=subsub * 2.0, sigma=1.0)
        return m

    m = build_model()
    ref_fn = m.compile_logp()
    point = {"y": 0.7}
    ref = logsumexp(
        [
            ref_fn({"idx": i, "sub": s, "subsub": ss, **point})
            for i, s, ss in itertools.product((0, 1), repeat=3)
        ]
    )

    joint_m = marginalize(build_model(), ["idx", "sub", "subsub"])
    assert not _find_unallowed_rvs_in_graph([joint_m.logp()])
    np.testing.assert_allclose(joint_m.compile_logp()(joint_m.initial_point() | point), ref)

    seq_m = marginalize(marginalize(marginalize(build_model(), "idx"), "sub"), "subsub")
    np.testing.assert_allclose(seq_m.compile_logp()(seq_m.initial_point() | point), ref)


def test_sequential_marginalization_outside_dependent():
    """Sequential marginalization where the second target's dependent was not absorbed.

    y depends only on sub_idx, so the first marginalize call absorbs sub_idx
    but not y. The second call must rebuild y's graph over the inlined variables.
    """

    def build_model():
        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
            y = pm.Normal("y", mu=sub_idx)
        return m

    joint_m = marginalize(build_model(), ["idx", "sub_idx"])
    seq_m = marginalize(marginalize(build_model(), "idx"), "sub_idx")

    test_point = joint_m.initial_point()
    np.testing.assert_allclose(
        seq_m.compile_logp()(test_point),
        joint_m.compile_logp()(test_point),
    )


def test_interdependent_rvs():
    """Test Marginalization when dependent RVs are interdependent."""
    with Model() as m:
        idx = pm.Bernoulli("idx", p=0.75)
        x = pm.Normal("x", mu=idx * 2, sigma=1e-3)
        # Y depends on both x and idx
        y = pm.Normal("y", mu=x * idx * 2, sigma=1e-3)

    marginal_m = marginalize(m, "idx")

    marginal_x = marginal_m["x"]
    marginal_y = marginal_m["y"]
    assert isinstance(marginal_x.owner.op, MarginalRV)
    assert isinstance(marginal_y.owner.op, MarginalRV)
    assert marginal_x.owner is marginal_y.owner

    # Test forward draws
    x_draws, y_draws = draw([marginal_x, marginal_y], draws=1000, random_seed=54)
    assert sorted(np.unique(x_draws.round())) == [0, 2]
    assert sorted(np.unique(y_draws.round())) == [0, 4]
    assert np.unique(y_draws[x_draws < 1].round()) == [0]
    assert np.unique(y_draws[x_draws > 1].round()) == [4]

    # Test initial_point
    ips = make_initial_point_expression(
        free_rvs=[marginal_m["x"], marginal_m["y"]],
        rvs_to_transforms={},
        initval_strategies={},
    )
    # After simplification, we should have only constants in the graph
    ip_x, ip_y = constant_fold(ips)
    np.testing.assert_allclose(ip_x, 2.0)
    np.testing.assert_allclose(ip_y, 4.0)

    # Test custom initval strategy
    ips = make_initial_point_expression(
        # Test that order does not matter
        free_rvs=[marginal_m["y"], marginal_m["x"]],
        rvs_to_transforms={},
        initval_strategies={marginal_x: pt.constant(5.0)},
    )
    ip_y, ip_x = constant_fold(ips)
    np.testing.assert_allclose(ip_x, 5.0)
    np.testing.assert_allclose(ip_y, 10.0)

    # Test logp
    test_point = marginal_m.initial_point()
    ref_logp_fn = m.compile_logp([m["idx"], m["x"], m["y"]])
    ref_logp = logsumexp([ref_logp_fn({**test_point, **{"idx": idx}}) for idx in (0, 1)])
    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        logp = marginal_m.compile_logp([marginal_m["x"], marginal_m["y"]])(test_point)
    np.testing.assert_almost_equal(logp, ref_logp)


@pytest.mark.parametrize("advanced_indexing", (False, True))
def test_marginalized_index_as_key(advanced_indexing):
    """Test we can marginalize graphs where indexing is used as a mapping."""

    w = [0.1, 0.3, 0.6]
    mu = pt.as_tensor([-1, 0, 1])

    if advanced_indexing:
        y_val = pt.as_tensor([[-1, -1], [0, 1]])
        shape = (2, 2)
    else:
        y_val = -1
        shape = ()

    with Model() as m:
        x = pm.Categorical("x", p=w, shape=shape)
        y = pm.Normal("y", mu[x].T, sigma=1, observed=y_val)

    marginal_m = marginalize(m, x)

    marginal_logp = marginal_m.compile_logp(sum=False)({})[0]
    ref_logp = pm.logp(pm.NormalMixture.dist(w=w, mu=mu.T, sigma=1, shape=shape), y_val).eval()

    np.testing.assert_allclose(marginal_logp, ref_logp)


def test_marginalized_index_as_value_and_key():
    """Test we can marginalize graphs were marginalized_rv is indexed."""

    def build_model(build_batched: bool) -> Model:
        with Model() as m:
            if build_batched:
                latent_state = pm.Bernoulli("latent_state", p=0.3, size=(4,))
            else:
                latent_state = pm.math.stack(
                    [pm.Bernoulli(f"latent_state_{i}", p=0.3) for i in range(4)]
                )
            # latent state is used as the indexed variable
            latent_intensities = pt.where(latent_state[:, None], [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
            picked_intensity = pm.Categorical("picked_intensity", p=[0.2, 0.2, 0.6])
            # picked intensity is used as the indexing variable
            pm.Normal(
                "intensity",
                mu=latent_intensities[:, picked_intensity],
                observed=[0.5, 1.5, 5.0, 15.0],
            )
        return m

    # We compare with the equivalent but less efficient batched model
    m = build_model(build_batched=True)
    ref_m = build_model(build_batched=False)

    m = marginalize(m, ["latent_state"])
    ref_m = marginalize(ref_m, [f"latent_state_{i}" for i in range(4)])
    test_point = {"picked_intensity": 1}
    np.testing.assert_allclose(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )

    m = marginalize(m, ["picked_intensity"])
    ref_m = marginalize(ref_m, ["picked_intensity"])
    test_point = {}
    np.testing.assert_allclose(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )


class TestNotSupportedMixedDims:
    """Test lack of support for models where batch dims of marginalized variables are mixed."""

    def test_mixed_dims_via_transposed_dot(self):
        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx @ idx.T)

        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

    def test_mixed_dims_via_indexing(self):
        mean = pt.as_tensor([[0.1, 0.9], [0.6, 0.4]])

        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=mean[idx, :] + mean[:, idx])
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=mean[idx, None] + mean[None, idx])
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            mu = pt.specify_broadcastable(mean[:, None][idx], 1) + pt.specify_broadcastable(
                mean[None, :][:, idx], 0
            )
            y = pm.Normal("y", mu=mu)
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx[0] + idx[1])
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

    def test_mixed_dims_via_vector_indexing(self):
        with Model() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx[[0, 1, 0, 0]])
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

        with Model() as m:
            idx = pm.Categorical("key", p=[0.1, 0.3, 0.6], shape=(2, 2))
            y = pm.Normal("y", pt.as_tensor([[0, 1], [2, 3]])[idx.astype(bool)])
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, idx)

    def test_mixed_dims_via_support_dimension(self):
        with Model() as m:
            x = pm.Bernoulli("x", p=0.7, shape=3)
            y = pm.Dirichlet("y", a=x * 10 + 1)
        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, x)

    def test_mixed_dims_via_nested_marginalization(self):
        with Model() as m:
            x = pm.Bernoulli("x", p=0.7, shape=(3,))
            y = pm.Bernoulli("y", p=0.7, shape=(2,))
            z = pm.Normal("z", mu=pt.add.outer(x, y), shape=(3, 2))

        with pytest.raises((ValueError, NotImplementedError)):
            marginalize(m, [x, y])


def test_marginalized_deterministic_and_potential():
    rng = np.random.default_rng(299)

    with Model() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        z = pm.Normal("z", x)
        det = pm.Deterministic("det", y + z)
        pot = pm.Potential("pot", y + z + 1)

    marginal_m = marginalize(m, [x])

    y_draw, z_draw, det_draw, pot_draw = pm.draw([y, z, det, pot], draws=5, random_seed=rng)
    np.testing.assert_almost_equal(y_draw + z_draw, det_draw)
    np.testing.assert_almost_equal(det_draw, pot_draw - 1)

    y_value = marginal_m.rvs_to_values[marginal_m["y"]]
    z_value = marginal_m.rvs_to_values[marginal_m["z"]]
    det_value, pot_value = marginal_m.replace_rvs_by_values([marginal_m["det"], marginal_m["pot"]])
    assert set(inputvars([det_value, pot_value])) == {y_value, z_value}
    assert det_value.eval({y_value: 2, z_value: 5}) == 7
    assert pot_value.eval({y_value: 2, z_value: 5}) == 8


def test_not_supported_marginalized_deterministic_and_potential():
    with Model() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        det = pm.Deterministic("det", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Deterministic det"
    ):
        marginalize(m, [x])

    with Model() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        pot = pm.Potential("pot", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Potential pot"
    ):
        marginalize(m, [x])


@pytest.mark.parametrize(
    "transform, expected_warning",
    (
        (None, does_not_warn()),
        (UNSET, does_not_warn()),
        (transforms.log, does_not_warn()),
        (transforms.Chain([transforms.logodds, transforms.log]), does_not_warn()),
        (
            transforms.Interval(0, 2),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
        (
            transforms.Chain([transforms.log, transforms.Interval(-1, 1)]),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
    ),
)
def test_marginalized_transforms(transform, expected_warning):
    w = [0.1, 0.3, 0.6]
    data = [0, 5, 10]
    initval = 0.7  # Value that will be negative on the unconstrained space

    with pm.Model() as m_ref:
        sigma = pm.Mixture(
            "sigma",
            w=w,
            comp_dists=pm.HalfNormal.dist([1, 2, 3]),
            initval=initval,
            default_transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with Model() as m:
        idx = pm.Categorical("idx", p=w)
        sigma = pm.HalfNormal(
            "sigma",
            pt.switch(
                pt.eq(idx, 0),
                1,
                pt.switch(
                    pt.eq(idx, 1),
                    2,
                    3,
                ),
            ),
            default_transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with expected_warning:
        marginal_m = marginalize(m, [idx])

    marginal_m.set_initval(marginal_m["sigma"], initval)
    ip = marginal_m.initial_point()
    if transform is not None:
        if transform is UNSET:
            transform_name = "log"
        else:
            transform_name = transform.name
        assert -np.inf < ip[f"sigma_{transform_name}__"] < 0.0
    np.testing.assert_allclose(marginal_m.compile_logp()(ip), m_ref.compile_logp()(ip))


def test_data_container():
    """Test that MarginalModel can handle Data containers."""
    with Model(coords={"obs": [0]}) as m:
        x = pm.Data("x", 2.5)
        idx = pm.Bernoulli("idx", p=0.7, dims="obs")
        y = pm.Normal("y", idx * x, dims="obs")

    marginal_m = marginalize(m, [idx])

    logp_fn = marginal_m.compile_logp()

    with pm.Model(coords={"obs": [0]}) as m_ref:
        x = pm.Data("x", 2.5)
        y = pm.NormalMixture("y", w=[0.3, 0.7], mu=[0, x], dims="obs")

    ref_logp_fn = m_ref.compile_logp()

    for i, x_val in enumerate((-1.5, 2.5, 3.5), start=1):
        for m in (marginal_m, m_ref):
            m.set_dim("obs", new_length=i, coord_values=tuple(range(i)))
            pm.set_data({"x": x_val}, model=m)

        ip = marginal_m.initial_point()
        np.testing.assert_allclose(logp_fn(ip), ref_logp_fn(ip))


def test_unmarginalize():
    with pm.Model() as m:
        idx = pm.Bernoulli("idx", p=0.5)
        sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
        x = pm.Normal("x", mu=(idx + sub_idx) - 1)

    marginal_m = marginalize(m, [idx, sub_idx])
    assert not equivalent_models(marginal_m, m)

    unmarginal_m = unmarginalize(marginal_m)
    assert equivalent_models(unmarginal_m, m)

    unmarginal_idx_explicit = unmarginalize(marginal_m, ("idx", "sub_idx"))
    assert equivalent_models(unmarginal_idx_explicit, m)

    # Test partial unmarginalize
    unmarginal_idx = unmarginalize(marginal_m, "idx")
    assert equivalent_models(unmarginal_idx, marginalize(m, "sub_idx"))

    unmarginal_sub_idx = unmarginalize(marginal_m, "sub_idx")
    assert equivalent_models(unmarginal_sub_idx, marginalize(m, "idx"))


def test_forward_after_sampling():
    # Regression test where forward graph was modified during sampling
    # Specifically `model.initial_point`, but we test the whole pipeline

    with pm.Model() as m:
        p_outlier = pm.Beta("p_outlier", 1, 1)
        is_outlier = pm.Bernoulli("is_outlier", p=p_outlier, shape=(10,))
        sigma = pm.Exponential("sigma", 1, shape=(2,))
        pm.Normal("y_hat", mu=0, sigma=sigma[is_outlier])

    marginalized_mod = marginalize(m, [is_outlier])

    # Check that model.initial_point() does not modify the inner graph of the marginalization Op
    marginal_rv = marginalized_mod["y_hat"]
    inner_outputs_before = marginal_rv.owner.op.fgraph.unfreeze().outputs
    marginalized_mod.initial_point()
    inner_outputs_after = marginal_rv.owner.op.fgraph.outputs
    assert equal_computations_up_to_root(inner_outputs_before, inner_outputs_after)

    assert pm.draw(marginalized_mod["y_hat"]).shape == (10,)

    with marginalized_mod:
        pm.sample(tune=1, chains=1, draws=1, compute_convergence_checks=False)

    assert pm.draw(marginalized_mod["y_hat"]).shape == (10,)
