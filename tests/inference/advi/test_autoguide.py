import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pytensor import function as pytensor_function

from pymc_extras.inference.advi.autoguide import (
    AutoDiagonalNormal,
    AutoGuideModel,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
)
from pymc_extras.inference.advi.compile import compile_sampling_fn
from pymc_extras.inference.advi.objective import get_logp_logq

# TODO: This is a magic number from AutoDiagonalNormal's scale initialization
SCALE_INIT = 0.1


@pytest.fixture
def simple_model():
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
    return model


@pytest.fixture
def multi_rv_model():
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
        pm.Normal("y", 0, 1, shape=(3,))
    return model


class TestAutoDiagonalNormal:
    def test_creates_guide_variables(self):
        with pm.Model() as model:
            pm.Normal("mu", 0, 1)
            pm.Exponential("sigma", 1)

        guide = AutoDiagonalNormal(model)

        assert isinstance(guide, AutoGuideModel)
        expected_vars = {"mu", "sigma", "mu_z", "sigma_z"}
        assert expected_vars <= set(guide.model.named_vars.keys())

    @pytest.mark.parametrize(
        "rv_shapes, expected_param_shapes",
        [
            (
                [(), (3,), (2, 4)],
                {
                    "x_loc": (),
                    "x_scale": (),
                    "y_loc": (3,),
                    "y_scale": (3,),
                    "z_loc": (2, 4),
                    "z_scale": (2, 4),
                },
            ),
        ],
    )
    def test_params_have_correct_shapes(self, rv_shapes, expected_param_shapes):
        with pm.Model() as model:
            for i, (name, shape) in enumerate(zip(["x", "y", "z"], rv_shapes)):
                pm.Normal(name, 0, 1, shape=shape if shape else None)

        guide = AutoDiagonalNormal(model)
        param_shapes = {p.name: v.shape for p, v in guide.params_init_values.items()}

        for param_name, expected_shape in expected_param_shapes.items():
            assert param_shapes[param_name] == expected_shape

    def test_preserves_coords_and_dims(self):
        coords = {"city": ["A", "B", "C"]}
        with pm.Model(coords=coords) as model:
            pm.Normal("mu", 0, 1, dims=["city"])

        guide = AutoDiagonalNormal(model)

        assert tuple(guide.model.coords["city"]) == tuple(coords["city"])
        assert guide.model.named_vars_to_dims["mu"] == ("city",)


class TestTransformedRVs:
    def test_shape_changing_transform(self):
        # https://github.com/pymc-devs/pymc-extras/issues/646
        with pm.Model() as model:
            p = pm.Dirichlet("p", np.ones(3))
            pm.Categorical("obs", p=p, observed=[0, 1, 2])

        guide = AutoDiagonalNormal(model)

        # The guide parameterizes the simplex-transformed value variable, of size n - 1
        param_shapes = {p.name: v.shape for p, v in guide.params_init_values.items()}
        assert param_shapes["p_loc"] == (2,)
        assert param_shapes["p_scale"] == (2,)

        logp, logq = get_logp_logq(model, guide)
        f = pytensor_function(list(guide.params), [logp, logq])
        res = f(*[guide.params_init_values[p] for p in guide.params])
        assert np.all(np.isfinite(res))

    def test_elemwise_transform_preserves_dims(self):
        coords = {"city": ["A", "B", "C"]}
        with pm.Model(coords=coords) as model:
            pm.Exponential("sigma", 1, dims="city")

        guide = AutoDiagonalNormal(model)

        assert guide.model.named_vars_to_dims["sigma"] == ("city",)

    def test_shape_changing_transform_drops_dims(self):
        coords = {"cat": ["a", "b", "c"]}
        with pm.Model(coords=coords) as model:
            pm.Dirichlet("p", np.ones(3), dims="cat")

        guide = AutoDiagonalNormal(model)

        assert "p" not in guide.model.named_vars_to_dims

    def test_sampling_fn_maps_to_constrained_space(self):
        with pm.Model() as model:
            pm.Uniform("u", lower=-1, upper=3, shape=(2,))
            pm.Dirichlet("p", np.ones(3))

        guide = AutoDiagonalNormal(model)
        draws = 100
        f_sample = compile_sampling_fn(model, guide, draws=draws)

        u_draws, p_draws = f_sample(*[guide.params_init_values[p] for p in guide.params])

        assert u_draws.shape == (draws, 2)
        assert p_draws.shape == (draws, 3)
        assert np.all((u_draws > -1) & (u_draws < 3))
        assert np.all(p_draws >= 0)
        np.testing.assert_allclose(p_draws.sum(axis=-1), 1.0)


class TestAutoGuideModel:
    def test_params_returns_all_loc_and_scale(self, multi_rv_model):
        guide = AutoDiagonalNormal(multi_rv_model)

        param_names = {p.name for p in guide.params}
        assert param_names == {"x_loc", "x_scale", "y_loc", "y_scale"}

    def test_getitem_returns_param_by_name(self, simple_model):
        guide = AutoDiagonalNormal(simple_model)

        loc = guide["x_loc"]
        scale = guide["x_scale"]

        assert loc.name == "x_loc"
        assert scale.name == "x_scale"

    def test_stochastic_logq_returns_scalar(self, multi_rv_model):
        guide = AutoDiagonalNormal(multi_rv_model)
        logq = guide.stochastic_logq()

        f = pytensor_function(list(guide.params), logq)
        result = f(*[guide.params_init_values[p] for p in guide.params])

        assert result.shape == ()
        assert np.isfinite(result)


class TestAutoDiagonalNormalSampling:
    def test_samples_have_expected_variance(self, simple_model):
        """Samples from guide should have std ≈ exp(scale_init)."""
        guide = AutoDiagonalNormal(simple_model)
        x_det = guide.model["x"]

        z_rv = guide.model["x_z"]
        rng = z_rv.owner.inputs[0]
        updates = {rng: z_rv.owner.outputs[0]}

        f = pytensor_function(list(guide.params), x_det, updates=updates)
        samples = np.array(
            [f(*[guide.params_init_values[p] for p in guide.params]) for _ in range(1000)]
        )

        EXPECTED_STD = np.exp(SCALE_INIT)

        np.testing.assert_allclose(np.std(samples), EXPECTED_STD, rtol=0.1)

    def test_loc_shifts_output_mean(self, simple_model):
        guide = AutoDiagonalNormal(simple_model)
        x_det = guide.model["x"]

        loc_var, scale_var = guide["x_loc"], guide["x_scale"]
        f = pytensor_function([loc_var, scale_var], x_det)

        init_scale = guide.params_init_values[scale_var]
        val_at_0 = f(np.array(0.0), init_scale)
        val_at_5 = f(np.array(5.0), init_scale)

        np.testing.assert_allclose(val_at_5 - val_at_0, 5.0)

    def test_scale_affects_output_variance(self, simple_model):
        guide = AutoDiagonalNormal(simple_model)
        x_det = guide.model["x"]

        z_rv = guide.model["x_z"]
        rng = z_rv.owner.inputs[0]
        updates = {rng: z_rv.owner.outputs[0]}

        loc_var, scale_var = guide["x_loc"], guide["x_scale"]
        f = pytensor_function([loc_var, scale_var], x_det, updates=updates)

        def sample_std(scale_val, n=500):
            samples = [f(np.array(0.0), np.array(scale_val)) for _ in range(n)]
            return np.std(samples)

        std_small = sample_std(0.1)
        std_large = sample_std(2.0)

        assert std_large > std_small * 2


class TestAutoMultivariateNormal:
    def test_full_rank_guide(self):
        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            pm.Normal("y", 0, 1, shape=(3,))
            pm.Dirichlet("p", np.ones(3))  # transformed: value-space size 2

        guide = AutoMultivariateNormal(model, random_seed=0)

        # D = 1 + 3 + 2 = 6 ; lower-triangular Cholesky packs 6 * 7 / 2 = 21 entries
        assert isinstance(guide, AutoGuideModel)
        assert {p.name for p in guide.params} == {"loc", "L_packed"}
        assert guide["loc"].type.shape == (None,)  # symbolic graph; size fixed by the init value
        assert guide.params_init_values[guide["loc"]].shape == (6,)
        assert guide.params_init_values[guide["L_packed"]].shape == (21,)

        # logq is the joint MvNormal density at the realized draw; reconstruct the effective
        # Cholesky factor for the reference.
        _, logq = get_logp_logq(model, guide, path_derivative_gradient=False)
        diag = pt.arange(6)
        chol = pt.zeros((6, 6))[pt.tril_indices(6)].set(guide["L_packed"])
        chol = chol[diag, diag].set(pt.exp(pt.diagonal(chol)))
        ref = pm.logp(pm.MvNormal.dist(mu=guide["loc"], chol=chol), guide.latent)

        f_logq = pytensor_function(list(guide.params), logq, on_unused_input="ignore")
        f_ref = pytensor_function(list(guide.params), ref, on_unused_input="ignore")
        vals = [guide.params_init_values[p] for p in guide.params]
        np.testing.assert_allclose(f_logq(*vals), f_ref(*vals))

        # assume(L, lower_triangular=True) lowers the MeasurableMatMul solve/slogdet to the
        # triangular fast path (checked on the logq graph alone, not the reference).
        op_names = {type(node.op).__name__ for node in f_logq.maker.fgraph.apply_nodes}
        assert any("Triangular" in name for name in op_names)

        # sampling maps back to the constrained space (the simplex sums to 1)
        f_sample = compile_sampling_fn(model, guide, draws=100)
        *_, p_draws = f_sample(*vals)
        assert p_draws.shape == (100, 3)
        np.testing.assert_allclose(p_draws.sum(axis=-1), 1.0)


class TestAutoLowRankMultivariateNormal:
    def test_low_rank_guide(self):
        with pm.Model() as model:
            pm.Normal("x", 0, 1)
            pm.Normal("y", 0, 1, shape=(3,))
            pm.Dirichlet("p", np.ones(3))  # transformed: value-space size 2

        guide = AutoLowRankMultivariateNormal(model, rank=2, random_seed=0)

        # D = 1 + 3 + 2 = 6, rank = 2
        assert isinstance(guide, AutoGuideModel)
        assert {p.name for p in guide.params} == {"loc", "cov_factor", "cov_diag_unconstrained"}
        assert guide.params_init_values[guide["cov_factor"]].shape == (6, 2)
        assert guide.params_init_values[guide["cov_diag_unconstrained"]].shape == (6,)

        # logq is the closed-form Woodbury density of cov = W W.T + diag(d**2); compare to the
        # dense MvNormal reference, with W perturbed off its zero init so the rank term is live.
        _, logq = get_logp_logq(model, guide, path_derivative_gradient=False)
        W, d_unc, loc = guide["cov_factor"], guide["cov_diag_unconstrained"], guide["loc"]
        cov = W @ W.T + pt.diag(pt.exp(d_unc) ** 2)
        ref = pm.logp(pm.MvNormal.dist(mu=loc, cov=cov), guide.latent)

        values = dict(guide.params_init_values)
        values[W] = np.full((6, 2), 0.3)  # perturb off the zero init
        vals = [values[p] for p in guide.params]
        f = pytensor_function(list(guide.params), [logq, ref], on_unused_input="ignore")
        got, want = f(*vals)
        assert got.shape == ()
        np.testing.assert_allclose(got, want)

        # sampling maps back to the constrained space (the simplex sums to 1)
        f_sample = compile_sampling_fn(model, guide, draws=100)
        *_, p_draws = f_sample(*[guide.params_init_values[p] for p in guide.params])
        assert p_draws.shape == (100, 3)
        np.testing.assert_allclose(p_draws.sum(axis=-1), 1.0)

    def test_default_rank_is_sqrt_d_clamped(self):
        with pm.Model() as model:
            pm.Normal("x", 0, 1, shape=(9,))

        guide = AutoLowRankMultivariateNormal(model)  # D = 9 -> round(sqrt(9)) = 3
        assert guide.params_init_values[guide["cov_factor"]].shape == (9, 3)
