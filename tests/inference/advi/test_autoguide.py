import numpy as np
import pymc as pm
import pytest

from pytensor import function as pytensor_function
from scipy import special

from pymc_extras.inference.advi.autoguide import AutoDiagonalNormal, AutoGuideModel

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
        """Samples from guide should have std ≈ softplus(scale_init)."""
        guide = AutoDiagonalNormal(simple_model)
        x_det = guide.model["x"]

        z_rv = guide.model["x_z"]
        rng = z_rv.owner.inputs[0]
        updates = {rng: z_rv.owner.outputs[0]}

        f = pytensor_function(list(guide.params), x_det, updates=updates)
        samples = np.array(
            [f(*[guide.params_init_values[p] for p in guide.params]) for _ in range(1000)]
        )

        EXPECTED_STD = special.softplus(SCALE_INIT)

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
