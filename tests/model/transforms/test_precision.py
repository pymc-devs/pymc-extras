import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sps

from pymc import Data, Deterministic, HalfNormal, Model, Normal
from pymc.logprob.transforms import Transform
from pytensor.graph import Apply, Op
from pytensor.graph.traversal import ancestors

from pymc_extras.model.transforms.precision import model_to_float32, model_to_float64


class ScaledTransform(Transform):
    """Bakes a float64 constant into its graphs and reads the RV inputs."""

    name = "scaled"
    scale = np.array(2.0, dtype="float64")

    def forward(self, value, *inputs):
        return (value - inputs[2]) * self.scale

    def backward(self, value, *inputs):
        return value / self.scale + inputs[2]

    def log_jac_det(self, value, *inputs):
        return -pt.log(self.scale) * pt.ones_like(value)


class JacobianOnlyTransform(ScaledTransform):
    """Stays float32 in forward/backward, but not in the jacobian."""

    name = "jac_only"

    def forward(self, value, *inputs):
        return value

    backward = forward


class Float64Op(Op):
    def make_node(self, x):
        return Apply(self, [pt.as_tensor(x)], [pt.dscalar()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.float64(inputs[0].sum())


def float64_ancestors(var):
    return [v for v in ancestors([var]) if getattr(v.type, "dtype", None) == "float64"]


def assert_logp_converted(m, m32, rtol=1e-5):
    """Check the float32 logp of `m32` is free of float64 and matches that of `m`."""
    ip = m.initial_point(0)
    with pytensor.config.change_flags(floatX="float32"):
        assert not float64_ancestors(m32.logp())
        ip32 = {k: v.astype("float32") if v.dtype.kind == "f" else v for k, v in ip.items()}
        logp32 = m32.compile_logp()(ip32)
    np.testing.assert_allclose(logp32, m.compile_logp()(ip), rtol=rtol)


class TestModelToFloat32:
    @staticmethod
    def _mixed_model():
        rng = np.random.default_rng(4)
        x_data = rng.normal(size=10)
        with Model(coords={"g": range(3)}) as m:
            x = Data("x", x_data, dims="obs")
            idx = Data("idx", np.arange(10))
            beta = Normal("beta")
            sigma = HalfNormal("sigma")
            z = pm.ZeroSumNormal("z", dims="g")
            det = Deterministic("det", beta * x + z.mean())
            Normal("y", mu=det[idx], sigma=sigma, observed=x_data * 2)
        return m

    def test_dtypes_converted(self):
        m = self._mixed_model()
        m32 = model_to_float32(m)

        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m32[name].type.dtype == "float32", name
        assert m32["idx"].type.dtype == m["idx"].type.dtype
        for rv in m32.free_RVs + m32.observed_RVs:
            assert m32.rvs_to_values[rv].type.dtype == "float32"
        # Static shapes and transforms are preserved
        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m32[name].type.shape == m[name].type.shape, name
        assert type(m32.rvs_to_transforms[m32["z"]]) is type(m.rvs_to_transforms[m["z"]])
        with Model() as m_static:
            pm.ZeroSumNormal("z", shape=(3,))
        assert model_to_float32(m_static)["z"].type.shape == (3,)

        assert {k: v.eval() for k, v in m32.dim_lengths.items()} == {"g": 3, "obs": 10}

    def test_logp_and_draws(self):
        m = self._mixed_model()
        m32 = model_to_float32(m)

        assert_logp_converted(m, m32)
        with pytensor.config.change_flags(floatX="float32"):
            ip32 = m32.initial_point()
            assert all(v.dtype == "float32" for v in ip32.values())
            assert np.asarray(m32.compile_dlogp()(ip32)).dtype == "float32"
        assert pm.draw(m32["z"], random_seed=1).dtype == "float32"

    def test_round_trip(self):
        m = self._mixed_model()
        m64 = model_to_float64(model_to_float32(m))
        for name in ("x", "beta", "sigma", "z", "det", "y"):
            assert m64[name].type.dtype == "float64", name
        np.testing.assert_allclose(
            m64.compile_logp()(m64.initial_point()),
            m.compile_logp()(m.initial_point()),
            rtol=1e-5,
        )

    def test_explicit_cast_redirected(self):
        with Model() as m:
            x = Data("x", np.arange(5))  # int64
            Normal("y", mu=x.astype("float64"), observed=np.zeros(5))
        y = model_to_float32(m)["y"]
        mu, _ = y.owner.op.dist_params(y.owner)
        assert y.type.dtype == mu.type.dtype == "float32"

    def test_integer_data(self):
        with Model() as m:
            slope = Normal("slope")
            det = Deterministic("det", slope * Data("t", np.arange(10)))
            Normal("y", mu=det, observed=np.zeros(10))
        m32 = model_to_float32(m)
        assert m32["t"].type.dtype == m["t"].type.dtype
        assert m32["det"].type.dtype == "float32"
        assert_logp_converted(m, m32)

    def test_sparse_data(self):
        A = sps.random(5, 5, density=0.5, format="csr", random_state=1)
        with Model() as m:
            beta = Normal("beta", shape=(5, 1))
            A_data, A_const = Data("A", A), pytensor.sparse.as_sparse_variable(A)
            A_int = Data("A_int", A.astype("int64")).astype("float64")
            Deterministic("det", pytensor.sparse.structured_dot(A_data + A_const + A_int, beta))
        m32 = model_to_float32(m)
        assert m32["A"].type == m["A"].type.clone(dtype="float32")
        assert m32["det"].type.dtype == "float32"
        assert not float64_ancestors(m32["det"])

    @pytest.mark.parametrize(
        "build",
        [
            lambda: pm.LKJCholeskyCov("x", n=3, eta=2.0, sd_dist=pm.Exponential.dist(1.0)),
            lambda: pm.AR("x", rho=[0.5, 0.2], shape=10, init_dist=Normal.dist(shape=2)),
            lambda: pm.EulerMaruyama(
                "x",
                0.1,
                lambda x, a: (a * x, 1.0),
                (Normal("a"),),
                shape=8,
                init_dist=Normal.dist(),
                initval=np.zeros(8),
            ),
            lambda: pmd.Normal("x", dims="g"),
        ],
        ids=["LKJCholeskyCov", "AR", "EulerMaruyama", "dims"],
    )
    def test_inner_graph_and_core_ops(self, build):
        with Model(coords={"g": range(3)}) as m:
            build()
        m32 = model_to_float32(m)
        assert all(rv.type.dtype == "float32" for rv in m32.basic_RVs + m32.value_vars)
        ip = m.initial_point(0)
        with pytensor.config.change_flags(floatX="float32"):
            logp32 = m32.compile_logp()({k: v.astype("float32") for k, v in ip.items()})
        np.testing.assert_allclose(logp32, m.compile_logp()(ip), rtol=1e-5)
        assert pm.draw(m32["x"]).dtype == "float32"
        m64 = model_to_float64(m32)
        assert all(rv.type.dtype == "float64" for rv in m64.basic_RVs + m64.value_vars)

    def test_unconvertible_op_raises(self):
        with Model() as m:
            Deterministic("det", Normal("x") + Float64Op()(Data("idx", np.arange(3))))
        with pytest.raises(NotImplementedError, match=r"Cannot convert .* graph of det"):
            model_to_float32(m)

    def test_preserves_initvals(self):
        with Model() as m:
            sigma = HalfNormal("sigma", initval=np.array(5.0))
            beta = Normal("beta", initval="prior")
        m32 = model_to_float32(m)
        initval = m32.rvs_to_initial_values[m32["sigma"]]
        assert initval.dtype == "float32" and initval == 5.0
        assert m32.rvs_to_initial_values[m32["beta"]] == "prior"

    @pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie"])
    def test_sample_smoke(self, nuts_sampler):
        pytest.importorskip(nuts_sampler)
        with Model() as m:
            Normal("y", Normal("beta"), HalfNormal("sigma"), observed=np.zeros(5))
        m32 = model_to_float32(m)
        with pytensor.config.change_flags(floatX="float32"):
            with m32:
                idata = pm.sample(
                    draws=10,
                    tune=10,
                    chains=1,
                    nuts_sampler=nuts_sampler,
                    progressbar=False,
                    random_seed=1,
                    compute_convergence_checks=False,
                )
        if nuts_sampler == "pymc":
            assert idata.posterior["beta"].dtype == "float32"
        assert np.isfinite(idata.posterior["beta"]).all()

    @pytest.mark.parametrize("transform", [ScaledTransform(), JacobianOnlyTransform()])
    def test_transform_with_foreign_dtype_constants(self, transform):
        # Transforms travel with the model as objects and may bake float64 constants
        # into value-space graphs; model_to_float32 must keep those graphs float32.
        with Model() as m:
            Normal("x", Normal("mu"), 1, default_transform=transform)

        m32 = model_to_float32(m)
        assert_logp_converted(m, m32)
        with pytensor.config.change_flags(floatX="float32"):
            assert all(v.dtype == "float32" for v in m32.initial_point().values())
        m64 = model_to_float64(m32)
        assert m64.rvs_to_transforms[m64["x"]] is transform
        ip = m.initial_point()
        np.testing.assert_allclose(m64.compile_logp()(ip), m.compile_logp()(ip))
