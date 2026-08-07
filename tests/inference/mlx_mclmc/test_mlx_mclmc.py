import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

mx = pytest.importorskip("mlx.core", reason="MCLMC requires mlx, which needs Apple Silicon")

from pymc_extras.inference.mlx_mclmc import fit_mlx_mclmc
from pymc_extras.inference.mlx_mclmc.kernel import (
    TunedParameters,
    _optimize_to_mode,
    sample,
    tune_step_size,
    warmup,
    warmup_and_sample,
)
from pymc_extras.inference.mlx_mclmc.logp import (
    MLXLogp,
    check_model_is_sampleable,
    draws_to_datasets,
)
from pymc_extras.inference.mlx_mclmc.mlx_mclmc import _warn_if_adaptation_failed


@pytest.fixture
def float32():
    with pytensor.config.change_flags(floatX="float32"):
        yield


@pytest.fixture
def conjugate_model(float32):
    """A normal-normal model whose posterior for ``mu`` is available in closed form."""
    prior_sd, sigma, n_obs, dim = 2.0, 0.7, 40, 3
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_obs, dim)) * sigma

    coords = {"group": ["a", "b", "c"], "obs": range(n_obs)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", 0.0, prior_sd, dims="group")
        pm.Deterministic("mu_sum", mu.sum())
        pm.Normal("y", mu, sigma, observed=data, dims=("obs", "group"))

    precision = 1.0 / prior_sd**2 + n_obs / sigma**2
    posterior_mean = (data.sum(axis=0) / sigma**2) / precision
    posterior_sd = np.sqrt(1.0 / precision)

    return model, posterior_mean, posterior_sd


def test_kernel_recovers_correlated_gaussian():
    dim = 6
    scales = mx.exp(mx.linspace(-1.0, 1.0, dim))
    factor = mx.random.normal(shape=(dim, dim), key=mx.random.key(7)) * 0.4
    covariance = (scales[:, None] * (factor @ factor.T + mx.eye(dim))) * scales[None, :]
    precision = mx.linalg.inv(covariance, stream=mx.cpu)
    mx.eval(covariance, precision)

    def logdensity_fn(x):
        return -0.5 * mx.sum(x * (precision @ x))

    output, tuned = warmup_and_sample(
        logdensity_fn, mx.zeros((dim,)), num_tune=2000, draws=4000, chains=4, seed=0
    )
    draws = np.asarray(output.samples).reshape(-1, dim)
    true_sd = np.sqrt(np.diag(np.asarray(covariance)))

    assert tuned.step_size > 0
    assert np.isfinite(np.asarray(output.energy_errors)).all()
    assert not np.asarray(output.diverging).any()
    np.testing.assert_allclose(draws.std(axis=0), true_sd, rtol=0.1)
    np.testing.assert_array_less(np.abs(draws.mean(axis=0)) / true_sd, 0.15)


def test_sample_rejects_discarding_every_draw():
    with pytest.raises(ValueError, match="leaves no draws"):
        sample(mx.sum, mx.zeros((1, 2)), L=1.0, step_size=0.1, n_steps=10, discard=10)


def test_sample_rejects_zero_decoherence_scale():
    """L = 0 would silently become an infinite refresh rate via 1 / L."""
    with pytest.raises(ValueError, match="L must be non-zero"):
        sample(mx.sum, mx.zeros((1, 2)), L=0.0, step_size=0.1, n_steps=10)


def test_logp_matches_pymc(conjugate_model):
    model, *_ = conjugate_model
    logdensity_fn = MLXLogp(model)
    point = mx.array(np.array([0.3, -1.2, 0.8], dtype="float32"))

    expected = model.compile_logp()({"mu": np.asarray(point)})

    np.testing.assert_allclose(np.asarray(logdensity_fn(point)), expected, rtol=1e-5)


def test_check_model_is_sampleable():
    with pm.Model() as model:
        pm.Normal("x", dtype="float64")

    with pytest.raises(ValueError, match="not float32"):
        check_model_is_sampleable(model)


def test_check_model_is_sampleable_rejects_discrete(float32):
    with pm.Model() as model:
        pm.Poisson("counts", 3.0)

    with pytest.raises(ValueError, match="discrete"):
        check_model_is_sampleable(model)


def test_one_dimensional_model_is_rejected(float32):
    """The isokinetic update divides by (dim - 1), so a single parameter must not sample."""
    with pm.Model() as model:
        pm.HalfNormal("sigma", 3.0)

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        fit_mlx_mclmc(draws=10, tune=100, chains=1, model=model)


def test_fit_mlx_mclmc_recovers_conjugate_posterior(conjugate_model):
    model, posterior_mean, posterior_sd = conjugate_model

    idata = fit_mlx_mclmc(
        draws=2000,
        tune=2000,
        chains=2,
        model=model,
        random_seed=42,
        include_transformed=True,
    )
    posterior = idata["posterior"].dataset

    assert posterior["mu"].shape == (2, 2000, 3)
    assert posterior["mu"].coords["group"].values.tolist() == ["a", "b", "c"]
    assert idata["sample_stats"]["energy_error"].shape == (2, 2000)
    assert np.isfinite(idata["sample_stats"]["energy_error"].values).all()
    assert "mu" in idata["unconstrained_posterior"].dataset
    assert posterior.attrs["step_size"] > 0

    np.testing.assert_allclose(
        posterior["mu"].mean(dim=("chain", "draw")), posterior_mean, atol=0.15 * posterior_sd
    )
    np.testing.assert_allclose(posterior["mu"].std(dim=("chain", "draw")), posterior_sd, rtol=0.1)
    np.testing.assert_allclose(posterior["mu_sum"], posterior["mu"].sum(dim="group"), atol=1e-6)


def test_fit_mlx_mclmc_transformed_variable(float32):
    rng = np.random.default_rng(1)
    data = rng.normal(loc=1.0, scale=2.0, size=400)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 5.0)
        sigma = pm.HalfNormal("sigma", 3.0)
        pm.Normal("y", mu, sigma, observed=data)

    idata = fit_mlx_mclmc(draws=2000, tune=2000, chains=2, model=model, random_seed=0)
    posterior = idata["posterior"]

    assert (posterior["sigma"] > 0).all()
    assert np.isfinite(idata["sample_stats"]["energy_error"].values).all()

    np.testing.assert_allclose(posterior["mu"].mean(), data.mean(), atol=0.1)
    np.testing.assert_allclose(posterior["sigma"].mean(), data.std(), atol=0.1)

    # The spread is what a sampler is for; a frozen chain passes every mean-only assertion.
    np.testing.assert_allclose(
        posterior["sigma"].std(), data.std() / np.sqrt(2 * data.size), rtol=0.25
    )


def test_fit_mlx_mclmc_warmup_kwargs_reach_the_adaptation(conjugate_model):
    """warmup_kwargs must override the named arguments it shadows, not collide with them."""
    model, *_ = conjugate_model
    settings = dict(draws=100, tune=400, chains=1, model=model, random_seed=0)

    default = fit_mlx_mclmc(**settings, warmup_kwargs={"optimize_steps": 20})
    overridden = fit_mlx_mclmc(
        **settings,
        # desired_energy_var shadows a named argument; frac_tune3 makes the override observable
        # by dropping the third adaptation phase from the step count.
        warmup_kwargs={"desired_energy_var": 1e-3, "optimize_steps": 20, "frac_tune3": 0.0},
    )

    assert overridden["posterior"].attrs["num_tuning_steps"] == (
        default["posterior"].attrs["num_tuning_steps"] - 40
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diagonal_preconditioning": False},
        {"integrator": "velocity_verlet"},
        {"compile_step": False},
    ],
    ids=["no_preconditioning", "velocity_verlet", "uncompiled"],
)
def test_fit_mlx_mclmc_alternate_settings(conjugate_model, kwargs):
    model, posterior_mean, posterior_sd = conjugate_model

    idata = fit_mlx_mclmc(draws=2000, tune=2000, chains=2, model=model, random_seed=3, **kwargs)

    assert np.isfinite(idata["sample_stats"]["energy_error"].values).all()
    np.testing.assert_allclose(
        idata["posterior"]["mu"].mean(dim=("chain", "draw")),
        posterior_mean,
        atol=0.2 * posterior_sd,
    )
    np.testing.assert_allclose(
        idata["posterior"]["mu"].std(dim=("chain", "draw")), posterior_sd, rtol=0.15
    )


def test_draws_round_trip_through_mixed_shapes_and_transforms(float32):
    """Draws must land in the right variable regardless of value_vars order or transform."""
    coords = {"g": ["a", "b", "c"], "r": [0, 1], "c": ["x", "y", "z"], "k": list("wxyz")}
    with pm.Model(coords=coords) as model:
        pm.Normal("mu", 0.0, 1.0, dims="g")
        pm.Normal("B", 0.0, 1.0, dims=("r", "c"))
        pm.HalfNormal("sigma", 1.0)
        pm.Dirichlet("p", np.ones(4), dims="k")
        pm.Uniform("lo", -1.0, 1.0, dims="g")

    logdensity_fn = MLXLogp(model)
    rng = np.random.default_rng(0)
    flat_draws = rng.normal(size=(2, 5, logdensity_fn.dim)).astype("float32") * 0.3

    posterior, unconstrained = draws_to_datasets(
        flat_draws, logdensity_fn.model, include_transformed=True
    )

    # PyMC's own forward map from the same flat vector is the reference.
    forward = logdensity_fn.model.compile_fn(
        logdensity_fn.model.replace_rvs_by_values(logdensity_fn.model.free_RVs),
        inputs=logdensity_fn.model.value_vars,
        on_unused_input="ignore",
    )
    blocks = np.split(flat_draws[1, 3], np.cumsum(logdensity_fn.sizes)[:-1])
    expected = forward(
        {
            name: block.reshape(shape)
            for name, block, shape in zip(
                logdensity_fn.names, blocks, logdensity_fn.shapes, strict=True
            )
        }
    )

    for free_RV, expected_value in zip(logdensity_fn.model.free_RVs, expected, strict=True):
        np.testing.assert_allclose(posterior[free_RV.name].values[1, 3], expected_value, rtol=1e-5)

    np.testing.assert_allclose(posterior["p"].sum(dim="k"), 1.0, rtol=1e-5)
    assert ((posterior["lo"] > -1) & (posterior["lo"] < 1)).all()

    # The simplex transform drops a coordinate, so it must not borrow the constrained dim.
    assert unconstrained["p_simplex__"].shape[-1] == 3
    assert "k" not in unconstrained["p_simplex__"].dims


def test_sampler_is_absent_from_the_inference_namespace():
    """It needs mlx, so nothing importable on a non-Apple machine may reach it."""
    import pymc_extras.inference as inference

    assert not hasattr(inference, "fit_mlx_mclmc")
    with pytest.raises(ValueError, match="not supported"):
        inference.fit(method="mlx_mclmc")


def test_sample_reverts_and_reports_non_finite_steps():
    """A step whose log-density comes back nan is reverted, as blackjax's kernel does."""

    def logdensity_fn(x):
        return mx.where(x[0] > 0, -0.5 * mx.sum(x**2), mx.array(float("nan")))

    output = sample(
        logdensity_fn,
        np.array([[0.05, 0.0, 0.0]], dtype="float32"),
        L=1.0,
        step_size=0.8,
        n_steps=200,
        seed=1,
    )

    assert np.asarray(output.diverging).any(), "the target should push the chain out of support"
    assert np.isfinite(np.asarray(output.samples)).all()
    assert np.isfinite(np.asarray(output.energy_errors)).all()


def test_warns_on_divergences_and_on_collapsed_adaptation():
    healthy = TunedParameters(
        position=mx.zeros((2,)),
        L=1.0,
        step_size=0.5,
        inverse_mass_matrix=mx.ones((2,)),
        num_tuning_steps=10,
    )
    with pytest.warns(RuntimeWarning, match="divergent"):
        diverging = np.zeros((2, 100), dtype=bool)
        diverging[0, :10] = True
        _warn_if_adaptation_failed(healthy, diverging)

    with pytest.warns(RuntimeWarning, match="collapsed"):
        collapsed = healthy._replace(step_size=1e-12)
        _warn_if_adaptation_failed(collapsed, np.zeros((2, 100), dtype=bool))


def test_warmup_recovers_the_diagonal_metric():
    """Diagonal preconditioning exists to put the marginal variances in the mass matrix."""
    variances = np.array([0.25, 1.0, 4.0, 16.0], dtype="float32")
    precision = mx.array(np.diag(1.0 / variances))

    def logdensity_fn(x):
        return -0.5 * mx.sum(x * (precision @ x))

    tuned = warmup(logdensity_fn, np.zeros(len(variances)), num_steps=8000, seed=0)

    # A streaming estimate over a short window, so the tolerance is wide -- the point is that it
    # tracks a 64x spread in scale rather than that it nails any one coordinate.
    np.testing.assert_allclose(np.asarray(tuned.inverse_mass_matrix), variances, rtol=0.35)


def test_tune_step_size_reaches_the_target_energy_variance():
    dim = 4
    precision = mx.eye(dim)

    def logdensity_fn(x):
        return -0.5 * mx.sum(x * (precision @ x))

    initial_positions = mx.random.normal(shape=(4, dim), key=mx.random.key(0))
    step_size = tune_step_size(
        logdensity_fn, initial_positions, L=1.5 * np.sqrt(dim), desired_energy_var=5e-4
    )

    energy_errors = sample(
        logdensity_fn,
        initial_positions,
        L=1.5 * np.sqrt(dim),
        step_size=step_size,
        n_steps=600,
        discard=300,
        seed=1,
    ).energy_errors
    energy_var = float(mx.mean(mx.var(energy_errors, axis=0)) / dim)

    assert 0.5 * 5e-4 < energy_var < 2.0 * 5e-4


def test_initial_point_accepts_a_dict_or_a_flat_vector(conjugate_model):
    """The flat form is in value_vars order, which is not the order the model declares."""
    model, *_ = conjugate_model
    settings = dict(draws=100, tune=400, chains=1, model=model, random_seed=0)
    start = {"mu": np.array([0.5, -0.5, 1.5], dtype="float32")}

    from_dict = fit_mlx_mclmc(**settings, initial_point=start)
    from_vector = fit_mlx_mclmc(**settings, initial_point=start["mu"])

    np.testing.assert_array_equal(
        from_dict["posterior"]["mu"].values, from_vector["posterior"]["mu"].values
    )


def test_burn_in_is_dropped_from_both_the_draws_and_the_diagnostics(conjugate_model):
    model, *_ = conjugate_model

    idata = fit_mlx_mclmc(draws=200, tune=400, burn_in=300, chains=2, model=model, random_seed=0)

    assert idata["posterior"]["mu"].shape == (2, 200, 3)
    assert idata["sample_stats"]["energy_error"].shape == (2, 200)
    assert idata["sample_stats"]["diverging"].shape == (2, 200)


def test_non_finite_initial_gradient_is_rejected(float32):
    """A finite log-density with a non-finite gradient must fail loudly, not sample nans."""
    with pm.Model() as model:
        x = pm.Flat("x", shape=2)
        pm.Potential("kink", pt.sqrt(pt.abs(x)).sum())

    with pytest.raises(ValueError, match="gradient is not"):
        fit_mlx_mclmc(draws=10, tune=100, chains=2, model=model)


def test_non_finite_initial_logdensity_is_rejected(float32):
    with pm.Model() as model:
        x = pm.Flat("x", shape=2)
        pm.Potential("undefined", pt.log(x).sum())

    with pytest.raises(ValueError, match="log-density is not finite"):
        fit_mlx_mclmc(draws=10, tune=100, chains=2, model=model)


def test_ascent_skips_non_finite_steps_instead_of_absorbing_them():
    """A nan gradient must leave the position and the Adam moments untouched, not poison them."""

    def logdensity_fn(x):
        in_band = (x[0] > 1.0) & (x[0] < 1.2)
        return mx.where(in_band, mx.array(float("nan")), -0.5 * mx.sum((x - 3.0) ** 2))

    reached = _optimize_to_mode(
        mx.vmap(mx.value_and_grad(logdensity_fn)),
        mx.array([[0.0]]),
        steps=400,
        learning_rate=0.05,
    )

    # Absorbing the nan into the moments would freeze the ascent short of the band at 1.0.
    np.testing.assert_allclose(np.asarray(reached).ravel(), [3.0], atol=1e-2)


def test_ascent_gives_up_when_the_gradient_never_becomes_finite():
    always_nan = mx.vmap(mx.value_and_grad(lambda x: mx.sum(x) * mx.array(float("nan"))))
    start = mx.array([[0.7, -0.3]])

    unchanged = _optimize_to_mode(always_nan, start, steps=400, learning_rate=0.05)

    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(start))
