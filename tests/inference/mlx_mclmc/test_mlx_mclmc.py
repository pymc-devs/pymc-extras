import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.special import gamma

mx = pytest.importorskip("mlx.core", reason="MCLMC requires mlx, which needs Apple Silicon")

from pymc_extras.inference.mlx_mclmc import fit_mlx_mclmc
from pymc_extras.inference.mlx_mclmc.kernel import (
    AdaptationSettings,
    Metric,
    TunedParameters,
    _accumulate,
    _empty_moments,
    _ess_per_dim,
    _fit_metric,
    _low_rank_metric,
    _optimize_to_mode,
    _window_switch_steps,
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

    assert (tuned.step_size > 0).all()
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
    assert (posterior.attrs["step_size"] > 0).all()

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


def test_adaptation_settings_reach_the_warmup(conjugate_model):
    """Settings must be threaded to warmup, not silently defaulted."""
    model, *_ = conjugate_model
    shared = dict(draws=100, tune=400, chains=1, model=model, random_seed=0)

    default = fit_mlx_mclmc(**shared)
    without_phase_three = fit_mlx_mclmc(**shared, adaptation=AdaptationSettings(frac_tune3=0.0))

    # blackjax's budget: 40 + 40 for phases 1 and 2, 40 // 3 to re-adjust under the metric, and
    # 40 for phase 3, which frac_tune3=0 drops.
    assert default["posterior"].attrs["num_tuning_steps"] == 133
    assert without_phase_three["posterior"].attrs["num_tuning_steps"] == 93


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adaptation": AdaptationSettings(diagonal_preconditioning=False)},
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
        metric=Metric(scale=mx.ones((2,))),
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
    np.testing.assert_allclose(np.asarray(tuned.metric.scale) ** 2, variances, rtol=0.35)


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


def test_optimize_steps_moves_the_adapting_chain_to_the_mode():
    """The ascent is off by default, so the opt-in has to be seen to reach warmup."""
    mode = np.array([5.0, -5.0], dtype="float32")

    def logdensity_fn(x):
        return -0.5 * mx.sum((x - mx.array(mode)) ** 2) * 100.0

    # A budget this small gives the dynamics no chance to cross 5 units on their own.
    tuned = warmup(
        logdensity_fn,
        np.zeros(2),
        num_steps=20,
        settings=AdaptationSettings(optimize_steps=300, optimize_learning_rate=0.1),
        seed=0,
    )

    np.testing.assert_allclose(np.asarray(tuned.position)[0], mode, atol=0.5)


def test_ascent_gives_up_when_the_gradient_never_becomes_finite():
    always_nan = mx.vmap(mx.value_and_grad(lambda x: mx.sum(x) * mx.array(float("nan"))))
    start = mx.array([[0.7, -0.3]])

    unchanged = _optimize_to_mode(always_nan, start, steps=400, learning_rate=0.05)

    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(start))


def test_fit_falls_back_to_an_unfused_step_past_the_metal_limit(conjugate_model, monkeypatch):
    """A graph too large for mx.compile must retry unfused, not fail."""
    import pymc_extras.inference.mlx_mclmc.kernel as kernel

    model, *_ = conjugate_model
    attempted = []
    real = kernel.warmup_and_sample

    def fail_once_when_fused(*args, compile_step, **kwargs):
        attempted.append(compile_step)
        if compile_step:
            raise RuntimeError("[compile] Too many inputs/outputs fused in the Metal Compiled")
        return real(*args, compile_step=False, **kwargs)

    monkeypatch.setattr(kernel, "warmup_and_sample", fail_once_when_fused)

    with pytest.warns(RuntimeWarning, match="unfused"):
        idata = fit_mlx_mclmc(draws=100, tune=200, chains=1, model=model, random_seed=0)

    assert attempted == [True, False]
    assert idata["posterior"]["mu"].shape == (1, 100, 3)


def test_logp_leaves_the_model_unfrozen_and_follows_set_data(float32):
    """Freezing is the caller's choice, so pm.Data stays live and set_data is picked up."""
    with pm.Model(coords={"obs": range(4)}) as model:
        x = pm.Data("x", np.ones(4, dtype="float32"), dims="obs")
        mu = pm.Normal("mu")
        pm.Normal("y", mu * x, 1.0, observed=np.zeros(4, dtype="float32"), dims="obs")

    logdensity_fn = MLXLogp(model)
    assert logdensity_fn.model is model

    point = mx.array(np.array([0.4], dtype="float32"))
    reference = model.compile_logp()
    before = np.asarray(logdensity_fn(point))
    np.testing.assert_allclose(before, reference({"mu": 0.4}), rtol=1e-5)

    pm.set_data({"x": np.full(4, 3.0, dtype="float32")}, model=model)

    after = np.asarray(logdensity_fn(point))
    np.testing.assert_allclose(after, reference({"mu": 0.4}), rtol=1e-5)
    assert not np.isclose(before, after)


def test_fit_samples_a_model_with_live_data_and_dims(float32):
    """An unfrozen model, whose dim lengths and data are still shared variables, must sample."""
    rng = np.random.default_rng(1)
    observed = rng.normal(loc=2.0, scale=0.5, size=60).astype("float32")

    with pm.Model(coords={"obs": range(60)}) as model:
        x = pm.Data("x", np.ones(60, dtype="float32"), dims="obs")
        mu = pm.Normal("mu", 0.0, 3.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu * x, sigma, observed=observed, dims="obs")

    idata = fit_mlx_mclmc(draws=400, tune=400, chains=2, model=model, random_seed=0)

    assert idata["posterior"]["mu"].shape == (2, 400)
    np.testing.assert_allclose(idata["posterior"]["mu"].mean(), 2.0, atol=0.1)
    np.testing.assert_allclose(idata["posterior"]["sigma"].mean(), 0.5, atol=0.1)


def test_fit_recovers_the_funnel_of_a_centered_hierarchical_model(float32):
    """Eight schools, centered. A warmup that seeks the mode ends up in the neck and never leaves."""
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype="float32")
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype="float32")

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfCauchy("tau", 5.0)
        theta = pm.Normal("theta", mu, tau, shape=8)
        pm.Normal("obs", theta, sigma, observed=y)

    idata = fit_mlx_mclmc(draws=2000, tune=2000, chains=4, model=model, random_seed=0)
    tau_draws = idata["posterior"]["tau"]

    # NUTS puts tau's mean near 3.6 with a standard deviation near 3.2. The neck is at tau < 0.1.
    assert 2.0 < float(tau_draws.mean()) < 6.0
    assert float(tau_draws.std()) > 1.5


def test_warmup_survives_a_non_finite_step_in_phase_three():
    """Phase 3 measures autocorrelation from raw positions, so one nan there must be reverted."""

    class BandedGaussian:
        """A nan band straddling the mode, so any long run lands in it. The gradient is nan there
        too, as PyTensor's would be, where MLX autodiff through ``mx.where`` would give zero."""

        def __call__(self, x):
            return self.value_and_grad(x[None])[0][0]

        def value_and_grad(self, x):
            in_band = mx.abs(x[:, :1]) < 0.05
            nan = mx.array(float("nan"))
            value = mx.where(in_band[:, 0], nan, -0.5 * mx.sum(x**2, axis=-1))
            return value, mx.where(in_band, nan, -x)

    tuned = warmup(
        BandedGaussian(),
        np.array([1.0, 0.0, 0.0], dtype="float32"),
        num_steps=3000,
        settings=AdaptationSettings(frac_tune1=0.05, frac_tune2=0.05, frac_tune3=0.9),
        seed=1,
    )

    assert np.isfinite(tuned.L).all()
    assert np.isfinite(np.asarray(tuned.position)).all()


def test_low_rank_metric_beats_the_diagonal_on_a_rotated_gaussian():
    """The low-rank correction exists to precondition ridges the diagonal cannot see."""
    dim = 12
    rng = np.random.default_rng(3)
    factor = rng.normal(size=(dim, dim))
    covariance = factor @ factor.T + 0.05 * np.eye(dim)
    precision = mx.array(np.linalg.inv(covariance).astype("float32"))

    def logdensity_fn(x):
        return -0.5 * mx.sum(x * (precision @ x))

    ess = {}
    for mass_matrix in ("gradient", "low_rank"):
        output, tuned = warmup_and_sample(
            logdensity_fn,
            np.zeros(dim),
            num_tune=4000,
            draws=4000,
            chains=4,
            seed=0,
            settings=AdaptationSettings(mass_matrix=mass_matrix),
        )
        draws = np.asarray(output.samples)
        ess[mass_matrix] = np.mean([_ess_per_dim(draws[:, chain]).mean() for chain in range(4)])

    assert tuned.metric.correction is not None
    # Measured 330 against 2306; a factor of 3 leaves room for the seed.
    assert ess["low_rank"] > 3 * ess["gradient"]


def test_low_rank_fit_rejects_a_window_it_cannot_trust(caplog):
    dim = 4
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(dim, 20))
    draws = clean.copy()
    draws[0, 0] = np.nan

    assert _low_rank_metric(draws, grads=-draws) is None

    def as_row(column):
        return mx.array(column[None].astype("float32"))

    # The running moments are clean; only the retained window carries the nan.
    moments = _empty_moments(dim)
    for column in clean.T:
        moments = _accumulate(
            moments, position=as_row(column), grad=-as_row(column), weight=mx.ones((1,))
        )
    retained = [(as_row(column), -as_row(column)) for column in draws.T[:3]]

    with caplog.at_level("WARNING"):
        metric = _fit_metric(
            moments, retained, dim=dim, mass_matrix="low_rank", allow_low_rank=True
        )

    assert metric.correction is None
    assert "rejected as unreliable" in caplog.text


def test_low_rank_fit_drops_the_correction_when_no_direction_qualifies():
    """An axis-aligned window has nothing for the correction to add, so it must not carry an
    empty one that costs two matmuls per step."""
    dim = 4
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(dim, 400))

    metric = _low_rank_metric(draws, grads=-draws)

    assert metric is not None
    assert metric.correction is None


def test_window_switch_schedule():
    """Short early windows, then windows that start at switch_freq and grow by window_growth,
    stopping before a window that could not finish inside num_steps."""
    schedule = dict(early_switch_freq=10, switch_freq=80, window_growth=1.5)

    assert _window_switch_steps(num_steps=300, early_end=30, **schedule) == {10, 20, 30, 110, 230}
    assert _window_switch_steps(num_steps=200, early_end=0, **schedule) == {80, 200}


def test_unpreconditioned_L_is_the_root_summed_position_variance():
    """Without preconditioning, L is sqrt(sum Var[x]) as in blackjax. The gradient-ratio proxy
    the metric uses agrees on a Gaussian only, so the target here is a quartic."""
    dim = 4
    quartic_variance = 2.0 * gamma(0.75) / gamma(0.25)

    def logdensity_fn(x):
        return -0.25 * mx.sum(x**4)

    tuned = warmup(
        logdensity_fn,
        np.zeros(dim),
        num_steps=20000,
        seed=0,
        settings=AdaptationSettings(diagonal_preconditioning=False, frac_tune3=0.0),
    )

    # The proxy would give sqrt(dim * 0.568) = 1.51 against the true 1.64.
    np.testing.assert_allclose(tuned.L, np.sqrt(dim * quartic_variance), rtol=0.04)


def test_fused_momentum_update_matches_the_mlx_path():
    """The Metal kernel must reproduce the MLX ops it replaces, including the per-chain step."""
    from pymc_extras.inference.mlx_mclmc.kernel import _momentum_update, _momentum_update_fused

    dim, chains = 200, 4
    rng = np.random.default_rng(0)
    momentum = rng.normal(size=(chains, dim)).astype("float32")
    momentum /= np.linalg.norm(momentum, axis=1, keepdims=True)
    grad = (3.0 * rng.normal(size=(chains, dim))).astype("float32")
    metric = Metric(scale=mx.array(rng.uniform(0.5, 2.0, dim).astype("float32")))

    # A per-chain step spanning the small-delta branch, where the kernel uses a series for
    # expm1, and the large-delta branch.
    step = mx.array([1e-3, 0.05, 0.5, 5.0], dtype=mx.float32).reshape(-1, 1)
    reference = _momentum_update(mx.array(momentum), mx.array(grad), step, metric, dim)
    fused = _momentum_update_fused(mx.array(momentum), mx.array(grad), step, metric.scale, dim)
    mx.eval(*reference, *fused)

    for left, right in zip(reference, fused, strict=True):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=1e-4, atol=1e-6)
