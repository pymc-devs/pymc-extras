import numpy as np
import pymc as pm
import pytest
import xarray as xr

from xarray import DataTree

import pymc_extras as pmx
import pymc_extras.inference.consensus_mc.consensus_mc as cmc

from pymc_extras.inference.consensus_mc import (
    estimate_parametric,
    fit_consensus_mc,
    merge_consensus,
    merge_parametric,
)
from pymc_extras.inference.consensus_mc.consensus_mc import (
    _build_shard_model,
    _resolve_shards,
    _validate_likelihood_coverage,
    _validate_model_supported,
)


def _expected_covariances(samples):
    covariances = []
    for subposterior in samples:
        centered = subposterior - subposterior.mean(axis=0, keepdims=True)
        covariances.append(centered.T @ centered / (subposterior.shape[0] - 1))
    return np.asarray(covariances)


def _expected_full_consensus(samples):
    covariances = _expected_covariances(samples)
    precisions = np.array([np.linalg.inv(covariance) for covariance in covariances])
    total_cov = np.linalg.inv(precisions.sum(axis=0))
    weights = np.einsum("ij,kjl->kil", total_cov, precisions)
    return np.einsum("kij,knj->ni", weights, samples)


def _expected_parametric(samples):
    means = samples.mean(axis=1)
    covariances = _expected_covariances(samples)
    precisions = np.array([np.linalg.inv(covariance) for covariance in covariances])
    cov = np.linalg.inv(precisions.sum(axis=0))
    weights = np.einsum("ij,kjl->kil", cov, precisions)
    mean = np.einsum("kij,kj->i", weights, means)
    return mean, cov


def _model_with_data(n=6):
    coords = {"obs": np.arange(n)}
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", np.arange(float(n)), dims="obs")
        theta = pm.Normal("theta")
        pm.Normal("y_obs", theta, 1, observed=y, dims="obs")
    return model


def _patch_sample(monkeypatch, values_by_call=None, fail_on_call=None):
    calls = []

    def fake_sample(*, draws, chains, model, **kwargs):
        calls.append((draws, chains, tuple(np.asarray(model["y"].get_value()).tolist())))
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("induced sampling failure")
        if values_by_call is None:
            base = float(np.mean(model["y"].get_value()))
            values = np.arange(chains * draws, dtype=float).reshape(chains, draws) + base
        else:
            values = np.asarray(values_by_call[len(calls) - 1], dtype=float).reshape(chains, draws)
        return DataTree.from_dict({"posterior": xr.Dataset({"theta": (("chain", "draw"), values)})})

    monkeypatch.setattr(pm, "sample", fake_sample)
    return calls


def test_diagonal_consensus_matches_known_univariate_result():
    subposteriors = np.zeros((2, 3, 1))
    subposteriors[0, :, 0] = [0.0, 2.0, 4.0]
    subposteriors[1, :, 0] = [10.0, 11.0, 12.0]

    merged = merge_consensus(subposteriors, draws=None, diagonal=True)

    np.testing.assert_allclose(merged, np.array([[8.0], [9.2], [10.4]]))


@pytest.mark.parametrize("shape", [(3, 20, 1), (3, 20, 2)])
def test_full_consensus_matches_numpy_for_univariate_and_multivariate(shape):
    rng = np.random.default_rng(829)
    subposteriors = rng.normal(size=shape)

    merged = merge_consensus(subposteriors, draws=None, diagonal=False)

    np.testing.assert_allclose(merged, _expected_full_consensus(subposteriors))


def test_parametric_estimates_match_numpy_diagonal_and_full():
    rng = np.random.default_rng(132)
    subposteriors = rng.normal(size=(3, 20, 2))

    mean, var = estimate_parametric(subposteriors, diagonal=True)
    submeans = subposteriors.mean(axis=1)
    subvars = subposteriors.var(axis=1, ddof=1)
    precision = 1.0 / subvars
    expected_var = 1.0 / precision.sum(axis=0)
    expected_mean = np.einsum("kp,kp->p", expected_var * precision, submeans)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)

    full_mean, full_cov = estimate_parametric(subposteriors, diagonal=False)
    expected_full_mean, expected_full_cov = _expected_parametric(subposteriors)
    np.testing.assert_allclose(full_mean, expected_full_mean)
    np.testing.assert_allclose(full_cov, expected_full_cov)

    univariate = rng.normal(size=(3, 20, 1))
    mean_1d, cov_1d = estimate_parametric(univariate, diagonal=False)
    expected_mean_1d, expected_cov_1d = _expected_parametric(univariate)
    np.testing.assert_allclose(mean_1d, expected_mean_1d)
    np.testing.assert_allclose(cov_1d, expected_cov_1d)
    assert cov_1d.shape == (1, 1)


def test_parametric_draws_have_expected_shape_and_mean():
    rng = np.random.default_rng(91)
    subposteriors = rng.normal(size=(3, 200, 2))
    mean, _ = estimate_parametric(subposteriors, diagonal=False)

    draws = merge_parametric(subposteriors, draws=2_000, random_seed=123)

    assert draws.shape == (2_000, 2)
    assert np.isfinite(draws).all()
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.08)


@pytest.mark.parametrize(
    "subposteriors, match",
    [
        (np.ones((2, 3)), "shape"),
        (np.ones((1, 3, 1)), "at least two subposteriors"),
        (np.ones((2, 1, 1)), "at least two samples"),
        (np.ones((2, 3, 4)), "more samples"),
        (np.array([[[np.nan], [1.0]], [[1.0], [2.0]]]), "finite"),
    ],
)
def test_merge_validation_errors(subposteriors, match):
    with pytest.raises(ValueError, match=match):
        merge_consensus(subposteriors, diagonal=False)


def test_resampled_consensus_validates_resampled_draw_count():
    subposteriors = np.arange(20.0).reshape(2, 5, 2)

    with pytest.raises(ValueError, match="at least two samples"):
        merge_consensus(subposteriors, draws=1, diagonal=True, random_seed=53)

    with pytest.raises(ValueError, match="more samples"):
        merge_consensus(subposteriors, draws=2, diagonal=False, random_seed=53)


def test_prior_scaling_untransformed_model():
    y = np.array([0.2, -0.1])
    with pm.Model() as model:
        y_data = pm.Data("y_data", y)
        theta = pm.Normal("theta", 0, 1)
        pm.Normal("obs", theta, 1, observed=y_data)

    shard_model = _build_shard_model(
        model, num_shards=4, prior_potentials=[], likelihood_potentials=[]
    )
    point = shard_model.initial_point()
    point["theta"] = 0.25

    shard_logp = shard_model.compile_logp()(point)
    observed_logp = shard_model.compile_logp(vars=shard_model.observed_RVs)(point)
    prior_logp_nojac = shard_model.compile_logp(vars=shard_model.free_RVs, jacobian=False)(point)
    np.testing.assert_allclose(shard_logp, observed_logp + 0.25 * prior_logp_nojac)


def test_prior_scaling_transformed_model_keeps_jacobian_once():
    y = np.array([0.4, 0.8])
    with pm.Model() as model:
        y_data = pm.Data("y_data", y)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("obs", 0, sigma, observed=y_data)

    num_shards = 5
    shard_model = _build_shard_model(
        model, num_shards=num_shards, prior_potentials=[], likelihood_potentials=[]
    )
    point = shard_model.initial_point()
    point["sigma_log__"] = np.log(0.7)

    shard_logp = shard_model.compile_logp()(point)
    observed_logp = shard_model.compile_logp(vars=shard_model.observed_RVs)(point)
    prior_nojac = shard_model.compile_logp(vars=shard_model.free_RVs, jacobian=False)(point)
    prior_with_jac = shard_model.compile_logp(vars=shard_model.free_RVs, jacobian=True)(point)
    jacobian_once = prior_with_jac - prior_nojac
    np.testing.assert_allclose(
        shard_logp,
        observed_logp + prior_nojac / num_shards + jacobian_once,
    )


def test_potential_classification_is_strict():
    with pm.Model() as model:
        theta = pm.Normal("theta")
        pm.Potential("soft_prior", -(theta**2))

    with pytest.raises(ValueError, match="classify every potential"):
        _build_shard_model(model, num_shards=2, prior_potentials=None, likelihood_potentials=None)

    _build_shard_model(
        model,
        num_shards=2,
        prior_potentials=["soft_prior"],
        likelihood_potentials=[],
    )


def test_likelihood_coverage_requires_every_likelihood_to_be_sharded():
    with pm.Model(coords={"obs": np.arange(4)}) as model:
        y1 = pm.Data("y1", np.arange(4.0), dims="obs")
        y2 = pm.Data("y2", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta")
        pm.Normal("obs1", theta, 1, observed=y1, dims="obs")
        pm.Normal("obs2", theta, 1, observed=y2, dims="obs")

    with pytest.raises(ValueError, match="depends on unsharded data variables"):
        _validate_likelihood_coverage(
            model,
            sharded_data_names={"y1"},
            global_data=[],
            likelihood_potentials=[],
        )


def test_likelihood_coverage_rejects_direct_ndarray_observations():
    with pm.Model() as model:
        theta = pm.Normal("theta")
        pm.Normal("obs", theta, 1, observed=np.arange(4.0))

    with pytest.raises(TypeError, match=r"backed by mutable pm\.Data"):
        _validate_likelihood_coverage(
            model,
            sharded_data_names={"y"},
            global_data=[],
            likelihood_potentials=[],
        )


def test_likelihood_coverage_allows_scalar_global_but_not_vector_global_by_default():
    with pm.Model(coords={"obs": np.arange(4)}) as model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        offset = pm.Data("offset", np.array(1.0))
        theta = pm.Normal("theta")
        pm.Normal("y_like", theta + offset, 1, observed=y, dims="obs")

    _validate_likelihood_coverage(
        model,
        sharded_data_names={"y"},
        global_data=["offset"],
        likelihood_potentials=[],
    )

    with pm.Model(coords={"obs": np.arange(4)}) as vector_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        z = pm.Data("z", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta")
        pm.Normal("y_like", theta + z, 1, observed=y, dims="obs")

    with pytest.raises(ValueError, match="depends on unsharded data variables"):
        _validate_likelihood_coverage(
            vector_model,
            sharded_data_names={"y"},
            global_data=[],
            likelihood_potentials=[],
        )


def test_split_shards_and_coords_guardrails():
    model = _model_with_data(10)
    shards, coords, dims = _resolve_shards(
        model,
        shards=None,
        shard_coords=None,
        split_data={"y": "obs"},
        num_shards=3,
    )
    assert [shard["y"].shape[0] for shard in shards] == [4, 3, 3]
    assert [len(coord["obs"]) for coord in coords] == [4, 3, 3]
    assert dims == {"obs"}

    with pytest.raises(ValueError, match="empty shards"):
        _resolve_shards(
            model,
            shards=None,
            shard_coords=None,
            split_data={"y": "obs"},
            num_shards=11,
        )

    with pm.Model() as unnamed_model:
        y = pm.Data("y", np.arange(4.0))
        theta = pm.Normal("theta")
        pm.Normal("obs", theta, 1, observed=y)

    with pytest.raises(ValueError, match="requires named model dimensions"):
        _resolve_shards(
            unnamed_model,
            shards=None,
            shard_coords=None,
            split_data={"y": 0},
            num_shards=2,
        )


def test_shard_local_and_unnamed_free_rvs_are_rejected():
    with pm.Model(coords={"obs": np.arange(4)}) as local_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta", dims="obs")
        pm.Normal("y_like", theta, 1, observed=y, dims="obs")

    with pytest.raises(ValueError, match="indexed by sharded dimension"):
        _validate_model_supported(local_model, {"obs"})

    with pm.Model(coords={"obs": np.arange(4)}) as unnamed_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta", shape=2)
        pm.Normal("y_like", theta[0], 1, observed=y, dims="obs")

    with pytest.raises(ValueError, match="cannot prove unnamed non-scalar free RV"):
        _validate_model_supported(unnamed_model, {"obs"})


def test_explicit_shards_num_shards_and_sharded_dims():
    with pm.Model(coords={"obs": np.arange(6), "feature": ["a", "b"]}) as model:
        x = pm.Data("x", np.ones((6, 2)), dims=("obs", "feature"))
        y = pm.Data("y", np.arange(6.0), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        pm.Normal("y_like", x @ beta, 1, observed=y, dims="obs")

    explicit = [
        {"x": np.ones((3, 2)), "y": np.arange(3.0)},
        {"x": np.ones((3, 2)), "y": np.arange(3.0, 6.0)},
    ]
    shards, _, dims = _resolve_shards(
        model,
        shards=explicit,
        shard_coords=None,
        split_data=None,
        num_shards=None,
        sharded_dims=["obs"],
    )
    assert len(shards) == 2
    assert dims == {"obs"}
    _validate_model_supported(model, dims)

    with pytest.raises(ValueError, match="num_shards must match"):
        _resolve_shards(
            model,
            shards=explicit,
            shard_coords=None,
            split_data=None,
            num_shards=3,
            sharded_dims=["obs"],
        )
    with pytest.raises(ValueError, match="at least two explicit shards"):
        _resolve_shards(
            model,
            shards=explicit[:1],
            shard_coords=None,
            split_data=None,
            num_shards=None,
            sharded_dims=["obs"],
        )

    with pm.Model(coords={"obs": np.arange(6)}) as local_model:
        y = pm.Data("y", np.arange(6.0), dims="obs")
        theta = pm.Normal("theta", dims="obs")
        pm.Normal("y_like", theta, 1, observed=y, dims="obs")
    with pytest.raises(ValueError, match="indexed by sharded dimension"):
        _validate_model_supported(local_model, {"obs"})

    with pytest.raises(ValueError, match="non-scalar data require sharded_dims"):
        _resolve_shards(
            model,
            shards=[{"x": np.ones((6, 2)), "y": np.arange(6.0)}] * 2,
            shard_coords=None,
            split_data=None,
            num_shards=None,
            sharded_dims=None,
        )


def test_explicit_shards_without_coords_generate_shard_local_coords(monkeypatch):
    calls = []

    def fake_sample(*, draws, chains, model, **kwargs):
        calls.append(
            (
                tuple(model.coords["obs"]),
                model["x"].get_value().shape,
                model["y"].get_value().shape,
            )
        )
        values = np.ones((chains, draws, 2))
        return DataTree.from_dict(
            {"posterior": xr.Dataset({"beta": (("chain", "draw", "feature"), values)})}
        )

    monkeypatch.setattr(pm, "sample", fake_sample)
    with pm.Model(coords={"obs": np.arange(6), "feature": ["a", "b"]}) as model:
        x = pm.Data("x", np.ones((6, 2)), dims=("obs", "feature"))
        y = pm.Data("y", np.arange(6.0), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        pm.Normal("y_like", x @ beta, 1, observed=y, dims="obs")

    idata = fit_consensus_mc(
        model=model,
        shards=[
            {"x": np.ones((3, 2)), "y": np.arange(3.0)},
            {"x": np.ones((3, 2)), "y": np.arange(3.0, 6.0)},
        ],
        sharded_dims=["obs"],
        draws=2,
        sample_draws=2,
        tune=0,
        diagonal=True,
        attach_data=False,
        progressbar=False,
    )

    assert calls == [((0, 1, 2), (3, 2), (3,)), ((0, 1, 2), (3, 2), (3,))]
    np.testing.assert_array_equal(idata["consensus_mc"].dataset["shard_size"].values, [3, 3])
    np.testing.assert_array_equal(model["y"].get_value(), np.arange(6.0))
    assert tuple(model.coords["obs"]) == tuple(np.arange(6))


def test_shard_size_uses_named_split_axis_not_axis_zero(monkeypatch):
    def fake_sample(*, draws, chains, **kwargs):
        values = np.ones((chains, draws, 2))
        return DataTree.from_dict(
            {"posterior": xr.Dataset({"beta": (("chain", "draw", "feature"), values)})}
        )

    monkeypatch.setattr(pm, "sample", fake_sample)
    with pm.Model(coords={"feature": ["a", "b"], "obs": np.arange(6)}) as model:
        x = pm.Data("x", np.ones((2, 6)), dims=("feature", "obs"))
        y = pm.Data("y", np.arange(6.0), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        pm.Normal("y_like", beta @ x, 1, observed=y, dims="obs")

    idata = fit_consensus_mc(
        model=model,
        split_data={"x": "obs", "y": "obs"},
        num_shards=2,
        draws=2,
        sample_draws=2,
        tune=0,
        diagonal=True,
        attach_data=False,
        progressbar=False,
    )

    np.testing.assert_array_equal(idata["consensus_mc"].dataset["shard_size"].values, [3, 3])


def test_minibatch_and_total_size_likelihoods_are_rejected():
    with pm.Model(coords={"obs": np.arange(4)}) as total_size_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta")
        pm.Normal("y_like", theta, 1, observed=y, dims="obs", total_size=8)

    with pytest.raises(ValueError, match="unscaled shard likelihoods"):
        fit_consensus_mc(
            model=total_size_model,
            split_data={"y": "obs"},
            num_shards=2,
            draws=2,
            tune=0,
            attach_data=False,
        )

    with pm.Model(coords={"obs": np.arange(4)}) as minibatch_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        y_mb = pm.Minibatch(np.arange(4.0), batch_size=2)
        theta = pm.Normal("theta")
        pm.Normal("mb_like", theta, 1, observed=y_mb, total_size=4)

    with pytest.raises(ValueError, match="unscaled shard likelihoods"):
        fit_consensus_mc(
            model=minibatch_model,
            split_data={"y": "obs"},
            num_shards=2,
            draws=2,
            tune=0,
            attach_data=False,
        )


def test_data_and_coords_are_restored_after_success_and_failure(monkeypatch):
    model = _model_with_data(4)
    original_y = model["y"].get_value().copy()
    original_coords = tuple(model.coords["obs"])
    _patch_sample(monkeypatch)

    fit_consensus_mc(
        model=model,
        split_data={"y": "obs"},
        num_shards=2,
        draws=2,
        sample_draws=2,
        tune=0,
        diagonal=True,
        attach_data=False,
        progressbar=False,
    )
    np.testing.assert_array_equal(model["y"].get_value(), original_y)
    assert tuple(model.coords["obs"]) == original_coords

    failing_model = _model_with_data(4)
    failing_y = failing_model["y"].get_value().copy()
    failing_coords = tuple(failing_model.coords["obs"])
    _patch_sample(monkeypatch, fail_on_call=2)
    with pytest.raises(RuntimeError, match="induced"):
        fit_consensus_mc(
            model=failing_model,
            split_data={"y": "obs"},
            num_shards=2,
            draws=2,
            sample_draws=2,
            tune=0,
            diagonal=True,
            attach_data=False,
            progressbar=False,
        )
    np.testing.assert_array_equal(failing_model["y"].get_value(), failing_y)
    assert tuple(failing_model.coords["obs"]) == failing_coords


def test_non_default_initval_is_rejected_before_clone(monkeypatch):
    model = _model_with_data(4)
    with pm.Model(coords={"obs": np.arange(4)}) as init_model:
        y = pm.Data("y", np.arange(4.0), dims="obs")
        theta = pm.Normal("theta", initval=1.0)
        pm.Normal("y_like", theta, 1, observed=y, dims="obs")

    def fail_clone(_model):
        raise AssertionError("clone_model should not be called")

    monkeypatch.setattr(cmc, "clone_model", fail_clone)
    with pytest.raises(ValueError, match="non-default initvals"):
        fit_consensus_mc(
            model=init_model,
            split_data={"y": "obs"},
            num_shards=2,
            draws=2,
            tune=0,
            attach_data=False,
        )
    assert model.free_RVs[0].name == "theta"


def test_multi_chain_draw_count_uses_requested_final_draws(monkeypatch):
    model = _model_with_data(4)
    _patch_sample(monkeypatch)

    idata = fit_consensus_mc(
        model=model,
        split_data={"y": "obs"},
        num_shards=2,
        draws=2,
        sample_draws=2,
        tune=0,
        chains=2,
        diagonal=True,
        attach_data=False,
        progressbar=False,
    )

    assert idata["posterior"].dataset.sizes["draw"] == 2


def test_constrained_support_warning_and_pmx_fit_smoke(monkeypatch):
    calls = []

    def fake_sample(*, draws, chains, model, **kwargs):
        calls.append(model["y"].get_value().copy())
        values = np.linspace(0.5, 1.5, chains * draws).reshape(chains, draws)
        return DataTree.from_dict({"posterior": xr.Dataset({"sigma": (("chain", "draw"), values)})})

    monkeypatch.setattr(pm, "sample", fake_sample)
    with pm.Model(coords={"obs": np.arange(4)}) as model:
        y = pm.Data("y", np.ones(4), dims="obs")
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y_like", 0, sigma, observed=y, dims="obs")

    with pytest.warns(UserWarning, match="merges constrained posterior samples"):
        idata = pmx.fit(
            method="consensus_mc",
            model=model,
            split_data={"y": "obs"},
            num_shards=2,
            draws=2,
            sample_draws=2,
            tune=0,
            diagonal=True,
            attach_data=False,
            progressbar=False,
        )

    assert idata["posterior"].dataset.sizes["draw"] == 2
    assert len(calls) == 2


def test_public_inference_import_smoke():
    from pymc_extras.inference import (
        estimate_parametric,
        fit_consensus_mc,
        merge_consensus,
        merge_parametric,
    )

    assert fit_consensus_mc.__name__ == "fit_consensus_mc"
    assert merge_consensus.__name__ == "merge_consensus"
    assert estimate_parametric.__name__ == "estimate_parametric"
    assert merge_parametric.__name__ == "merge_parametric"
