import numpy as np
import pytest

from pymc_extras.inference.pathfinder.importance_sampling import importance_sampling


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "importance_sampling")))


def test_importance_sampling_none_returns_raw_samples(rng):
    samples = rng.normal(size=(4, 100, 3))
    logP = rng.normal(size=(4, 100))
    logQ = rng.normal(size=(4, 100))

    result = importance_sampling(samples, logP, logQ, num_draws=50, method=None)

    # method=None passes the per-path samples through untouched (num_draws is ignored)
    np.testing.assert_array_equal(result.samples, samples)
    assert result.method is None
    assert any("disabled" in w.lower() for w in result.warnings)


def test_importance_sampling_identity_shape_contract(rng):
    num_paths, M, N = 4, 100, 3
    samples = rng.normal(size=(num_paths, M, N))
    logP = rng.normal(size=(num_paths, M))
    logQ = rng.normal(size=(num_paths, M))

    result = importance_sampling(samples, logP, logQ, num_draws=50, method="identity")

    # Resampling collapses the path dimension: (L, M, N) -> (num_draws, N)
    assert result.samples.shape == (50, N)


def test_importance_sampling_falls_back_to_replacement(rng):
    # Only two samples carry finite logP, so there are fewer non-zero weights than requested
    # draws (but still fewer draws than the population) — this forces the with-replacement
    # (psir) fallback rather than the "larger than population" error.
    num_paths, M, N = 1, 20, 2
    samples = rng.normal(size=(num_paths, M, N))
    logP = np.full((num_paths, M), -np.inf)
    logP[0, :2] = [-1.0, -2.0]
    logQ = np.full((num_paths, M), -3.0)

    result = importance_sampling(
        samples, logP, logQ, num_draws=10, method="identity", random_seed=1
    )

    assert result.samples.shape == (10, N)
    assert any("psir" in w.lower() for w in result.warnings)


@pytest.mark.parametrize("method", ["psis", "psir"])
def test_importance_sampling_handles_nonfinite_weights(rng, method):
    # Out-of-support draws score logP = -inf and degenerate Gaussian logdets give non-finite logQ.
    # Both used to poison the Pareto fit (-logiw = +inf) and yield NaN resampling probabilities,
    # crashing rng.choice. They must instead get zero weight and leave a valid resample.
    num_paths, M, N = 2, 100, 3
    samples = rng.normal(size=(num_paths, M, N))
    logP = rng.normal(size=(num_paths, M))
    logQ = rng.normal(size=(num_paths, M))
    logP[0, :30] = -np.inf  # out-of-support draws
    logQ[1, :10] = np.nan  # degenerate proposal density
    logQ[1, 10:20] = -np.inf

    result = importance_sampling(samples, logP, logQ, num_draws=50, method=method, random_seed=1)

    assert result.samples.shape == (50, N)
    assert np.isfinite(result.samples).all()


def test_importance_sampling_all_nonfinite_raises(rng):
    samples = rng.normal(size=(2, 50, 3))
    logP = np.full((2, 50), -np.inf)
    logQ = rng.normal(size=(2, 50))

    with pytest.raises(ValueError, match="non-finite"):
        importance_sampling(samples, logP, logQ, num_draws=10, method="psis")


def test_importance_sampling_does_not_mutate_inputs(rng):
    # The num_paths mixture adjustment must not mutate the caller's logP/logQ in place -- those
    # arrays remain the result's user-facing diagnostics.
    num_paths, M, N = 3, 40, 2
    samples = rng.normal(size=(num_paths, M, N))
    logP = rng.normal(size=(num_paths, M))
    logQ = rng.normal(size=(num_paths, M))
    logP_before, logQ_before = logP.copy(), logQ.copy()

    importance_sampling(samples, logP, logQ, num_draws=50, method="psis", random_seed=1)

    np.testing.assert_array_equal(logP, logP_before)
    np.testing.assert_array_equal(logQ, logQ_before)


def test_importance_sampling_oversized_num_draws_falls_back(rng):
    # num_draws exceeds the pooled population, so replace=False cannot draw that many distinct rows
    # and numpy raises "larger sample than population"; the fallback must retry with replacement.
    num_paths, M, N = 1, 5, 2
    samples = rng.normal(size=(num_paths, M, N))
    logP = rng.normal(size=(num_paths, M))
    logQ = rng.normal(size=(num_paths, M))

    result = importance_sampling(
        samples, logP, logQ, num_draws=20, method="identity", random_seed=1
    )

    assert result.samples.shape == (20, N)  # pool is only 5 -> required the replacement fallback
    assert any("psir" in w.lower() for w in result.warnings)
