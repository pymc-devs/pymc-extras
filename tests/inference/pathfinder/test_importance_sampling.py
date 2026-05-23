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
