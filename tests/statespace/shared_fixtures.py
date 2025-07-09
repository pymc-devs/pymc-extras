import numpy as np
import pytest

TEST_SEED = sum(map(ord, "statespace"))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(TEST_SEED)
