import numpy as np
import pytest

from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)

TEST_SEED = sum(map(ord, "Structural Statespace"))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(TEST_SEED)


def _assert_basic_coords_correct(mod):
    assert mod.coords[ALL_STATE_DIM] == mod.state_names
    assert mod.coords[ALL_STATE_AUX_DIM] == mod.state_names
    assert mod.coords[SHOCK_DIM] == mod.shock_names
    assert mod.coords[SHOCK_AUX_DIM] == mod.shock_names
    expected_obs = mod.observed_state_names if hasattr(mod, "observed_state_names") else ["data"]

    assert mod.coords[OBS_STATE_DIM] == expected_obs
    assert mod.coords[OBS_STATE_AUX_DIM] == expected_obs
