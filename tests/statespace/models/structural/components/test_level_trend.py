import numpy as np

from numpy.testing import assert_allclose
from pytensor import config

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


def test_level_trend_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1, atol=ATOL, rtol=RTOL)

    # Check coords
    mod = mod.build(verbose=False)
    _assert_basic_coords_correct(mod)
    assert mod.coords["trend_state"] == ["level", "trend"]
