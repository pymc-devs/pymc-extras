import numpy as np
import pytest

from numpy.testing import assert_allclose
from pytensor import config

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
    ar = st.AutoregressiveComponent(order=order)
    params = {
        "ar_params": np.full((sum(ar.order),), 0.5, dtype=config.floatX),
        "sigma_ar": 0.0,
    }

    x, y = simulate_from_numpy_model(ar, rng, params, steps=100)

    # Check coords
    ar.build(verbose=False)
    _assert_basic_coords_correct(ar)
    lags = np.arange(len(order) if isinstance(order, list) else order, dtype="int") + 1
    if isinstance(order, list):
        lags = lags[np.flatnonzero(order)]
    assert_allclose(ar.coords["ar_lag"], lags)
