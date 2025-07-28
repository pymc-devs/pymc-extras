import numpy as np

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct


def test_measurement_error(rng):
    mod = st.MeasurementError("obs") + st.LevelTrendComponent(order=2)
    mod = mod.build(verbose=False)

    _assert_basic_coords_correct(mod)
    assert "sigma_obs" in mod.param_names


def test_measurement_error_multiple_observed():
    mod = st.MeasurementError("obs", observed_state_names=["data_1", "data_2"])
    assert mod.k_endog == 2
    assert mod.coords["endog_obs"] == ["data_1", "data_2"]
    assert mod.param_dims["sigma_obs"] == ("endog_obs",)


def test_build_with_measurement_error_subset():
    ll = st.LevelTrendComponent(order=2, observed_state_names=["data_1", "data_2", "data_3"])
    me = st.MeasurementError("obs", observed_state_names=["data_1", "data_3"])
    mod = (ll + me).build()

    H = mod.ssm["obs_cov"]
    assert H.type.shape == (3, 3)
    np.testing.assert_allclose(
        H.eval({"sigma_obs": [1.0, 3.0]}),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
    )
