import numpy as np
import pytensor

from pytensor.graph.basic import explicit_graph_inputs

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


def test_measurement_error_share_states():
    mod = st.MeasurementError("obs", observed_state_names=["data_1", "data_2"], share_states=True)
    mod.build(verbose=False)

    assert mod.k_endog == 2
    assert mod.param_names == ["sigma_obs", "P0"]
    assert "endog_obs" not in mod.coords

    # Check that the parameter is shared across the observed states
    assert mod.param_info["sigma_obs"]["shape"] == ()

    outputs = mod.ssm["obs_cov"]

    H = pytensor.function(list(explicit_graph_inputs([outputs])), outputs)(sigma_obs=np.array(0.5))
    np.testing.assert_allclose(H, np.diag([0.5, 0.5]) ** 2)


def test_measurement_error_shared_and_not_shared():
    shared = st.MeasurementError(
        "error_shared", observed_state_names=["data_1", "data_2"], share_states=True
    )
    individual = st.MeasurementError("error_individual", observed_state_names=["data_1", "data_2"])
    mod = (shared + individual).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.param_names == ["sigma_error_shared", "sigma_error_individual", "P0"]
    assert mod.coords["endog_error_individual"] == ["data_1", "data_2"]

    assert mod.param_info["sigma_error_shared"]["shape"] == ()
    assert mod.param_info["sigma_error_individual"]["shape"] == (2,)

    outputs = mod.ssm["obs_cov"]

    H = pytensor.function(list(explicit_graph_inputs([outputs])), outputs)(
        sigma_error_shared=np.array(0.5), sigma_error_individual=np.array([0.1, 0.9])
    )
    np.testing.assert_allclose(H, np.diag([0.5, 0.5]) ** 2 + np.diag([0.1, 0.9]) ** 2)


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
