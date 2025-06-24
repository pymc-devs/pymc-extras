from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct


def test_measurement_error(rng):
    mod = st.MeasurementError("obs") + st.LevelTrendComponent(order=2)
    mod = mod.build(verbose=False)

    _assert_basic_coords_correct(mod)
    assert "sigma_obs" in mod.param_names
