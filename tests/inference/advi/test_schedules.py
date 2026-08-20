import numpy as np

from pymc_extras.inference.advi.schedules import linear_onecycle_schedule


def test_linear_onecycle_schedule_shape():
    schedule = linear_onecycle_schedule(
        transition_steps=1000, peak_value=0.01, pct_start=0.2, div_factor=25.0
    )

    np.testing.assert_allclose(schedule(0), 0.01 / 25)
    np.testing.assert_allclose(schedule(200), 0.01)  # peak at pct_start
    assert schedule(100) < schedule(200)  # warmup
    assert schedule(500) < schedule(200)  # anneal
    assert schedule(1000) < schedule(0)  # final decay below init
