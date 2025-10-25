import numpy as np
import pymc as pm

import pymc_extras as pmx


def test_logp():
    """Compare standard PyMC `with pm.Model()` context API against `pmx.model` decorator
    and a functional syntax. Checks whether the kwarg `coords` can be passed.
    """
    coords = {"obs": ["a", "b"]}

    with pm.Model(coords=coords) as model:
        pm.Normal("x", 0.0, 1.0, dims="obs")

    @pmx.as_model(coords=coords)
    def model_wrapped():
        pm.Normal("x", 0.0, 1.0, dims="obs")

    mw = model_wrapped()

    @pmx.as_model()
    def model_wrapped2():
        pm.Normal("x", 0.0, 1.0, dims="obs")

    mw2 = model_wrapped2(coords=coords)

    @pmx.as_model()
    def model_wrapped3(mu):
        pm.Normal("x", mu, 1.0, dims="obs")

    mw3 = model_wrapped3(0.0, coords=coords)
    mw4 = model_wrapped3(np.array([np.nan]), coords=coords)

    np.testing.assert_equal(model.point_logps(), mw.point_logps())
    np.testing.assert_equal(mw.point_logps(), mw2.point_logps())
    assert mw3["mu"] in mw3.data_vars
    assert "mu" not in mw4
