import numpy as np
import pytest
import scipy
import pymc as pm

from pymc_extras import marginalize



def test_normal_normal():
    with pm.Model() as m:
        x = pm.Normal("x", mu=0, sigma=1)
        y = pm.Normal("y", mu=x + np.pi - 1, sigma=1.0)
        z = pm.Normal("z", mu=y + 2 * np.pi, sigma=np.sqrt(np.e))

    marginal_m = marginalize(m, m["y"])

    test_point = {"x": 1, "z": -1}

    np.testing.assert_allclose(
        marginal_m.compile_logp([marginal_m["z"]])(test_point),
        scipy.stats.norm.logpdf(test_point["z"], np.pi * 3, np.sqrt(1 + np.e))
    )

def test_normal_normal_does_not_apply():
    # If these cases become supported, the test should be repurposed

    with pm.Model() as m1:
        y = pm.Normal("y", mu=1)
        z = pm.Normal("z", mu=y * 2)

    with pytest.raises(RuntimeError, match="could not be derived"):
        marginalize(m1, y).logp()

    with pm.Model() as m2:
        y = pm.Normal("y", mu=1)
        z = pm.Normal("z", mu=y)
        w = pm.Normal("w", mu=y)

    with pytest.raises(RuntimeError, match="could not be derived"):
        marginalize(m2, y).logp()
