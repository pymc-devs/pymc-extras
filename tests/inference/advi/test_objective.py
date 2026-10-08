import numpy as np
import pymc as pm
import pytensor

from pymc_extras.inference.advi.autoguide import AutoDiagonalNormal
from pymc_extras.inference.advi.objective import get_logp_logq


def test_logp_scaling_keeps_the_logp_dtype():
    with pytensor.config.change_flags(floatX="float32"):
        with pm.Model() as model:
            theta = pm.Normal("theta", mu=0, sigma=1)
            y = pm.Normal("y", mu=theta, sigma=1, observed=np.ones(4, dtype="float32"))
        logp, _ = get_logp_logq(model, AutoDiagonalNormal(model), logp_scalings={y: 250.0})

    assert logp.dtype == "float32"
