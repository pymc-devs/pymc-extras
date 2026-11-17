import numpy as np
import pymc as pm
import pytest

from xarray import DataTree

import pymc_extras as pmx


def test_find_MAP_is_deprecated_alias():
    with pm.Model():
        pm.Normal("mu", 3.0)
        with pytest.warns(FutureWarning, match="use `pymc.find_MAP`"):
            idata = pmx.find_MAP(progressbar=False)
    assert isinstance(idata, DataTree)
    np.testing.assert_allclose(idata.posterior["mu"].item(), 3.0, atol=1e-6)
