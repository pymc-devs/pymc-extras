import numpy as np
import pandas as pd
import pymc as pm
import pytest

from model.modular.utilities import encode_categoricals

from pymc_experimental.model.modular.components import Intercept, PoolingType


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def model(rng):
    city = ["A", "B", "C"]
    race = ["white", "black", "hispanic"]

    df = pd.DataFrame(
        {
            "city": np.random.choice(city, 1000),
            "age": rng.normal(size=1000),
            "race": rng.choice(race, size=1000),
            "income": rng.normal(size=1000),
        }
    )

    coords = {"feature": df.columns, "obs_idx": df.index}

    df, coords = encode_categoricals(df, coords)

    with pm.Model(coords=coords) as m:
        X = pm.Data("X_data", df, dims=["obs_idx", "features"])

    return m


@pytest.mark.parametrize("pooling", ["partial", "none", "complete"], ids=str)
@pytest.mark.parametrize("prior", ["Normal", "Laplace", "StudentT"], ids=str)
def test_intercept(pooling: PoolingType, prior, model):
    intercept = Intercept(name=None, pooling=pooling, pooling_columns="city", prior=prior)

    x = intercept.build(model.copy()).eval()

    if pooling != "complete":
        assert x.shape[0] == len(model.coords["obs_idx"])
        assert np.unique(x).shape[0] == len(model.coords["city"])
    else:
        assert np.unique(x).shape[0] == 1


def test_regression():
    pass
