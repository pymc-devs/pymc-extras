import numpy as np
import pandas as pd
import pymc as pm
import pytest

from model.modular.utilities import at_least_list, encode_categoricals

from pymc_experimental.model.modular.components import Intercept, PoolingType, Regression, Spline


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


@pytest.mark.parametrize("pooling", ["partial", "none", "complete"], ids=str)
@pytest.mark.parametrize("prior", ["Normal", "Laplace", "StudentT"], ids=str)
@pytest.mark.parametrize(
    "feature_columns", ["income", ["age", "income"]], ids=["single", "multiple"]
)
def test_regression(pooling: PoolingType, prior, feature_columns, model):
    regression = Regression(
        name=None,
        feature_columns=feature_columns,
        prior=prior,
        pooling=pooling,
        pooling_columns="city",
    )

    temp_model = model.copy()
    xb = regression.build(temp_model)
    assert f"Regression({feature_columns})_features" in temp_model.coords.keys()

    if pooling != "complete":
        assert f"Regression({feature_columns})_city_effect" in temp_model.named_vars
        assert f"Regression({feature_columns})_city_effect_sigma" in temp_model.named_vars

        if pooling == "partial":
            assert (
                f"Regression({feature_columns})_city_effect_offset" in temp_model.named_vars_to_dims
            )
    else:
        assert f"Regression({feature_columns})" in temp_model.named_vars

    xb_val = xb.eval()

    X, beta = xb.owner.inputs[0].owner.inputs
    beta_val = beta.eval()
    n_features = len(at_least_list(feature_columns))

    if pooling != "complete":
        assert xb_val.shape[0] == len(model.coords["obs_idx"])
        assert np.unique(beta_val).shape[0] == len(model.coords["city"]) * n_features
    else:
        assert np.unique(beta_val).shape[0] == n_features


@pytest.mark.parametrize("pooling", ["partial", "none", "complete"], ids=str)
@pytest.mark.parametrize("prior", ["Normal", "Laplace", "StudentT"], ids=str)
def test_spline(pooling: PoolingType, prior, model):
    spline = Spline(
        name=None, feature_column="income", prior=prior, pooling=pooling, pooling_columns="city"
    )

    temp_model = model.copy()
    xb = spline.build(temp_model)

    assert "Spline(income, df=10, degree=3)_knots" in temp_model.coords.keys()
