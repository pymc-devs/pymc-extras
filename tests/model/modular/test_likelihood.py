import numpy as np
import pandas as pd
import pytest

from pymc_experimental.model.modular.likelihood import NormalLikelihood


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def data(rng):
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
    return df


def test_normal_likelihood(data):
    model = NormalLikelihood(mu=None, sigma=None, target_col="income", data=data)
    idata = model.sample_prior_predictive()
