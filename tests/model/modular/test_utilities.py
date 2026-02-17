import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

from numpy.testing import assert_allclose

from pymc_experimental.model.modular.utilities import (
    encode_categoricals,
    make_level_maps,
    make_next_level_hierarchy_variable,
    make_partial_pooled_hierarchy,
    make_unpooled_hierarchy,
    select_data_columns,
)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def df(rng):
    N = 1000

    level_0_labels = ["Beverage", "Snack"]
    level_1_labels = {
        "Beverage": ["Soft Drinks", "Milk", "Smoothies", "Sports Drinks", "Alcoholic Beverages"],
        "Snack": ["Jerky", "Pretzels", "Nuts", "Tea"],
    }
    level_2_labels = {
        "Soft Drinks": ["Lemonade", "Cola", "Root Beer", "Ginger Ale"],
        "Milk": ["Oat Milk", "Cow Milk", "Soy Milk"],
        "Smoothies": ["Green Smoothies", "Berry Smoothies"],
        "Sports Drinks": ["Gatorade", "Powerade"],
        "Alcoholic Beverages": ["Beer", "Wine", "Spirits"],
        "Jerky": ["Vegan Jerky", "Beef Jerky"],
        "Pretzels": ["Salted Pretzels", "Unsalted Pretzels"],
        "Nuts": ["Peanuts", "Almonds", "Cashews", "Pistachios", "Walnuts"],
        "Tea": ["Black Tea", "Green Tea", "Herbal Tea"],
    }

    level_0_data = rng.choice(level_0_labels, N)
    level_1_data = [rng.choice(level_1_labels[level_0]) for level_0 in level_0_data]
    level_2_data = [rng.choice(level_2_labels[level_1]) for level_1 in level_1_data]

    df = pd.DataFrame(
        {
            "level_0": level_0_data,
            "level_1": level_1_data,
            "level_2": level_2_data,
            "A": rng.normal(size=N),
            "B": rng.normal(size=N),
            "C": rng.normal(size=N),
            "sales": rng.normal(size=N),
        }
    )

    return df


@pytest.fixture(scope="session")
def encoded_df_and_coords(df):
    return encode_categoricals(df, {"feature": df.columns.tolist(), "obs_idx": df.index.tolist()})


@pytest.fixture(scope="session")
def model(encoded_df_and_coords, rng):
    df, coords = encoded_df_and_coords

    with pm.Model(coords=coords) as m:
        X = pm.Data("X", df[coords["feature"]], dims=["obs_idx", "features"])

    return m


@pytest.mark.parametrize("cols", ["A", ["A", "B"]], ids=["single", "multiple"])
def test_select_data_columns(model, cols):
    col = select_data_columns(cols, model, data_name="X")

    idxs = [model.coords["feature"].index(col) for col in cols]
    assert_allclose(col.eval(), model["X"].get_value()[:, idxs].squeeze())


def test_select_missing_column_raises(model):
    with pytest.raises(ValueError):
        select_data_columns("D", model, data_name="X")


def test_make_level_maps(model, encoded_df_and_coords, df):
    df_encoded, coords = encoded_df_and_coords
    data = pytensor.shared(df_encoded.values)

    level_maps = make_level_maps(data, coords, ordered_levels=["level_0", "level_1", "level_2"])

    level_maps = [x.eval() for x in level_maps[1:]]
    m0, m1, m2 = level_maps

    # Rebuild the labels from the level maps
    new_labels = np.array(coords["level_0"])[m0]
    new_labels = np.array([x + "_" + y for x, y in zip(new_labels, coords["level_1"])])[m1]
    new_labels = np.array([x + "_" + y for x, y in zip(new_labels, coords["level_2"])])[m2]

    new_labels = pd.Series(new_labels).apply(
        lambda x: pd.Series(x.split("_"), index=["level_0", "level_1", "level_2"])
    )

    pd.testing.assert_frame_equal(new_labels, df[["level_0", "level_1", "level_2"]])


def test_make_simple_hierarchical_variable():
    with pm.Model(coords={"level_0": ["A", "B", "C"]}) as m:
        mapping = np.random.choice(3, size=(10,))
        effect_mu = pm.Normal("effect_mu")
        mu = make_next_level_hierarchy_variable(
            "effect", mu=effect_mu, offset_dims="level_0", sigma_dims=None, mapping=None
        )
        mu = mu[mapping]

    expected_names = ["effect_mu", "effect_sigma", "effect_offset"]
    assert all([x.name in expected_names for x in m.free_RVs])


def test_make_two_level_hierarchical_variable():
    with pm.Model(coords={"level_0": ["A", "B", "C"], "level_1": range(9)}) as m:
        mapping = np.random.choice(3, size=(10,))

        level_0_mu = pm.Normal("level_0_effect_mu")
        mu = make_next_level_hierarchy_variable(
            "level_0_effect", mu=level_0_mu, offset_dims="level_0", sigma_dims=None, mapping=None
        )

        mu = make_next_level_hierarchy_variable(
            "level_1_effect", mu=mu, offset_dims="level_1", sigma_dims="level_0", mapping=mapping
        )

    expected_names = [
        "level_0_effect_mu",
        "level_0_effect_sigma",
        "level_0_effect_offset",
        "level_0_effect",
        "level_1_effect_sigma",
        "level_1_effect_offset",
    ]
    assert all([x.name in expected_names for x in m.free_RVs])

    m0, s0, o0, m1, s1, o1 = (m[name] for name in expected_names)
    assert m1.shape.eval() == (3,)
    assert s1.shape.eval() == (3,)
    assert o1.shape.eval() == (9,)


def test_make_next_level_no_pooling():
    data = pd.DataFrame({"level_0": np.random.choice(["A", "B", "C"], size=(10,))})
    data, coords = encode_categoricals(data, {"feature": data.columns, "obs_idx": data.index})

    with pm.Model(coords=coords) as m:
        X = pm.Data("X_data", data, dims=["obs_idx", "feature"])
        mu = make_unpooled_hierarchy(
            "effect",
            X=X,
            levels=["level_0"],
            model=m,
            sigma_dims=None,
        )

    assert "effect_sigma" not in m.named_vars
    assert "effect_offset" not in m.named_vars
    assert mu.shape.eval() == (10,)


@pytest.mark.parametrize(
    "pooling_columns", [["level_0"], ["level_0", "level_1"], ["level_0", "level_1", "level_2"]]
)
def test_hierarchical_prior_to_requested_depth(model, encoded_df_and_coords, pooling_columns):
    temp_model = model.copy()
    with temp_model:
        intercept = make_partial_pooled_hierarchy(
            name="intercept", X=model["X"], pooling_columns=pooling_columns, model=temp_model
        )

    intercept = intercept.eval()
    assert intercept.shape[0] == len(model.coords["obs_idx"])
    assert len(np.unique(intercept)) == len(model.coords[pooling_columns[-1]])
